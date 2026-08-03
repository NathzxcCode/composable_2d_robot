import gtsam
import numpy as np
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from gtsam_examples.gtsam_factors import make_fixed_kinematics_factor
from gtsam_gbp import GBPOptimizer, GBPParams

# Position sigma of 0.05 m keeps the arm chain well-connected while keeping the
# information within ~3 orders of magnitude of the dynamics factors (~1), which
# is required for GBP message passing to stay numerically stable.
# (1e-4 sigmas give 1e8 information, causing 9-OOM spread that breaks GBP.)
KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.02]))
ANCHOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.02]))
LOOSE_ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 10.0]))
GOAL_NOISE         = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.06, 0.06, 1000.0]))


def _J(i, t):  return gtsam.Symbol('j', i * 1000 + t).key()
def _E(i, t):  return gtsam.Symbol('e', i * 1000 + t).key()
def _V(i, t):  return gtsam.Symbol('o', i * 1000 + t).key()
def _VE(i, t): return gtsam.Symbol('v', i * 1000 + t).key()


def _make_task_space_dynamics_factor(key_p1, key_v1, key_p2, key_v2, dt, sigma):
    I = np.eye(2)
    Qc_inv = (sigma ** -2.0) * I
    Qi_inv = np.zeros((4, 4))
    Qi_inv[0:2, 0:2] =  12.0 * (dt ** -3.0) * Qc_inv
    Qi_inv[0:2, 2:4] =  -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[2:4, 0:2] =  -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[2:4, 2:4] =   4.0 / dt           * Qc_inv
    noise_model = gtsam.noiseModel.Gaussian.Information(Qi_inv)

    def error_func(this, values, jacobians):
        p1 = values.atPose2(this.keys()[0])
        v1 = values.atVector(this.keys()[1])
        p2 = values.atPose2(this.keys()[2])
        v2 = values.atVector(this.keys()[3])

        def z(r, c): return np.zeros((r, c), order='F')
        H_t1 = z(2, 3); H_t2 = z(2, 3)
        pos1 = p1.translation(H_t1)
        pos2 = p2.translation(H_t2)

        error = np.hstack((pos1 + v1 * dt - pos2, v1 - v2))

        if jacobians is not None:
            J1 = z(4, 3); J1[0:2, :] = H_t1
            J2 = z(4, 2); J2[0:2, :] = I * dt; J2[2:4, :] = I
            J3 = z(4, 3); J3[0:2, :] = -H_t2
            J4 = z(4, 2); J4[2:4, :] = -I
            jacobians[0] = J1; jacobians[1] = J2
            jacobians[2] = J3; jacobians[3] = J4
        return error

    keys = gtsam.KeyVector()
    for k in [key_p1, key_v1, key_p2, key_v2]: keys.append(k)
    return gtsam.CustomFactor(noise_model, keys, error_func)


def _make_joint_space_dynamics_factor(key_p1, key_v1, key_p2, key_v2, dt, sigma):
    Qc_inv = sigma ** -2.0
    Qi_inv = np.zeros((2, 2))
    Qi_inv[0, 0] =  12.0 * (dt ** -3.0) * Qc_inv
    Qi_inv[0, 1] =  -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[1, 0] =  -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[1, 1] =   4.0 / dt           * Qc_inv
    noise_model = gtsam.noiseModel.Gaussian.Information(Qi_inv)

    def error_func(this, values, jacobians):
        p1 = values.atPose2(this.keys()[0])
        v1 = values.atVector(this.keys()[1])[0]
        p2 = values.atPose2(this.keys()[2])
        v2 = values.atVector(this.keys()[3])[0]

        raw = p1.theta() + v1 * dt - p2.theta()
        error = np.array([np.arctan2(np.sin(raw), np.cos(raw)), v1 - v2])

        if jacobians is not None:
            def z(r, c): return np.zeros((r, c), order='F')
            J1 = z(2, 3); J1[0, 2] =  1.0
            J2 = z(2, 1); J2[0, 0] =  dt; J2[1, 0] = 1.0
            J3 = z(2, 3); J3[0, 2] = -1.0
            J4 = z(2, 1); J4[1, 0] = -1.0
            jacobians[0] = J1; jacobians[1] = J2
            jacobians[2] = J3; jacobians[3] = J4
        return error

    keys = gtsam.KeyVector()
    for k in [key_p1, key_v1, key_p2, key_v2]: keys.append(k)
    return gtsam.CustomFactor(noise_model, keys, error_func)


class PlanningGraph:
    """
    Receding-horizon motion planner over a fixed-topology GTSAM factor graph.

    Each call to centralised_solve(current_qpos, joint_poses):
      1. Updates k=0 joint variable values from ground-truth MuJoCo poses.
      2. Replaces the k=0 root anchor and inter-joint kinematic factors with
         the measured joint angles, firmly grounding the current arm state.
      3. Runs LM optimisation.
      4. Shifts all values one timestep forward (receding window).
      5. Returns planned relative joint angles for the next timestep.
    """

    def __init__(self, limbs, time_horizon=5, dt=0.5,
                 goal_xy=(0.3, 0.2),
                 sigma_endpoint=10.0, sigma_joint=1.0):
        self.num_limbs    = len(limbs)
        self.limb_lengths = [l.length for l in limbs]
        self.time_horizon = time_horizon
        self.dt           = dt

        self.graph    = gtsam.NonlinearFactorGraph()
        self.initial  = gtsam.Values()
        self.params   = gtsam.LevenbergMarquardtParams()
        self._dof_map = {}  # key -> tangent-space DOF, used by GBPOptimizer

        self._k0_root_idx   = None
        self._k0_kin_indices = []

        # Cumulative X positions for straight-arm initialisation
        x_positions = [0.0]
        for L in self.limb_lengths:
            x_positions.append(x_positions[-1] + L)

        # --- Build one kinematic chain per timestep ---
        for k in range(time_horizon):
            anchor_noise = ANCHOR_NOISE if k == 0 else LOOSE_ANCHOR_NOISE
            self.graph.add(gtsam.PriorFactorPose2(
                _J(1, k), gtsam.Pose2(0.0, 0.0, 0.0), anchor_noise))
            if k == 0:
                self._k0_root_idx = self.graph.size() - 1

            for i in range(1, self.num_limbs):
                self.graph.add(make_fixed_kinematics_factor(
                    _J(i, k), _J(i + 1, k),
                    self.limb_lengths[i - 1], 0.0, LOOSE_ANCHOR_NOISE))
                if k == 0:
                    self._k0_kin_indices.append(self.graph.size() - 1)

            self.graph.add(make_fixed_kinematics_factor(
                _J(self.num_limbs, k), _E(self.num_limbs, k),
                self.limb_lengths[-1], 0.0, KINEMATIC_NOISE))

            for i in range(1, self.num_limbs + 1):
                self.initial.insert(_J(i, k), gtsam.Pose2(x_positions[i - 1], 0.0, 0.0))
                self.initial.insert(_V(i, k), np.array([0.0]))
                self._dof_map[_J(i, k)] = 3
                self._dof_map[_V(i, k)] = 1
            self.initial.insert(_E(self.num_limbs, k),
                                gtsam.Pose2(x_positions[self.num_limbs], 0.0, 0.0))
            self.initial.insert(_VE(self.num_limbs, k), np.array([0.0, 0.0]))
            self._dof_map[_E(self.num_limbs, k)] = 3
            self._dof_map[_VE(self.num_limbs, k)] = 2

        # --- Connect chains across timesteps via dynamics ---
        for k in range(time_horizon - 1):
            self.graph.add(_make_task_space_dynamics_factor(
                _E(self.num_limbs, k),   _VE(self.num_limbs, k),
                _E(self.num_limbs, k+1), _VE(self.num_limbs, k+1),
                dt, sigma_endpoint))
            for i in range(1, self.num_limbs + 1):
                self.graph.add(_make_joint_space_dynamics_factor(
                    _J(i, k), _V(i, k), _J(i, k+1), _V(i, k+1),
                    dt, sigma_joint))

        # Goal prior on the horizon endpoint
        self.graph.add(gtsam.PriorFactorPose2(
            _E(self.num_limbs, time_horizon - 1),
            gtsam.Pose2(goal_xy[0], goal_xy[1], 0.0),
            GOAL_NOISE))

    def _pre_solve(self, current_qpos: np.ndarray, joint_poses: np.ndarray) -> None:
        """Update k=0 variable values and replace k=0 factors with measured state."""
        for i, xyt in enumerate(joint_poses):
            self.initial.update(_J(i + 1, 0),
                                gtsam.Pose2(float(xyt[0]), float(xyt[1]), float(xyt[2])))

        last = joint_poses[-1]
        last_pose = gtsam.Pose2(float(last[0]), float(last[1]), float(last[2]))
        end_pose  = last_pose.compose(gtsam.Pose2(self.limb_lengths[-1], 0.0, 0.0))
        self.initial.update(_E(self.num_limbs, 0), end_pose)

        self.graph.replace(self._k0_root_idx,
            gtsam.PriorFactorPose2(
                _J(1, 0),
                gtsam.Pose2(float(joint_poses[0][0]),
                            float(joint_poses[0][1]),
                            float(joint_poses[0][2])),
                ANCHOR_NOISE))

        for graph_idx, i in zip(self._k0_kin_indices, range(1, self.num_limbs)):
            self.graph.replace(graph_idx,
                make_fixed_kinematics_factor(
                    _J(i, 0), _J(i + 1, 0),
                    self.limb_lengths[i - 1],
                    float(current_qpos[i]),
                    KINEMATIC_NOISE))

    def _post_solve(self, result: gtsam.Values) -> np.ndarray:
        """Shift the receding window forward and extract relative joint angles at k=1."""
        for k in range(self.time_horizon):
            next_k = k + 1 if k < self.time_horizon - 1 else k
            for i in range(1, self.num_limbs + 1):
                self.initial.update(_J(i, k), result.atPose2(_J(i, next_k)))
                self.initial.update(_V(i, k), result.atVector(_V(i, next_k)))
            self.initial.update(_E(self.num_limbs, k),
                                result.atPose2(_E(self.num_limbs, next_k)))
            self.initial.update(_VE(self.num_limbs, k),
                                result.atVector(_VE(self.num_limbs, next_k)))

        thetas = [result.atPose2(_J(i + 1, 1)).theta() for i in range(self.num_limbs)]
        ctrl = np.zeros(self.num_limbs)
        ctrl[0] = thetas[0]
        for i in range(1, self.num_limbs):
            ctrl[i] = thetas[i] - thetas[i - 1]
        return ctrl

    def centralised_solve(self, current_qpos: np.ndarray,
                          joint_poses: np.ndarray) -> np.ndarray:
        """
        current_qpos : (num_limbs,) relative joint angles from encoders.
        joint_poses  : (num_limbs, 3) ground-truth [x, y, theta] per joint.
        """
        self._pre_solve(current_qpos, joint_poses)
        result = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial, self.params).optimize()
        return self._post_solve(result)

    def gbp_solve(self, current_qpos: np.ndarray,
                  joint_poses: np.ndarray,
                  n_outer: int = 5, n_inner: int = 10,
                  damping: float = 0.0) -> np.ndarray:
        """GBP variant — drop-in replacement for centralised_solve."""
        self._pre_solve(current_qpos, joint_poses)
        result = GBPOptimizer(
            self.graph, self.initial,
            GBPParams(n_outer=n_outer, n_inner=n_inner, damping=damping),
            dof_map=self._dof_map,
        ).optimize()
        return self._post_solve(result)
