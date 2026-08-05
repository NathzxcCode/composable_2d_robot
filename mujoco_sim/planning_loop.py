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
KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.005, 0.005, 0.02]))
ANCHOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.005, 0.005, 0.02]))
LOOSE_ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.005, 0.005, 10.0]))
GOAL_NOISE         = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.06, 0.06, 1000.0]))


# Keys encode robot_id * 100000 + limb_index * 1000 + timestep
def _J(r, i, t):  return gtsam.Symbol('j', r * 100000 + i * 1000 + t).key()
def _E(r, i, t):  return gtsam.Symbol('e', r * 100000 + i * 1000 + t).key()
def _V(r, i, t):  return gtsam.Symbol('o', r * 100000 + i * 1000 + t).key()
def _VE(r, i, t): return gtsam.Symbol('v', r * 100000 + i * 1000 + t).key()


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


def _make_ellipsoid_collision_factor(key_A1, key_A2, key_B1, key_B2,
                                     k=4.0, r=0.03, cost_sigma=0.1):
    """
    Smooth self-collision avoidance factor between two limb segments.
    Each segment is defined by two Pose2 endpoint keys (start, end).
    Uses Bhattacharyya distance between ellipsoid representations of the segments.
    Only add between non-adjacent limbs (adjacent limbs share a joint endpoint).
    """
    Qi_inv = np.array([[cost_sigma ** -2.0]])
    noise_model = gtsam.noiseModel.Gaussian.Information(Qi_inv)

    def error_func(this, values, jacobians):
        pA1 = values.atPose2(this.keys()[0])
        pA2 = values.atPose2(this.keys()[1])
        pB1 = values.atPose2(this.keys()[2])
        pB2 = values.atPose2(this.keys()[3])

        def z(rows, cols): return np.zeros((rows, cols), order='F')
        H_tA1 = z(2, 3); H_tA2 = z(2, 3)
        H_tB1 = z(2, 3); H_tB2 = z(2, 3)
        xA1 = pA1.translation(H_tA1); xA2 = pA2.translation(H_tA2)
        xB1 = pB1.translation(H_tB1); xB2 = pB2.translation(H_tB2)

        mu_A = 0.5 * (xA1 + xA2);  mu_B = 0.5 * (xB1 + xB2)
        v_A  = xA2 - xA1;          v_B  = xB2 - xB1
        Sigma_A = 0.25 * np.outer(v_A, v_A) + (r**2) * np.eye(2)
        Sigma_B = 0.25 * np.outer(v_B, v_B) + (r**2) * np.eye(2)
        Sigma   = 0.5 * (Sigma_A + Sigma_B)
        Sigma_inv = np.linalg.inv(Sigma)

        d_mu       = mu_A - mu_B
        mahalanobis = 0.125 * np.dot(d_mu, np.dot(Sigma_inv, d_mu))
        shape_term  = 0.5 * np.log(np.linalg.det(Sigma) /
                                    np.sqrt(np.linalg.det(Sigma_A) * np.linalg.det(Sigma_B)))
        D_B   = mahalanobis + shape_term
        error = np.array([np.exp(-k * D_B)])

        if jacobians is not None:
            dE_dDB   = -k * error[0]
            dDB_dmuA = 0.25 * np.dot(Sigma_inv, d_mu)
            dDB_dmuB = -dDB_dmuA
            dE_dxA1_pos = dE_dDB * 0.5 * dDB_dmuA
            dE_dxA2_pos = dE_dDB * 0.5 * dDB_dmuA
            dE_dxB1_pos = dE_dDB * 0.5 * dDB_dmuB
            dE_dxB2_pos = dE_dDB * 0.5 * dDB_dmuB
            dDB_dSigma  = (-0.125 * np.outer(np.dot(Sigma_inv, d_mu),
                                              np.dot(Sigma_inv, d_mu))
                           + 0.5 * Sigma_inv)
            dE_dSigma   = dE_dDB * dDB_dSigma
            dE_dxA2_cov = np.zeros(2); dE_dxB2_cov = np.zeros(2)
            for idx in range(2):
                M_A = np.zeros((2, 2)); M_A[idx, :] += v_A; M_A[:, idx] += v_A
                dE_dxA2_cov[idx] = np.sum(dE_dSigma * (0.5 * 0.25 * M_A))
                M_B = np.zeros((2, 2)); M_B[idx, :] += v_B; M_B[:, idx] += v_B
                dE_dxB2_cov[idx] = np.sum(dE_dSigma * (0.5 * 0.25 * M_B))
            jacobians[0] = np.dot(dE_dxA1_pos - dE_dxA2_cov, H_tA1).reshape(1, 3)
            jacobians[1] = np.dot(dE_dxA2_pos + dE_dxA2_cov, H_tA2).reshape(1, 3)
            jacobians[2] = np.dot(dE_dxB1_pos - dE_dxB2_cov, H_tB1).reshape(1, 3)
            jacobians[3] = np.dot(dE_dxB2_pos + dE_dxB2_cov, H_tB2).reshape(1, 3)
        return error

    keys = gtsam.KeyVector()
    for key in [key_A1, key_A2, key_B1, key_B2]: keys.append(key)
    return gtsam.CustomFactor(noise_model, keys, error_func)


class PlanningGraph:
    """
    Receding-horizon motion planner supporting one or more robots in a shared
    factor graph.

    robots  : list of LimbSpec lists, one per robot.
    goals   : list of (x, y) goal positions, one per robot.

    centralised_solve(robot_states) and gbp_solve(robot_states, ...)
    both accept robot_states = [(qpos_0, joint_poses_0), ...] and return
    a list of ctrl arrays, one per robot.
    """

    def __init__(self, robots, goals,
                 time_horizon=5, dt=0.5,
                 sigma_endpoint=10.0, sigma_joint=1.0,
                 enable_collision_avoidance=False,
                 collision_radius=0.03, collision_k=4.0, collision_sigma=0.1):

        self.time_horizon = time_horizon
        self.dt           = dt
        self._num_robots  = len(robots)

        self.graph   = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.params  = gtsam.LevenbergMarquardtParams()
        self._dof_map = {}

        # Per-robot state (keyed by robot_id int)
        self._num_limbs      = {}
        self.limb_lengths    = {}
        self._k0_root_idx    = {}
        self._k0_kin_indices = {}
        self._goal_idx       = {}

        for robot_id, (limbs, goal_xy) in enumerate(zip(robots, goals)):
            self._setup_robot(robot_id, limbs, goal_xy, sigma_endpoint, sigma_joint)

        # Inter-robot collision avoidance (requires at least 2 robots)
        if enable_collision_avoidance and self._num_robots > 1:
            for r_a in range(self._num_robots):
                for r_b in range(r_a + 1, self._num_robots):
                    n_a = self._num_limbs[r_a]
                    n_b = self._num_limbs[r_b]
                    for k in range(1, time_horizon):
                        for i in range(1, n_a + 1):
                            a1 = _J(r_a, i, k)
                            a2 = _J(r_a, i + 1, k) if i < n_a else _E(r_a, n_a, k)
                            for j in range(1, n_b + 1):
                                b1 = _J(r_b, j, k)
                                b2 = _J(r_b, j + 1, k) if j < n_b else _E(r_b, n_b, k)
                                self.graph.add(_make_ellipsoid_collision_factor(
                                    a1, a2, b1, b2,
                                    k=collision_k, r=collision_radius,
                                    cost_sigma=collision_sigma))

    def _setup_robot(self, robot_id, limbs, goal_xy, sigma_endpoint, sigma_joint):
        """Build the kinematic chain, dynamics, and goal prior for one robot."""
        n = len(limbs)
        self._num_limbs[robot_id]      = n
        self.limb_lengths[robot_id]    = [l.length for l in limbs]
        self._k0_kin_indices[robot_id] = []

        x_pos = [0.0]
        for L in self.limb_lengths[robot_id]:
            x_pos.append(x_pos[-1] + L)

        for k in range(self.time_horizon):
            anchor_noise = ANCHOR_NOISE if k == 0 else LOOSE_ANCHOR_NOISE
            self.graph.add(gtsam.PriorFactorPose2(
                _J(robot_id, 1, k), gtsam.Pose2(0.0, 0.0, 0.0), anchor_noise))
            if k == 0:
                self._k0_root_idx[robot_id] = self.graph.size() - 1

            for i in range(1, n):
                self.graph.add(make_fixed_kinematics_factor(
                    _J(robot_id, i, k), _J(robot_id, i + 1, k),
                    self.limb_lengths[robot_id][i - 1], 0.0, LOOSE_ANCHOR_NOISE))
                if k == 0:
                    self._k0_kin_indices[robot_id].append(self.graph.size() - 1)

            self.graph.add(make_fixed_kinematics_factor(
                _J(robot_id, n, k), _E(robot_id, n, k),
                self.limb_lengths[robot_id][-1], 0.0, KINEMATIC_NOISE))

            for i in range(1, n + 1):
                self.initial.insert(_J(robot_id, i, k), gtsam.Pose2(x_pos[i - 1], 0.0, 0.0))
                self.initial.insert(_V(robot_id, i, k), np.array([0.0]))
                self._dof_map[_J(robot_id, i, k)] = 3
                self._dof_map[_V(robot_id, i, k)] = 1
            self.initial.insert(_E(robot_id, n, k), gtsam.Pose2(x_pos[n], 0.0, 0.0))
            self.initial.insert(_VE(robot_id, n, k), np.array([0.0, 0.0]))
            self._dof_map[_E(robot_id, n, k)] = 3
            self._dof_map[_VE(robot_id, n, k)] = 2

        for k in range(self.time_horizon - 1):
            self.graph.add(_make_task_space_dynamics_factor(
                _E(robot_id, n, k),   _VE(robot_id, n, k),
                _E(robot_id, n, k+1), _VE(robot_id, n, k+1),
                self.dt, sigma_endpoint))
            for i in range(1, n + 1):
                self.graph.add(_make_joint_space_dynamics_factor(
                    _J(robot_id, i, k), _V(robot_id, i, k),
                    _J(robot_id, i, k+1), _V(robot_id, i, k+1),
                    self.dt, sigma_joint))

        self.graph.add(gtsam.PriorFactorPose2(
            _E(robot_id, n, self.time_horizon - 1),
            gtsam.Pose2(goal_xy[0], goal_xy[1], 0.0),
            GOAL_NOISE))
        self._goal_idx[robot_id] = self.graph.size() - 1

    def _update_goal(self, robot_id: int, goal_xy: tuple) -> None:
        n = self._num_limbs[robot_id]
        self.graph.replace(
            self._goal_idx[robot_id],
            gtsam.PriorFactorPose2(
                _E(robot_id, n, self.time_horizon - 1),
                gtsam.Pose2(goal_xy[0], goal_xy[1], 0.0),
                GOAL_NOISE))

    def _pre_solve(self, robot_id: int, current_qpos: np.ndarray,
                   joint_poses: np.ndarray) -> None:
        """Ground robot_id's k=0 state to the measured configuration."""
        n = self._num_limbs[robot_id]

        for i, xyt in enumerate(joint_poses):
            self.initial.update(_J(robot_id, i + 1, 0),
                                gtsam.Pose2(float(xyt[0]), float(xyt[1]), float(xyt[2])))

        last     = joint_poses[-1]
        end_pose = gtsam.Pose2(float(last[0]), float(last[1]), float(last[2])).compose(
                       gtsam.Pose2(self.limb_lengths[robot_id][-1], 0.0, 0.0))
        self.initial.update(_E(robot_id, n, 0), end_pose)

        self.graph.replace(self._k0_root_idx[robot_id],
            gtsam.PriorFactorPose2(
                _J(robot_id, 1, 0),
                gtsam.Pose2(float(joint_poses[0][0]),
                            float(joint_poses[0][1]),
                            float(joint_poses[0][2])),
                ANCHOR_NOISE))

        for graph_idx, i in zip(self._k0_kin_indices[robot_id], range(1, n)):
            self.graph.replace(graph_idx,
                make_fixed_kinematics_factor(
                    _J(robot_id, i, 0), _J(robot_id, i + 1, 0),
                    self.limb_lengths[robot_id][i - 1],
                    float(current_qpos[i]),
                    KINEMATIC_NOISE))

    def _post_solve(self, robot_id: int, result: gtsam.Values) -> np.ndarray:
        """Shift robot_id's receding window and extract k=1 joint angles."""
        n = self._num_limbs[robot_id]
        for k in range(self.time_horizon):
            next_k = k + 1 if k < self.time_horizon - 1 else k
            for i in range(1, n + 1):
                self.initial.update(_J(robot_id, i, k), result.atPose2(_J(robot_id, i, next_k)))
                self.initial.update(_V(robot_id, i, k), result.atVector(_V(robot_id, i, next_k)))
            self.initial.update(_E(robot_id, n, k), result.atPose2(_E(robot_id, n, next_k)))
            self.initial.update(_VE(robot_id, n, k), result.atVector(_VE(robot_id, n, next_k)))

        thetas = [result.atPose2(_J(robot_id, i + 1, 1)).theta() for i in range(n)]
        ctrl = np.zeros(n)
        ctrl[0] = thetas[0]
        for i in range(1, n):
            ctrl[i] = thetas[i] - thetas[i - 1]
        return ctrl

    def centralised_solve(self, robot_states: list, goals: list = None) -> list:
        """
        robot_states: [(qpos_0, joint_poses_0), (qpos_1, joint_poses_1), ...]
        goals:        optional [(x, y), ...], one per robot. If provided, replaces
                      all robots' goal priors before solving.
        Returns:      [ctrl_0, ctrl_1, ...] one numpy array per robot.
        """
        if goals is not None:
            for robot_id, goal_xy in enumerate(goals):
                self._update_goal(robot_id, goal_xy)
        for robot_id, (qpos, joint_poses) in enumerate(robot_states):
            self._pre_solve(robot_id, qpos, joint_poses)
        result = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial, self.params).optimize()
        return [self._post_solve(robot_id, result) for robot_id in range(len(robot_states))]

    def gbp_solve(self, robot_states: list, goals: list = None,
                  n_outer: int = 5, n_inner: int = 10,
                  damping: float = 0.0) -> list:
        """GBP variant — drop-in replacement for centralised_solve."""
        if goals is not None:
            for robot_id, goal_xy in enumerate(goals):
                self._update_goal(robot_id, goal_xy)
        for robot_id, (qpos, joint_poses) in enumerate(robot_states):
            self._pre_solve(robot_id, qpos, joint_poses)
        result = GBPOptimizer(
            self.graph, self.initial,
            GBPParams(n_outer=n_outer, n_inner=n_inner, damping=damping),
            dof_map=self._dof_map,
        ).optimize()
        return [self._post_solve(robot_id, result) for robot_id in range(len(robot_states))]
