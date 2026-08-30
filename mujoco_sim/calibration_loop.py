import gtsam
from gtsam import symbol
import numpy as np
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from gtsam_examples.gtsam_factors import make_calib_kinematics_factor, make_fixed_kinematics_factor
from gtsam_gbp import GBPOptimizer, GBPParams

RIGID_KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,  1e-4,  1e-4]))
CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))
# RIGID_KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5,  0.5,  1e-4]))
# CALIB_NOISE        = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))

def J(joint_id, t): # joint
    index = (joint_id * 10000) + t
    return gtsam.Symbol('j', index).key()

def E(effector_id, t): # end effector
    index = (effector_id * 10000) + t
    return gtsam.Symbol('e', index).key()

def S(sensor_id, t): # sensor
    index = (sensor_id * 10000) + t
    return gtsam.Symbol('s', index).key()

def CJ(joint_id): # joint calibration. id refers to child joint
    index = (joint_id)
    return gtsam.Symbol('c', index).key()

def CS(sensor_id): # sensor calibration
    index = (sensor_id)
    return gtsam.Symbol('z', index).key()


class FactorGraph():
    def __init__(self, sigma_range=0.001, sigma_encoder=0.0009, window_size=10):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.params = gtsam.LevenbergMarquardtParams()
        self.t = 0
        self.num_limbs = 0
        self.window_size = window_size
        self._slot_inter_kin = {}
        self._slot_range = {}
        self._slot_anchor = {}

        self.CALIB_KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,  1e-4,  sigma_encoder])) # rotation dependant on encoder accuracy
        self.ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, sigma_encoder])) # rotation dependant on encoder accuracy
        self.SENSOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_range])) # dependant on range sensing noise
        # self.CALIB_KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5,  0.5,  1e-4]))
        # self.ANCHOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 0.001]))
        # self.SENSOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5 ]))

    def update_factor_graph(self, data) -> None:
        limbs            = data["limbs"]
        joint_angles     = data["joint_angles"]
        sensor_distances = data["sensor_distances"]
        self.num_limbs   = len(limbs)

        slot      = self.t % self.window_size
        replacing = self.t >= self.window_size

        # Forward kinematics: initial pose estimates for this step
        joint_poses, sensor_poses = [], []
        current_parent_pose = gtsam.Pose2()
        for i, limb in enumerate(limbs):
            angle    = joint_angles[i]
            T_attach = (gtsam.Pose2(0.0, 0.0, 0.0) if i == 0
                        else gtsam.Pose2(limbs[i - 1].length, 0.0, 0.0))
            # compose estimated next joint pose based on parent joint
            j_pose = current_parent_pose.compose(T_attach).compose(gtsam.Pose2(0.0, 0.0, angle))
            # estimate sensor pose based on next joint
            s_pose = j_pose.compose(
                gtsam.Pose2(limb.sensor_pos[0], limb.sensor_pos[1], limb.sensor_euler[2]))
            joint_poses.append(j_pose)
            sensor_poses.append(s_pose)
            current_parent_pose = j_pose

        # Variables 
        for i in range(self.num_limbs):
            if replacing:
                self.values.update(J(i, slot), joint_poses[i])
                self.values.update(S(i, slot), sensor_poses[i])
            else:
                self.values.insert(J(i, slot), joint_poses[i])
                self.values.insert(S(i, slot), sensor_poses[i])
                # Calibration variables are presistant insert and prior once only
                if i > 0 and not self.values.exists(CJ(i)):
                    self.values.insert(CJ(i), gtsam.Pose2())
                    self.graph.add(gtsam.PriorFactorPose2(
                        CJ(i), gtsam.Pose2(), CALIB_NOISE))

        # Factors 
        if replacing:
            # joint-joint calibration kinematics and joint anchor contain observations measurements and update
            # sensor kinematics, calibration priors remains constant
            for graph_idx, i in self._slot_inter_kin[slot]:
                self.graph.replace(
                    graph_idx,
                    make_calib_kinematics_factor(
                        J(i - 1, slot), CJ(i), J(i, slot),
                        limbs[i - 1].length, joint_angles[i], self.CALIB_KINEMATIC_NOISE))

            for graph_idx, i in self._slot_range[slot]:
                self.graph.replace(
                    graph_idx,
                    gtsam.RangeFactorPose2(
                        S(i - 1, slot), S(i, slot),
                        sensor_distances[i - 1], self.SENSOR_NOISE))

            for graph_idx in self._slot_anchor[slot]:
                self.graph.replace(
                    graph_idx,
                    gtsam.PriorFactorPose2(
                        J(0, slot), gtsam.Pose2(0.0, 0.0, joint_angles[0]), self.ANCHOR_NOISE))
        
        else:
            # add all factors and record indices of the observation dependant ones
            inter_kin_indices, range_indices, anchor_indices = [], [], []

            for i, limb in enumerate(limbs):
                # Sensor-to-body kinematics — sensor_pos is constant; never replaced.
                self.graph.add(make_fixed_kinematics_factor(
                    J(i, slot), S(i, slot),
                    limb.sensor_pos[0], 0, RIGID_KINEMATIC_NOISE))

                if i == 0:
                    # Anchor prior — measurement always (0,0,0); never replaced.
                    anchor_idx = self.graph.size()
                    self.graph.add(gtsam.PriorFactorPose2(
                        J(0, slot), gtsam.Pose2(0.0, 0.0, joint_angles[0]), self.ANCHOR_NOISE))
                    anchor_indices.append((anchor_idx))
                else:
                    # joint-joint calibration kinematics, save index for replacing later
                    inter_kin_idx = self.graph.size()
                    self.graph.add(make_calib_kinematics_factor(
                        J(i - 1, slot), CJ(i), J(i, slot),
                        limbs[i - 1].length, joint_angles[i], self.CALIB_KINEMATIC_NOISE))
                    inter_kin_indices.append((inter_kin_idx, i))

                    # Range factor between sensors, save index for later
                    range_idx = self.graph.size()
                    self.graph.add(gtsam.RangeFactorPose2(
                        S(i - 1, slot), S(i, slot),
                        sensor_distances[i - 1], self.SENSOR_NOISE))
                    range_indices.append((range_idx, i))

            self._slot_inter_kin[slot] = inter_kin_indices
            self._slot_range[slot] = range_indices
            self._slot_anchor[slot] = anchor_indices

        self.t += 1


    def centralised_solve(self):
        if self.values.size() == 0:
            return
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, self.params)
        self.values = optimizer.optimize()
        # print(self.values.atPose2(CJ(1)), self.values.atPose2(CJ(2)))

    def gbp_solve(self, n_outer=5, n_inner=10, damping=0.0):
        if self.values.size() == 0:
            return
        params = GBPParams(n_outer=n_outer, n_inner=n_inner, damping=damping, dof=3)
        self.values = GBPOptimizer(self.graph, self.values, params).optimize()


    def extract_calibrations(self) -> list:
        """
        Extract joint calibration estimates from the factor graph.

        Returns a list of dicts, one per joint-calibration node:
        {
            "id":     "calib2",
            "mean":   [x, y],
            "cov_xy": [[cxx, cxy], [cyx, cyy]]
        }
        """
        results = []

        # Attempt to compute marginals for covariances (may fail on under-constrained graphs)
        marginals = None
        if self.values.size() > 0:
            try:
                marginals = gtsam.Marginals(self.graph, self.values)
            except Exception:
                marginals = None

        for joint_id in range(1, self.num_limbs):
            c_key = CJ(joint_id)
            label = f"calib{joint_id}"

            try:
                pose = self.values.atPose2(c_key)
                mean_x = pose.x()
                mean_y = pose.y()

                if marginals is not None:
                    cov = marginals.marginalCovariance(c_key)   # 3x3
                    cov_xy = [
                        [float(cov[0, 0]), float(cov[0, 1])],
                        [float(cov[1, 0]), float(cov[1, 1])],
                    ]
                else:
                    cov_xy = [[1000.0, 0.0], [0.0, 1000.0]]

                results.append({
                    "id":     label,
                    "mean":   [float(mean_x), float(mean_y)],
                    "cov_xy": cov_xy,
                })

            except Exception:
                results.append({
                    "id":     label,
                    "mean":   [0.0, 0.0],
                    "cov_xy": [[1000.0, 0.0], [0.0, 1000.0]],
                })

        return results