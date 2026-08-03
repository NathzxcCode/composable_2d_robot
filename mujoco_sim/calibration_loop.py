import gtsam
from gtsam import symbol
import numpy as np
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from gtsam_examples.gtsam_factors import make_calib_kinematics_factor, make_fixed_kinematics_factor
from gtsam_gbp import GBPOptimizer, GBPParams

KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5,  0.5,  1e-4]))
ANCHOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1.0]))
CALIB_NOISE        = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.001]))
SENSOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5 ]))

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
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.params = gtsam.LevenbergMarquardtParams()
        self.t = 0
        self.num_limbs = 0

    def update_factor_graph(self, data) -> None:

        if self.t > 100:
            return
        
        limbs = data["limbs"]
        joint_angles = data["joint_angles"]
        sensor_distances = data["sensor_distances"]
        self.num_limbs = len(limbs)

        # initialise limb poses
        # initialise sensor poses per limb
        # connect sensors to limbs
        # J(i) represents the PIVOT FRAME of joint i in world coordinates.
        # J(0) is at the base pivot = origin, so we start the chain there.
        current_parent_pose = gtsam.Pose2()
        for i, limb in enumerate(limbs):
            angle = joint_angles[i]
            
            # 1. Build the Static Attachment Transform (Parent Pivot -> Child Pivot)
            # For limb 0: its pivot IS the base origin, so T_attach = identity.
            # For limb i>0: walk the NOMINAL (assumed) parent length to reach the child pivot.
            # We use limb.length (nominal model assumption), NOT limb.attach_pos (true value).
            # The discrepancy between nominal and true is what CJ(i) is calibrated to absorb.
            # X-Z plane mapping: MuJoCo X -> GTSAM X, MuJoCo Z -> GTSAM Y.
            if i == 0:
                T_attach = gtsam.Pose2(0.0, 0.0, 0.0)
            else:
                parent_limb = limbs[i - 1]
                T_attach = gtsam.Pose2(parent_limb.length, 0.0, 0.0)
            
            # 2. Build the Dynamic Joint Rotation Transform
            T_joint = gtsam.Pose2(0.0, 0.0, angle)
            
            # 3. Calculate Global Pivot Pose of Joint i
            # Global_J(i) = Global_J(i-1) * T_Attach_Nominal * T_Joint_Rotation
            global_limb_pose = current_parent_pose.compose(T_attach).compose(T_joint)
            
            # 4. Calculate Global Pose of Sensor i
            x_sensor = limb.sensor_pos[0]
            y_sensor = limb.sensor_pos[1]  # X-Y plane: GTSAM Y maps to MuJoCo world Y
            theta_sensor = limb.sensor_euler[2]
            
            T_sensor_offset = gtsam.Pose2(x_sensor, y_sensor, theta_sensor)
            
            # Global_Sensor = Global_Limb * T_Sensor_Offset
            global_sensor_pose = global_limb_pose.compose(T_sensor_offset)
            
            # Initialise variables for joint and sensor pose estimates
            self.values.insert(J(i, self.t), global_limb_pose)
            self.values.insert(S(i, self.t), global_sensor_pose)
            if not self.values.exists(CS(i)):
                self.values.insert(CS(i), gtsam.Pose2(0.0, 0.0, 0.0))
                self.graph.add(gtsam.PriorFactorPose2(CS(i), gtsam.Pose2(0.0, 0.0, 0.0), SENSOR_CALIB_NOISE))
            
            # 5. Move the Chain Forward
            current_parent_pose = global_limb_pose

            # connect sensor to limb with kinematics factor
            self.graph.add(make_calib_kinematics_factor(J(i, self.t), CS(i), S(i, self.t), limb.sensor_pos[0], 0, KINEMATIC_NOISE)) # sensor connected to body

            # setup connection factors between limbs
            # add anchor prior on base limb
            if i == 0:
                # Pin the base pivot to the world origin (0,0). Theta is left free
                # (large sigma) so the measured joint angle can be expressed naturally.
                self.graph.add(gtsam.PriorFactorPose2(J(i, self.t), gtsam.Pose2(0.0, 0.0, 0.0), ANCHOR_NOISE))
            # add kinematics between joint i-1 and joint i
            else:
                if not self.values.exists(CJ(i)):
                    self.values.insert(CJ(i), gtsam.Pose2(0.0, 0.0, 0.0))
                    self.graph.add(gtsam.PriorFactorPose2(CJ(i), gtsam.Pose2(0.0, 0.0, 0.0), CALIB_NOISE))
                # Use the NOMINAL (assumed) parent limb length, not the true attach_pos.
                # The calibration CJ(i) will absorb the difference.
                self.graph.add(make_calib_kinematics_factor(J(i-1, self.t), CJ(i), J(i, self.t), limbs[i-1].length, angle, KINEMATIC_NOISE))
                self.graph.add(gtsam.RangeFactorPose2(S(i-1, self.t), S(i, self.t), sensor_distances[i-1], SENSOR_NOISE))

        self.t += 1

    def centralised_solve(self):
        if self.values.size() == 0:
            return
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, self.params)
        self.values = optimizer.optimize()
        print(self.values.atPose2(CJ(1)), self.values.atPose2(CJ(2)))

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