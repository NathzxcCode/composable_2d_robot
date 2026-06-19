import gtsam
import numpy as np
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from gtsam_examples.gtsam_factors import make_calib_kinematics_factor_3d

# ---------------------------------------------------------------------------
# Noise models  (6-DOF: [rot_x, rot_y, rot_z, tx, ty, tz])
# GTSAM Pose3 tangent-space order is [rotation (3), translation (3)].
# ---------------------------------------------------------------------------
KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4, 0.5, 0.5, 0.5]))
# Anchor pins the BASE PIVOT position (tx,ty,tz) tightly but leaves all
# rotations free (large sigma) so the joint frame can rotate freely.
ANCHOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,  1.0,  1.0,  1e-4, 1e-4, 1e-4]))
CALIB_NOISE        = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 10.0, 10.0, 10.0]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1,  0.1,  0.1]))
SENSOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5]))


# ---------------------------------------------------------------------------
# Key helpers  (same encoding as 2D version)
# ---------------------------------------------------------------------------
def J(joint_id, t):
    """Joint pivot frame (Pose3), one per limb per timestep."""
    return gtsam.Symbol('j', joint_id * 10000 + t).key()

def S(sensor_id, t):
    """Sensor frame (Pose3), one per limb per timestep."""
    return gtsam.Symbol('s', sensor_id * 10000 + t).key()

def CJ(joint_id):
    """Joint calibration offset (Pose3), singleton shared across all timesteps."""
    return gtsam.Symbol('c', joint_id).key()

def CS(sensor_id):
    """Sensor calibration offset (Pose3), singleton shared across all timesteps."""
    return gtsam.Symbol('z', sensor_id).key()


# ---------------------------------------------------------------------------
# FactorGraph3D
# ---------------------------------------------------------------------------
class FactorGraph3D():
    def __init__(self):
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.params = gtsam.LevenbergMarquardtParams()
        self.t         = 0
        self.num_limbs = 0

    # ------------------------------------------------------------------
    def update_factor_graph(self, data) -> None:
        """
        Ingest one observation into the 3-D factor graph.

        data dict keys:
            "limbs"            – list[LimbSpec]
            "joint_angles"     – list[float], one per limb (radians)
            "sensor_distances" – list[float], len = num_limbs - 1
                                 sensor_distances[k] = dist(sensor_k, sensor_k+1)
        """
        if self.t > 100:
            return

        limbs            = data["limbs"]
        joint_angles     = data["joint_angles"]
        sensor_distances = data["sensor_distances"]
        self.num_limbs   = len(limbs)

        # J(i) represents the PIVOT FRAME of joint i in world coordinates.
        # J(0) is at the base pivot = world origin, so we start the chain there.
        current_parent_pose = gtsam.Pose3()

        for i, limb in enumerate(limbs):
            angle = joint_angles[i]

            # ----------------------------------------------------------
            # 1. Build the Static Attachment Transform (Parent Pivot -> Child Pivot)
            #
            # For limb 0: its pivot IS the base origin – T_attach = identity.
            # For limb i>0: walk the NOMINAL (assumed) parent length along the
            # parent's local X-axis to reach the child pivot.
            # We use parent_limb.length (nominal model assumption), NOT
            # limb.attach_pos (true value).  The discrepancy is what CJ(i)
            # is calibrated to absorb.
            # ----------------------------------------------------------
            if i == 0:
                T_attach = gtsam.Pose3()                          # identity
            else:
                parent_limb = limbs[i - 1]
                T_attach = gtsam.Pose3(
                    gtsam.Rot3(),
                    gtsam.Point3(parent_limb.length, 0.0, 0.0)
                )

            # ----------------------------------------------------------
            # 2. Build the Dynamic Joint Rotation Transform
            #
            # A hinge joint rotates around limb.joint_axis by `angle` radians.
            # Rodrigues vector = axis * angle.
            # ----------------------------------------------------------
            ax  = np.array(limb.joint_axis, dtype=float)
            rod = ax * angle
            rot_joint = gtsam.Rot3.Rodrigues(rod[0], rod[1], rod[2])
            T_joint   = gtsam.Pose3(rot_joint, gtsam.Point3(0.0, 0.0, 0.0))

            # ----------------------------------------------------------
            # 3. Calculate Global Pivot Pose of Joint i (nominal FK)
            #    Global_J(i) = Global_J(i-1) * T_attach * T_joint
            # ----------------------------------------------------------
            global_limb_pose = current_parent_pose.compose(T_attach).compose(T_joint)

            # ----------------------------------------------------------
            # 4. Calculate Global Pose of Sensor i
            #    Sensor sits at a fixed offset in the limb's local frame.
            # ----------------------------------------------------------
            # RzRyRx(roll, pitch, yaw) applies extrinsic X->Y->Z rotations,
            # matching LimbSpec.sensor_euler = [roll, pitch, yaw]
            rot_sensor = gtsam.Rot3.RzRyRx(
                limb.sensor_euler[0], limb.sensor_euler[1], limb.sensor_euler[2]
            )
            pos_sensor = gtsam.Point3(
                limb.sensor_pos[0], limb.sensor_pos[1], limb.sensor_pos[2]
            )
            T_sensor_offset  = gtsam.Pose3(rot_sensor, pos_sensor)
            global_sensor_pose = global_limb_pose.compose(T_sensor_offset)

            # ----------------------------------------------------------
            # 5. Insert variable initial estimates
            # ----------------------------------------------------------
            self.values.insert(J(i, self.t), global_limb_pose)
            self.values.insert(S(i, self.t), global_sensor_pose)

            # Sensor calibration singleton (first timestep only)
            if not self.values.exists(CS(i)):
                self.values.insert(CS(i), gtsam.Pose3())
                self.graph.add(
                    gtsam.PriorFactorPose3(CS(i), gtsam.Pose3(), SENSOR_CALIB_NOISE)
                )

            # ----------------------------------------------------------
            # 6. Move the kinematic chain forward for the next iteration
            # ----------------------------------------------------------
            current_parent_pose = global_limb_pose

            # ----------------------------------------------------------
            # 7. Sensor-to-limb kinematic factor
            #    J(i) * Pose3(sensor_pos) * CS(i) * I = S(i)
            #    Uses nominal sensor offset along X; CS(i) absorbs 3-D error.
            # ----------------------------------------------------------
            self.graph.add(
                make_calib_kinematics_factor_3d(
                    J(i, self.t), CS(i), S(i, self.t),
                    L=limb.sensor_pos[0],
                    joint_axis=limb.joint_axis,   # no rotation: angle=0
                    joint_angle=0.0,
                    noise_model=KINEMATIC_NOISE,
                )
            )

            # ----------------------------------------------------------
            # 8. Inter-limb factors
            # ----------------------------------------------------------
            if i == 0:
                # Anchor: pin base pivot position to world origin.
                # Rotation sigmas are large so the joint angle is free.
                self.graph.add(
                    gtsam.PriorFactorPose3(
                        J(i, self.t), gtsam.Pose3(), ANCHOR_NOISE
                    )
                )
            else:
                # Joint calibration singleton (first timestep only)
                if not self.values.exists(CJ(i)):
                    self.values.insert(CJ(i), gtsam.Pose3())
                    self.graph.add(
                        gtsam.PriorFactorPose3(CJ(i), gtsam.Pose3(), CALIB_NOISE)
                    )

                # Kinematic factor: J(i-1) * T_nominal * CJ(i) * T_joint = J(i)
                # Uses parent's nominal length; joint_angle baked in.
                self.graph.add(
                    make_calib_kinematics_factor_3d(
                        J(i - 1, self.t), CJ(i), J(i, self.t),
                        L=limbs[i - 1].length,
                        joint_axis=limb.joint_axis,
                        joint_angle=angle,
                        noise_model=KINEMATIC_NOISE,
                    )
                )

                # Range factor: measured Euclidean distance between sensor i-1 and i
                self.graph.add(
                    gtsam.RangeFactorPose3(
                        S(i - 1, self.t), S(i, self.t),
                        float(sensor_distances[i - 1]),
                        SENSOR_NOISE,
                    )
                )

        self.t += 1

    # ------------------------------------------------------------------
    def centralised_solve(self) -> None:
        if self.values.size() == 0:
            return
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.values, self.params
        )
        self.values = optimizer.optimize()

        # Debug: print calibration poses after each solve
        for joint_id in range(1, self.num_limbs):
            p = self.values.atPose3(CJ(joint_id))
            print(
                f"CJ({joint_id}): t=({p.x():.4f}, {p.y():.4f}, {p.z():.4f})  "
                f"R={np.round(p.rotation().matrix().flatten()[:3], 3)}"
            )

    # ------------------------------------------------------------------
    def extract_calibrations(self) -> list:
        """
        Extract joint calibration estimates from the factor graph.

        Returns a list of dicts, one per non-root joint calibration node:
        {
            "id":     "calib1",
            "mean":   [tx, ty, tz],          -- translation part of CJ Pose3
            "cov_xyz": [[...], [...], [...]]  -- 3x3 translation covariance
                                                  (rows/cols 3:6 of 6x6 marginal)
        }
        """
        results = []

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
                pose   = self.values.atPose3(c_key)
                mean_t = [float(pose.x()), float(pose.y()), float(pose.z())]

                if marginals is not None:
                    cov66  = marginals.marginalCovariance(c_key)   # 6x6
                    # Translation block is indices [3:6, 3:6] in GTSAM's
                    # Pose3 tangent ordering [rot(3), trans(3)]
                    cov_xyz = cov66[3:6, 3:6].tolist()
                else:
                    cov_xyz = [[1000.0, 0.0, 0.0],
                               [0.0, 1000.0, 0.0],
                               [0.0, 0.0, 1000.0]]

                results.append({
                    "id":      label,
                    "mean":    mean_t,
                    "cov_xyz": cov_xyz,
                })

            except Exception:
                results.append({
                    "id":      label,
                    "mean":    [0.0, 0.0, 0.0],
                    "cov_xyz": [[1000.0, 0.0, 0.0],
                                [0.0, 1000.0, 0.0],
                                [0.0, 0.0, 1000.0]],
                })

        return results
