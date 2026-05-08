import math
import random
import json

import gtsam
from gtsam import symbol
import numpy as np
import torch  # kept only for forward_kinematics_step (unchanged from original)

# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------
KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5,  0.5,  1e-4]))
ANCHOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-6]))
CALIB_NOISE        = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.001]))
SENSOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5 ]))


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
# Keys are encoded as symbol(char, integer_index).
# Limb pose    per observation : symbol('L', limb_id * 10000 + step)
# Sensor pose  per observation : symbol('S', limb_id * 10000 + step)
# Joint calib  singleton       : symbol('C', limb_id)          -- shared across all steps
# Sensor calib singleton       : symbol('Z', limb_id)          -- shared across all steps

def _limb_key(limb_id: int, step: int) -> int:
    return symbol('L', limb_id * 10000 + step)

def _sensor_key(limb_id: int, step: int) -> int:
    return symbol('S', limb_id * 10000 + step)

def _calib_key(limb_id: int) -> int:
    """Singleton joint-calibration key – same key every step."""
    return symbol('C', limb_id)

def _sensor_calib_key(limb_id: int) -> int:
    """Singleton sensor-calibration key – same key every step."""
    return symbol('Z', limb_id)


# ---------------------------------------------------------------------------
# Custom kinematics factor
# ---------------------------------------------------------------------------
def make_kinematics_factor(key_n1: int, key_c: int, key_n2: int,
                            L: float, theta_joint: float,
                            noise_model) -> gtsam.CustomFactor:
    """
    3-node factor: T_pred = N1 * Pose2(L,0,0) * C * Pose2(0,0,theta_joint)
    error = N2.localCoordinates(T_pred)
    Analytic Jacobians via GTSAM Pose2 chain rule.
    """
    L_link  = gtsam.Pose2(float(L), 0.0, 0.0)
    J_joint = gtsam.Pose2(0.0, 0.0, float(theta_joint))

    def error_func(this, values, jacobians):
        n1 = values.atPose2(this.keys()[0])
        c  = values.atPose2(this.keys()[1])
        n2 = values.atPose2(this.keys()[2])

        def z():
            return np.zeros((3, 3), order='F')

        H_A_N1 = z(); H_A_L = z()
        A = n1.compose(L_link, H_A_N1, H_A_L)

        H_B_A = z(); H_B_C = z()
        B = A.compose(c, H_B_A, H_B_C)

        H_T_B = z(); H_T_J = z()
        T_pred = B.compose(J_joint, H_T_B, H_T_J)

        H_e_N2 = z(); H_e_T = z()
        error = n2.localCoordinates(T_pred, H_e_N2, H_e_T)

        if jacobians is not None:
            jacobians[0] = H_e_T @ H_T_B @ H_B_A @ H_A_N1  # de/dN1
            jacobians[1] = H_e_T @ H_T_B @ H_B_C            # de/dC
            jacobians[2] = H_e_N2                            # de/dN2

        return error

    keys = gtsam.KeyVector()
    keys.append(key_n1)
    keys.append(key_c)
    keys.append(key_n2)
    return gtsam.CustomFactor(noise_model, keys, error_func)


# ---------------------------------------------------------------------------
# GTSAMSolver — wraps graph + values + step counter, mirrors old fg interface
# ---------------------------------------------------------------------------
class GTSAMSolver:
    """
    Drop-in replacement for the old FactorGraph object expected by live_server.py.

    Public interface mirroring the old fg object:
        fg.step_count          int
        fg.var_nodes           dict  (limb_id -> metadata, for logging)
        fg.factors             list  (for logging – len only)
        update_factor_graph(data, fg)
        fg.gbp_solve(n_iters=N)  -> runs LM optimisation
        fg.energy()             -> float graph error
        extract_calibrations(fg) -> list of dicts
    """

    def __init__(self):
        self.graph      = gtsam.NonlinearFactorGraph()
        self.values     = gtsam.Values()
        self.step_count = 0

        # Metadata stores: limb_id (int) -> limb properties dict
        # Used for logging and to reconstruct calib output.
        self._limb_meta: dict[int, dict] = {}   # limb_id -> properties
        self._calib_ids: list[int] = []          # limb_ids that have a calib node
        self._sensor_calib_ids: list[int] = []   # limb_ids that have a sensor calib node

        # Mirrors for live_server.py log calls
        self.var_nodes: dict = {}  # populated in update_factor_graph
        self.factors:   list = []  # populated in update_factor_graph (appended per factor)

    # ------------------------------------------------------------------
    def gbp_solve(self, n_iters: int = 20) -> None:
        """Run Levenberg-Marquardt optimisation and update internal values."""
        if self.values.size() == 0:
            return
        params = gtsam.LevenbergMarquardtParams()
        # params.setMaxIterations(n_iters)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.values, params)
        self.values = optimizer.optimize()

    # ------------------------------------------------------------------
    def energy(self) -> float:
        """Return the current graph error (sum of squared whitened residuals)."""
        if self.values.size() == 0:
            return 0.0
        return self.graph.error(self.values)


# ---------------------------------------------------------------------------
# Noise helpers (unchanged from original)
# ---------------------------------------------------------------------------
def add_noise_to_measurement(angle_rad: float, noise_std: float) -> float:
    if noise_std > 0:
        return angle_rad + random.gauss(0, noise_std)
    return angle_rad


def add_noise(data, angle_noise=True, angle_noise_deg=2.0,
              distance_noise=False, dist_noise_units=2.0):
    limbs       = data["limbs"]
    connections = data["connections"]
    limbs       = sorted(limbs,       key=lambda x: x['depth'])
    connections = sorted(connections, key=lambda x: x['depth'])

    limbs_dict = {limb["id"]: limb for limb in limbs}

    if angle_noise:
        for limb in limbs_dict.values():
            limb["local_angle"] = add_noise_to_measurement(
                limb["local_angle"], angle_noise_deg)

    if distance_noise:
        for conn in connections:
            conn["distance"] = add_noise_to_measurement(
                conn["distance"], dist_noise_units)

    return limbs_dict, connections


# ---------------------------------------------------------------------------
# Forward kinematics (unchanged from original – returns torch tensors)
# ---------------------------------------------------------------------------
def forward_kinematics_step(prev_conn_pose, joint_angle_rad, limb_length):
    joint_angle_rad = torch.tensor(float(joint_angle_rad))
    current_theta   = prev_conn_pose[2] + joint_angle_rad
    current_theta   = torch.atan2(torch.sin(current_theta),
                                  torch.cos(current_theta))

    current_limb_pose = torch.tensor([
        prev_conn_pose[0].item() if hasattr(prev_conn_pose[0], 'item') else float(prev_conn_pose[0]),
        prev_conn_pose[1].item() if hasattr(prev_conn_pose[1], 'item') else float(prev_conn_pose[1]),
        current_theta.item(),
    ])

    next_x = float(current_limb_pose[0]) + limb_length * float(torch.cos(current_theta))
    next_y = float(current_limb_pose[1]) + limb_length * float(torch.sin(current_theta))

    next_conn_pose = torch.tensor([next_x, next_y, current_theta.item()])
    return current_limb_pose, next_conn_pose


# ---------------------------------------------------------------------------
# create_gbp_solver  – returns a GTSAMSolver (same name, same call site)
# ---------------------------------------------------------------------------
def create_gbp_solver() -> GTSAMSolver:
    """
    Create and return a fresh GTSAMSolver ready to receive poses.
    Call once per calibration session (e.g. on WebSocket connect or reset).
    """
    return GTSAMSolver()


# ---------------------------------------------------------------------------
# update_factor_graph
# ---------------------------------------------------------------------------
def update_factor_graph(data: dict, fg: GTSAMSolver) -> None:
    """
    Ingest one pose observation into the factor graph.

    For each step a new set of limb-pose and sensor-pose nodes is added.
    Calibration nodes (joint and sensor) are singletons: created only on the
    first step and reused (connected to) every subsequent step.

    Args:
        data: dict with keys "limbs" and "connections".
        fg:   GTSAMSolver instance (from create_gbp_solver()).
    """
    step = fg.step_count

    limbs_dict, connections = add_noise(
        data,
        angle_noise=False,  angle_noise_deg=1,
        distance_noise=False, dist_noise_units=1,
    )

    next_conn_pose = torch.tensor([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #
    # 2. Insert variable nodes into Values and add singleton factors      #
    # ------------------------------------------------------------------ #
    for limb_id, limb in limbs_dict.items():

        length = float(limb["limb_length"])
        theta_rad = math.radians(float(limb["local_angle"]))
        position_estimate, next_conn_pose = forward_kinematics_step(next_conn_pose, theta_rad, length)

        L_key  = _limb_key(limb_id, step)
        S_key  = _sensor_key(limb_id, step)
        C_key  = _calib_key(limb_id)
        Z_key  = _sensor_calib_key(limb_id)

        # Limb pose node (new every step)
        fg.values.insert(
            L_key,
            gtsam.Pose2(float(position_estimate[0]), float(position_estimate[1]), float(position_estimate[2]))
        )

        # Sensor pose node — initially same as limb pose (refined via factors)
        fg.values.insert(
            S_key,
            gtsam.Pose2(float(next_conn_pose[0]), float(next_conn_pose[1]), float(next_conn_pose[2]))
        )

        # Anchor: pin base limb (depth 0) to origin
        if limb["depth"] == 0:
            fg.graph.add(gtsam.PriorFactorPose2(
                L_key, gtsam.Pose2(0.0, 0.0, 0.0), ANCHOR_NOISE))
            fg.factors.append(('anchor', L_key))

        # Singleton calibration nodes — created only on step 0
        if step == 0:
            if limb["depth"] != 0:
                # Joint calibration (only non-root limbs have a parent joint)
                fg.values.insert(C_key, gtsam.Pose2(0.0, 0.0, 0.0))
                fg.graph.add(gtsam.PriorFactorPose2(
                    C_key, gtsam.Pose2(0.0, 0.0, 0.0), CALIB_NOISE))
                fg.factors.append(('calib_prior', C_key))
                if limb_id not in fg._calib_ids:
                    fg._calib_ids.append(limb_id)

            # Sensor calibration (all limbs that carry a sensor)
            fg.values.insert(Z_key, gtsam.Pose2(0.0, 0.0, 0.0))
            fg.graph.add(gtsam.PriorFactorPose2(
                Z_key, gtsam.Pose2(0.0, 0.0, 0.0), SENSOR_CALIB_NOISE))
            fg.factors.append(('sensor_calib_prior', Z_key))
            if limb_id not in fg._sensor_calib_ids:
                fg._sensor_calib_ids.append(limb_id)

        # Update var_nodes mirror for live_server.py logging
        key_str = f"L{limb_id}"
        fg.var_nodes.setdefault(key_str, []).append({'step': step, 'limb_id': limb_id})

    # ------------------------------------------------------------------ #
    # 3. Add kinematic and distance factors for each connection           #
    # ------------------------------------------------------------------ #
    for conn in connections:
        parent_id = conn["parent_id"]
        child_id  = conn["child_id"]

        parent_limb = limbs_dict[parent_id]
        child_limb  = limbs_dict[child_id]

        parent_L = float(parent_limb["limb_length"])
        angle_rad = math.radians(float(child_limb["local_angle"]))
        dist_meas = float(conn["distance"])

        parent_sensor_offset = parent_limb["sensor_offset"]
        child_sensor_offset  = child_limb["sensor_offset"]
        # Sensor offset is stored as {x, y} in the limb's local frame.
        # We model the sensor as a point at distance sqrt(x^2+y^2) along a
        # direction given by atan2(y, x) from the limb origin.
        # s_parent_dist  = math.hypot(parent_sensor_offset["x"], parent_sensor_offset["y"])
        # s_parent_angle = math.atan2(parent_sensor_offset["y"], parent_sensor_offset["x"])
        # s_child_dist   = math.hypot(child_sensor_offset["x"],  child_sensor_offset["y"])
        # s_child_angle  = math.atan2(child_sensor_offset["y"],  child_sensor_offset["x"])

        L_parent = _limb_key(parent_id, step)
        L_child  = _limb_key(child_id,  step)
        C_child  = _calib_key(child_id)           # singleton
        S_parent = _sensor_key(parent_id, step)
        S_child  = _sensor_key(child_id,  step)
        Z_parent = _sensor_calib_key(parent_id)   # singleton
        Z_child  = _sensor_calib_key(child_id)    # singleton

        # Kinematic factor: parent limb -> joint calib -> child limb
        fg.graph.add(make_kinematics_factor(
            L_parent, C_child, L_child,
            L=parent_L, theta_joint=angle_rad,
            noise_model=KINEMATIC_NOISE,
        ))
        fg.factors.append(('kinematics', L_parent, C_child, L_child))

        # Sensor placement factors: limb origin -> sensor calib -> sensor pose
        fg.graph.add(make_kinematics_factor(
            L_parent, Z_parent, S_parent,
            L=parent_sensor_offset["x"], theta_joint=0.0,
            noise_model=KINEMATIC_NOISE,
        ))
        fg.factors.append(('sensor_kin', L_parent, Z_parent, S_parent))

        fg.graph.add(make_kinematics_factor(
            L_child, Z_child, S_child,
            L=child_sensor_offset["x"], theta_joint=0.0,
            noise_model=KINEMATIC_NOISE,
        ))
        fg.factors.append(('sensor_kin', L_child, Z_child, S_child))

        # Distance measurement: range between sensor origins
        fg.graph.add(gtsam.RangeFactorPose2(S_parent, S_child, dist_meas, SENSOR_NOISE))
        fg.factors.append(('range', S_parent, S_child))

    fg.step_count += 1


# ---------------------------------------------------------------------------
# extract_calibrations  – queries gtsam.Values + gtsam.Marginals
# ---------------------------------------------------------------------------
def extract_calibrations(fg: GTSAMSolver) -> list:
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
    if fg.values.size() > 0:
        try:
            marginals = gtsam.Marginals(fg.graph, fg.values)
        except Exception:
            marginals = None

    for limb_id in fg._calib_ids:
        c_key = _calib_key(limb_id)
        label = f"calib{limb_id}"

        try:
            pose = fg.values.atPose2(c_key)
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


# ---------------------------------------------------------------------------
# __main__ – smoke-test matching the original gbp_implementation.py style
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fg = create_gbp_solver()

    with open("pose_data.json", "r") as f:
        poses = json.load(f)

    for pose in poses:
        update_factor_graph(pose, fg)

    print(f"Graph built: {fg.values.size()} variables, {len(fg.factors)} factors")

    fg.gbp_solve(n_iters=100)

    print("Optimisation complete")
    print(f"Energy: {fg.energy():.4f}")

    calibrations = extract_calibrations(fg)
    for c in calibrations:
        print(f"  {c['id']}: mean={[round(v,4) for v in c['mean']]}")
