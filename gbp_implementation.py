import torch
import math
import random
import json
from gbp_utilities import MeasModel, SquaredLoss, TukeyLoss, HuberLoss
from gbp import GBPSettings, FactorGraph
from gbp_factors import KinematicCalibModel, AnchorModel, DistanceMeasurementModel, EndpointModel, AngleMeasurementModel

# ===================================================================
# Setup: shared settings and loss functions
# ===================================================================
gbp_settings = GBPSettings(
    damping=0.95,
    beta=0.2,
    num_undamped_iters=3,
    min_linear_iters=10,
    dropout=0.0,
)

# loss functions for the factors
kinematic_loss = HuberLoss(3, torch.tensor([0.5, 0.5, 1e-4]), 3.0)
anchor_loss = SquaredLoss(3, torch.tensor([1e-4, 1e-4, 1e-6]))
endpoint_loss = SquaredLoss(3, torch.tensor([1e-3, 1e-3, 1e-5]))
distance_loss = TukeyLoss(1, torch.tensor([0.5]), 3.0)


def create_gbp_solver():
    """
    Create and return a fresh FactorGraph ready to receive poses.
    Call this once per calibration session (e.g. on WebSocket connect or reset).
    """
    fg = FactorGraph(gbp_settings)
    fg.step_count = 0   # tracks how many poses have been added
    return fg


def add_noise_to_measurement(angle_rad, noise_std):
    """Add Gaussian noise to a measurement."""
    if noise_std > 0:
        noise_rad = random.gauss(0, noise_std)
        return angle_rad + noise_rad
    return angle_rad

def add_noise(data, angle_noise=True, angle_noise_deg=2.0, distance_noise=False, dist_noise_units=2.0):
    """adds noise to measurments taken from the robot"""

    limbs = data["limbs"]
    connections = data["connections"]
    limbs = sorted(limbs, key=lambda x: x['depth'])
    connections = sorted(connections, key=lambda x: x['depth'])
    
    if angle_noise:
        for limb in limbs:
            limb["local_angle"] = add_noise_to_measurement(limb["local_angle"], angle_noise_deg)

    if distance_noise:
        for connection in connections:
            connection["distance"] = add_noise_to_measurement(connection["distance"], dist_noise_units)

    return limbs, connections


def forward_kinematics_step(prev_conn_pose, joint_angle_rad, limb_length):
    """
    Calculates the pose of the current limb and the next joint location.
    
    Args:
        prev_conn_pose: Tensor [x, y, theta] of the previous joint/connection.
        joint_angle_rad: The motor measurement (relative rotation).
        limb_length: The nominal length of the bone.
        
    Returns:
        current_limb_pose: Tensor [x, y, theta] (The start/base of the limb)
        next_conn_pose: Tensor [x, y, theta] (The end of the limb/next joint)
    """
    # 1. Calculate the current limb's orientation
    # The limb points in the direction of the previous joint + the motor's turn
    joint_angle_rad = torch.tensor(joint_angle_rad)
    current_theta = prev_conn_pose[2] + joint_angle_rad

    # If you want to keep nodes near [-pi, pi] for human readability 
    # ONLY do it using atan2 to ensure the jump is handled smoothly
    # during the INITIALIZATION phase.
    current_theta = torch.atan2(torch.sin(current_theta), torch.cos(current_theta))
    
    # The base of the current limb is the same x,y as the previous connection
    current_limb_pose = torch.tensor([
        prev_conn_pose[0],
        prev_conn_pose[1],
        current_theta
    ])
    
    # 2. Project forward along the limb length to find the next connection
    # We move 'limb_length' units in the direction of current_theta
    next_x = current_limb_pose[0] + limb_length * torch.cos(current_theta)
    next_y = current_limb_pose[1] + limb_length * torch.sin(current_theta)
    
    # The orientation at the next connection matches the limb's orientation
    next_conn_pose = torch.tensor([next_x, next_y, current_theta])
    
    return current_limb_pose, next_conn_pose


def update_factor_graph(data, fg):
    """
    Ingest one pose observation into the factor graph.

    Adds a new pose variable node per limb and a singleton calibration node
    per non-root connection (created only on first observation).
    Adds anchor, kinematic-calibration, and distance factors.

    Args:
        data: dict with keys "limbs" and "connections" (same schema as the
              simulator JSON payload).
        fg:   FactorGraph instance (from create_gbp_solver()).
    """
    limbs, connections = add_noise(data, angle_noise=True, angle_noise_deg=1, distance_noise=True, dist_noise_units=1)

    next_conn_pose = torch.tensor([0., 0., 0.])
    position_cov = torch.tensor([1000., 1000., 0.01])

    for limb in limbs:
        id = str(limb["id"])
        length = limb["limb_length"]
        theta_rad = math.radians(limb["local_angle"])

        position_estimate, next_conn_pose = forward_kinematics_step(next_conn_pose, theta_rad, length)

        # add limb nodes to the factor graph (global x, y, theta of endpoint)
        fg.add_var_node(id=id,
                        dofs=3,
                        prior_mean=position_estimate,
                        prior_diag_cov=position_cov,  # Large variance = weak prior
                        properties=limb)
        
        # add calibration nodes for non-base limbs (singleton – created once)
        if fg.var_nodes.get("calib"+id) is None:
            if limb["depth"] != 0:
                fg.add_var_node(id="calib"+id,
                                dofs=3,
                                prior_mean=torch.tensor([0., 0., 0.]), 
                                prior_diag_cov=torch.tensor([1000., 1000., 0.01]),
                                properties={})
        # add anchor to base node
        if limb["depth"] == 0:
            base_node = fg.var_nodes[id][-1]  # last node is current node
            fg.add_factor(measurement=torch.tensor([0., 0., 0.]),
                          meas_model=AnchorModel(anchor_loss, T_origin=position_estimate),
                          adj_var_nodes=[base_node],
                          properties={})
            
    for connection in connections:
        parent_id = str(connection["parent_id"])
        child_id = str(connection["child_id"])
        
        parent_node = fg.var_nodes[parent_id][-1]
        child_node = fg.var_nodes[child_id][-1]
        calib_node = fg.var_nodes["calib"+child_id][-1]   # only one node in list shared across factor graph
        parent_limb_length = parent_node.properties["limb_length"]

        angle_measure = child_node.properties["local_angle"]
        angle_measure_rad = math.radians(angle_measure)
        distance_measure = connection["distance"]

        # add kinematics factors
        fg.add_factor(measurement=torch.tensor([0., 0., 0.]),
                      meas_model=KinematicCalibModel(kinematic_loss, L=parent_limb_length, theta_joint=angle_measure_rad),
                      adj_var_nodes=[parent_node, calib_node, child_node], 
                      properties={})
        
        # add distance measurement factors
        parent_limb_sensor_offset = parent_node.properties["sensor_offset"]
        child_limb_sensor_offset = child_node.properties["sensor_offset"]
        s1 = torch.tensor([parent_limb_sensor_offset["x"], parent_limb_sensor_offset["y"]])
        s2 = torch.tensor([child_limb_sensor_offset["x"], child_limb_sensor_offset["y"]])
        fg.add_factor(measurement=torch.tensor([distance_measure]),
                      meas_model=DistanceMeasurementModel(distance_loss, s1=s1, s2=s2),
                      adj_var_nodes=[parent_node, child_node], 
                      properties={})

    fg.step_count += 1


def extract_calibrations(fg):
    """
    Extract calibration estimates from the factor graph.

    Returns a list of dicts, one per calibration variable node:
      {
        "id":     "calib2",          # the variable node id
        "mean":   [x, y],            # 2-D positional mean (parent-endpoint-relative offset)
        "cov_xy": [[cxx, cxy],       # 2x2 positional covariance
                   [cyx, cyy]]
      }

    The mean [x, y] is the calibration offset in the parent limb's local frame,
    measured from the parent's endpoint (limb end). This matches the convention
    in KinematicCalibModel: T_pred = N1 @ L_link @ C2 @ J1, where C2 encodes
    the offset applied after walking the full limb length L.
    """
    results = []
    for key, node_list in fg.var_nodes.items():
        if not key.startswith("calib"):
            continue
        node = node_list[-1]  # singleton – only one entry
        try:
            mean = node.belief.mean()          # shape [3]: [x, y, theta]
            cov  = node.belief.cov()           # shape [3, 3]
            results.append({
                "id": key,
                "mean": [mean[0].item(), mean[1].item()],
                "cov_xy": [
                    [cov[0, 0].item(), cov[0, 1].item()],
                    [cov[1, 0].item(), cov[1, 1].item()]
                ]
            })
        except Exception:
            # belief not yet initialised (e.g. before first solve)
            results.append({
                "id": key,
                "mean": [0.0, 0.0],
                "cov_xy": [[1000.0, 0.0], [0.0, 1000.0]]
            })
    return results

if __name__ == "__main__":
    fg = create_gbp_solver()

    count = 0
    with open("pose_data.json", "r") as f:
        poses = json.load(f)

    N = len(poses)
    for pose in poses[:N]:
        update_factor_graph(pose, fg)
        count += 1

    fg.gbp_solve(n_iters=100)

    print("Factor graph updated successfully!")
    print(f"Variables: {len(fg.var_nodes)}")
    print(f"Factors: {len(fg.factors)}")
    fg.print()
