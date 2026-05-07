import torch
import math
import random
import json
from gbp_utilities import MeasModel, SquaredLoss, TukeyLoss, HuberLoss
from gbp import GBPSettings, FactorGraph
from gbp_factors import KinematicCalibModel, AnchorModel, DistanceMeasurementModel, EndpointModel, AngleMeasurementModel

import gtsam
from gtsam import Pose2, symbol
import numpy as np

# Create noise models
KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5]))
ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4]))
CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
SENSOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

num_iters = 0

def create_gbp_solver():
    """
    Create and return a fresh FactorGraph ready to receive poses.
    Call this once per calibration session (e.g. on WebSocket connect or reset).
    """
    fg = gtsam.NonlinearFactorGraph()
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

    limbs_dict = {}
    for limb in limbs:
        limbs_dict[limb["id"]] = limb
    
    if angle_noise:
        for limb in limbs_dict.values():
            limb["local_angle"] = add_noise_to_measurement(limb["local_angle"], angle_noise_deg)

    if distance_noise:
        for connection in connections:
            connection["distance"] = add_noise_to_measurement(connection["distance"], dist_noise_units)

    return limbs_dict, connections


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


def update_factor_graph(data, graph, values):
    limbs, connections = add_noise(data, angle_noise=True, angle_noise_deg=1, distance_noise=True, dist_noise_units=1)

    next_conn_pose = torch.tensor([0., 0., 0.])

    for connection in connections:
        parent_id = connection["parent_id"]
        child_id = connection["child_id"]
        
        parent_limb_length = limbs[parent_id]["limb_length"]
        parent_limb_sensor_offset = limbs[parent_id]["sensor_offset"]
        child_limb_sensor_offset = limbs[child_id]["sensor_offset"]
        angle_measure = limbs[child_id]["local_angle"]
        angle_measure_rad = math.radians(angle_measure)
        distance_measure = connection["distance"]

        parent_id = int(str(parent_id) + str(num_iters))
        child_id = int(str(child_id) + str(num_iters))
        
        L1_key = symbol('L', parent_id)
        L2_key = symbol('L', child_id)
        CL2_key = symbol('C', child_id)
        S1_key = symbol('S', parent_id)
        CS1_key = symbol('Z', parent_id)
        S2_key = symbol('S', child_id)
        CS2_key = symbol('Z', child_id)

        # 2. Define custom factor using kinematics model
        graph.add(make_kinematics_factor(L1_key, CL2_key, L2_key, parent_limb_length, angle_measure_rad, KINEMATIC_NOISE)) # joint connecting parent and child
        graph.add(make_kinematics_factor(L1_key, CS1_key, S1_key, parent_limb_sensor_offset, 0, KINEMATIC_NOISE)) # parent sensor position
        graph.add(make_kinematics_factor(L2_key, CS2_key, S2_key, child_limb_sensor_offset, 0, KINEMATIC_NOISE)) # child sensor position

        # 3. Add sensor distance measurements as factor
        graph.add(gtsam.RangeFactorPose2(S1_key, S2_key, distance_measure, SENSOR_NOISE))

    for id, limb in limbs.keys():
        length = limb["limb_length"]
        theta_rad = math.radians(limb["local_angle"])

        position_estimate, next_conn_pose = forward_kinematics_step(next_conn_pose, theta_rad, length)

        id = int(str(id) + str(num_iters))
        L_key = symbol('L', id)
        CL_key = symbol('C', id)
        S_key = symbol('S', id)
        CS_key = symbol('Z', id)

        values.insert(L_key, gtsam.Pose2(position_estimate)) # limb
        values.insert(S_key, gtsam.Pose2(next_conn_pose)) # sensor
        
        if num_iters == 0:
            priorMean = gtsam.Pose2(0.0, 0.0, 0.0)  # prior of zero
            if limb["depth"] != 0:
                graph.add(gtsam.PriorFactorPose2(CL_key, priorMean, CALIB_NOISE)) # joint calibration to graph
                values.insert(CL_key, gtsam.Pose2(0.0, 0.0, 0.0)) # limb calibration initial value

            graph.add(gtsam.PriorFactorPose2(CS_key, priorMean, SENSOR_CALIB_NOISE)) # sensor 1 calibration
            values.insert(CS_key, gtsam.Pose2(0.0, 0.0, 0.0)) # sensor 1 calibration

        if limb["depth"] == 0:
            anchorPrior = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
            graph.add(gtsam.PriorFactorPose2(L_key, anchorPrior, ANCHOR_NOISE))

    num_iters += 1


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
