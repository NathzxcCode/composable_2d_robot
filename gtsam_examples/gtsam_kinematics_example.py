import gtsam
from gtsam import Pose2, symbol
import numpy as np

from gtsam_factors import make_calib_kinematics_factor, make_fixed_kinematics_factor
from utils import get_connections, plot_chain, plot_side_by_side

# Create noise models
KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5]))
ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4]))
CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
SENSOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

def main():
    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    # 1. Define Symbols for your variables
    L1_key = symbol('L', 1)
    CL2_key  = symbol('C', 2)
    L2_key = symbol('L', 2)
    S1_key = symbol('S', 1)
    CS1_key = symbol('Z', 1)
    S2_key = symbol('S', 2)
    CS2_key = symbol('Z', 2)
    E2_key = symbol('E', 2)

    # Add a prior on the first pose, setting it to the origin
    # A prior factor consists of a mean and a noise model (covariance matrix)
    priorMean = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
    graph.add(gtsam.PriorFactorPose2(L1_key, priorMean, ANCHOR_NOISE))

    # Add prior on the calibration nodes
    graph.add(gtsam.PriorFactorPose2(CL2_key, priorMean, CALIB_NOISE)) # joint 2 calibration
    graph.add(gtsam.PriorFactorPose2(CS1_key, priorMean, SENSOR_CALIB_NOISE)) # sensor 1 calibration
    graph.add(gtsam.PriorFactorPose2(CS2_key, priorMean, SENSOR_CALIB_NOISE)) # sensor 2 calibration

    # 2. Define custom factor using kinematics model
    graph.add(make_calib_kinematics_factor(L1_key, CL2_key, L2_key, 20, np.pi/3, KINEMATIC_NOISE)) # joint connecting 1,2
    graph.add(make_calib_kinematics_factor(L1_key, CS1_key, S1_key, 20, 0, KINEMATIC_NOISE)) # sensor 1
    graph.add(make_calib_kinematics_factor(L2_key, CS2_key, S2_key, 20, 0, KINEMATIC_NOISE)) # sensor 2
    # add endpoint for limb 2
    graph.add(make_fixed_kinematics_factor(L2_key, E2_key, 20, 0, KINEMATIC_NOISE))

    # 3. Add sensor distance measurements as factor
    graph.add(gtsam.RangeFactorPose2(S1_key, S2_key, 20, SENSOR_NOISE))

    # 3. Add initial values
    initial = gtsam.Values()
    initial.insert(L1_key, gtsam.Pose2(0.5, 0.0, 0.2)) # limb 1
    initial.insert(CL2_key, gtsam.Pose2(0.0, 0.0, 0.0)) # limb 2 calibration
    initial.insert(L2_key, gtsam.Pose2(4.1, 0.1, 0.1)) # limb 2
    initial.insert(S1_key, gtsam.Pose2(4.0, 0.0, 0.0)) # sensor 1
    initial.insert(CS1_key, gtsam.Pose2(0.0, 0.0, 0.0)) # sensor 1 calibration
    initial.insert(S2_key, gtsam.Pose2(4.1, 0.1, 0.1)) # sensor 2
    initial.insert(CS2_key, gtsam.Pose2(0.0, 0.0, 0.0)) # sensor 2 calibration
    initial.insert(E2_key, gtsam.Pose2(20.0, 0.0, 0.0)) # limb 2 endpoint
    print("\nInitial Estimate:\n{}".format(initial))

    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print(result)

    plot_keys = [L1_key, CL2_key, L2_key, E2_key]
    conns = get_connections(graph)
    plot_side_by_side(initial, result, plot_keys, conns)
    


if __name__=="__main__":
    main()