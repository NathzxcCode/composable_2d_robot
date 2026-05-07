import gtsam
from gtsam import Pose2, symbol
import numpy as np

# Create noise models
KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5]))
ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4]))
CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
SENSOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

def make_kinematics_factor(key_n1, key_c, key_n2, L: float, theta_joint: float, noise_model):
    """
    3-node kinematic calibration factor:
      T_pred = N1 * Pose2(L,0,0) * C * Pose2(0,0,theta_joint)
      error  = N2.localCoordinates(T_pred)   [3-vector in tangent space of N2]
    Jacobians computed analytically via GTSAM Pose2 chain rule.
    """
    L_link  = gtsam.Pose2(L, 0.0, 0.0)
    J_joint = gtsam.Pose2(0.0, 0.0, theta_joint)
    def error_func(this, values, jacobians):
        n1 = values.atPose2(this.keys()[0])
        c  = values.atPose2(this.keys()[1])
        n2 = values.atPose2(this.keys()[2])
        def z():
            return np.zeros((3, 3), order='F')  # MUST be Fortran-order
        # Step 1: A = N1 * L_link
        H_A_N1 = z(); H_A_L = z()
        A = n1.compose(L_link, H_A_N1, H_A_L)   # H_A_L unused (L is constant)
        # Step 2: B = A * C
        H_B_A = z(); H_B_C = z()
        B = A.compose(c, H_B_A, H_B_C)
        # Step 3: T_pred = B * J_joint
        H_T_B = z(); H_T_J = z()
        T_pred = B.compose(J_joint, H_T_B, H_T_J)  # H_T_J unused (J is constant)
        # Step 4: error = N2.localCoordinates(T_pred)
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

    # Add a prior on the first pose, setting it to the origin
    # A prior factor consists of a mean and a noise model (covariance matrix)
    priorMean = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
    graph.add(gtsam.PriorFactorPose2(L1_key, priorMean, ANCHOR_NOISE))

    # Add prior on the calibration nodes
    graph.add(gtsam.PriorFactorPose2(CL2_key, priorMean, CALIB_NOISE)) # joint 2 calibration
    graph.add(gtsam.PriorFactorPose2(CS1_key, priorMean, SENSOR_CALIB_NOISE)) # sensor 1 calibration
    graph.add(gtsam.PriorFactorPose2(CS2_key, priorMean, SENSOR_CALIB_NOISE)) # sensor 2 calibration

    # 2. Define custom factor using kinematics model
    graph.add(make_kinematics_factor(L1_key, CL2_key, L2_key, 20, np.pi/2, KINEMATIC_NOISE)) # joint connecting 1,2
    graph.add(make_kinematics_factor(L1_key, CS1_key, S1_key, 20, 0, KINEMATIC_NOISE)) # sensor 1
    graph.add(make_kinematics_factor(L2_key, CS2_key, S2_key, 20, 0, KINEMATIC_NOISE)) # sensor 2

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
    print("\nInitial Estimate:\n{}".format(initial))

    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print(result)


if __name__=="__main__":
    main()