import gtsam
from gtsam import Pose2, symbol
import numpy as np

from gtsam_factors import make_fixed_kinematics_factor
from utils import get_connections, plot_chain, plot_side_by_side

# Create noise models
KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 0.02]))
ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 0.02]))
LOOSE_ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1000]))
GOAL_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))

## task space dynamics use cartesian velocity x_dot, y_dot. used to plan end-effector in task space
def make_task_space_dynamics_factor(key_p1, key_v1, key_p2, key_v2, dt: float, sigma: float):
    """
    4-node dynamics factor (constant velocity model):
      p1, p2: Pose2 (We extract only the translation)
      v1, v2: Vector2 (Linear velocity)
    """
    
    # -------------------------------------------------------------------------
    # 1. NOISE MODEL: Matching the paper's Qi_inv
    # -------------------------------------------------------------------------
    I = np.eye(2)
    Qc_inv = (sigma ** -2.0) * I

    # Build the 4x4 precision (Information) matrix exactly as in the C++ code
    Qi_inv = np.zeros((4, 4))
    Qi_inv[0:2, 0:2] = 12.0 * (dt ** -3.0) * Qc_inv
    Qi_inv[0:2, 2:4] = -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[2:4, 0:2] = -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[2:4, 2:4] =  4.0 / dt * Qc_inv

    # GTSAM's Information noise model directly accepts the inverse covariance matrix
    noise_model = gtsam.noiseModel.Gaussian.Information(Qi_inv)

    # -------------------------------------------------------------------------
    # 2. ERROR & JACOBIAN FUNCTION
    # -------------------------------------------------------------------------
    def error_func(this, values, jacobians):
        p1 = values.atPose2(this.keys()[0])
        v1 = values.atVector(this.keys()[1])  # 2D numpy array
        p2 = values.atPose2(this.keys()[2])
        v2 = values.atVector(this.keys()[3])  # 2D numpy array

        # Helper to create Fortran-order zero matrices for GTSAM out-parameters
        def z(rows, cols):
            return np.zeros((rows, cols), order='F')

        # Extract translations and their Jacobians wrt the Pose2 tangent space
        H_t1 = z(2, 3)
        H_t2 = z(2, 3)
        pos1 = p1.translation(H_t1)
        pos2 = p2.translation(H_t2)

        # Compute the continuous-velocity errors
        err_pos = pos1 + (v1 * dt) - pos2
        err_vel = v1 - v2
        
        # Stack into the final 4D error vector
        error = np.hstack((err_pos, err_vel))

        # -------------------------------------------------------------------------
        # 3. JACOBIANS: Mapping the C++ linear matrix to the Pose2 Manifold
        # -------------------------------------------------------------------------
        if jacobians is not None:
            # J1: Derivative of 4D error wrt Pose1 (3 DOF) -> 4x3 matrix
            J1 = z(4, 3)
            J1[0:2, :] = H_t1
            # Bottom 2 rows stay 0 (changing pose doesn't instantly change velocity)
            jacobians[0] = J1

            # J2: Derivative of 4D error wrt Vel1 (2 DOF) -> 4x2 matrix
            J2 = z(4, 2)
            J2[0:2, :] = I * dt
            J2[2:4, :] = I
            jacobians[1] = J2

            # J3: Derivative of 4D error wrt Pose2 (3 DOF) -> 4x3 matrix
            J3 = z(4, 3)
            J3[0:2, :] = -H_t2
            jacobians[2] = J3

            # J4: Derivative of 4D error wrt Vel2 (2 DOF) -> 4x2 matrix
            J4 = z(4, 2)
            J4[2:4, :] = -I
            jacobians[3] = J4

        return error
    
    keys = gtsam.KeyVector()
    keys.append(key_p1)
    keys.append(key_v1)
    keys.append(key_p2)
    keys.append(key_v2)
    
    return gtsam.CustomFactor(noise_model, keys, error_func)

## joint space dynamics uses rotation velocity. Used to control robot joints by angular velocity
def make_joint_space_dynamics_factor(key_p1, key_v1, key_p2, key_v2, dt: float, sigma: float):
    """
    4-node 1D dynamics factor for joint angles:
      p1, p2: Pose2 (We extract ONLY the theta rotation component)
      v1, v2: Vector1 (Scalar angular velocity, e.g., np.array([omega]))
    """
    # 1. Information Matrix for 1D (scalar) system
    Qc_inv = sigma ** -2.0
    Qi_inv = np.zeros((2, 2))
    Qi_inv[0, 0] = 12.0 * (dt ** -3.0) * Qc_inv
    Qi_inv[0, 1] = -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[1, 0] = -6.0 * (dt ** -2.0) * Qc_inv
    Qi_inv[1, 1] =  4.0 / dt * Qc_inv

    noise_model = gtsam.noiseModel.Gaussian.Information(Qi_inv)

    def error_func(this, values, jacobians):
        p1 = values.atPose2(this.keys()[0])
        v1 = values.atVector(this.keys()[1])[0] # Extract scalar velocity
        p2 = values.atPose2(this.keys()[2])
        v2 = values.atVector(this.keys()[3])[0]

        # Extract only the angular values
        theta1 = p1.theta()
        theta2 = p2.theta()

        # Compute 1D errors (wrapped safely if your joints spin fully)
        raw_error = theta1 + (v1 * dt) - theta2
        err_pos = np.arctan2(np.sin(raw_error), np.cos(raw_error))
        err_vel = v1 - v2
        error = np.array([err_pos, err_vel])

        if jacobians is not None:
            def z(rows, cols): return np.zeros((rows, cols), order='F')

            # J1: wrt Pose1 tangent space [dx, dy, dtheta]. Only dtheta affects error.
            J1 = z(2, 3)
            J1[0, 2] = 1.0  # d(err_pos) / dtheta1
            jacobians[0] = J1

            # J2: wrt Vel1 (1 DOF scalar)
            J2 = z(2, 1)
            J2[0, 0] = dt
            J2[1, 0] = 1.0
            jacobians[1] = J2

            # J3: wrt Pose2 tangent space
            J3 = z(2, 3)
            J3[0, 2] = -1.0  # d(err_pos) / dtheta2
            jacobians[2] = J3

            # J4: wrt Vel2 (1 DOF scalar)
            J4 = z(2, 1)
            J4[1, 0] = -1.0
            jacobians[3] = J4

        return error

    keys = gtsam.KeyVector()
    keys.append(key_p1)
    keys.append(key_v1)
    keys.append(key_p2)
    keys.append(key_v2)
    
    return gtsam.CustomFactor(noise_model, keys, error_func)


def main():
    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # 1. Define Symbols for variables
    def J(joint_id, t):
        index = (joint_id * 1000) + t
        return gtsam.Symbol('l', index).key()

    def E(effector_id, t):
        index = (effector_id * 1000) + t
        return gtsam.Symbol('e', index).key()
    
    def VE(endpoint_velocity_id, t):
        index = (endpoint_velocity_id * 1000) + t
        return gtsam.Symbol('v', index).key()

    def V(joint_velocity_id, t):
        index = (joint_velocity_id * 1000) + t
        return gtsam.Symbol('o', index).key()

    dt = 0.1
    sigma_endpoint = 5 # how stiff the endpoints states are from the optimal straight line path. larger values allows them to move further from the optimal straight line
    sigma_joint = 5 # how stiff the joints are, larger makes them looser and allows joints to bend more during movement
    time_horizon = 10

    # build the robot chains for each timestep
    for k in range(time_horizon):
        # 1. Anchor only the first timesteps root to the origin
        if k == 0:
            priorMean = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
            graph.add(gtsam.PriorFactorPose2(J(1, k), priorMean, ANCHOR_NOISE))
        else:
            priorMean = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
            graph.add(gtsam.PriorFactorPose2(J(1, k), priorMean, LOOSE_ANCHOR_NOISE))

        # 2. Define fixed kinematics factors of the robot
        graph.add(make_fixed_kinematics_factor(J(1, k), J(2, k), 20.0, np.pi/2, KINEMATIC_NOISE)) # joint connecting 1,2
        graph.add(make_fixed_kinematics_factor(J(2, k), E(2, k), 20.0, 0.0, KINEMATIC_NOISE)) # endpoint for limb 2

        # 3. Add initial values
        initial.insert(J(1, k), gtsam.Pose2(0.0, 0.0, 0.0)) # limb 1
        initial.insert(J(2, k), gtsam.Pose2(20.0, 0.0, 0.0)) # limb 2
        initial.insert(E(2, k), gtsam.Pose2(40.0, 0.0, 0.0)) # limb 2 endpoint
        
        initial.insert(V(1, k), np.array([0.0], dtype=float)) # Joint 1 velocity scalar
        initial.insert(V(2, k), np.array([0.0], dtype=float)) # Joint 2 velocity scalar
        initial.insert(VE(2, k), gtsam.Point2(0.0, 0.0)) # Endpoint velocity vector

    # connect root chaings over times steps
    for k in range(time_horizon-1):
        # Add task space dynamics between endpoints
        graph.add(make_task_space_dynamics_factor(E(2, k), VE(2, k), E(2, k+1), VE(2, k+1), dt, sigma_endpoint))
        
        # Add joint space 1d rotation dynamics between joints
        graph.add(make_joint_space_dynamics_factor(J(1, k), V(1, k), J(1, k+1), V(1, k+1), dt, sigma_joint))
        graph.add(make_joint_space_dynamics_factor(J(2, k), V(2, k), J(2, k+1), V(2, k+1), dt, sigma_joint))

    # add pior onto initial endpoint to hold the initial position stationary
    priorMean = gtsam.Pose2(20.0, 20.0, 0)  # prior at origin
    graph.add(gtsam.PriorFactorPose2(E(2, 0), priorMean, ANCHOR_NOISE))

    # add pior onto time horizon endpoint to pull robot to goal
    priorMean = gtsam.Pose2(40.0, 00.0, 0)  # prior at origin
    graph.add(gtsam.PriorFactorPose2(E(2, time_horizon-1), priorMean, GOAL_NOISE))

    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print(result)

    plot_keys = []
    for k in range(time_horizon):
        plot_keys.extend([J(1, k), J(2, k), E(2, k)])
    conns = get_connections(graph)
    plot_side_by_side(initial, result, plot_keys, conns)
    


if __name__=="__main__":
    main()