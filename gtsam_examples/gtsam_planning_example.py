import gtsam
from gtsam import Pose2, symbol
import numpy as np

from gtsam_factors import make_fixed_kinematics_factor
from utils import get_connections, plot_chain, plot_side_by_side

# Create noise models
KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 0.02]))
ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 0.02]))
LOOSE_ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 10]))
GOAL_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.06, 0.06, 1000]))

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
        return gtsam.Symbol('j', index).key()

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
    sigma_endpoint = 10 # how stiff the endpoints states are from the optimal straight line path. larger values allows them to move further from the optimal straight line
    sigma_joint = 1 # how stiff the joints are, larger makes them looser and allows joints to bend more during movement
    time_horizon = 3
    
    # limb details
    num_limbs = 3
    limb_len = 20.0

    start = np.array([40.0, 20.0, 0.0])
    end = np.array([60.0, 0.0, 0.0])

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
        for i in range(1,num_limbs):
            graph.add(make_fixed_kinematics_factor(J(i, k), J(i+1, k), limb_len, 0.0, LOOSE_ANCHOR_NOISE)) # joint connecting i,i+1
        graph.add(make_fixed_kinematics_factor(J(num_limbs, k), E(num_limbs, k), limb_len, 0.0, KINEMATIC_NOISE)) # endpoint for last limb (num_limbs)

        # 3. Add initial values
        for i in range(1,num_limbs+1):
            initial.insert(J(i, k), gtsam.Pose2(limb_len*(i-1), 0.0, 0.0)) # limb i at its estimated position
        initial.insert(E(num_limbs, k), gtsam.Pose2(limb_len*num_limbs, 0.0, 0.0)) # last limb endpoint (num_limbs)
        
        for i in range(1,num_limbs+1):
            initial.insert(V(i, k), np.array([0.0], dtype=float)) # Joint i velocity scalar
        initial.insert(VE(num_limbs, k), gtsam.Point2(0.0, 0.0)) # Endpoint velocity vector

    # connect kinematics chains over times steps
    for k in range(time_horizon-1):
        # Add task space dynamics between endpoints
        graph.add(make_task_space_dynamics_factor(E(num_limbs, k), VE(num_limbs, k), E(num_limbs, k+1), VE(num_limbs, k+1), dt, sigma_endpoint))
        
        # Add joint space 1d rotation dynamics between joints
        for i in range(1,num_limbs+1):
            graph.add(make_joint_space_dynamics_factor(J(i, k), V(i, k), J(i, k+1), V(i, k+1), dt, sigma_joint))

    # add pior onto initial endpoint to hold the initial position stationary
    priorMean = gtsam.Pose2(start[0], start[1], start[2])  # prior at origin
    graph.add(gtsam.PriorFactorPose2(E(num_limbs, 0), priorMean, ANCHOR_NOISE))
    PRIOR_INDEX = graph.size() - 1 

    # add pior onto time horizon endpoint to pull robot to goal
    priorMean = gtsam.Pose2(end[0], end[1], end[2])  # prior at origin
    graph.add(gtsam.PriorFactorPose2(E(num_limbs, time_horizon-1), priorMean, GOAL_NOISE))

    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()

    steps = 10
    endpoint_paths = {"r1": []}
    for step in range(steps):
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()
        print(result)

        plot_keys = []
        for k in range(time_horizon):
            for i in range(1,num_limbs+1):
                plot_keys.append(J(i, k))
            plot_keys.append(E(num_limbs, k))
        conns = get_connections(graph)
        # store endpoint positions over time to track path, if loop because i dont initialise the robot to start in the position its priored on
        if step == 0:
            endpoint_paths["r1"].append((start[0], start[1]))
        else:
            er1 = initial.atPose2(E(num_limbs,0))
            endpoint_paths["r1"].append((er1.x(), er1.y()))
        plot_side_by_side(initial, result, plot_keys, conns, paths=endpoint_paths)
    
        # update the values for joints,endpoint,velocities to their next future state
        for k in range(time_horizon):
            next_k = k + 1 if k != time_horizon - 1 else k # set all states to their next state and the end state to itself
            
            # Shift Poses and Velocities for all Joints
            for i in range(1, num_limbs + 1):
                initial.update(J(i, k), result.atPose2(J(i, next_k)))
                initial.update(V(i, k), result.atVector(V(i, next_k)))
                
            # Shift Poses and Velocities for the Endpoint
            initial.update(E(num_limbs, k), result.atPose2(E(num_limbs, next_k)))
            initial.update(VE(num_limbs, k), result.atPoint2(VE(num_limbs, next_k)))

        # Update the start prior of the endpoint at k=0
        new_start_endpoint = result.atPose2(E(num_limbs, 1))
        new_factor = gtsam.PriorFactorPose2(E(num_limbs, 0), new_start_endpoint, ANCHOR_NOISE)
        graph.replace(PRIOR_INDEX, new_factor)

if __name__=="__main__":
    main()