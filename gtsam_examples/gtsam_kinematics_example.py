import gtsam
from gtsam import Pose2, symbol
import numpy as np
import matplotlib.pyplot as plt
from gtsam import symbolChr, symbolIndex

# Create noise models
KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5]))
ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4]))
CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
SENSOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

def plot_chain(values, keys_to_plot, connections, title="Kinematic Chain", arrow_len=1.0, ax=None):
    """Plot selected poses as (x,y) points with orientation arrows, connected by lines."""
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(7, 7))
    idx = {}
    xs, ys, thetas, labels = [], [], [], []
    for i, key in enumerate(keys_to_plot):
        p = values.atPose2(key)
        xs.append(p.x())
        ys.append(p.y())
        thetas.append(p.theta())
        labels.append(f"{chr(symbolChr(key))}{symbolIndex(key)}")
        idx[key] = i
    for k1, k2 in connections:
        if k1 in idx and k2 in idx:
            i1, i2 = idx[k1], idx[k2]
            ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]], 'b-', alpha=0.4, lw=2)
    for x, y, th in zip(xs, ys, thetas):
        dx = arrow_len * np.cos(th)
        dy = arrow_len * np.sin(th)
        ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='r', ec='r', alpha=0.7)
    ax.scatter(xs, ys, s=60, c='blue', zorder=3)
    for lab, x, y in zip(labels, xs, ys):
        ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)
    if own_fig:
        plt.show()

def get_connections(graph):
    """Extract kinematic connections from graph factors (skipping priors/1-key factors)."""
    connections = []
    for i in range(graph.size()):
        factor = graph.at(i)
        keys = list(factor.keys())
        if len(keys) >= 2:
            for j in range(len(keys) - 1):
                connections.append((keys[j], keys[j+1]))
    return connections

## relate 2 poses via a transformation and calibration allowind for adjustment. ued to relate joints of joint and sensor
def make_calib_kinematics_factor(key_n1, key_c, key_n2, L: float, theta_joint: float, noise_model):
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

## relate 2 poses by a strict translation with no calibration. used for constant offsets like limb endpoints
def make_fixed_kinematics_factor(key_n1, key_n2, L: float, theta_joint: float, noise_model):
    """
    2-node kinematic calibration factor:
      T_pred = N1 * Pose2(L,0,0) * Pose2(0,0,theta_joint)
      error  = N2.localCoordinates(T_pred)   [3-vector in tangent space of N2]
    Jacobians computed analytically via GTSAM Pose2 chain rule.
    """
    L_link  = gtsam.Pose2(L, 0.0, 0.0)
    J_joint = gtsam.Pose2(0.0, 0.0, theta_joint)
    def error_func(this, values, jacobians):
        n1 = values.atPose2(this.keys()[0])
        n2 = values.atPose2(this.keys()[1])
        def z():
            return np.zeros((3, 3), order='F')  # MUST be Fortran-order
        # Step 1: A = N1 * L_link
        H_A_N1 = z(); H_A_L = z()
        A = n1.compose(L_link, H_A_N1, H_A_L)   # H_A_L unused (L is constant)
        # Step 2: T_pred = A * J_joint
        H_T_A = z(); H_T_J = z()
        T_pred = A.compose(J_joint, H_T_A, H_T_J)  # H_T_J unused (J is constant)
        # Step 3: error = N2.localCoordinates(T_pred)
        H_e_N2 = z(); H_e_T = z()
        error = n2.localCoordinates(T_pred, H_e_N2, H_e_T)
        if jacobians is not None:
            jacobians[0] = H_e_T @ H_T_A @ H_A_N1  # de/dN1
            jacobians[1] = H_e_N2                  # de/dN2
        return error
    keys = gtsam.KeyVector()
    keys.append(key_n1)
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

    plot_keys = [L1_key, L2_key, E2_key]
    conns = get_connections(graph)
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_chain(initial, plot_keys, conns, "Initial Estimate", ax=axes[0])
    plot_chain(result, plot_keys, conns, "Optimized Result", ax=axes[1])
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()