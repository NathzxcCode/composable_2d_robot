import gtsam
import numpy as np

# ---------------------------------------------------------------------------
# 3-D version  (Pose3 nodes)
# ---------------------------------------------------------------------------

def make_calib_kinematics_factor_3d(key_n1, key_c, key_n2,
                                     L: float,
                                     joint_axis, joint_angle: float,
                                     noise_model):
    """
    3-node kinematic calibration factor for Pose3 nodes.

    Predicts:
        T_pred = N1 * T_link * C * T_joint
    where
        T_link  = Pose3(I, Point3(L, 0, 0))          -- nominal link length along local X
        C       = Pose3 calibration variable           -- absorbs attachment error
        T_joint = Pose3(Rot3.Rodrigues(axis*angle), 0) -- baked-in joint rotation

    Error (6-vector in tangent space):
        error = Log( N2^{-1} * T_pred )

    Analytic Jacobians via Pose3.between + Pose3.Logmap chain rule.

    Args:
        key_n1:       GTSAM key for parent joint Pose3
        key_c:        GTSAM key for calibration Pose3 (singleton)
        key_n2:       GTSAM key for child joint Pose3
        L:            Nominal link length (metres) along the parent local X-axis
        joint_axis:   3-element array-like, unit vector of the hinge axis in the parent frame
        joint_angle:  Current joint angle (radians) – baked in as a constant
        noise_model:  6-DOF noise model (sigmas on [rot3, trans3])
    """
    T_link  = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(float(L), 0.0, 0.0))
    ax      = np.asarray(joint_axis, dtype=float)
    rod     = ax * float(joint_angle)
    T_joint = gtsam.Pose3(gtsam.Rot3.Rodrigues(rod[0], rod[1], rod[2]),
                           gtsam.Point3(0.0, 0.0, 0.0))

    def error_func(this, values, jacobians):
        n1 = values.atPose3(this.keys()[0])
        c  = values.atPose3(this.keys()[1])
        n2 = values.atPose3(this.keys()[2])

        def z66():
            return np.zeros((6, 6), order='F')

        # Step 1: A = N1 * T_link
        H_A_N1 = z66(); H_A_L = z66()
        A = n1.compose(T_link, H_A_N1, H_A_L)          # H_A_L unused (constant)

        # Step 2: B = A * C
        H_B_A = z66(); H_B_C = z66()
        B = A.compose(c, H_B_A, H_B_C)

        # Step 3: T_pred = B * T_joint
        H_T_B = z66(); H_T_J = z66()
        T_pred = B.compose(T_joint, H_T_B, H_T_J)      # H_T_J unused (constant)

        # Step 4: error = Log( N2^{-1} * T_pred ) via between + Logmap
        H_b_N2 = z66(); H_b_T = z66()
        between = n2.between(T_pred, H_b_N2, H_b_T)

        H_log = z66()
        error = gtsam.Pose3.Logmap(between, H_log)

        if jacobians is not None:
            # de/dN2  = H_log @ H_b_N2
            # de/dT_pred = H_log @ H_b_T
            H_e_N2 = H_log @ H_b_N2
            H_e_T  = H_log @ H_b_T

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
# 2-D version  (Pose2 nodes) — unchanged
# ---------------------------------------------------------------------------

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