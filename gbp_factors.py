import torch
from gbp_utilities import MeasModel, SquaredLoss, HuberLoss, TukeyLoss

def get_SE2_matrix(x, y, theta):
    """Helper to build an SE(2) 3x3 transformation matrix."""
    theta = torch.tensor([theta]) if not torch.is_tensor(theta) else theta
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.tensor([
        [cos_t, -sin_t, x],
        [sin_t,  cos_t, y],
        [0.0,    0.0,   1.0]
    ])

def SE2_log_map(T: torch.Tensor, eps: float = 1e-6):
    """
    Upgraded SE(2) Log Map using the V-matrix inverse.
    T: 3x3 SE(2) matrix
    eps: Threshold for Taylor expansion to avoid division by zero
    Returns: [dx, dy, dtheta]
    """
    # 1. Extract rotation error (The "Heading")
    # T[1,0] is sin(theta), T[0,0] is cos(theta)
    d_theta = torch.atan2(T[1, 0], T[0, 0])
    
    # 2. Extract raw translation components
    tx = T[0, 2]
    ty = T[1, 2]
    t_vec = torch.stack([tx, ty])

    # 3. Handle the Singularity using Taylor Series
    if torch.abs(d_theta) < eps:
        # As theta -> 0:
        # A = sin(theta)/theta -> 1 - theta^2/6
        # B = (1 - cos(theta))/theta -> theta/2 - theta^3/24
        # At the limit, V^-1 is just the Identity matrix.
        # We use a 2nd order expansion for stability near zero.
        A = 1.0 - (d_theta**2) / 6.0
        B = d_theta / 2.0 - (d_theta**3) / 24.0
    else:
        # Standard formulas
        A = torch.sin(d_theta) / d_theta
        B = (1.0 - torch.cos(d_theta)) / d_theta

    # 4. Construct V^-1 coefficients
    # Formula: V_inv = (1 / (A^2 + B^2)) * [[A, B], [-B, A]]
    denom = A**2 + B**2
    
    # Apply coefficients to recover Lie Algebra translation (rho)
    dx = (1.0 / denom) * (A * tx + B * ty)
    dy = (1.0 / denom) * (-B * tx + A * ty)

    return torch.stack([dx, dy, d_theta])

def SE2_adjoint(T):
    """Calculates the 3x3 Adjoint matrix of an SE(2) pose."""
    x = T[0, 2]
    y = T[1, 2]
    cos_t = T[0, 0]
    sin_t = T[1, 0]
    return torch.tensor([
        [cos_t, -sin_t,  y],
        [sin_t,  cos_t, -x],
        [0.0,    0.0,    1.0]
    ])


## kinematics factor joint-joint ##
def kin_calib_meas_fn(x: torch.Tensor, L: float, theta_joint: float): # 
    """
    Residual function using the SE(2) Log Map.
    x is [x1, y1, theta1, Cx, Cy, Ctheta, x2, y2, theta2]
    """
    # 1. Parse current states into Pose Matrices
    N1 = get_SE2_matrix(x[0], x[1], x[2])
    C2 = get_SE2_matrix(x[3], x[4], x[5])
    N2 = get_SE2_matrix(x[6], x[7], x[8])
    
    # 2. Define the fixed Joint and Limb geometry
    # Joint rotation from sensor + Limb length translation
    J1 = get_SE2_matrix(0, 0, torch.tensor(theta_joint))
    L_link = get_SE2_matrix(torch.tensor(L), 0, 0) # Limb along local x-axis
    
    # 3. Predict Node 2: T_pred = N1 * J1 * L * C2
    T_pred = N1 @ L_link @ C2 @ J1
    
    # 4. Calculate relative error on manifold: Delta_T = N2^-1 * T_pred
    # Note: Invert N2 to compare T_pred in the local frame of N2
    N2_inv = torch.inverse(N2)
    delta_T = N2_inv @ T_pred
    
    # 5. Residual is the Log map (tangent space vector)
    return SE2_log_map(delta_T)

def kin_calib_jac_fn(x: torch.Tensor, L: float, theta_joint: float): #  
    """
    Analytic Jacobian based on Lie Group Adjoints.
    The Jacobian is 3x9 (3-DOF residual, 9-DOF state).
    """
    # Parse states for Adjoint calculations
    N1 = get_SE2_matrix(x[0], x[1], x[2])
    C2 = get_SE2_matrix(x[3], x[4], x[5])
    N2 = get_SE2_matrix(x[6], x[7], x[8])
    J1 = get_SE2_matrix(0, 0, torch.tensor(theta_joint))
    L_link = get_SE2_matrix(torch.tensor(L), 0, 0)
    
    N2_inv = torch.inverse(N2)
    
    # Jacobian wrt Node 2 (N2)
    # Since N2 is the inverse in our Log map: r = Log(N2^-1 * T_pred)
    # The local perturbation Jacobian is approximately -Identity
    J_n2 = -torch.eye(3)
    
    # Jacobian wrt Node 1 (N1)
    # We use the Adjoint to map the perturbation of N1 through the chain to N2's frame
    J_n1 = SE2_adjoint(N2_inv)
    
    # Jacobian wrt Calibration (C2)
    # The perturbation is at the end of the chain, so we map it into N2's frame
    T_chain = N2_inv @ N1 @ L_link
    J_c2 = SE2_adjoint(T_chain)
    
    # Concatenate into a 3x9 matrix
    return torch.cat([J_n1, J_c2, J_n2], dim=1)

class KinematicCalibModel(MeasModel):
    def __init__(self, loss: HuberLoss, L: float, theta_joint: float) -> None: # 
        # Pass L and theta_joint as extra args for the meas/jac functions
        MeasModel.__init__(self, kin_calib_meas_fn, kin_calib_jac_fn, loss, L, theta_joint) #
        self.linear = False

## anchor factor origin-base_joint ##
def anchor_meas_fn(x: torch.Tensor, T_origin: torch.Tensor = None):
    """
    Anchor Residual: r = Log(N1^-1 * T_origin)
    x: [x1, y1, theta1] (the node we are anchoring)
    T_origin: 3x3 SE(2) matrix (defaults to Identity if None)
    """
    # get origin matrix 
    T_origin = get_SE2_matrix(T_origin[0], T_origin[1], T_origin[2])

    # 1. Convert current state to Pose Matrix
    N1 = get_SE2_matrix(x[0], x[1], x[2])
    
    # 2. Calculate relative error: Delta_T = N1^-1 * T_origin
    N1_inv = torch.inverse(N1)
    delta_T = N1_inv @ T_origin
    
    # 3. Apply the upgraded Log map to get the 3D residual vector
    return SE2_log_map(delta_T)

def anchor_jac_fn(x: torch.Tensor, T_origin: torch.Tensor = None):
    """
    The Jacobian for the anchor.
    Since r = Log(N1^-1 * T_origin), the gradient wrt N1 is -Identity.
    """
    # The output is a 3x3 matrix (3-DOF residual wrt 3-DOF state)
    return -torch.eye(3)

class AnchorModel(MeasModel):
    def __init__(self, loss: SquaredLoss, T_origin: torch.tensor) -> None:
        # T_origin defaults to Identity in our anchor_meas_fn
        MeasModel.__init__(self, anchor_meas_fn, anchor_jac_fn, loss, T_origin)
        self.linear = False

## limb endpoint factor joint-endpoint ##
def endpoint_meas_fn(x: torch.Tensor, L_last: float):
    """
    Endpoint Residual: r = Log(E^-1 * (N_last * L_last))
    x: [x_j, y_j, th_j, x_e, y_e, th_e]
    L_last: Length of the final limb
    """
    # 1. Parse into Pose Matrices
    N_last = get_SE2_matrix(x[0], x[1], x[2])
    E = get_SE2_matrix(x[3], x[4], x[5])
    
    # 2. Fixed limb transform (translation along local x-axis)
    L_link = get_SE2_matrix(torch.tensor(L_last), 0, 0)
    
    # 3. Predict where endpoint should be
    T_pred = N_last @ L_link
    
    # 4. Calculate relative error: Delta_T = E^-1 * T_pred
    E_inv = torch.inverse(E)
    delta_T = E_inv @ T_pred
    
    return SE2_log_map(delta_T)

def endpoint_jac_fn(x: torch.Tensor, L_last: float):
    """
    Jacobian for the Endpoint Factor.
    Output: 3x6 matrix (Residual is 3D, States are Joint + Endpoint)
    """
    # Parse for Adjoint
    E = get_SE2_matrix(x[3], x[4], x[5])
    E_inv = torch.inverse(E)
    
    # 1. Jacobian wrt Parent Joint (N_last)
    # We use the Adjoint to map N_last's perturbation into E's frame
    J_joint = SE2_adjoint(E_inv)
    
    # 2. Jacobian wrt Endpoint Node (E)
    # Since E is the reference frame being inverted:
    J_endpoint = -torch.eye(3)
    
    # Concatenate into a 3x6 matrix
    return torch.cat([J_joint, J_endpoint], dim=1)

class EndpointModel(MeasModel):
    def __init__(self, loss: SquaredLoss, L_last: float) -> None:
        MeasModel.__init__(self, endpoint_meas_fn, endpoint_jac_fn, loss, L_last)
        self.linear = False


## distance measurement factor endpoint-endpoint ##
## TO:DO figure out how the measurement it inserted into this function whether though class or other
def distance_meas_fn(x: torch.Tensor, s1_local: torch.Tensor, s2_local: torch.Tensor):
    """
    Distance Residual: r = ||P2 - P1|| - d_measured
    x: [x1, y1, th1, x2, y2, th2]
    s1_local, s2_local: [x, y] offsets of sensors relative to their joints
    """
    # 1. Calculate global sensor positions
    # P1 = R1 * s1 + t1
    cos1, sin1 = torch.cos(x[2]), torch.sin(x[2])
    p1_x = x[0] + cos1 * s1_local[0] - sin1 * s1_local[1]
    p1_y = x[1] + sin1 * s1_local[0] + cos1 * s1_local[1]

    # P2 = R2 * s2 + t2
    cos2, sin2 = torch.cos(x[5]), torch.sin(x[5])
    p2_x = x[3] + cos2 * s2_local[0] - sin2 * s2_local[1]
    p2_y = x[4] + sin2 * s2_local[0] + cos2 * s2_local[1]

    # 2. Predicted Euclidean distance
    d_pred = torch.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
    
    # The actual measurement (d_measured) is usually passed via args or subtracted outside
    return d_pred

def distance_jac_fn(x: torch.Tensor, s1_local: torch.Tensor, s2_local: torch.Tensor):
    """
    Jacobian for Distance Factor (1x6 matrix).
    Relates [dx1, dy1, dth1, dx2, dy2, dth2] to change in distance.
    """
    # Re-calculate points for the gradient
    cos1, sin1 = torch.cos(x[2]), torch.sin(x[2])
    p1 = torch.tensor([x[0] + cos1 * s1_local[0] - sin1 * s1_local[1],
                       x[1] + sin1 * s1_local[0] + cos1 * s1_local[1]])

    cos2, sin2 = torch.cos(x[5]), torch.sin(x[5])
    p2 = torch.tensor([x[3] + cos2 * s2_local[0] - sin2 * s2_local[1],
                       x[4] + sin2 * s2_local[0] + cos2 * s2_local[1]])

    diff = p2 - p1
    d = torch.norm(diff)
    
    # Unit vector from P1 to P2 (the direction the distance grows)
    u = diff / (d + 1e-6)

    # Gradient wrt P1 is -u, wrt P2 is u
    # Now use chain rule: d(dist)/d(theta) = d(dist)/dP * dP/d(theta)
    
    # Derivatives of P1 wrt [x1, y1, th1]
    dp1_dth = torch.tensor([-sin1 * s1_local[0] - cos1 * s1_local[1],
                             cos1 * s1_local[0] - sin1 * s1_local[1]])
    
    j_n1 = torch.tensor([-u[0], -u[1], torch.dot(-u, dp1_dth)])

    # Derivatives of P2 wrt [x2, y2, th2]
    dp2_dth = torch.tensor([-sin2 * s2_local[0] - cos2 * s2_local[1],
                             cos2 * s2_local[0] - sin2 * s2_local[1]])
    
    j_n2 = torch.tensor([u[0], u[1], torch.dot(u, dp2_dth)])

    return torch.cat([j_n1, j_n2]).view(1, 6)

class DistanceMeasurementModel(MeasModel):
    def __init__(self, loss: TukeyLoss, s1: torch.Tensor, s2: torch.Tensor) -> None:
        # Args: sensor 1 local offset, sensor 2 local offset
        MeasModel.__init__(self, distance_meas_fn, distance_jac_fn, loss, s1, s2)
        self.linear = False


## joint angle measurement factor joint1-joint2-angle_measurement ##
# def angle_meas_fn(x: torch.Tensor):
#     # x is composed of two nodes: [x1, y1, theta1, x2, y2, theta2]
#     # We predict the encoder measurement: theta2 - theta1
#     diff = x[5] - x[2]
    
#     # Wrap the angle to [-pi, pi] to prevent 360-degree error jumps
#     return torch.tensor([torch.atan2(torch.sin(diff), torch.cos(diff))])
# def angle_jac_fn(x: torch.Tensor):
#     # The Jacobian is a 1x6 matrix. 
#     # Derivative with respect to theta1 (index 2) is -1
#     # Derivative with respect to theta2 (index 5) is 1
#     return torch.tensor([[0., 0., -1., 0., 0., 1.]])
# class AngleMeasurementModel(MeasModel):
#     def __init__(self, loss: SquaredLoss) -> None:
#         MeasModel.__init__(self, angle_meas_fn, angle_jac_fn, loss)
#         self.linear = True  # Linear relations between angles


## joint angle measurement factor joint1-joint2-angle_measurement ##
def angle_meas_fn(x: torch.Tensor, measured_angle: torch.Tensor):
    """
    x: [x1, y1, th1, x2, y2, th2]
    measured_angle: the constant encoder reading (rad)
    """
    # 1. Predicted relative angle from current node states
    predicted_diff = x[5] - x[2]
    
    # 2. Raw residual (error)
    error = predicted_diff - measured_angle
    
    # 3. Wrap the error to [-pi, pi] 
    # This ensures the solver takes the "short way" around the circle
    wrapped_error = torch.atan2(torch.sin(error), torch.cos(error))
    
    return wrapped_error.view(1)

def angle_jac_fn(x: torch.Tensor, measured_angle: torch.Tensor):
    # The derivative of the wrapped error is still 1 and -1 
    # everywhere except exactly at the wrap-around point.
    return torch.tensor([[0., 0., -1., 0., 0., 1.]])

class AngleMeasurementModel(MeasModel):
    def __init__(self, loss: HuberLoss, measured_angle: torch.Tensor) -> None:
        # Pass the actual measurement (constant) into the model args
        MeasModel.__init__(self, angle_meas_fn, angle_jac_fn, loss, measured_angle)
        
        # CRITICAL: Set to False. 
        # The jump at pi means the solver MUST re-evaluate the 
        # residual logic at every iteration.
        self.linear = False