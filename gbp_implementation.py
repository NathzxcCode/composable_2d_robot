import torch
import math
import random
from gbp_utilities import MeasModel, SquaredLoss, TukeyLoss, HuberLoss
from gbp import GBPSettings, FactorGraph

# =============================================================================
# Noise Control Parameters
# =============================================================================
# Angle measurement noise standard deviation in degrees
# Set to 0.0 to disable noise (deterministic)
ANGLE_NOISE_STD_DEG = 2.0

def add_noise_to_angle(angle_rad):
    """Add Gaussian noise to an angle measurement."""
    if ANGLE_NOISE_STD_DEG > 0:
        noise_rad = random.gauss(0, math.radians(ANGLE_NOISE_STD_DEG))
        return angle_rad + noise_rad
    return angle_rad
### custom factors designed for the composable 2d robot problem ###
#Custom Factor 1: Angle Measurement
def angle_meas_fn(x: torch.Tensor):
    # x is composed of two nodes: [x1, y1, theta1, x2, y2, theta2]
    # We predict the encoder measurement: theta2 - theta1
    diff = x[5] - x[2]
    
    # Wrap the angle to [-pi, pi] to prevent 360-degree error jumps
    return torch.tensor([torch.atan2(torch.sin(diff), torch.cos(diff))])
def angle_jac_fn(x: torch.Tensor):
    # The Jacobian is a 1x6 matrix. 
    # Derivative with respect to theta1 (index 2) is -1
    # Derivative with respect to theta2 (index 5) is 1
    return torch.tensor([[0., 0., -1., 0., 0., 1.]])
class AngleMeasurementModel(MeasModel):
    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, angle_meas_fn, angle_jac_fn, loss)
        self.linear = True  # Linear relations between angles
#Custom Factor 2: Kinematic Calibration
def kin_calib_meas_fn(x: torch.Tensor, L: float):
    # x is composed of: [x1, y1, theta1, C12x, C12y, x2, y2, theta2]
    # Predict where Node 2 should be based on Node 1, Calib, and L
    pred_x2 = x[0] + x[3] + L * torch.sin(x[7])
    pred_y2 = x[1] + x[4] + L * torch.cos(x[7])
    
    # Residual error: actual Node 2 position minus predicted position
    return torch.tensor([x[5] - pred_x2, x[6] - pred_y2])
def kin_calib_jac_fn(x: torch.Tensor, L: float):
    # The Jacobian is a 2x8 matrix
    J = torch.zeros(2, 8)
    
    # Partial derivatives for h_0 (the x2 equation)
    J[0, 0] = -1.  # wrt x1
    J[0, 3] = -1.  # wrt C12x
    J[0, 5] = 1.   # wrt x2
    J[0, 7] = -L * torch.cos(x[7])  # wrt theta2
    
    # Partial derivatives for h_1 (the y2 equation)
    J[1, 1] = -1.  # wrt y1
    J[1, 4] = -1.  # wrt C12y
    J[1, 6] = 1.   # wrt y2
    J[1, 7] = L * torch.sin(x[7])   # wrt theta2
    
    return J
class KinematicCalibModel(MeasModel):
    def __init__(self, loss: SquaredLoss, L: float) -> None:
        MeasModel.__init__(self, kin_calib_meas_fn, kin_calib_jac_fn, loss, L)
        self.linear = False
#custom factor 3: anchor factor for the base node
def endpoint_anchor_meas_fn(x: torch.Tensor, L: float):
    # x = [x_tip, y_tip, theta]
    # We calculate where the BASE would be given this tip and angle
    base_x = x[0] - L * torch.cos(x[2])
    base_y = x[1] - L * torch.sin(x[2])
    
    # The measurement we want is for the base to be at (0,0)
    return torch.tensor([base_x, base_y])
def endpoint_anchor_jac_fn(x: torch.Tensor, L: float):
    J = torch.tensor([
        [1.0, 0.0, L * torch.sin(x[2])],
        [0.0, 1.0, -L * torch.cos(x[2])]
    ])
    return J
class EndpointAnchorModel(MeasModel):
    def __init__(self, loss: SquaredLoss, L: float) -> None:
        MeasModel.__init__(self, endpoint_anchor_meas_fn, endpoint_anchor_jac_fn, loss, L)
        self.linear = False

def update_factor_graph(data, fg):
    limbs = data["limbs"]
    connections = data["connections"]
    limbs = sorted(limbs, key=lambda x: x['depth'])
    connections = sorted(connections, key=lambda x: x['depth'])
    for limb in limbs:
        limb_id = str(limb["id"])
        
        # add limb nodes to the factor graph (global x, y, theta of endpoint)
        fg.add_var_node(id=limb_id,
                        dofs=3,
                        # prior_mean=torch.tensor([limb["endpoint"]["x"], limb["endpoint"]["y"], math.radians(limb["global_angle"])]),
                        prior_mean=torch.tensor([limb["limb_length"]*limb["depth"], 0., 0.]),
                        prior_diag_cov=torch.tensor([1000., 1000., 10.]),  # Large variance = weak prior
                        properties=limb)
        
        # add calibration nodes for non-base limbs
        if fg.var_nodes.get("calib"+limb_id) is None:
            if limb["depth"] != 0:
                calib = limb.get("calibration", {})
                fg.add_var_node(id="calib"+limb_id,
                                dofs=2,
                                prior_mean=torch.tensor([calib.get("offset_x", 0.), calib.get("offset_y", 0.)]), 
                                prior_diag_cov=torch.tensor([1000., 1000.]),  # Large variance = weak prior
                                properties={})
        # add factor to base node to anchor its position
        if limb["depth"] == 0:
            base_node = fg.var_nodes[limb_id][-1]  # Use last node for temporal compatibility
            fg.add_factor(measurement=torch.tensor([0., 0.]),  # anchor base at (0, 0)
                          meas_model=anchor_model,
                          adj_var_nodes=[base_node],
                          properties={})
    for connection in connections:
        parent_id = str(connection["parent_id"])
        child_id = str(connection["child_id"])
        parent_node = fg.var_nodes[parent_id][-1]
        child_node = fg.var_nodes[child_id][-1]
        calib_node = fg.var_nodes["calib"+child_id][-1]   # Fixed: use last node (temporal compatible)
        angle_measure = child_node.properties["local_angle"]
        # Convert degrees to radians and add noise
        angle_measure_rad = math.radians(angle_measure)
        angle_measure_rad_noisy = add_noise_to_angle(angle_measure_rad)
        # add angle measurement factors
        fg.add_factor(measurement=torch.tensor([angle_measure_rad_noisy]), 
                      meas_model=angle_model,
                      adj_var_nodes=[parent_node, child_node],
                      properties={})
        # add kinematics factors
        fg.add_factor(measurement=torch.tensor([0.0, 0.0]),
                      meas_model=kinematic_model,
                      adj_var_nodes=[parent_node, calib_node, child_node], 
                      properties={})
# ===================================================================
# Setup
# ===================================================================
gbp_settings = GBPSettings(
    damping=0.5,
    beta=1.0,
    num_undamped_iters=3,
    min_linear_iters=5,
    dropout=0.0,
)
# loss functions for the factors - normalized variance scales
angle_loss = HuberLoss(1, torch.tensor([1.0]), 2.0)  # was 0.01
kinematic_loss = HuberLoss(2, torch.tensor([5.0, 5.0]), 2.0)  # was 0.05
anchor_loss = SquaredLoss(2, torch.tensor([1.0, 1.0]))  # was 1e-6
# Instantiate the models
angle_model = AngleMeasurementModel(angle_loss)
kinematic_model = KinematicCalibModel(kinematic_loss, L=140)
anchor_model = EndpointAnchorModel(anchor_loss, L=140)
# initialise the factor graph
fg = FactorGraph(gbp_settings)
data = {
    'limbs': [
        {'id': 1, 'local_angle': 95, 'global_angle': 190, 'position': {'x': 422.21, 'y': 440.92}, 
         'endpoint': {'x': 284.34, 'y': 416.61}, 'limb_length': 140, 'depth': 1},
        {'id': 2, 'local_angle': 95, 'global_angle': 95, 'position': {'x': 429, 'y': 306}, 
         'endpoint': {'x': 416.80, 'y': 445.47}, 'limb_length': 140, 'depth': 0},
        {'id': 3, 'local_angle': 265, 'global_angle': 455, 'position': {'x': 284.34, 'y': 416.61}, 
         'endpoint': {'x': 272.14, 'y': 556.08}, 'limb_length': 140, 'depth': 2}
    ],
    'connections': [
        {'child_id': 1, 'parent_id': 2, 'depth': 1, 'calibration': {'offset_x': 135, 'offset_y': -5}},
        {'child_id': 3, 'parent_id': 1, 'depth': 2, 'calibration': {'offset_x': 140, 'offset_y': 0}}
    ]
}

data2 = {
    'limbs': [
        {'id': 1, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 404, 'y': 394}, 
         'endpoint': {'x': 544, 'y': 394}, 'limb_length': 140, 'depth': 0}, 
        {'id': 2, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 544, 'y': 395},
         'endpoint': {'x': 684, 'y': 395}, 'limb_length': 140, 'depth': 1}, 
        {'id': 3, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 664, 'y': 395}, 
         'endpoint': {'x': 804, 'y': 395}, 'limb_length': 140, 'depth': 2}, 
        {'id': 4, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 801, 'y': 417}, 
         'endpoint': {'x': 941, 'y': 417}, 'limb_length': 140, 'depth': 3}, 
        {'id': 5, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 941, 'y': 440}, 
         'endpoint': {'x': 1081, 'y': 440}, 'limb_length': 140, 'depth': 4}
    ], 
    'connections': [
        {'child_id': 2, 'parent_id': 1, 'depth': 1, 'calibration': {'offset_x': 140, 'offset_y': 1}}, 
        {'child_id': 3, 'parent_id': 2, 'depth': 2, 'calibration': {'offset_x': 120, 'offset_y': 0}}, 
        {'child_id': 4, 'parent_id': 3, 'depth': 3, 'calibration': {'offset_x': 137, 'offset_y': 22}}, 
        {'child_id': 5, 'parent_id': 4, 'depth': 4, 'calibration': {'offset_x': 140, 'offset_y': 23}}
    ]
}

data3 = {
    'limbs': [
        {'id': 1, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 558, 'y': 395}, 
         'endpoint': {'x': 698, 'y': 395}, 'limb_length': 140, 'depth': 1}, 
        {'id': 2, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 431, 'y': 386}, 
         'endpoint': {'x': 571, 'y': 386}, 'limb_length': 140, 'depth': 0}, 
        {'id': 3, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 683, 'y': 372}, 
         'endpoint': {'x': 823, 'y': 372}, 'limb_length': 140, 'depth': 2}
    ], 
    'connections': [
        {'child_id': 1, 'parent_id': 2, 'depth': 1, 'calibration': {'offset_x': 127, 'offset_y': 9}}, 
        {'child_id': 3, 'parent_id': 1, 'depth': 2, 'calibration': {'offset_x': 125, 'offset_y': -23}}
    ]
}

update_factor_graph(data3, fg)
print("Factor graph updated successfully!")
print(f"Variables: {len(fg.var_nodes)}")
print(f"Factors: {len(fg.factors)}")

fg.gbp_solve(n_iters=50)
fg.print()