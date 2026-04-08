import torch
from gbp_utilities import MeasModel, SquaredLoss, TukeyLoss, HuberLoss
from gbp import GBPSettings, FactorGraph

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
    J[0, 7] = -L * torch.cos(x[7])  # wrt theta2: d/d0 (-L * sin(0))
    
    # Partial derivatives for h_1 (the y2 equation)
    J[1, 1] = -1.  # wrt y1
    J[1, 4] = -1.  # wrt C12y
    J[1, 6] = 1.   # wrt y2
    J[1, 7] = L * torch.sin(x[7])   # wrt theta2: d/d0 (-L * cos(0))
    
    return J

class KinematicCalibModel(MeasModel):
    def __init__(self, loss: SquaredLoss, L: float) -> None:
        MeasModel.__init__(self, kin_calib_meas_fn, kin_calib_jac_fn, loss, L)
        self.linear = False  # Mark as False to trigger JIT linearizations


#custom factor 3: anchor factor for the base node
def endpoint_anchor_meas_fn(x: torch.Tensor, L: float):
    # x = [x_tip, y_tip, theta]
    # We calculate where the BASE would be given this tip and angle
    base_x = x[0] - L * torch.cos(x[2])
    base_y = x[1] - L * torch.sin(x[2])
    
    # The measurement we want is for the base to be at (0,0)
    return torch.tensor([base_x, base_y])

def endpoint_anchor_jac_fn(x: torch.Tensor, L: float):
    # Jacobian of [base_x, base_y] with respect to [x_tip, y_tip, theta]
    # d(base_x)/dx_tip = 1, d(base_x)/dy_tip = 0, d(base_x)/d_theta = L*sin(theta)
    # d(base_y)/dx_tip = 0, d(base_y)/dy_tip = 1, d(base_y)/d_theta = -L*cos(theta)
    
    J = torch.tensor([
        [1.0, 0.0, L * torch.sin(x[2])],
        [0.0, 1.0, -L * torch.cos(x[2])]
    ])
    return J

class EndpointAnchorModel(MeasModel):
    def __init__(self, loss: SquaredLoss, L: float) -> None:
        # We pass the limb length L as a parameter
        MeasModel.__init__(self, endpoint_anchor_meas_fn, endpoint_anchor_jac_fn, loss, L)
        self.linear = False


def update_factor_graph(data, fg):
    # 1) add limb nodes x,y,theta
    # 2) get calibration nodes, if none then set them
    # 3) add angle measurement factors. needs: parent node, child node, angle measurement
    # 4) add kinematic factors. needs: parent node, calibration node, child node

    limbs = data["limbs"]
    connections = data["connections"]
    limbs = sorted(limbs, key=lambda x: x['depth'])
    connections = sorted(connections, key=lambda x: x['depth'])

    for limb in limbs:
        # add limb nodes to the factor graph. represent current pose
        limb_id = str(limb["id"])
        fg.add_var_node(id=limb_id,
                        dofs=3,
                        prior_mean=torch.tensor([limb["limb_length"]*limb["depth"],0.,0.]),
                        prior_diag_cov=torch.tensor([10.,10.,2.]),
                        properties=limb)
        
        # if this is the first set of nodes then add calib nodes for each child node
        if fg.var_nodes.get("calib"+limb_id) == None:
            if limb["depth"] != 0:
                fg.add_var_node(id="calib"+limb_id,
                                dofs=2,
                                prior_mean=torch.tensor([0.,0.]), 
                                prior_diag_cov=torch.tensor([limb["limb_length"]/2, 10, 2]),
                                properties={})

        # add factor to base node to anchor its position
        if limb["depth"] == 0:
            fg.add_factor(measurement=torch.tensor([0., 0., 0.]),# The world coordinate of the anchor
                          meas_model=anchor_model,
                          adj_var_nodes=fg.var_nodes[limb_id],
                          properties={})

    for connection in connections:
        parent_id = str(connection["parent_id"])
        child_id = str(connection["child_id"])
        parent_node = fg.var_nodes[parent_id][-1]
        child_node = fg.var_nodes[child_id][-1]
        clalib_node = fg.var_nodes["calib"+child_id]

        angle_measure = child_node.properties["local_angle"]  ## TO:DO add noise to this measurement to make it more realistic

        # add angle measurement factors
        fg.add_factor(measurement=torch.tenson([angle_measure]),
                      meas_model=angle_model,
                      adj_var_nodes=[parent_node, child_node],
                      properties={})

        # add kinematics factors
        fg.add_factor(measurement=torch.tensor([0.0, 0.0]),
                      meas_model=kinematic_model,
                      adj_var_nodes=[parent_node, clalib_node, child_node],
                      properties={})

# initialise gbp instance
gbp_settings = GBPSettings(
    damping = 0.1,
    beta = 0.01,
    num_undamped_iters = 1,
    min_linear_iters = 1,
    dropout = 0.0,
  )

# loss functions for the factors
angle_loss = HuberLoss(1, torch.tensor([0.01]), 2.0)
kinematic_loss = HuberLoss(2, torch.tensor([0.05, 0.05]), 2.0)
anchor_loss = SquaredLoss(2, torch.tensor([1e-6, 1e-6]))

# Instantiate the models
angle_model = AngleMeasurementModel(angle_loss)
kinematic_model = KinematicCalibModel(kinematic_loss, L=140) ## L limb length TO:DO make this a variable 
anchor_model = EndpointAnchorModel(anchor_loss, L=140)

# initialise the factor graph
fg = FactorGraph(gbp_settings)


data = {'limbs': [{'id': 1, 'local_angle': 95, 'global_angle': 190, 'position': {'x': 422.21494821952484, 'y': 440.9220629561239}, 'endpoint': {'x': 284.3418627978157, 'y': 416.61131808275366}, 'limb_length': 140, 'depth': 1}, {'id': 2, 'local_angle': 95, 'global_angle': 95, 'position': {'x': 429, 'y': 306}, 'endpoint': {'x': 416.79819601532785, 'y': 445.46725773284436}, 'limb_length': 140, 'depth': 0}, {'id': 3, 'local_angle': 265, 'global_angle': 455, 'position': {'x': 284.3418627978157, 'y': 416.61131808275366}, 'endpoint': {'x': 272.1400588131436, 'y': 556.078575815598}, 'limb_length': 140, 'depth': 2}], 'connections': [{'child_id': 1, 'parent_id': 2, 'depth': 1, 'calibration': {'offset_x': 135, 'offset_y': -5}}, {'child_id': 3, 'parent_id': 1, 'depth': 2, 'calibration': {'offset_x': 140, 'offset_y': 0}}]}

update_factor_graph(data, fg)