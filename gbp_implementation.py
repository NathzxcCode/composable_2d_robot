import torch
import math
import random
from gbp_utilities import MeasModel, SquaredLoss, TukeyLoss, HuberLoss
from gbp import GBPSettings, FactorGraph
from gbp_factors import KinematicCalibModel, AnchorModel, DistanceMeasurementModel, EndpointModel, AngleMeasurementModel

# Angle measurement noise standard deviation in degrees
# Set to 0.0 to disable noise (deterministic)
ANGLE_NOISE_STD_DEG = 2.0

def add_noise_to_angle(angle_rad):
    """Add Gaussian noise to an angle measurement."""
    if ANGLE_NOISE_STD_DEG > 0:
        noise_rad = random.gauss(0, math.radians(ANGLE_NOISE_STD_DEG))
        return angle_rad + noise_rad
    return angle_rad


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
                        prior_mean=torch.tensor([limb["position"]["x"], limb["position"]["y"], math.radians(limb["global_angle"])]),
                        # prior_mean=torch.tensor([limb["limb_length"]*limb["depth"], 0., 0.]),
                        prior_diag_cov=torch.tensor([0.1, 0.1, 0.01]),  # Large variance = weak prior
                        properties=limb)
        
        # add calibration nodes for non-base limbs
        if fg.var_nodes.get("calib"+limb_id) is None:
            if limb["depth"] != 0:
                fg.add_var_node(id="calib"+limb_id,
                                dofs=3,
                                prior_mean=torch.tensor([0., 0., 0.]), 
                                prior_diag_cov=torch.tensor([100., 100., 0.1]),  # Large variance = weak prior
                                properties={})
        # add anchor to base node
        if limb["depth"] == 0:
            base_node = fg.var_nodes[limb_id][-1]  # last node is current node
            fg.add_factor(measurement=torch.tensor([0., 0., 0.]), # minimise the risidual direct from factor
                          meas_model=AnchorModel(anchor_loss, T_origin=torch.eye(3)),
                          adj_var_nodes=[base_node],
                          properties={})
    for connection in connections:
        parent_id = str(connection["parent_id"])
        child_id = str(connection["child_id"])
        
        parent_node = fg.var_nodes[parent_id][-1]
        child_node = fg.var_nodes[child_id][-1]
        calib_node = fg.var_nodes["calib"+child_id][-1]   # only one node in list shared accross factor graph
        parent_limb_length = parent_node.properties["limb_length"]

        angle_measure = child_node.properties["local_angle"]
        distance_measure = connection["distance"]
        angle_measure_rad = math.radians(angle_measure) # Convert degrees to radians and add noise

        # add angle measurement factors
        # fg.add_factor(measurement=torch.tensor([angle_measure_rad]), 
        #               meas_model=AngleMeasurementModel(angle_loss),
        #               adj_var_nodes=[parent_node, child_node],
        #               properties={})

        # add kinematics factors
        fg.add_factor(measurement=torch.tensor([0., 0., 0.]),
                      meas_model=KinematicCalibModel(kinematic_loss, L=parent_limb_length, theta_joint=angle_measure_rad),
                      adj_var_nodes=[parent_node, calib_node, child_node], 
                      properties={})
        
        # add distance measurement factors
        parent_limb_sensor_offset = parent_node.properties["sensor_offset"]
        child_limb_sensor_offset = child_node.properties["sensor_offset"]
        s1 = torch.tensor([parent_limb_sensor_offset["x"], parent_limb_sensor_offset["y"]])
        s2 = torch.tensor([child_limb_sensor_offset["x"], child_limb_sensor_offset["y"]])
        fg.add_factor(measurement=torch.tensor([distance_measure]),
                      meas_model=DistanceMeasurementModel(distance_loss, s1=s1, s2=s2),
                      adj_var_nodes=[parent_node, child_node], 
                      properties={})
# ===================================================================
# Setup
# ===================================================================
gbp_settings = GBPSettings(
    damping=0.95,
    beta=0.2,
    num_undamped_iters=3,
    min_linear_iters=10,
    dropout=0.0,
)

# loss functions for the factors
kinematic_loss = HuberLoss(3, torch.tensor([0.1, 0.1, 1e-4]), 3.0)
anchor_loss = SquaredLoss(3, torch.tensor([1e-4, 1e-4, 1e-6]))
endpoint_loss = SquaredLoss(3, torch.tensor([1e-3, 1e-3, 1e-5]))
distance_loss = TukeyLoss(1, torch.tensor([1.0]), 3.0)
# angle_loss = HuberLoss(1, torch.tensor([1e-4]), 3.0)

# Instantiate the models
# endpoint_model = EndpointModel(endpoint_loss, L_last=)

# initialise the factor graph
fg = FactorGraph(gbp_settings)

data = {
    'limbs': [
        {'id': 1, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 0, 'y': 0}, 
         'endpoint': {'x': 140, 'y': 0}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 0}, 
        {'id': 2, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 110, 'y': 22}, 
         'endpoint': {'x': 250, 'y': 22}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 1}, 
        {'id': 3, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 236, 'y': 5},
         'endpoint': {'x': 376, 'y': 5}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 2}], 
    'connections': [
        {'child_id': 2, 'parent_id': 1, 'depth': 1, 'calibration': {'offset_x': 110, 'offset_y': 22}, 'distance': 112.17842929904127}, 
        {'child_id': 3, 'parent_id': 2, 'depth': 2, 'calibration': {'offset_x': 126, 'offset_y': -17}, 'distance': 127.1416532848303}]}

data1 = {
    'limbs': [
        {'id': 1, 'local_angle': 0, 'global_angle': 0, 'position': {'x': 0, 'y': 0}, 
         'endpoint': {'x': 140, 'y': 0}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 0}, 
        {'id': 2, 'local_angle': 2, 'global_angle': 2, 'position': {'x': 136, 'y': 4}, 
         'endpoint': {'x': 275.9147157826734, 'y': 8.885929538350126}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 1}, 
        {'id': 3, 'local_angle': 357, 'global_angle': 359, 'position': {'x': 270.8130631574704, 'y': 11.709604535894925}, 
         'endpoint': {'x': 410.7917404793652, 'y': 9.266267634675103}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 2}], 
    'connections': [
        {'child_id': 2, 'parent_id': 1, 'depth': 1, 'calibration': {'offset_x': 136, 'offset_y': 4}, 'distance': 136.1732877351051}, 
        {'child_id': 3, 'parent_id': 2, 'depth': 2, 'calibration': {'offset_x': 135, 'offset_y': 3}, 'distance': 134.87544045014846}]}

update_factor_graph(data, fg)
print("Factor graph updated successfully!")
print(f"Variables: {len(fg.var_nodes)}")
print(f"Factors: {len(fg.factors)}")

fg.gbp_solve(n_iters=25)
# for i in range(100):
#     fg.gradient_descent_step(lr=0.0001)
# print(f"Energy: {fg.energy()}")
fg.print()