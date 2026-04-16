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

def forward_kinematics_step(prev_conn_pose, joint_angle_rad, limb_length):
    """
    Calculates the pose of the current limb and the next joint location.
    
    Args:
        prev_conn_pose: Tensor [x, y, theta] of the previous joint/connection.
        joint_angle_rad: The motor measurement (relative rotation).
        limb_length: The nominal length of the bone.
        
    Returns:
        current_limb_pose: Tensor [x, y, theta] (The start/base of the limb)
        next_conn_pose: Tensor [x, y, theta] (The end of the limb/next joint)
    """
    # 1. Calculate the current limb's orientation
    # The limb points in the direction of the previous joint + the motor's turn
    joint_angle_rad = torch.tensor(joint_angle_rad)
    current_theta = prev_conn_pose[2] + joint_angle_rad

    # If you want to keep nodes near [-pi, pi] for human readability 
    # ONLY do it using atan2 to ensure the jump is handled smoothly
    # during the INITIALIZATION phase.
    current_theta = torch.atan2(torch.sin(current_theta), torch.cos(current_theta))
    
    # The base of the current limb is the same x,y as the previous connection
    current_limb_pose = torch.tensor([
        prev_conn_pose[0],
        prev_conn_pose[1],
        current_theta
    ])
    
    # 2. Project forward along the limb length to find the next connection
    # We move 'limb_length' units in the direction of current_theta
    next_x = current_limb_pose[0] + limb_length * torch.cos(current_theta)
    next_y = current_limb_pose[1] + limb_length * torch.sin(current_theta)
    
    # The orientation at the next connection matches the limb's orientation
    next_conn_pose = torch.tensor([next_x, next_y, current_theta])
    
    return current_limb_pose, next_conn_pose


def update_factor_graph(data, fg):
    limbs = data["limbs"]
    connections = data["connections"]
    limbs = sorted(limbs, key=lambda x: x['depth'])
    connections = sorted(connections, key=lambda x: x['depth'])

    next_conn_pose = torch.tensor([0., 0., 0.])
    position_cov = torch.tensor([1000., 1000., 0.01])

    for limb in limbs:
        id = str(limb["id"])
        length = limb["limb_length"]
        theta_rad = math.radians(limb["local_angle"])

        position_estimate, next_conn_pose = forward_kinematics_step(next_conn_pose, theta_rad, length)
        print(id, position_estimate, next_conn_pose)

        # add limb nodes to the factor graph (global x, y, theta of endpoint)
        fg.add_var_node(id=id,
                        dofs=3,
                        prior_mean=position_estimate,
                        prior_diag_cov=position_cov,  # Large variance = weak prior
                        properties=limb)
        
        # add calibration nodes for non-base limbs
        if fg.var_nodes.get("calib"+id) is None:
            if limb["depth"] != 0:
                fg.add_var_node(id="calib"+id,
                                dofs=3,
                                prior_mean=torch.tensor([0., 0., 0.]), 
                                prior_diag_cov=torch.tensor([1000., 1000., 0.01]),  # Large variance = weak prior
                                properties={})
        # add anchor to base node
        if limb["depth"] == 0:
            base_node = fg.var_nodes[id][-1]  # last node is current node
            fg.add_factor(measurement=torch.tensor([0., 0., 0.]), # minimise the risidual direct from factor
                          meas_model=AnchorModel(anchor_loss, T_origin=position_estimate), # pos_estimate = [0,0,theta]
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
        angle_measure_rad = math.radians(angle_measure) # Convert degrees to radians and add noise
        distance_measure = connection["distance"]
        
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
distance_loss = TukeyLoss(1, torch.tensor([0.1]), 3.0)
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

data2 = {
    'limbs': [
        {'id': 1, 'local_angle': 5, 'global_angle': 5, 'position': {'x': 0, 'y': 0}, 
        'endpoint': {'x': 139.46725773284436, 'y': 12.201803984672154}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 0}, 
        {'id': 2, 'local_angle': 0, 'global_angle': 5, 'position': {'x': 135.13385596948683, 'y': 15.837959806048502}, 
        'endpoint': {'x': 274.6011137023312, 'y': 28.039763790720656}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 1}, 
        {'id': 3, 'local_angle': 357, 'global_angle': 362, 'position': {'x': 269.3586729836295, 'y': 30.592569171257537}, 
        'endpoint': {'x': 409.2733887663029, 'y': 35.47849870960761}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 2}], 
    'connections': [
        {'child_id': 2, 'parent_id': 1, 'depth': 1, 'calibration': {'offset_x': 136, 'offset_y': 4}, 'distance': 136.05881081355966}, 
        {'child_id': 3, 'parent_id': 2, 'depth': 2, 'calibration': {'offset_x': 135, 'offset_y': 3}, 'distance': 134.87544045014846}]}

data3 = {
    'limbs': [
        {'id': 1, 'local_angle': 5, 'global_angle': 5, 'position': {'x': 0, 'y': 0}, 
         'endpoint': {'x': 139.46725773284436, 'y': 12.201803984672154}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 0}, 
        {'id': 2, 'local_angle': 2, 'global_angle': 7, 'position': {'x': 135.13385596948683, 'y': 15.837959806048502}, 
         'endpoint': {'x': 274.0903171992719, 'y': 32.899667882769165}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 1}, 
        {'id': 3, 'local_angle': 357, 'global_angle': 364, 'position': {'x': 268.7619784108499, 'y': 35.2679596206674}, 
         'endpoint': {'x': 408.42094544722534, 'y': 45.03386594484493}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 2}], 
    'connections': [
        {'child_id': 2, 'parent_id': 1, 'depth': 1, 'calibration': {'offset_x': 136, 'offset_y': 4}, 'distance': 136.17328773510513}, 
        {'child_id': 3, 'parent_id': 2, 'depth': 2, 'calibration': {'offset_x': 135, 'offset_y': 3}, 'distance': 134.87544045014857}]}

data4 = {
    'limbs': [
        {'id': 1, 'local_angle': 5, 'global_angle': 5, 'position': {'x': 0, 'y': 0}, 
         'endpoint': {'x': 139.46725773284436, 'y': 12.201803984672154}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 0}, 
        {'id': 2, 'local_angle': 2, 'global_angle': 7, 'position': {'x': 135.13385596948683, 'y': 15.837959806048502}, 
         'endpoint': {'x': 274.0903171992719, 'y': 32.899667882769165}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 1}, 
        {'id': 3, 'local_angle': 350, 'global_angle': 357, 'position': {'x': 268.7619784108499, 'y': 35.2679596206674}, 
         'endpoint': {'x': 408.57011327649025, 'y': 27.940925746655182}, 'sensor_offset': {'x': 120, 'y': 0}, 'limb_length': 140, 'depth': 2}], 
    'connections': [
        {'child_id': 2, 'parent_id': 1, 'depth': 1, 'calibration': {'offset_x': 136, 'offset_y': 4}, 'distance': 136.17328773510513}, 
        {'child_id': 3, 'parent_id': 2, 'depth': 2, 'calibration': {'offset_x': 135, 'offset_y': 3}, 'distance': 134.3662205426787}]}

update_factor_graph(data1, fg)
update_factor_graph(data2, fg)
update_factor_graph(data3, fg)
update_factor_graph(data4, fg)

print("Factor graph updated successfully!")
print(f"Variables: {len(fg.var_nodes)}")
print(f"Factors: {len(fg.factors)}")

fg.gbp_solve(n_iters=200)
# for i in range(100):
#     fg.gradient_descent_step(lr=0.0001)
# print(f"Energy: {fg.energy()}")
fg.print()