from typing import Callable, Optional, List
import time
import numpy as np
import mujoco
import mujoco.viewer

from limb_spec import LimbSpec
from robot_builder import build_robot_xml
from utils import render_covariance_ellipses

Controller = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def run_simulation(
    limbs: List[LimbSpec],
    controller: Optional[Controller] = None,
    time_limit: float = float("inf"),
    real_time: bool = True,
    base_pos: tuple = (0, 0, 0),
    kp: float = 20.0,
):
    xml = build_robot_xml(limbs, base_pos=base_pos, kp=kp)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"Limbs: {len(limbs)}")
    print(f"Joints: {[model.joint(i).name for i in range(model.njnt)]}")
    print(f"Actuators (ctrl dim): {model.nu}")
    print("Move the view with drag/scroll. Press Ctrl+C to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < time_limit:
            step_start = time.time()

            joint_angles = data.qpos.copy()
            joint_velocities = data.qvel.copy()

            sensor_positions = []
            joint_positions = []
            joint_rotations = []
            expected_connection_positions = []
            for i in range(len(limbs)):
                sensor_positions.append(data.sensor(f"sensor_pos_{i}").data.copy())
                joint_positions.append(data.joint(f"joint_{i}").xanchor.copy())
                joint_rotations.append(data.body(f"limb_{i}").xmat.copy().reshape(3, 3))
                expected_connection_positions.append(np.array([limbs[i].length, 0.0, 0.0]))
            sensor_positions = np.array(sensor_positions)

            global_joint_connection_positions = []
            for i in range(1, len(limbs)):
                # Nominal connection point = parent joint anchor + parent rotation * parent tip offset
                # expected_connection_positions[i-1] is the nominal tip of limb i-1 in its local frame
                global_joint_connection_positions.append(joint_positions[i-1] + (joint_rotations[i-1] @ expected_connection_positions[i-1]))
            

            if controller is not None:
                ctrl, calibrations = controller(joint_angles, joint_velocities, sensor_positions, data.time)
                data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
                render_covariance_ellipses(viewer, calibrations, global_joint_connection_positions, joint_rotations, sigma=2.0)

            mujoco.mj_step(model, data)
            viewer.sync()

            if real_time:
                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)