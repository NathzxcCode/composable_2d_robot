from typing import Callable, Optional, List
import time
import numpy as np
import mujoco
import mujoco.viewer

from limb_spec import LimbSpec
from robot_builder import build_robot_xml

Controller = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def run_simulation(
    limbs: List[LimbSpec],
    controller: Optional[Controller] = None,
    time_limit: float = float("inf"),
    real_time: bool = True,
    base_pos: tuple = (0, 0.5, 0),
    kp: float = 100.0,
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

            joint_positions = data.qpos.copy()
            joint_velocities = data.qvel.copy()

            if controller is not None:
                ctrl = controller(joint_positions, joint_velocities, data.time)
                data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)

            mujoco.mj_step(model, data)
            viewer.sync()

            if real_time:
                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)