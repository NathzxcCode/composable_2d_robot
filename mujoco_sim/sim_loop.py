from typing import Callable, Optional, List
from collections import deque
import time
import numpy as np
import mujoco
import mujoco.viewer

from limb_spec import LimbSpec
from robot_builder import build_robot_xml, build_multi_robot_xml
from utils import render_covariance_ellipses, render_covariance_ellipses_3d

Controller = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def run_simulation(
    limbs: List[LimbSpec],
    controller: Optional[Controller] = None,
    time_limit: float = float("inf"),
    real_time: bool = True,
    base_pos: tuple = (0, 0, 0),
    kp: float = 30.0,
    control_hz: float = 50.0,
    trail_length: int = 0,
):
    """
    Run the MuJoCo simulation loop.

    The physics integrator always runs at the timestep defined in the XML
    (currently 2 ms = 500 Hz).  The controller is called at a lower, fixed
    rate given by `control_hz` (default 50 Hz).  Between controller calls the
    last commanded ctrl values are held, exactly as a real servo drive would.

    Args:
        limbs:       List of LimbSpec objects defining the robot.
        controller:  Callable(qpos, qvel, sensor_positions, sim_time)
                     -> (ctrl_array, calibrations).
        time_limit:  Stop after this many simulated seconds.
        real_time:   If True, sleep to match wall-clock real time.
        base_pos:    World position of the robot base.
        kp:          Position-actuator gain passed to the XML builder.
        control_hz:  Frequency at which the controller is called (Hz).
                     The physics still integrates at the full timestep rate.
    """
    xml = build_robot_xml(limbs, base_pos=base_pos, kp=kp)
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    sim_dt      = model.opt.timestep          # e.g. 0.002 s
    control_dt  = 1.0 / control_hz            # e.g. 0.020 s at 50 Hz
    # How many physics steps between controller calls (rounded to nearest int)
    steps_per_ctrl = max(1, round(control_dt / sim_dt))
    actual_control_hz = 1.0 / (steps_per_ctrl * sim_dt)

    print(f"Limbs: {len(limbs)}")
    print(f"Joints: {[model.joint(i).name for i in range(model.njnt)]}")
    print(f"Actuators (ctrl dim): {model.nu}")
    print(f"Physics timestep: {sim_dt*1000:.1f} ms  ({1/sim_dt:.0f} Hz)")
    print(f"Controller rate:   {actual_control_hz:.1f} Hz  "
          f"({steps_per_ctrl} physics steps per control step)")
    print("Move the view with drag/scroll. Press Ctrl+C to exit.")

    # Last calibrations from controller, held between control calls
    last_calibrations: list = []
    trail: deque = deque(maxlen=trail_length) if trail_length > 0 else None

    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < time_limit:
            # ----------------------------------------------------------------
            # Physics + real-time budget is measured per CONTROL step, not per
            # physics step.  This gives the controller a full control_dt of
            # wall-clock budget, so GTSAM solves fit comfortably.
            # ----------------------------------------------------------------
            ctrl_step_start = time.time()

            # Run `steps_per_ctrl` physics steps with the current ctrl held
            for _ in range(steps_per_ctrl):
                mujoco.mj_step(model, data)

            step_count += steps_per_ctrl

            # ----------------------------------------------------------------
            # Read state ONCE after the physics batch
            # ----------------------------------------------------------------
            joint_angles      = data.qpos.copy()
            joint_velocities  = data.qvel.copy()

            sensor_positions         = []
            joint_positions          = []
            joint_rotations          = []
            expected_connection_pos  = []
            for i in range(len(limbs)):
                sensor_positions.append(data.sensor(f"sensor_pos_{i}").data.copy())
                joint_positions.append(data.joint(f"joint_{i}").xanchor.copy())
                joint_rotations.append(data.body(f"limb_{i}").xmat.copy().reshape(3, 3))
                expected_connection_pos.append(np.array([limbs[i].length, 0.0, 0.0]))
            sensor_positions = np.array(sensor_positions)

            global_joint_connection_positions = []
            for i in range(1, len(limbs)):
                global_joint_connection_positions.append(
                    joint_positions[i - 1] + (joint_rotations[i - 1] @ expected_connection_pos[i - 1])
                )

            if trail is not None:
                tip = joint_positions[-1] + joint_rotations[-1] @ np.array([limbs[-1].length, 0.0, 0.0])
                trail.append(tip.copy())

            # ----------------------------------------------------------------
            # Call controller at control_hz
            # ----------------------------------------------------------------
            if controller is not None:
                ctrl, last_calibrations = controller(
                    joint_angles, joint_velocities, sensor_positions,
                    joint_positions, joint_rotations, data.time
                )
                data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)

            # ----------------------------------------------------------------
            # Render (covariance ellipses + viewer sync)
            # ----------------------------------------------------------------
            render_covariance_ellipses_3d(
                viewer, last_calibrations,
                global_joint_connection_positions, joint_rotations,
                sigma=2.0,
            )

            if trail is not None and len(trail) > 1:
                pts = list(trail)
                for i in range(1, len(pts)):
                    idx = viewer.user_scn.ngeom
                    if idx >= viewer.user_scn.maxgeom:
                        break
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.004,
                        pts[i - 1], pts[i],
                    )
                    viewer.user_scn.geoms[idx].rgba[:] = [1.0, 0.4, 0.0, 0.8]
                    viewer.user_scn.ngeom += 1

            viewer.sync()

            # ----------------------------------------------------------------
            # Real-time pacing: sleep for whatever wall time remains in
            # the control period.  This is robust to occasional long solves
            # because the budget is control_dt (e.g. 20 ms), not sim_dt (2 ms).
            # ----------------------------------------------------------------
            if real_time:
                elapsed = time.time() - ctrl_step_start
                remaining = control_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)


_TRAIL_COLORS = [
    [1.0, 0.4, 0.0, 0.8],   # orange
    [0.0, 0.8, 0.8, 0.8],   # cyan
    [1.0, 0.9, 0.0, 0.8],   # yellow
]


def run_multi_robot_simulation(
    robot_specs: list,
    controllers: list,
    time_limit: float = float("inf"),
    real_time: bool = True,
    kp: float = 30.0,
    control_hz: float = 50.0,
    trail_length: int = 0,
):
    """
    Run a MuJoCo simulation with multiple robots in one scene.

    robot_specs:  list of (limbs, base_pos) tuples, one per robot.
    controllers:  list of callables, one per robot.
                  Each has the same signature as the single-robot controller:
                  (qpos, qvel, spos, joint_positions, joint_rotations, t)
                  -> (ctrl_array, calibrations)
    """
    xml   = build_multi_robot_xml(robot_specs, kp=kp)
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    sim_dt         = model.opt.timestep
    control_dt     = 1.0 / control_hz
    steps_per_ctrl = max(1, round(control_dt / sim_dt))

    # Per-robot joint/actuator slice boundaries (flat qpos/ctrl are ordered by robot)
    limb_counts = [len(spec[0]) for spec in robot_specs]
    starts      = [sum(limb_counts[:r]) for r in range(len(robot_specs))]

    last_calibrations = [[] for _ in robot_specs]
    trails = [deque(maxlen=trail_length) if trail_length > 0 else None
              for _ in robot_specs]

    print(f"Robots: {len(robot_specs)}  total joints: {sum(limb_counts)}")
    print("Move the view with drag/scroll. Press Ctrl+C to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < time_limit:
            ctrl_step_start = time.time()

            for _ in range(steps_per_ctrl):
                mujoco.mj_step(model, data)

            # ----------------------------------------------------------------
            # Read and dispatch per-robot state
            # ----------------------------------------------------------------
            for r_id, (limbs, _) in enumerate(robot_specs):
                p   = f"r{r_id}_"
                s   = starts[r_id]
                n   = limb_counts[r_id]

                qpos_r = data.qpos[s:s + n].copy()
                qvel_r = data.qvel[s:s + n].copy()

                spos_r  = []
                jpos_r  = []
                jrot_r  = []
                exp_r   = []
                for i in range(n):
                    spos_r.append(data.sensor(f"{p}sensor_pos_{i}").data.copy())
                    jpos_r.append(data.joint(f"{p}joint_{i}").xanchor.copy())
                    jrot_r.append(data.body(f"{p}limb_{i}").xmat.copy().reshape(3, 3))
                    exp_r.append(np.array([limbs[i].length, 0.0, 0.0]))
                spos_r = np.array(spos_r)

                if trails[r_id] is not None:
                    tip = jpos_r[-1] + jrot_r[-1] @ exp_r[-1]
                    trails[r_id].append(tip.copy())

                ctrl_r, last_calibrations[r_id] = controllers[r_id](
                    qpos_r, qvel_r, spos_r, jpos_r, jrot_r, data.time
                )
                data.ctrl[s:s + n] = np.asarray(ctrl_r, dtype=np.float64)

            # ----------------------------------------------------------------
            # Render: clear scene then draw per-robot trails
            # ----------------------------------------------------------------
            render_covariance_ellipses_3d(viewer, [], [], [], sigma=2.0)

            for r_id, trail in enumerate(trails):
                if trail is None or len(trail) < 2:
                    continue
                color = _TRAIL_COLORS[r_id % len(_TRAIL_COLORS)]
                pts   = list(trail)
                for i in range(1, len(pts)):
                    idx = viewer.user_scn.ngeom
                    if idx >= viewer.user_scn.maxgeom:
                        break
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.004,
                        pts[i - 1], pts[i],
                    )
                    viewer.user_scn.geoms[idx].rgba[:] = color
                    viewer.user_scn.ngeom += 1

            viewer.sync()

            if real_time:
                elapsed   = time.time() - ctrl_step_start
                remaining = control_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
