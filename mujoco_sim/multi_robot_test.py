import numpy as np

from limb_spec import LimbSpec
from sim_loop import run_multi_robot_simulation


def _make_planar_limbs():
    pi = np.pi
    return [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.06, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.15, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.15, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
    ]


def _make_random_controller(limbs):
    targets   = [0.0] * len(limbs)
    threshold = 0.05

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        actions = np.zeros_like(qpos)
        for i, limb in enumerate(limbs):
            if abs(qpos[i] - targets[i]) < threshold:
                targets[i] = np.random.uniform(-limb.joint_range, limb.joint_range)
            actions[i] = targets[i]
        return actions, []

    return controller


def main():
    limbs_0 = _make_planar_limbs()
    limbs_1 = _make_planar_limbs()
    limbs_2 = _make_planar_limbs()

    robot_specs = [
        (limbs_0, (0.0, 0.0, 0.0)),
        (limbs_1, (0.6, 0.0, 0.0)),
        (limbs_2, (0.3, 0.6, 0.0)),
    ]

    controllers = [
        _make_random_controller(limbs_0),
        _make_random_controller(limbs_1),
        _make_random_controller(limbs_2),
    ]

    run_multi_robot_simulation(
        robot_specs, controllers, control_hz=50.0, trail_length=150
    )


if __name__ == "__main__":
    main()
