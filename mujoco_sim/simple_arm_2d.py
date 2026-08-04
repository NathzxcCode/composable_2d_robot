import numpy as np

from limb_spec import LimbSpec
from sim_loop import run_simulation
from planning_loop import PlanningGraph


def main():
    limbs = [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.06, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=np.pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.15, 0.0, 0.0],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=np.pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.15, 0.0, 0.0],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=np.pi / 2,
        ),
        # LimbSpec(
        #     length=0.15, radius=0.015, density=500.0,
        #     attach_pos=[0.15, 0.0, 0.0],
        #     attach_euler=[0.0, 0.0, 0.0],
        #     joint_axis=[0.0, 0.0, 1.0],
        #     joint_damping=0.4,
        #     sensor_pos=[0.12, 0.0, 0.0],
        #     joint_centre=0.0,
        #     joint_range=np.pi / 2,
        # ),
        # LimbSpec(
        #     length=0.15, radius=0.015, density=500.0,
        #     attach_pos=[0.15, 0.0, 0.0],
        #     attach_euler=[0.0, 0.0, 0.0],
        #     joint_axis=[0.0, 0.0, 1.0],
        #     joint_damping=0.4,
        #     sensor_pos=[0.12, 0.0, 0.0],
        #     joint_centre=0.0,
        #     joint_range=np.pi / 2,
        # ),
        #  LimbSpec(
        #     length=0.15, radius=0.015, density=500.0,
        #     attach_pos=[0.15, 0.0, 0.0],
        #     attach_euler=[0.0, 0.0, 0.0],
        #     joint_axis=[0.0, 0.0, 1.0],
        #     joint_damping=0.4,
        #     sensor_pos=[0.12, 0.0, 0.0],
        #     joint_centre=0.0,
        #     joint_range=np.pi / 2,
        # ),
    ]

    fg = PlanningGraph(limbs, time_horizon=2, dt=0.1, goal_xy=(0.0, -0.15))

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        # Build ground-truth [x, y, theta] per joint from MuJoCo state.
        # joint_positions[i] = world-space anchor of joint i (3D).
        # joint_rotations[i] = 3x3 world-to-body rotation matrix; for a Z-axis
        # hinge, atan2(R[1,0], R[0,0]) recovers the accumulated rotation angle.
        joint_poses = np.array([
            [pos[0], pos[1], np.arctan2(rot[1, 0], rot[0, 0])]
            for pos, rot in zip(joint_positions, joint_rotations)
        ])
        ctrl = fg.gbp_solve(qpos, joint_poses, damping=0.7)
        return ctrl, []

    run_simulation(limbs, controller=controller, control_hz=10.0)


if __name__ == "__main__":
    main()
