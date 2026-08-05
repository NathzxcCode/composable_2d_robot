import numpy as np

from limb_spec import LimbSpec
from sim_loop import run_simulation
from planning_loop import PlanningGraph

def circle_target(t, cx, cy, radius, period):
    a = 2 * np.pi * t / period
    return np.array([cx + radius * np.cos(a), cy + radius * np.sin(a)])

def figure8_target(t, cx, cy, rx, ry, period, angle_deg=45.0):
    a = 2 * np.pi * t / period
    x = rx * np.sin(2 * a)
    y = ry * np.sin(a)
    # rotate the shape by angle_deg around its centre
    rad = np.radians(angle_deg)
    cos_r, sin_r = np.cos(rad), np.sin(rad)
    return np.array([
        cx + cos_r * x - sin_r * y,
        cy + sin_r * x + cos_r * y,
    ])

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

    fg = PlanningGraph([limbs], [(0.2, 0.3)], time_horizon=2, dt=0.1)

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        # Build ground-truth [x, y, theta] per joint from MuJoCo state.
        # joint_positions[i] = world-space anchor of joint i (3D).
        # joint_rotations[i] = 3x3 world-to-body rotation matrix; for a Z-axis
        # hinge, atan2(R[1,0], R[0,0]) recovers the accumulated rotation angle.
        # target = circle_target(t, cx=0.25, cy=0.1, radius=0.15, period=10.0)
        target = figure8_target(t, cx=0.25, cy=0.1, rx=0.1, ry=0.15, period=10.0)

        joint_poses = np.array([
            [pos[0], pos[1], np.arctan2(rot[1, 0], rot[0, 0])]
            for pos, rot in zip(joint_positions, joint_rotations)
        ])
        ctrl = fg.gbp_solve([(qpos, joint_poses)], goals=[target])[0]
        return ctrl, []

    run_simulation(limbs, controller=controller, control_hz=10.0)


if __name__ == "__main__":
    main()
