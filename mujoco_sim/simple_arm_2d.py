import numpy as np
import matplotlib.pyplot as plt
import time

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

    waypoints = [
            # [(0.2, 0.35), (0.15, 0.25), (0.0, 0.43)],   # robot 0
            [(0.15, 0.25), (0.3, 0.43), (0.2, 0.35), (0.3, 0.35)]
        ]
    goal_idx = [0]
    ARRIVAL_THR = 0.02

    fg = PlanningGraph([limbs], [waypoints[0][0]], time_horizon=4, dt=[0.1, 0.2, 0.4],
                       base_positions=[(0.2,0,0)], sigma_endpoint=5, sigma_joint=1)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    _ROBOT_COLORS = ['tab:blue']

    def _draw_plan(ax):
        ax.cla()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Planned horizon')
        for r_id in range(len([limbs])):
            timesteps = fg.planned_positions(r_id)
            # print("timesteps ", timesteps)
            color = _ROBOT_COLORS[r_id]
            n_steps = len(timesteps)
            for k, pts in enumerate(timesteps):
                alpha = 0.25 + 0.75 * k / max(n_steps - 1, 1)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, '-o', color=color, alpha=alpha,
                        linewidth=2, markersize=4)
        plt.pause(0.001)

    

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        # Build ground-truth [x, y, theta] per joint from MuJoCo state.
        # joint_positions[i] = world-space anchor of joint i (3D).
        # joint_rotations[i] = 3x3 world-to-body rotation matrix; for a Z-axis
        # hinge, atan2(R[1,0], R[0,0]) recovers the accumulated rotation angle.
        # target = circle_target(t, cx=0.25, cy=0.1, radius=0.15, period=10.0)
        # target = figure8_target(t+6, cx=0.25, cy=0.1, rx=0.1, ry=0.15, period=10.0)
        target = waypoints[0][goal_idx[0]]

        joint_poses = np.array([
            [pos[0], pos[1], np.arctan2(rot[1, 0], rot[0, 0])]
            for pos, rot in zip(joint_positions, joint_rotations)
        ])
        ctrl = fg.gbp_solve([(qpos, joint_poses)], goals=[target],
                            n_inner=25, 
                            n_outer=1, damping=0.52)[0]
        # ctrl = fg.centralised_solve([(qpos, joint_poses)], goals=[target])[0]
        print("ctrl ", ctrl)

        # if usign pre-set waypoints iterate them here when robot reaches a distance to the goal
        for r in range(1):
            timesteps = fg.planned_positions(r)
            if not timesteps:
                continue
            ee = np.array(timesteps[0][-1])  # k=0 end effector (x, y)
            goal = np.array(target)
            # print(np.linalg.norm(ee - goal))
            if np.linalg.norm(ee - goal) < ARRIVAL_THR:
                print("robot: ", r, "reached goal: ", waypoints[r][goal_idx[r]])
                goal_idx[r] = (goal_idx[r] + 1) % len(waypoints[r])

        _draw_plan(ax)
        return ctrl, []

    run_simulation(limbs, controller=controller, control_hz=10.0, trail_length=200, 
                   initial_qpos=np.array([np.pi/2,0,0]), base_pos=(0.2,0,0))


if __name__ == "__main__":
    main()
