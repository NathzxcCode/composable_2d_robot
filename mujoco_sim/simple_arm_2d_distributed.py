import gtsam
import numpy as np
import matplotlib.pyplot as plt

from limb_spec import LimbSpec
from sim_loop import run_simulation
from limb_module import LimbModule, run_distributed_gbp

# planning_loop's own tuning rule for this loopy grid-shaped graph: keep
# n_inner * damping roughly >= n_limbs * time_horizon (effective hops needed for
# information to cross the whole grid), same values simple_arm_2d.py already uses.
N_OUTER, N_INNER, DAMPING = 3, 15, 0


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
    ]

    waypoints = [(0.15, 0.25), (0.3, 0.43), (0.2, 0.35), (0.3, 0.35)]
    goal_idx = [0]
    ARRIVAL_THR = 0.02
    base_xy = (0.2, 0.0)
    time_horizon = 4
    dt = [0.1, 0.2, 0.4]

    modules = [
        LimbModule(i, limbs[i], is_root=(i == 0), parent_id=(i - 1 if i > 0 else None),
                   child_id=(i + 1 if i < len(limbs) - 1 else None))
        for i in range(len(limbs))
    ]
    guesses = []
    x = base_xy[0]
    for limb in limbs:
        guesses.append(gtsam.Pose2(x, base_xy[1], 0.0))
        x += limb.length
    for i, module in enumerate(modules):
        has_child = i < len(limbs) - 1
        module.initialise_planning(
            guesses[i], child_pose_guess=guesses[i + 1] if has_child else None,
            goal_xy=waypoints[goal_idx[0]] if not has_child else None,
            time_horizon=time_horizon, dt=dt, sigma_endpoint=2, sigma_joint=1,
            base_xy=base_xy)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    def _draw_plan(ax):
        ax.cla()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Planned horizon (distributed)')
        n_steps = time_horizon
        for k in range(n_steps):
            alpha = 0.25 + 0.75 * k / max(n_steps - 1, 1)
            pts = [m.planned_position(k) for m in modules]
            pts.append(modules[-1].planned_endpoint(k))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, '-o', color='tab:blue', alpha=alpha, linewidth=2, markersize=4)
        plt.pause(0.001)

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        target = waypoints[goal_idx[0]]

        joint_poses = np.array([
            [pos[0], pos[1], np.arctan2(rot[1, 0], rot[0, 0])]
            for pos, rot in zip(joint_positions, joint_rotations)
        ])

        modules[-1].update_goal(target)
        for i, module in enumerate(modules):
            has_child = i < len(limbs) - 1
            own_pose0 = gtsam.Pose2(joint_poses[i][0], joint_poses[i][1], joint_poses[i][2])
            module.pre_solve_planning(qpos[i], own_pose0,
                                       child_qpos0=qpos[i + 1] if has_child else None)

        run_distributed_gbp(modules, n_outer=N_OUTER, n_inner=N_INNER,
            build_fn=lambda m: m.build_planning_optimizer(N_OUTER, N_INNER, DAMPING),
            apply_fn=lambda m: m.apply_planning_optimizer_result())

        thetas = [m.extract_planned_theta(1) for m in modules]
        ctrl = np.zeros(len(limbs))
        ctrl[0] = thetas[0]
        for i in range(1, len(limbs)):
            ctrl[i] = thetas[i] - thetas[i - 1]
        print("ctrl ", ctrl)

        ee = np.array(modules[-1].planned_endpoint(0))
        goal = np.array(target)
        if np.linalg.norm(ee - goal) < ARRIVAL_THR:
            print("reached goal: ", waypoints[goal_idx[0]])
            goal_idx[0] = (goal_idx[0] + 1) % len(waypoints)

        _draw_plan(ax)
        return ctrl, []

    run_simulation(limbs, controller=controller, control_hz=10.0, trail_length=200,
                   initial_qpos=np.array([np.pi / 2, 0, 0]), base_pos=(0.2, 0, 0))


if __name__ == "__main__":
    main()
