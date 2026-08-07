import numpy as np
import matplotlib.pyplot as plt

from limb_spec import LimbSpec
from planning_loop import PlanningGraph
from sim_loop import run_multi_robot_simulation


def _make_planar_limbs():
    return [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.06, 0.0, 0.0], joint_centre=0.0, joint_range=np.pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.15, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=np.pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.15, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=np.pi / 2,
        ),
    ]


ARRIVAL_THR = 0.05  # metres


def main():
    limbs_0 = _make_planar_limbs()
    limbs_1 = _make_planar_limbs()
    limbs_2 = _make_planar_limbs()

    # Equilateral triangle with 0.3 m side length.
    # Robots 0 and 1 on the bottom edge, robot 2 at the apex.
    # h = 0.3 * sqrt(3) / 2 ≈ 0.26 m
    robot_specs = [
        (limbs_0, (-0.15, 0.0,  0.0)),
        (limbs_1, ( 0.15, 0.0,  0.0)),
        (limbs_2, ( 0.0,  0.26, 0.0)),
    ]

    # Light interference: robots 0 and 1 sweep toward each other through the
    # shared centre; robot 2 dips down into the same region from above.
    waypoints = [
        [( 0.1,  0.2), (-0.3,  0.15)],   # robot 0: right toward centre, then out left
        [(-0.1,  0.2), ( 0.3,  0.15)],   # robot 1: left toward centre, then out right
        [( 0.1,  0.02), (-0.1, 0.02)],   # robot 2: dips right then left into shared space
    ]
    goal_idx = [0, 0, 0]

    pg = PlanningGraph(
        [limbs_0, limbs_1, limbs_2],
        [waypoints[0][0], waypoints[1][0], waypoints[2][0]],
        base_positions=[(-0.15, 0.0), (0.15, 0.0), (0.0, 0.26)],
        time_horizon=4, dt=[0.1, 0.2, 0.4],
        enable_collision_avoidance=True,
        collision_radius=0.03,
        collision_k=3,
    )

    all_limbs = [limbs_0, limbs_1, limbs_2]
    _ROBOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green']

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    def _draw_plan(ax):
        ax.cla()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Planned horizon — 3 robots')
        for r_id in range(len(all_limbs)):
            timesteps = pg.planned_positions(r_id)
            color = _ROBOT_COLORS[r_id]
            n_steps = len(timesteps)
            for k, pts in enumerate(timesteps):
                alpha = 0.25 + 0.75 * k / max(n_steps - 1, 1)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, '-o', color=color, alpha=alpha,
                        linewidth=2, markersize=4)
        plt.pause(0.001)

    def joint_controller(all_states, t):
        robot_states = []
        for (qpos, qvel, spos, jpos, jrot), limbs in zip(all_states, all_limbs):
            joint_poses = np.array([
                [p[0], p[1], np.arctan2(r[1, 0], r[0, 0])]
                for p, r in zip(jpos, jrot)
            ])
            robot_states.append((qpos, joint_poses))

        current_goals = [waypoints[r][goal_idx[r]] for r in range(len(all_limbs))]
        all_ctrls = pg.centralised_solve(robot_states, goals=current_goals)

        for r in range(len(all_limbs)):
            timesteps = pg.planned_positions(r)
            if not timesteps:
                continue
            ee = np.array(timesteps[0][-1])
            goal = np.array(current_goals[r])
            if np.linalg.norm(ee - goal) < ARRIVAL_THR:
                print("robot: ", r, "reached goal: ", waypoints[r][goal_idx[r]])
                goal_idx[r] = (goal_idx[r] + 1) % len(waypoints[r])

        _draw_plan(ax)
        return all_ctrls, []

    run_multi_robot_simulation(
        robot_specs,
        joint_controller=joint_controller,
        control_hz=10.0,
        trail_length=200,
        initial_qpos=[
            np.array([ np.pi / 2, 0.0, 0.0]),   # robot 0: bottom left,  pointing up
            np.array([ np.pi / 2, 0.0, 0.0]),   # robot 1: bottom right, pointing up
            np.array([-np.pi / 2, 0.0, 0.0]),   # robot 2: top,          pointing down
        ]
    )


if __name__ == "__main__":
    main()
