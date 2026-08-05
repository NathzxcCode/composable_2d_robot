import numpy as np

from limb_spec import LimbSpec
from planning_loop import PlanningGraph
from sim_loop import run_multi_robot_simulation

def linear_waypoint(start, end, speed, t, t_start=0.0):
    start = np.asarray(start, dtype=float)
    end   = np.asarray(end,   dtype=float)
    diff  = end - start
    dist  = np.linalg.norm(diff)
    if dist < 1e-9:
        return end.copy()
    moved = min(speed * (t - t_start), dist)
    return start + (diff / dist) * moved

def _make_planar_limbs():
    pi = np.pi
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


def main():
    limbs_0 = _make_planar_limbs()
    limbs_1 = _make_planar_limbs()

    robot_specs = [
        (limbs_0, (0.0, 0.0, 0.0)),
        (limbs_1, (0.2, 0.0, 0.0)),
    ]

    # Both robots share one factor graph and one solve call per control step.
    # Robot 0 goal: reach toward positive Y.
    # Robot 1 goal: reach toward negative Y (mirrored).
    pg = PlanningGraph(
        [limbs_0, limbs_1],
        [(0.05, 0.3),
         (-0.05, 0.3)],
        time_horizon=2, dt=0.1,
        enable_collision_avoidance=True,
        collision_radius=0.025,
        # collision_sigma=0.05,
        collision_k=3
    )

    all_limbs = [limbs_0, limbs_1]

    def joint_controller(all_states, t):
        robot_states = []
        for (qpos, qvel, spos, jpos, jrot), limbs in zip(all_states, all_limbs):
            joint_poses = np.array([
                [p[0], p[1], np.arctan2(r[1, 0], r[0, 0])]
                for p, r in zip(jpos, jrot)
            ])
            robot_states.append((qpos, joint_poses))

        # all_ctrls = pg.gbp_solve(robot_states, damping=0.95, n_inner=5, n_outer=8)
        all_ctrls = pg.centralised_solve(robot_states)
        return all_ctrls, []

    run_multi_robot_simulation(
        robot_specs,
        joint_controller=joint_controller,
        control_hz=10.0,
        trail_length=200,
        initial_qpos=[
        np.array([np.pi/2, 0.0, 0.0]),   # robot 0
        np.array([np.pi/2, 0.0, 0.0]),   # robot 1
    ]
    )


if __name__ == "__main__":
    main()
