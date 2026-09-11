import gtsam
import numpy as np

from limb_spec import LimbSpec
from limb_module import LimbModule, run_distributed_gbp
from planning_loop import PlanningGraph

N_LIMBS = 4
LIMB_LENGTH = 0.15
TIME_HORIZON = 5
DT = [0.1] * (TIME_HORIZON - 1)
GOAL_XY = (0.3, 0.2)
N_TICKS = 8
# planning_loop.py's own recommended GBP settings for this graph (see the commented-out
# call in multi_robot_planning.py) - this graph mixes very different information scales
# (tight k=0 grounding vs loose future-step priors) and needs damping to stay stable,
# unlike calibration's graph which converges fine with damping=0.
N_OUTER, N_INNER, DAMPING = 3, 25, 0.52


def straight_arm_guesses(limbs):
    guesses = []
    x = 0.0
    for limb in limbs:
        guesses.append(gtsam.Pose2(x, 0.0, 0.0))
        x += limb.length
    return guesses


def run_distributed(limbs):
    modules = [
        LimbModule(i, limbs[i], is_root=(i == 0), parent_id=(i - 1 if i > 0 else None),
                   child_id=(i + 1 if i < len(limbs) - 1 else None))
        for i in range(len(limbs))
    ]
    guesses = straight_arm_guesses(limbs)
    for i, module in enumerate(modules):
        has_child = i < len(limbs) - 1
        module.initialise_planning(
            guesses[i], child_pose_guess=guesses[i + 1] if has_child else None,
            goal_xy=GOAL_XY if not has_child else None,
            time_horizon=TIME_HORIZON, dt=DT)

    qpos = np.zeros(len(limbs))  # everything starts at rest, arm straight along x
    for _tick in range(N_TICKS):
        for i, module in enumerate(modules):
            has_child = i < len(limbs) - 1
            module.pre_solve_planning(
                qpos[i], gtsam.Pose2(guesses[i].x(), guesses[i].y(), float(qpos[i])),
                child_qpos0=qpos[i + 1] if has_child else None)

        run_distributed_gbp(modules, n_outer=N_OUTER, n_inner=N_INNER,
            build_fn=lambda m: m.build_planning_optimizer(N_OUTER, N_INNER, DAMPING),
            apply_fn=lambda m: m.apply_planning_optimizer_result())

        thetas = [m.extract_planned_theta(1) for m in modules]
        ctrl = np.zeros(len(limbs))
        ctrl[0] = thetas[0]
        for i in range(1, len(limbs)):
            ctrl[i] = thetas[i] - thetas[i - 1]
        qpos = ctrl  # ideal actuators - qpos tracks ctrl exactly, same as a perfect servo

        for module in modules:
            module.post_solve_rotate_planning()

    return qpos


def run_centralised(limbs):
    pg = PlanningGraph([limbs], [GOAL_XY], time_horizon=TIME_HORIZON, dt=DT)
    qpos = np.zeros(len(limbs))
    for _tick in range(N_TICKS):
        joint_poses = np.array([[straight_arm_guesses(limbs)[i].x(), 0.0, qpos[i]] for i in range(len(limbs))])
        ctrl = pg.gbp_solve([(qpos, joint_poses)], n_outer=N_OUTER, n_inner=N_INNER, damping=DAMPING)[0]
        qpos = ctrl
    return qpos


if __name__ == "__main__":
    limbs = [LimbSpec(length=LIMB_LENGTH) for _ in range(N_LIMBS)]

    dist_qpos = run_distributed(limbs)
    cent_qpos = run_centralised(limbs)

    print("distributed final qpos:", dist_qpos)
    print("centralised final qpos:", cent_qpos)
