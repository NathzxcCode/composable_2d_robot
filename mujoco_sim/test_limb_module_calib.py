import gtsam
import numpy as np

from limb_spec import LimbSpec
from limb_module import LimbModule, J, S, run_distributed_gbp
from calibration_loop import FactorGraph

CJ_TRUE = gtsam.Pose2(0.05, -0.02, 0.0)
N_TICKS = 15
WINDOW_SIZE = 5
# Crossing a module boundary costs one extra message round per hop (a value has to travel
# there and back), so a distributed solve needs more inner rounds than the monolithic one to
# fully propagate information that has to cross several hops (e.g. CJ, shared by every slot,
# routing another slot's information back out through a slot's own local factors).
N_INNER = 20


def synthetic_tick(t, root_limb, child_limb):
    root_encoder  = 0.2 * np.sin(0.3 * t)
    child_encoder = 0.3 * np.cos(0.2 * t)

    root_true  = gtsam.Pose2(0.0, 0.0, root_encoder)
    child_true = root_true.compose(gtsam.Pose2(root_limb.length, 0.0, 0.0)) \
                          .compose(CJ_TRUE) \
                          .compose(gtsam.Pose2(0.0, 0.0, child_encoder))

    def sensor_pose(j_pose, limb):
        return j_pose.compose(gtsam.Pose2(limb.sensor_pos[0], limb.sensor_pos[1], limb.sensor_euler[2]))

    root_sensor_true  = sensor_pose(root_true, root_limb)
    child_sensor_true = sensor_pose(child_true, child_limb)
    distance = np.linalg.norm(np.array(child_sensor_true.translation()) - np.array(root_sensor_true.translation()))

    # Reasonable per-tick guesses, ignoring the (unknown) calibration offset - same as the
    # forward-kinematics estimate calibration_loop.py already computes.
    root_j_guess  = gtsam.Pose2(0.0, 0.0, root_encoder)
    root_s_guess  = sensor_pose(root_j_guess, root_limb)
    child_j_guess = root_j_guess.compose(gtsam.Pose2(root_limb.length, 0.0, 0.0)) \
                                 .compose(gtsam.Pose2(0.0, 0.0, child_encoder))
    child_s_guess = sensor_pose(child_j_guess, child_limb)

    return dict(root_encoder=root_encoder, child_encoder=child_encoder, distance=distance,
                root_j_guess=root_j_guess, root_s_guess=root_s_guess,
                child_j_guess=child_j_guess, child_s_guess=child_s_guess)


def run_distributed(root_limb, child_limb):
    root  = LimbModule(0, root_limb,  is_root=True, child_id=1, window_size=WINDOW_SIZE)
    child = LimbModule(1, child_limb, parent_id=0,  window_size=WINDOW_SIZE)

    for t in range(N_TICKS):
        d = synthetic_tick(t, root_limb, child_limb)
        root.update_factor_graph(
            d["root_encoder"], d["root_j_guess"], d["root_s_guess"],
            child_encoder_angle=d["child_encoder"], child_j_guess=d["child_j_guess"],
            child_s_guess=d["child_s_guess"], sensor_distance_to_child=d["distance"])
        child.update_factor_graph(d["child_encoder"], d["child_j_guess"], d["child_s_guess"])

        run_distributed_gbp([root, child], n_outer=8, n_inner=N_INNER)

    return root.extract_calibration()


def run_centralised(root_limb, child_limb):
    fg = FactorGraph(window_size=WINDOW_SIZE)
    for t in range(N_TICKS):
        d = synthetic_tick(t, root_limb, child_limb)
        fg.update_factor_graph({
            "limbs": [root_limb, child_limb],
            "joint_angles": [d["root_encoder"], d["child_encoder"]],
            "sensor_distances": [d["distance"]],
        })
    fg.gbp_solve(n_outer=8, n_inner=N_INNER)
    return fg.extract_calibrations()[0]


if __name__ == "__main__":
    root_limb, child_limb = LimbSpec(), LimbSpec()

    dist_result = run_distributed(root_limb, child_limb)
    cent_result = run_centralised(root_limb, child_limb)

    print("true CJ:         ", [CJ_TRUE.x(), CJ_TRUE.y()])
    print("distributed CJ:  ", dist_result["mean"])
    print("centralised CJ:  ", cent_result["mean"])
