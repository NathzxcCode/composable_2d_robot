import gtsam
import numpy as np

from limb_spec import LimbSpec
from sim_loop import run_simulation
from limb_module import LimbModule, run_distributed_gbp
from demo_calib_2d import (
    _to_3d_calib, planar_fk_and_jacobian, ik_step, figure8_target,
)


def compute_fk_guesses(limbs, joint_angles):
    """
    Forward-kinematics guess for every limb's J/S pose, ignoring calibration -
    same computation calibration_loop.FactorGraph.update_factor_graph does
    internally, pulled out here since each LimbModule only sees its own and its
    child's slice of this chain.
    """
    j_guesses, s_guesses = [], []
    current_parent_pose = gtsam.Pose2()
    for i, limb in enumerate(limbs):
        angle = joint_angles[i]
        T_attach = gtsam.Pose2(0.0, 0.0, 0.0) if i == 0 else gtsam.Pose2(limbs[i - 1].length, 0.0, 0.0)
        j_pose = current_parent_pose.compose(T_attach).compose(gtsam.Pose2(0.0, 0.0, angle))
        s_pose = j_pose.compose(gtsam.Pose2(limb.sensor_pos[0], limb.sensor_pos[1], limb.sensor_euler[2]))
        j_guesses.append(j_pose)
        s_guesses.append(s_pose)
        current_parent_pose = j_pose
    return j_guesses, s_guesses


def main():
    pi = np.pi

    # Same limbs as demo_calib_2d.py, including limb 1's deliberate attachment offset.
    limbs = [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.13, 0.0, 0.0],   # miscalibrated: 3 cm short, 2 cm lateral
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.13, 0.0, 0.0],    # nominal (correct)
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.13, 0.0, 0.0],    # nominal (correct)
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
    ]

    # One LimbModule per limb: a chain, so each limb's parent/child are just its
    # neighbours in the list. Fed in manually for now, per the current scope
    # (topology discovery will supply this later).
    modules = [
        LimbModule(
            i, limbs[i],
            is_root=(i == 0),
            parent_id=(i - 1 if i > 0 else None),
            child_id=(i + 1 if i < len(limbs) - 1 else None),
            window_size=15
        )
        for i in range(len(limbs))
    ]

    last_fg_update_time = {"t": 0.0}
    FG_UPDATE_INTERVAL = 2.0
    limb_lengths = [l.length for l in limbs]

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        joint_angles     = qpos
        sensor_distances = [np.linalg.norm(spos[i] - spos[i + 1]) for i in range(len(spos) - 1)]

        if (t - last_fg_update_time["t"]) >= FG_UPDATE_INTERVAL:
            j_guesses, s_guesses = compute_fk_guesses(limbs, joint_angles)

            for i, module in enumerate(modules):
                has_child = i < len(limbs) - 1
                module.update_factor_graph(
                    joint_angles[i], j_guesses[i], s_guesses[i],
                    child_encoder_angle=joint_angles[i + 1] if has_child else None,
                    child_j_guess=j_guesses[i + 1] if has_child else None,
                    child_s_guess=s_guesses[i + 1] if has_child else None,
                    sensor_distance_to_child=sensor_distances[i] if has_child else None,
                )

            run_distributed_gbp(modules, n_outer=1, n_inner=10)
            last_fg_update_time["t"] = t

        # Same as demo_calib_2d.py: raw_calibs[k] = CJ(k+1) = offset at the end of link k.
        # A module reports None until its first solve (mirrors FactorGraph.extract_calibrations
        # returning [] before update_factor_graph has ever run).
        raw_calibs = [c for c in (m.extract_calibration() for m in modules[:-1]) if c is not None]
        calibrations = [_to_3d_calib(c) for c in raw_calibs]
        calib_offsets = [(0.0, 0.0)] * len(limbs)
        for k, c in enumerate(raw_calibs):
            calib_offsets[k] = tuple(c["mean"])

        target = figure8_target(t, cx=0.3, cy=0.1, rx=0.15, ry=0.2, period=10.0)
        actions = ik_step(qpos, limb_lengths, target, calib_offsets=calib_offsets)

        return actions, calibrations

    run_simulation(limbs, controller=controller, control_hz=50.0, trail_length=600)


if __name__ == "__main__":
    main()
