import numpy as np

from limb_spec import LimbSpec
from sim_loop import run_simulation
from calibration_loop import FactorGraph


def _to_3d_calib(calib):
    """
    Convert a 2D calibration dict (from FactorGraph.extract_calibrations) to
    the 3D format expected by render_covariance_ellipses_3d in sim_loop.
    The Z component of mean and covariance are set to zero; the ellipsoid
    renders as a flat disc in the arm's X-Y motion plane.
    """
    cov_3d = np.zeros((3, 3))
    cov_3d[:2, :2] = np.array(calib["cov_xy"])
    return {
        "id":      calib["id"],
        "mean":    [calib["mean"][0], calib["mean"][1], 0.0],
        "cov_xyz": cov_3d.tolist(),
    }


def main():
    pi = np.pi

    # Limb 1 has a deliberate attachment offset (true attach differs from nominal).
    # The calibration loop uses limb.length as the nominal attachment distance, so
    # CJ(1) should converge to approximately Pose2(-0.03, 0.02, 0).
    limbs = [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.06, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.12, 0.02, 0.0],   # miscalibrated: 3 cm short, 2 cm lateral
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.15, 0.0, 0.0],    # nominal (correct)
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
    ]

    robot_data = {
        "limbs": limbs,
        "last_fg_update_time": 0.0,
    }
    fg = FactorGraph()
    FG_UPDATE_INTERVAL = 2.0

    limb_targets = [limb.joint_centre for limb in limbs]
    arrival_threshold = 0.05

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        robot_data["joint_angles"] = qpos
        robot_data["sensor_distances"] = [
            np.linalg.norm(spos[i] - spos[i + 1]) for i in range(len(spos) - 1)
        ]

        if (t - robot_data["last_fg_update_time"]) >= FG_UPDATE_INTERVAL:
            fg.update_factor_graph(robot_data)
            fg.centralised_solve()
            robot_data["last_fg_update_time"] = t

        calibrations = [_to_3d_calib(c) for c in fg.extract_calibrations()]

        actions = np.zeros_like(qpos)
        for i, limb in enumerate(limbs):
            if abs(qpos[i] - limb_targets[i]) < arrival_threshold:
                limb_targets[i] = np.random.uniform(
                    low=limb.joint_centre - limb.joint_range,
                    high=limb.joint_centre + limb.joint_range,
                )
            actions[i] = limb_targets[i]

        return actions, calibrations

    run_simulation(limbs, controller=controller, control_hz=50.0)


if __name__ == "__main__":
    main()
