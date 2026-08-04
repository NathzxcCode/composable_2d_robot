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


def planar_fk_and_jacobian(q, lengths):
    """2D end-effector position and 2×n Jacobian for a Z-axis planar arm."""
    cum = np.cumsum(q)
    x = sum(L * np.cos(a) for L, a in zip(lengths, cum))
    y = sum(L * np.sin(a) for L, a in zip(lengths, cum))
    J = np.zeros((2, len(q)))
    for j in range(len(q)):
        for i in range(j, len(q)):
            J[0, j] -= lengths[i] * np.sin(cum[i])
            J[1, j] += lengths[i] * np.cos(cum[i])
    return np.array([x, y]), J


def ik_step(q, lengths, x_target, step_size=0.5, lam=0.05):
    x_cur, J = planar_fk_and_jacobian(q, lengths)
    J_dls = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(2))
    return q + step_size * J_dls @ (x_target - x_cur)


def circle_target(t, cx, cy, radius, period):
    a = 2 * np.pi * t / period
    return np.array([cx + radius * np.cos(a), cy + radius * np.sin(a)])


def figure8_target(t, cx, cy, rx, ry, period):
    a = 2 * np.pi * t / period
    return np.array([cx + rx * np.sin(a), cy + ry * np.sin(2 * a)])


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
    limb_lengths = [l.length for l in limbs]

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

        target = figure8_target(t, cx=0.25, cy=0.0, rx=0.15, ry=0.12, period=10.0)
        actions = ik_step(qpos, limb_lengths, target)

        return actions, calibrations

    run_simulation(limbs, controller=controller, control_hz=50.0)


if __name__ == "__main__":
    main()
