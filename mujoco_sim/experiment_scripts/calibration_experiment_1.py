import numpy as np
import pandas as pd
import time
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

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

def planar_fk_and_jacobian(q, lengths, calib_offsets=None):
    """
    2D end-effector position and 2×n Jacobian for a Z-axis planar arm.

    calib_offsets: list of (cx, cy) per link, where (cx, cy) is the calibration
                   offset at the end of that link in the parent's local frame.
                   cx adjusts effective length; cy adds a perpendicular offset.
                   Defaults to zero (nominal model).
    """
    n = len(q)
    if calib_offsets is None:
        calib_offsets = [(0.0, 0.0)] * n
    cum = np.cumsum(q)
    x, y = 0.0, 0.0
    J = np.zeros((2, n))
    for k in range(n):
        cx, cy = calib_offsets[k]
        Leff = lengths[k] + cx
        cos_k, sin_k = np.cos(cum[k]), np.sin(cum[k])
        x += Leff * cos_k - cy * sin_k
        y += Leff * sin_k + cy * cos_k
        for j in range(k + 1):
            J[0, j] += -Leff * sin_k - cy * cos_k
            J[1, j] +=  Leff * cos_k - cy * sin_k
    return np.array([x, y]), J

def ik_step(q, lengths, x_target, step_size=0.5, lam=0.05, calib_offsets=None):
    x_cur, J = planar_fk_and_jacobian(q, lengths, calib_offsets)
    J_dls = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(2))
    return q + step_size * J_dls @ (x_target - x_cur)

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

def apply_measurement_noise(gps_joint_poses, encoder_angles, sigma_pos, sigma_theta, sigma_encoder):
    noisy_poses = {
        i: gtsam.Pose2(
            p.x()     + np.random.normal(0, sigma_pos),
            p.y()     + np.random.normal(0, sigma_pos),
            p.theta() + np.random.normal(0, sigma_theta),
        )
        for i, p in gps_joint_poses.items()
    }
    noisy_encoders = {
        i: a + np.random.normal(0, sigma_encoder)
        for i, a in encoder_angles.items()
    }
    return noisy_poses, noisy_encoders

def main():
    pi = np.pi

    limbs = [
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.0, 0.0, 0.1],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.13, 0.0, 0.0], 
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.15, 0.019, 0.0],    
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.13, -0.019, 0.0],    
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.10, 0.01, 0.0],    
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
    ]

    n = len(limbs)
    targets   = [0.0] * n
    threshold = 0.05

    FG_UPDATE_INTERVAL = 0.5
    limb_lengths = [l.length for l in limbs]
    true_calibrations = [[round(limbs[i+1].attach_pos[0]-limbs[i].length,2),limbs[i+1].attach_pos[1]] for i in range(n-1)]

    Sigma_range = [0.001, 0.01, 0.05]
    Sigma_encoder = [0.0009, 0.009, 0.017]
    Window_size = [20]
    states = []
    rows = []
    for window_size in Window_size:
        for sigma_range in Sigma_range:
            for sigma_encoder in Sigma_encoder:
                states.append((sigma_range, sigma_encoder, window_size))

    current_state = states.pop()
    State = {
        "states": states,
        "current_state": current_state,
        "fg": FactorGraph(sigma_range=current_state[0], sigma_encoder=current_state[1], window_size=current_state[2]),
        "calibs_collected" : 0,
        "calibs_needed" : 5
    }

    robot_data = {
        "limbs": limbs,
        "last_fg_update_time": 0.0,
    }

    def controller(qpos, qvel, spos, joint_positions, joint_rotations, t):
        if State["calibs_collected"] >= State["calibs_needed"]:
            if len(State["states"]) == 0:
                OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "calib_results/results_calibration_1_lm.csv")
                df = pd.DataFrame(rows)
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"[INFO] Saved {len(rows)} rows to {OUTPUT_CSV}")
                print("end")
                return
            next_state = State["states"].pop()
            State["current_state"] = next_state
            State["fg"] = FactorGraph(sigma_range=next_state[0], sigma_encoder=next_state[1], window_size=next_state[2])
            State["calibs_collected"] = 0
            print("changing state: ", next_state)

        fg = State["fg"]
        sigma_range, sigma_encoder, window_size = State["current_state"]
        robot_data["joint_angles"] = [np.random.normal(angle, sigma_encoder) for angle in qpos]
        robot_data["sensor_distances"] = [
            np.random.normal(np.linalg.norm(spos[i] - spos[i + 1]), sigma_range) for i in range(len(spos) - 1)
        ]

        raw_calibs = fg.extract_calibrations()

        # operate calibration after small delays to give robot time to move and be in a different pose
        if (t - robot_data["last_fg_update_time"]) >= FG_UPDATE_INTERVAL:
            fg.update_factor_graph(robot_data)
            # fg.gbp_solve(n_outer=2, n_inner=15)
            fg.centralised_solve()
            robot_data["last_fg_update_time"] = t

            raw_calibs = fg.extract_calibrations()
            
            for i, c in enumerate(raw_calibs):
                # only save calibrations where the sliding window is full, for fainess
                if fg.t >= fg.window_size:
                    rows.append({
                        "calib_id": i,
                        "sigma_range":   sigma_range,
                        "sigma_encoder": sigma_encoder,
                        "window_size":   window_size,
                        "est_cj_x":      c["mean"][0],
                        "est_cj_y":      c["mean"][1],
                        "cj_x":          true_calibrations[i][0],
                        "cj_y":          true_calibrations[i][1],
                            })

            print(fg.t)
            # increment state counter every time we record a calibration with a full observation sliding window 
            if fg.t >= fg.window_size:
                State["calibs_collected"] += 1
            
        # Random joint motion
        for i in range(n):
            if abs(qpos[i] - targets[i]) < threshold:
                targets[i] = np.random.uniform(-limbs[i].joint_range, limbs[i].joint_range)
        actions = np.array(targets)
        calibrations = [_to_3d_calib(c) for c in raw_calibs]

        return actions, calibrations

    run_simulation(limbs, controller=controller, control_hz=50.0, trail_length=600)


if __name__ == "__main__":
    main()
