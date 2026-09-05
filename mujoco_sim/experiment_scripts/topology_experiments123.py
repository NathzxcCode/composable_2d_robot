import numpy as np
import gtsam
import time
import sys
import os
import pandas as pd

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from limb_spec import LimbSpec
from sim_loop import run_multi_robot_simulation
from topology_discovery import TopologyDiscovery, identify_root, CandidateStatus


_STATUS_COLORS = {
    CandidateStatus.CANDIDATE: "lightyellow",
    CandidateStatus.CONFIRMED: "lightgreen",
}


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


def _make_three_limbs_1():
    pi = np.pi
    return [
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.13, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.15, -0.01, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        )
    ]
def _make_three_limbs_2():
    pi = np.pi
    return [
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.13, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.15, -0.01, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        )
    ]
def _make_three_limbs_3():
    pi = np.pi
    return [
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.13, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.02, density=500.0,
            attach_pos=[0.15, -0.01, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        )
    ]


def main():
    rows = []
    robot_specs = [
        (_make_three_limbs_1(), (0.0, 0.0, 0.0)),
        (_make_three_limbs_2(), (0.0, 0.0, 0.2)),
        (_make_three_limbs_3(), (0.0, 0.0, 0.4)),
    ]

    # Flatten all limbs across all robots into one global list.
    # TopologyDiscovery operates on global limb indices and all limbs in the scene are considered together.
    all_limbs_flat = [l for robot_limbs, _ in robot_specs for l in robot_limbs]
    limb_counts    = [len(robot_limbs) for robot_limbs, _ in robot_specs]
    n              = len(all_limbs_flat)

    Sigma_pos     = [0.001, 0.01, 0.05]   # metres
    Sigma_theta   = [0.0017, 0.017, 0.09]   # radians
    Sigma_encoder = [0.0009, 0.009, 0.017]   # radians
    K = [5,10,15,20]
    states = []
    for k in K:
        for sigma_pos, sigma_theta in zip(Sigma_pos, Sigma_theta):
            for sigma_encoder in Sigma_encoder:
                states.append((sigma_pos, sigma_theta, sigma_encoder, k))

    def make_new_discoveries(state):
        # One TopologyDiscovery instance per limb.
        return [
            TopologyDiscovery(
                limb_id=i, 
                limbs=all_limbs_flat,
                sigma_gps_pos=state[0],
                sigma_gps_theta=state[1],
                sigma_encoder=state[2],
                n_sigma_search=1.5,
                K=state[3],
                TTL_max=15,
                T_max=state[3],
                cost_thr=0,       
                cov_thr=0,
                reject_thr=0,     
                K_stale=0,
            )
            for i in range(n)
        ]

    current_state = states.pop()
    State = {
        "states" : states,
        "current_state": current_state,
        "discoveries" : make_new_discoveries(current_state),
        "end_time" : time.time() + 25,
        "duration" : 25
    }

    # Random motion state — each joint independently seeks a new random target
    # when it arrives within threshold. Varied motion is needed for observability.
    targets   = [0.0] * n
    threshold = 0.05

    def joint_controller(all_states, t):
        if State["end_time"] <= time.time():
            if len(State["states"]) == 0:
                OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "topo_results/results_topology_discovery_123.csv")
                df = pd.DataFrame(rows)
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"[INFO] Saved {len(rows)} rows to {OUTPUT_CSV}")
                print("end")
                return
            next_state = State["states"].pop()
            State["current_state"] = next_state
            State["discoveries"] = make_new_discoveries(next_state)
            State["end_time"] = time.time() + State["duration"]
            print("changing state: ", next_state, State["discoveries"][0].sigma_encoder == next_state[2])

        # Get current state parameters
        sigma_pos, sigma_theta, sigma_encoder, k = State["current_state"]

        # Flatten per-robot states into global limb-indexed lists
        all_jpos, all_jrot, all_qpos_flat = [], [], []
        for (qpos_r, qvel_r, spos_r, jpos_r, jrot_r) in all_states:
            all_jpos.extend(jpos_r)
            all_jrot.extend(jrot_r)
            all_qpos_flat.extend(qpos_r)

        # Build GPS joint poses from MuJoCo state ground truth.
        gps_joint_poses = {
            i: gtsam.Pose2(
                float(all_jpos[i][0]),
                float(all_jpos[i][1]),
                float(np.arctan2(all_jrot[i][1, 0], all_jrot[i][0, 0])),
            )
            for i in range(n)
        }
        encoder_angles = {i: float(all_qpos_flat[i]) for i in range(n)}
        gps_joint_poses, encoder_angles = apply_measurement_noise(
            gps_joint_poses, encoder_angles, sigma_pos, sigma_theta, sigma_encoder
        )

        for disc in State["discoveries"]:
            diag = disc.update(gps_joint_poses, encoder_angles, get_failed_diagnostics=True)
            if diag:
                for cid, d in diag.items():
                    est = d["cj_estimate"]
                    connected = (disc.limb_id,cid) in {(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)}
                    rows.append({
                        "sigma_pos":     sigma_pos,
                        "sigma_theta":   sigma_theta,
                        "sigma_encoder": sigma_encoder,
                        "k":             k,
                        "connected":     connected,
                        "limb_id":       disc.limb_id,
                        "child_id":      cid,
                        "cost_per_obs":  d["cost_per_obs"],
                        "cov_trace":     d["cov_trace"],
                        "cj_x":          est.x()     if est is not None else float("nan"),
                        "cj_y":          est.y()     if est is not None else float("nan"),
                        "cj_theta":      est.theta() if est is not None else float("nan"),
                        "n_obs":         d["n_obs"]
                    })

        # Diagnostics — observe cost and cov_trace values to tune thresholds
        # for disc in State["discoveries"]:
        #     diag = disc.get_diagnostics()
        #     if diag:
        #         for cid, d in diag.items():
        #             est = d["cj_estimate"]
        #             connected = (disc.limb_id,cid) in {(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)}
        #             current_state = State["current_state"]
        #             rows.append({
        #                 "sigma_pos":     sigma_pos,
        #                 "sigma_theta":   sigma_theta,
        #                 "sigma_encoder": sigma_encoder,
        #                 "k":             k,
        #                 "connected":     connected,
        #                 "limb_id":       disc.limb_id,
        #                 "child_id":      cid,
        #                 "cost_per_obs":  d["cost_per_obs"],
        #                 "cov_trace":     d["cov_trace"],
        #                 "cj_x":          est.x()     if est is not None else float("nan"),
        #                 "cj_y":          est.y()     if est is not None else float("nan"),
        #                 "cj_theta":      est.theta() if est is not None else float("nan"),
        #             })
                    

        # Random joint motion
        for i in range(n):
            if abs(all_qpos_flat[i] - targets[i]) < threshold:
                targets[i] = np.random.uniform(-all_limbs_flat[i].joint_range, all_limbs_flat[i].joint_range)
        ctrl_flat = np.array(targets)

        # Split flat ctrl back into per-robot arrays
        all_ctrls, start = [], 0
        for count in limb_counts:
            all_ctrls.append(ctrl_flat[start:start + count])
            start += count

        return all_ctrls, []

    run_multi_robot_simulation(
        robot_specs,
        joint_controller=joint_controller,
        control_hz=10.0,
        trail_length=200
    )

if __name__ == "__main__":
    main()

                    