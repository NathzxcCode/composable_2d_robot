import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gtsam

from limb_spec import LimbSpec
from sim_loop import run_multi_robot_simulation
from topology_discovery import TopologyDiscovery, identify_root, CandidateStatus


_STATUS_COLORS = {
    CandidateStatus.CANDIDATE: "lightyellow",
    CandidateStatus.ACTIVE:    "orange",
    CandidateStatus.CONFIRMED: "lightgreen",
    CandidateStatus.STALE:     "lightcoral",
}


def _make_planar_limbs():
    pi = np.pi
    return [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            # Deliberate attachment offset on limb 1: true attach is 2 cm shorter
            # than nominal length. Topology discovery should confirm this connection
            # and CJ(0→1) should converge to approximately Pose2(-0.02, 0, 0).
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.13, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.12, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.08, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
    ]

def make_double_limb():
    pi = np.pi
    return [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        ),
        LimbSpec(
                    length=0.15, radius=0.015, density=500.0,
                    attach_pos=[0.12, 0.0, 0.0], attach_euler=[0.0, 0.0, 0.0],
                    joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
                    sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
                ),
    ]

def make_single_limb():
    pi = np.pi
    return [
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.1], attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0], joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0], joint_centre=0.0, joint_range=pi / 2,
        )
    ]


def main():
    robot_specs = [
        (_make_planar_limbs(), (0.0, 0.0, 0.0)),
        (make_single_limb(),   (0.0, 0.0, 0.2)),
        (make_double_limb(),   (0.0, 0.0, 0.4)),
    ]

    # Flatten all limbs across all robots into one global list.
    # TopologyDiscovery operates on global limb indices and has no concept of
    # robot boundaries — all limbs in the scene are considered together.
    all_limbs_flat = [l for robot_limbs, _ in robot_specs for l in robot_limbs]
    limb_counts    = [len(robot_limbs) for robot_limbs, _ in robot_specs]
    n              = len(all_limbs_flat)

    # One TopologyDiscovery instance per limb.
    # cost_thr and cov_thr are initial guesses — tune using the diagnostics
    # printed to console each control step.
    discoveries = [
        TopologyDiscovery(
            limb_id=i, limbs=all_limbs_flat,
            sigma_gps_pos=0.001,
            sigma_gps_theta=0.005,
            n_sigma_search=5.0,
            K=5,
            TTL_max=5,
            T_max=10,
            cost_thr=5.0,
            cov_thr=0.01,
        )
        for i in range(n)
    ]

    # Random motion state — each joint independently seeks a new random target
    # when it arrives within threshold. Varied motion is needed for observability.
    targets   = [0.0] * n
    threshold = 0.05

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))

    def _draw_status():
        ax.cla()
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("Child limb")
        ax.set_ylabel("Parent limb")
        ax.invert_yaxis()

        for disc in discoveries:
            pid   = disc.limb_id
            state = disc.get_persistence_state()
            diag  = disc.get_diagnostics()

            for cid in range(n):
                rect_kwargs = dict(width=1.0, height=1.0,
                                   xy=(cid - 0.5, pid - 0.5))
                if cid == pid:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        color="lightgrey", **rect_kwargs))
                    continue

                status = state.get(cid)
                color  = _STATUS_COLORS.get(status, "white")
                ax.add_patch(mpatches.FancyBboxPatch(color=color, **rect_kwargs))

                if cid in diag:
                    ax.text(cid, pid,
                            f"cost={diag[cid]['cost']:.2f}\ncov={diag[cid]['cov_trace']:.4f}",
                            ha="center", va="center", fontsize=6)

        confirmed = {disc.limb_id: list(disc.get_children().keys()) for disc in discoveries}
        root      = identify_root(discoveries)
        ax.set_title(f"root={root}  confirmed={confirmed}", fontsize=9)

        legend = [mpatches.Patch(color=c, label=s.value)
                  for s, c in _STATUS_COLORS.items()]
        ax.legend(handles=legend, loc="upper right", fontsize=7)
        plt.tight_layout()
        plt.pause(0.001)

    def joint_controller(all_states, t):
        # Flatten per-robot states into global limb-indexed lists
        all_jpos, all_jrot, all_qpos_flat = [], [], []
        for (qpos_r, qvel_r, spos_r, jpos_r, jrot_r) in all_states:
            all_jpos.extend(jpos_r)
            all_jrot.extend(jrot_r)
            all_qpos_flat.extend(qpos_r)

        # Build GPS joint poses from MuJoCo state.
        # Uses xanchor (true pivot position) directly — in physical deployment
        # this would be derived from sensor pose via J = S.compose(Pose2(-sx,-sy,0)).
        gps_joint_poses = {
            i: gtsam.Pose2(
                float(all_jpos[i][0]),
                float(all_jpos[i][1]),
                float(np.arctan2(all_jrot[i][1, 0], all_jrot[i][0, 0])),
            )
            for i in range(n)
        }
        encoder_angles = {i: float(all_qpos_flat[i]) for i in range(n)}

        for disc in discoveries:
            disc.update(gps_joint_poses, encoder_angles)

        # Diagnostics — observe cost and cov_trace values to tune thresholds
        print(f"\nt={t:.1f}")
        for disc in discoveries:
            diag = disc.get_diagnostics()
            if diag:
                for cid, d in diag.items():
                    print(f"  {disc.limb_id}→{cid}  "
                          f"cost={d['cost']:.3f}  cov={d['cov_trace']:.5f}  "
                          f"n={d['n_obs']}  [{d['status']}]"
                          f"cj estimate={d["cj_estimate"]}")
        print(f"  root={identify_root(discoveries)}")

        _draw_status()

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
        trail_length=200,
        initial_qpos=[
            np.array([np.pi / 2, 0.0, 0.0, 0.0]),
            np.array([np.pi / 2]),
        ],
    )


if __name__ == "__main__":
    main()
