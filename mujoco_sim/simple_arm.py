import numpy as np

from limb_spec import LimbSpec
from sim_loop import run_simulation
from calibration_loop_3d import FactorGraph3D


def main():
    """
    SO-101-inspired 5-DOF arm.

    Joint layout (all axes in the LOCAL body frame after attach_euler):
      0 – shoulder pan   axis=[1,0,0] local  = world Y (vertical)   → sweeps arm left/right
      1 – shoulder lift  axis=[0,1,0] local  = world -X              → raises/lowers arm
      2 – elbow          axis=[0,1,0] local                           → bends elbow
      3 – wrist pitch    axis=[0,1,0] local                           → tilts wrist
      4 – wrist roll     axis=[1,0,0] local  = along the arm         → rotates end effector

    Limb 0 uses attach_euler=[0, 0, pi/2] so its capsule (which runs along local X)
    stands upright along world Y.  All subsequent limbs inherit the rotated frame and
    need no extra euler correction.

    True attach positions (attach_pos) are set slightly shorter than limb.length so
    the calibration factor graph has a real offset to recover.
    """
    pi = np.pi

    limbs = [
        LimbSpec(
            length=0.08, radius=0.015, density=500.0,
            attach_pos=[0.0, 0.0, 0.0],
            attach_euler=[0.0, -pi/2, 0.0],
            joint_axis=[1.0, 0.0, 0.0],
            joint_damping=0.4,
            sensor_pos=[0.06, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.08, 0.0, 0.0],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 1.0, 0.0],
            joint_damping=0.4,
            sensor_pos=[0.09, 0.0, 0.0],
            joint_centre=-pi / 4,
            joint_range=pi / 4,
        ),
        LimbSpec(
            length=0.15, radius=0.015, density=500.0,
            attach_pos=[0.10, 0.0, 0.0],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 1.0, 0.0],
            joint_damping=0.4,
            sensor_pos=[0.12, 0.0, 0.0],
            joint_centre=pi / 4,
            joint_range=pi / 4,
        ),
        LimbSpec(
            length=0.07, radius=0.012, density=500.0,
            attach_pos=[0.13, 0.0, 0.0],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 1.0, 0.0],
            joint_damping=0.3,
            sensor_pos=[0.05, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 4,
        ),
        LimbSpec(
            length=0.04, radius=0.010, density=500.0,
            attach_pos=[0.06, 0.0, 0.0],
            attach_euler=[0.0, 0.0, 0.0],
            joint_axis=[0.0, 0.0, 1.0],
            joint_damping=0.2,
            sensor_pos=[0.03, 0.0, 0.0],
            joint_centre=0.0,
            joint_range=pi / 2,
        ),
    ]

    robot_data = {
        "limbs": limbs,
        "last_fg_update_time": 0.0,
    }
    fg = FactorGraph3D()
    FG_UPDATE_INTERVAL = 2   # seconds of sim time between graph updates

    # Read centre and range from each LimbSpec so adding/removing limbs
    # requires no changes here.
    limb_targets = [limb.joint_centre for limb in limbs]

    # arrival_threshold: must exceed the gravity-induced static position error.
    # With kp=30 and this arm, worst-case ≈ 0.015 rad; 0.05 gives safe margin.
    arrival_threshold = 0.05   # radians

    def controller(qpos, qvel, spos, t):
        robot_data["joint_angles"] = qpos
        robot_data["sensor_distances"] = [
            np.linalg.norm(spos[i] - spos[i + 1]) for i in range(len(spos) - 1)
        ]

        if (t - robot_data["last_fg_update_time"]) >= FG_UPDATE_INTERVAL:
            fg.update_factor_graph(robot_data)
            # fg.centralised_solve()
            fg.gbp_solve(n_outer=8, n_inner=8)
            robot_data["last_fg_update_time"] = t
        calibrations = fg.extract_calibrations()

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
