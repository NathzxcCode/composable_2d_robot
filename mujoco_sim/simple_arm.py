import numpy as np
import random

from limb_spec import LimbSpec
from sim_loop import run_simulation
from calibration_loop import FactorGraph



def main():
    limbs = [
        LimbSpec(attach_pos=[0.0,0.0,0.0]),
        LimbSpec(),
        LimbSpec(),
    ]

    robot_data = {"limbs": limbs,
                  "last_fg_update_time": 0.0}
    fg = FactorGraph()
    FG_UPDATE_INTERVAL = 1

    limb_centres = [-np.pi/2, 0, 0]
    limb_targets = [-np.pi/2, 0, 0]
    step_size = 0.01

    def controller(qpos, qvel, spos, t):
        robot_data["joint_angles"] = qpos
        robot_data["sensor_distances"] = [np.linalg.norm(spos[i] - spos[i+1]) for i in range(len(spos)-1)]

        if (t - robot_data["last_fg_update_time"]) >= FG_UPDATE_INTERVAL:
            fg.update_factor_graph(robot_data)
            fg.centralised_solve()
            robot_data["last_fg_update_time"] = t
        calibrations = fg.extract_calibrations()

        # Create an explicit container for our actuator targets
        actions = np.zeros_like(qpos)
        
        for i in range(len(limb_targets)):
            # 1. Check if we reached our current target checkpoint
            if abs(qpos[i] - limb_targets[i]) < step_size:
                # Pick a completely new long-term destination
                limb_targets[i] = np.random.uniform(
                    low=limb_centres[i] - np.pi/4, 
                    high=limb_centres[i] + np.pi/4
                )
            
            # 2. Command the actual target destination directly!
            # This allows the difference (limb_targets[i] - qpos[i]) to scale.
            # MuJoCo's Torq = Kp * (ctrl - qpos), so a larger gap creates more power.
            actions[i] = limb_targets[i]
            
        return actions, calibrations

    run_simulation(limbs, controller=controller)


if __name__ == "__main__":
    main()