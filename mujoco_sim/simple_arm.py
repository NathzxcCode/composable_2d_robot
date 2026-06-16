import numpy as np
import random

from limb_spec import LimbSpec
from sim_loop import run_simulation

def main():
    limbs = [
        LimbSpec(length=0.4, radius=0.022, color=[0.9, 0.2, 0.1, 1.0]),
        LimbSpec(length=0.35, radius=0.019, color=[0.1, 0.8, 0.2, 1.0]),
        LimbSpec(length=0.3, radius=0.016, color=[0.1, 0.2, 0.8, 1.0]),
    ]

    limb_centres = [-np.pi/2, 0, 0]
    limb_targets = [-np.pi/2, 0, 0]
    step_size = 0.01

    def controller(qpos, qvel, t):
        actions = qpos.copy()
        for i, target in enumerate(limb_targets):
            if abs(qpos[i] - target) < step_size:
                limb_targets[i] = np.random.uniform(low=limb_centres[i]-np.pi/4, high=limb_centres[i]+np.pi/4)
                target = limb_targets[i]

            if target > qpos[i] :
                actions[i] += step_size
            else:
                actions[i] -= step_size
        
        return actions

    run_simulation(limbs, controller=controller)


if __name__ == "__main__":
    main()