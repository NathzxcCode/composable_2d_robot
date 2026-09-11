# A Unified Factor Graph Framework For Modular Robotics Distributed Via GBP

A distributed factor graph framework for modular robotics, solving topology discovery, calibration, and motion planning problems. Each module in a physical robot assembly maintains its own local factor graph and communicates with neighbours via Gaussian Belief Propagation (GBP) message passing — meaning the full inference problem is solved in a decentralised way without any single module needing global knowledge of the structure.

The framework is implemented in Python using GTSAM for factor graph construction and a custom GBP solver built on top of it. A MuJoCo simulator provides the physical environment for testing.

---

## Repository structure

```
composable_2d_robot/
├── gtsam_gbp.py          # Custom GBP solver (GBPOptimizer, DistributedGBPOptimizer)
├── gbp_utilities.py      # Gaussian message utilities (canonical form, robust losses)
├── gtsam_examples/       # Standalone GTSAM factor examples (kinematics, planning, collision)
├── mujoco_sim/
│   ├── limb_spec.py              # Physical/kinematic description of one modular limb
│   ├── robot_builder.py          # Builds MuJoCo XML from a list of LimbSpec objects
│   ├── limb_module.py            # Distributed LimbModule class + run_distributed_gbp
│   ├── calibration_loop.py       # Centralised calibration factor graph (reference)
│   ├── planning_loop.py          # Centralised planning factor graph (reference)
│   ├── topology_discovery.py     # Per-limb topology discovery via pairwise GBP
│   ├── sim_loop.py               # MuJoCo simulation drivers
│   ├── demo_calib_2d.py          # → see Demos below
│   ├── demo_calib_distributed.py # → see Demos below
│   ├── demo_topology_discovery.py
│   ├── simple_arm_2d.py
│   ├── simple_arm.py
│   ├── multi_robot_planning.py
│   └── experiment_scripts/       # Scripts used to generate thesis experiment results
│       ├── calibration_experiment_{1,2,3}.py
│       ├── topology_experiment_4.py
│       ├── topology_experiments123.py
│       ├── planning_experiment_{1,3}.py
│       └── plots.ipynb
└── live_server.py        # Browser-based live visualisation server
```

---

## Demos

All demos live in `mujoco_sim/` and are run directly with Python. The venv at the repo root contains all dependencies:

```bash
source .venv/bin/activate
cd mujoco_sim
```

### Calibration

**`demo_calib_2d.py`** — A 4-limb planar arm traces a figure-8 trajectory. One limb has a deliberate attachment offset (miscalibration). The calibration factor graph converges to the true offset over time, progressively correcting the IK controller so the arm tracks the intended path more accurately. Covariance ellipses are rendered at each joint connection point showing calibration uncertainty shrinking as estimates converge.

**`demo_calib_distributed.py`** — The same scenario as above, now solved using the distributed `LimbModule` framework: each limb runs its own local factor graph and exchanges GBP messages with its neighbours rather than sharing a single global graph.

**`simple_arm.py`** — 3D calibration on a small robot arm executing random joint motions. Calibrates attachment offsets in 3D using `Pose3` factors.

### Topology Discovery

**`demo_topology_discovery.py`** — Three 2D arms are placed within connection range of one another. Each limb independently runs topology discovery, testing spatial co-occurrence and GBP-fit quality for all nearby limbs. Correctly identified parent-child connections are shown in green in a live adjacency matrix; spurious cross-robot candidate pairs are rejected. Each limb discovers only its own true connections without any global coordination.

### Planning and Collision Avoidance

**`simple_arm_2d.py`** — A 3-limb planar arm navigates a sequence of waypoints using a receding-horizon GBP planner. The planned horizon is visualised in matplotlib at each control step.

**`multi_robot_planning.py`** — Two robots plan trajectories simultaneously in a shared factor graph. Inter-robot collision-avoidance factors (Bhattacharyya-distance ellipsoid costs) are added between non-adjacent limb pairs, causing robots to take paths that avoid one another.

---

## Experiment scripts

`mujoco_sim/experiment_scripts/` contains all scripts used to produce the quantitative results in the thesis:

- **Calibration**: `calibration_experiment_{1,2,3}.py` — sweep noise levels, window sizes, and solver settings; results saved to `calib_results/`.
- **Topology**: `topology_experiment_4.py`, `topology_experiments123.py` — vary robot configurations and sensor noise; results saved to `topo_results/`.
- **Planning**: `planning_experiment_{1,3}.py` — trajectory tracking accuracy and collision metrics; results saved to `planning_results/`.
- **`plots.ipynb`** — Jupyter notebook that loads all saved results and generates the thesis figures.

---

## Key components

**`LimbModule`** (`mujoco_sim/limb_module.py`) — The core distributed agent class. Each physical module runs one `LimbModule` instance, which owns its own GTSAM factor graph and values for each active problem layer (calibration, planning). Cross-module factors are owned by the parent limb; child variables are represented as foreign placeholders whose beliefs arrive via GBP messages. `run_distributed_gbp()` drives the synchronous message-passing loop across all modules.

**`DistributedGBPOptimizer`** (`gtsam_gbp.py`) — Extends the base `GBPOptimizer` with support for foreign variables (owned by another module, belief received from inbox) and remote-link stubs (standing in for a factor owned on a neighbouring module). The outer/inner loop is exposed step-by-step so the driver can interleave message exchange between modules every inner iteration.

**`GBPOptimizer`** (`gtsam_gbp.py`) — Single-graph GBP solver wrapping a GTSAM `NonlinearFactorGraph`. Relinearizes each outer iteration, then runs synchronous sum-product message passing for `n_inner` rounds with optional damping before retracting the delta onto the manifold.
