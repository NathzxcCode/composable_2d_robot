import gtsam
import numpy as np
import torch
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from gtsam_examples.gtsam_factors import make_calib_kinematics_factor, make_fixed_kinematics_factor
from gtsam_gbp import GBPParams, DistributedGBPOptimizer
from planning_loop import (
    _J, _E, _V, _VE,
    KINEMATIC_NOISE, ANCHOR_NOISE, LOOSE_ANCHOR_NOISE, GOAL_NOISE,
    _make_task_space_dynamics_factor, _make_joint_space_dynamics_factor,
)

RIGID_KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4]))
CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))


def J(joint_id, t):  # joint
    index = (joint_id * 10000) + t
    return gtsam.Symbol('j', index).key()


def S(sensor_id, t):  # sensor
    index = (sensor_id * 10000) + t
    return gtsam.Symbol('s', index).key()


def CJ(joint_id):  # joint calibration. id refers to child joint
    return gtsam.Symbol('c', joint_id).key()


def key_owner(key):
    """Which limb id a J/S/CJ key belongs to, decoded from the key itself - same encoding
    calibration_loop.py already uses (limb_id folded into the numeric index)."""
    sym = gtsam.Symbol(key)
    ch, idx = chr(sym.chr()), sym.index()
    return idx if ch == 'c' else idx // 10000


class LimbModule:
    """
    One limb's own calibration factor graph + values, its own distributed GBP engine, and
    its mailbox. Cross-limb factors (kinematic + range) are always owned by the parent
    limb: a parent keeps placeholder variables for its child's J/S so it can linearize
    them locally, and a child keeps a "remote link" so its own belief update accounts for
    the parent's factor without ever computing it itself.
    """

    def __init__(self, limb_id, limb_spec, is_root=False, parent_id=None, child_id=None,
                 sigma_range=0.001, sigma_encoder=0.0009, window_size=10):
        self.limb_id    = limb_id
        self.limb_spec  = limb_spec
        self.is_root    = is_root
        self.parent_id  = parent_id
        self.child_id   = child_id
        self.has_child  = child_id is not None
        self.window_size = window_size

        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.t = 0

        self._idx_inter_kin = {}
        self._idx_range     = {}
        self._idx_anchor    = {}

        self.CALIB_KINEMATIC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, sigma_encoder]))
        self.ANCHOR_NOISE          = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, sigma_encoder]))
        self.SENSOR_NOISE          = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_range]))

        self.inbox = {}
        self._optimizer = None

        # Planning layer - separate graph/values from the calibration layer above, own
        # optimizer, built once by initialise_planning() rather than grown every tick.
        self.robot_id = 0  # single robot for now, kept for the key scheme's future use
        self.planning_graph = gtsam.NonlinearFactorGraph()
        self.planning_values = gtsam.Values()
        self._planning_dof_map = {}
        self._planning_optimizer = None
        self.time_horizon = None
        self.dt = None

    def _live_slots(self):
        return range(min(self.t, self.window_size))

    def foreign_keys(self) -> set:
        """Keys in this module's Values owned by its child - placeholders only, never
        solved locally."""
        if not self.has_child:
            return set()
        keys = set()
        for slot in self._live_slots():
            keys.add(J(self.child_id, slot))
            keys.add(S(self.child_id, slot))
        return keys

    def remote_link_keys(self) -> set:
        """This module's own keys that its parent's factors reference."""
        if self.is_root:
            return set()
        keys = set()
        for slot in self._live_slots():
            keys.add(J(self.limb_id, slot))
            keys.add(S(self.limb_id, slot))
        return keys

    def update_factor_graph(self, own_encoder_angle, own_j_guess, own_s_guess,
                             child_encoder_angle=None, child_j_guess=None, child_s_guess=None,
                             sensor_distance_to_child=None) -> None:
        slot      = self.t % self.window_size
        replacing = self.t >= self.window_size

        # This limb's own variables
        if replacing:
            self.values.update(J(self.limb_id, slot), own_j_guess)
            self.values.update(S(self.limb_id, slot), own_s_guess)
        else:
            self.values.insert(J(self.limb_id, slot), own_j_guess)
            self.values.insert(S(self.limb_id, slot), own_s_guess)
            # Sensor-to-body kinematics — constant, never replaced.
            self.graph.add(make_fixed_kinematics_factor(
                J(self.limb_id, slot), S(self.limb_id, slot),
                self.limb_spec.sensor_pos[0], 0, RIGID_KINEMATIC_NOISE))

        # Child's variables live here too — placeholders, used only so this limb can
        # linearize the factors connecting to them. Their belief comes from messages.
        if self.has_child:
            if replacing:
                self.values.update(J(self.child_id, slot), child_j_guess)
                self.values.update(S(self.child_id, slot), child_s_guess)
            else:
                self.values.insert(J(self.child_id, slot), child_j_guess)
                self.values.insert(S(self.child_id, slot), child_s_guess)
                if not self.values.exists(CJ(self.child_id)):
                    self.values.insert(CJ(self.child_id), gtsam.Pose2())
                    self.graph.add(gtsam.PriorFactorPose2(
                        CJ(self.child_id), gtsam.Pose2(), CALIB_NOISE))

        # Anchor prior — root only, measurement always (0,0,encoder); never replaced.
        if self.is_root:
            if replacing:
                self.graph.replace(self._idx_anchor[slot], gtsam.PriorFactorPose2(
                    J(self.limb_id, slot), gtsam.Pose2(0.0, 0.0, own_encoder_angle), self.ANCHOR_NOISE))
            else:
                self._idx_anchor[slot] = self.graph.size()
                self.graph.add(gtsam.PriorFactorPose2(
                    J(self.limb_id, slot), gtsam.Pose2(0.0, 0.0, own_encoder_angle), self.ANCHOR_NOISE))

        # Cross-limb factors — owned by the parent, referencing the child's placeholders.
        if self.has_child:
            if replacing:
                self.graph.replace(self._idx_inter_kin[slot], make_calib_kinematics_factor(
                    J(self.limb_id, slot), CJ(self.child_id), J(self.child_id, slot),
                    self.limb_spec.length, child_encoder_angle, self.CALIB_KINEMATIC_NOISE))
                self.graph.replace(self._idx_range[slot], gtsam.RangeFactorPose2(
                    S(self.limb_id, slot), S(self.child_id, slot),
                    sensor_distance_to_child, self.SENSOR_NOISE))
            else:
                self._idx_inter_kin[slot] = self.graph.size()
                self.graph.add(make_calib_kinematics_factor(
                    J(self.limb_id, slot), CJ(self.child_id), J(self.child_id, slot),
                    self.limb_spec.length, child_encoder_angle, self.CALIB_KINEMATIC_NOISE))

                self._idx_range[slot] = self.graph.size()
                self.graph.add(gtsam.RangeFactorPose2(
                    S(self.limb_id, slot), S(self.child_id, slot),
                    sensor_distance_to_child, self.SENSOR_NOISE))

        self.t += 1

    def build_optimizer(self, n_outer=8, n_inner=8, damping=0.0) -> DistributedGBPOptimizer:
        if self.values.size() == 0:
            self._optimizer = None
            return None
        params = GBPParams(n_outer=n_outer, n_inner=n_inner, damping=damping, dof=3)
        self._optimizer = DistributedGBPOptimizer(
            self.graph, self.values, params,
            foreign_keys=self.foreign_keys(),
            remote_link_keys=self.remote_link_keys())
        return self._optimizer

    def apply_optimizer_result(self) -> None:
        if self._optimizer is not None:
            self.values = self._optimizer.values

    def extract_calibration(self):
        # gtsam.Marginals needs a complete, well-posed graph - this module's own graph
        # deliberately isn't one (it's missing the child's local factors touching the
        # foreign placeholders), so it can't be used here the way calibration_loop.py
        # uses it. CJ is always locally owned though, and its GBP belief was already
        # built from proper cross-module message passing - use that precision directly.
        if not self.has_child or self.values.size() == 0:
            return None
        c_key = CJ(self.child_id)
        label = f"calib{self.child_id}"
        try:
            pose = self.values.atPose2(c_key)
        except Exception:
            return {"id": label, "mean": [0.0, 0.0], "cov_xy": [[1000.0, 0.0], [0.0, 1000.0]]}

        cov_xy = [[1000.0, 0.0], [0.0, 1000.0]]
        if self._optimizer is not None and c_key in self._optimizer._gbp_nodes:
            try:
                cov = torch.linalg.inv(self._optimizer._gbp_nodes[c_key].belief_lam).numpy()
                cov_xy = [[float(cov[0, 0]), float(cov[0, 1])], [float(cov[1, 0]), float(cov[1, 1])]]
            except Exception:
                pass
        return {"id": label, "mean": [float(pose.x()), float(pose.y())], "cov_xy": cov_xy}

    def connection_length(self) -> float:
        # Nominal transform from this limb's own joint to its child's - just the physical
        # length for now. Later this can compose in the learned CJ offset instead; every
        # kinematics factor below goes through this one method so that's a one-line change.
        return self.limb_spec.length

    def initialise_planning(self, own_pose_guess, child_pose_guess=None, goal_xy=None,
                             time_horizon=5, dt=None, sigma_endpoint=10.0, sigma_joint=1.0,
                             base_xy=(0.0, 0.0)) -> None:
        """Build this limb's slice of the receding-horizon planning graph, once. Mirrors
        PlanningGraph._setup_robot, scoped to one limb: own J/V chain + dynamics always;
        anchor if root; foreign J(child,k) placeholders + the connecting kinematics factor
        if non-leaf; end-effector J/V + task-space dynamics + goal prior if leaf."""
        assert dt is not None and len(dt) == time_horizon - 1
        self.time_horizon = time_horizon
        self.dt = dt

        for k in range(time_horizon):
            self.planning_values.insert(_J(self.robot_id, self.limb_id, k), own_pose_guess)
            self.planning_values.insert(_V(self.robot_id, self.limb_id, k), np.array([0.0]))
            self._planning_dof_map[_J(self.robot_id, self.limb_id, k)] = 3
            self._planning_dof_map[_V(self.robot_id, self.limb_id, k)] = 1

            if self.is_root:
                anchor_noise = ANCHOR_NOISE if k == 0 else LOOSE_ANCHOR_NOISE
                idx = self.planning_graph.size()
                self.planning_graph.add(gtsam.PriorFactorPose2(
                    _J(self.robot_id, self.limb_id, k),
                    gtsam.Pose2(base_xy[0], base_xy[1], 0.0), anchor_noise))
                if k == 0:
                    self._plan_idx_anchor0 = idx

            if self.has_child:
                self.planning_values.insert(_J(self.robot_id, self.child_id, k), child_pose_guess)
                self._planning_dof_map[_J(self.robot_id, self.child_id, k)] = 3
                idx = self.planning_graph.size()
                self.planning_graph.add(make_fixed_kinematics_factor(
                    _J(self.robot_id, self.limb_id, k), _J(self.robot_id, self.child_id, k),
                    self.connection_length(), 0.0, LOOSE_ANCHOR_NOISE))
                if k == 0:
                    self._plan_idx_kin0 = idx
            else:
                # Leaf owns the end-effector outright - no need to treat it as a separate
                # module, the connecting factor stays entirely local.
                end_pose = own_pose_guess.compose(gtsam.Pose2(self.connection_length(), 0.0, 0.0))
                self.planning_graph.add(make_fixed_kinematics_factor(
                    _J(self.robot_id, self.limb_id, k), _E(self.robot_id, self.limb_id, k),
                    self.connection_length(), 0.0, KINEMATIC_NOISE))
                self.planning_values.insert(_E(self.robot_id, self.limb_id, k), end_pose)
                self.planning_values.insert(_VE(self.robot_id, self.limb_id, k), np.array([0.0, 0.0]))
                self._planning_dof_map[_E(self.robot_id, self.limb_id, k)] = 3
                self._planning_dof_map[_VE(self.robot_id, self.limb_id, k)] = 2

        for k in range(time_horizon - 1):
            self.planning_graph.add(_make_joint_space_dynamics_factor(
                _J(self.robot_id, self.limb_id, k), _V(self.robot_id, self.limb_id, k),
                _J(self.robot_id, self.limb_id, k + 1), _V(self.robot_id, self.limb_id, k + 1),
                dt[k], sigma_joint))
            if not self.has_child:
                self.planning_graph.add(_make_task_space_dynamics_factor(
                    _E(self.robot_id, self.limb_id, k), _VE(self.robot_id, self.limb_id, k),
                    _E(self.robot_id, self.limb_id, k + 1), _VE(self.robot_id, self.limb_id, k + 1),
                    dt[k], sigma_endpoint))

        if not self.has_child and goal_xy is not None:
            self._plan_idx_goal = self.planning_graph.size()
            self.planning_graph.add(gtsam.PriorFactorPose2(
                _E(self.robot_id, self.limb_id, time_horizon - 1),
                gtsam.Pose2(goal_xy[0], goal_xy[1], 0.0), GOAL_NOISE))

    def update_goal(self, goal_xy) -> None:
        # Leaf only - the goal prior sits on the end-effector, which only the leaf owns.
        if self.has_child or not hasattr(self, "_plan_idx_goal"):
            return
        self.planning_graph.replace(self._plan_idx_goal, gtsam.PriorFactorPose2(
            _E(self.robot_id, self.limb_id, self.time_horizon - 1),
            gtsam.Pose2(goal_xy[0], goal_xy[1], 0.0), GOAL_NOISE))

    def planning_foreign_keys(self) -> set:
        if not self.has_child:
            return set()
        return {_J(self.robot_id, self.child_id, k) for k in range(self.time_horizon)}

    def planning_remote_link_keys(self) -> set:
        if self.is_root:
            return set()
        return {_J(self.robot_id, self.limb_id, k) for k in range(self.time_horizon)}

    def pre_solve_planning(self, own_qpos0, own_pose0, child_qpos0=None) -> None:
        """Ground k=0 to the measured state. Mirrors PlanningGraph._pre_solve: every limb
        updates its own k=0 pose; the parent also swaps its k=0 connecting factor's baked-in
        angle from the loose "free to plan" 0.0 to the real measured child encoder angle."""
        self.planning_values.update(_J(self.robot_id, self.limb_id, 0), own_pose0)

        if self.is_root:
            self.planning_graph.replace(self._plan_idx_anchor0, gtsam.PriorFactorPose2(
                _J(self.robot_id, self.limb_id, 0), own_pose0, ANCHOR_NOISE))

        if self.has_child:
            self.planning_graph.replace(self._plan_idx_kin0, make_fixed_kinematics_factor(
                _J(self.robot_id, self.limb_id, 0), _J(self.robot_id, self.child_id, 0),
                self.connection_length(), float(child_qpos0), KINEMATIC_NOISE))
        else:
            end_pose = own_pose0.compose(gtsam.Pose2(self.connection_length(), 0.0, 0.0))
            self.planning_values.update(_E(self.robot_id, self.limb_id, 0), end_pose)

    def build_planning_optimizer(self, n_outer=8, n_inner=20, damping=0.0) -> DistributedGBPOptimizer:
        if self.planning_values.size() == 0:
            self._planning_optimizer = None
            return None
        params = GBPParams(n_outer=n_outer, n_inner=n_inner, damping=damping)
        self._planning_optimizer = DistributedGBPOptimizer(
            self.planning_graph, self.planning_values, params,
            dof_map=self._planning_dof_map,
            foreign_keys=self.planning_foreign_keys(),
            remote_link_keys=self.planning_remote_link_keys())
        return self._planning_optimizer

    def apply_planning_optimizer_result(self) -> None:
        if self._planning_optimizer is not None:
            self.planning_values = self._planning_optimizer.values

    def post_solve_rotate_planning(self) -> None:
        """Shift this module's own receding window forward one step, including its own
        copy of the child's foreign placeholders. Mirrors PlanningGraph._post_solve_rotate,
        purely local - no messages needed for a time-shift within one module."""
        for k in range(self.time_horizon):
            next_k = k + 1 if k < self.time_horizon - 1 else k
            self.planning_values.update(_J(self.robot_id, self.limb_id, k),
                self.planning_values.atPose2(_J(self.robot_id, self.limb_id, next_k)))
            self.planning_values.update(_V(self.robot_id, self.limb_id, k),
                self.planning_values.atVector(_V(self.robot_id, self.limb_id, next_k)))
            if self.has_child:
                self.planning_values.update(_J(self.robot_id, self.child_id, k),
                    self.planning_values.atPose2(_J(self.robot_id, self.child_id, next_k)))
            else:
                self.planning_values.update(_E(self.robot_id, self.limb_id, k),
                    self.planning_values.atPose2(_E(self.robot_id, self.limb_id, next_k)))
                self.planning_values.update(_VE(self.robot_id, self.limb_id, k),
                    self.planning_values.atVector(_VE(self.robot_id, self.limb_id, next_k)))

    def extract_planned_theta(self, k=1) -> float:
        # Own absolute angle only - the demo driver subtracts adjacent modules' values to
        # get the relative ctrl an actuator expects (a module never reads a neighbour's own,
        # non-foreign variable directly - that would bypass the mailbox pattern entirely).
        return self.planning_values.atPose2(_J(self.robot_id, self.limb_id, k)).theta()

    def planned_position(self, k) -> tuple:
        # For drawing the planned horizon - mirrors PlanningGraph.planned_positions, one
        # module's own joint position at a time; the driver assembles the full arm's line.
        p = self.planning_values.atPose2(_J(self.robot_id, self.limb_id, k))
        return (p.x(), p.y())

    def planned_endpoint(self, k) -> tuple:
        # Leaf only - the arm's tip.
        p = self.planning_values.atPose2(_E(self.robot_id, self.limb_id, k))
        return (p.x(), p.y())


def run_distributed_gbp(modules, n_outer=8, n_inner=8, damping=0.0,
                         build_fn=None, apply_fn=None) -> None:
    """
    One synchronous distributed GBP solve across all modules: every module relinearizes
    locally, then for n_inner rounds every module computes its local messages and all
    modules exchange mailboxes before updating beliefs. Mirrors FactorGraph.gbp_solve,
    just fanned out over one DistributedGBPOptimizer per module.

    build_fn/apply_fn pick which per-module optimizer this drives - default to the
    calibration layer. For the planning layer pass e.g.
        build_fn=lambda m: m.build_planning_optimizer(n_outer, n_inner, damping)
        apply_fn=lambda m: m.apply_planning_optimizer_result()
    """
    build_fn = build_fn or (lambda m: m.build_optimizer(n_outer, n_inner, damping))
    apply_fn = apply_fn or (lambda m: m.apply_optimizer_result())

    by_id = {m.limb_id: m for m in modules}
    optimizers = {}
    for m in modules:
        opt = build_fn(m)
        if opt is not None:
            optimizers[m.limb_id] = opt
    if not optimizers:
        return

    for _outer in range(n_outer):
        for opt in optimizers.values():
            opt.relinearize()

        for _inner in range(n_inner):
            for opt in optimizers.values():
                opt.factor_step()

            inboxes = {lid: {} for lid in optimizers}
            for lid, opt in optimizers.items():
                module = by_id[lid]
                for key, msg in opt.get_outgoing().items():
                    # A foreign-key message goes to this module's child; a remote-link
                    # message (this module's own belief) goes up to its parent. Neither
                    # needs decoding the key itself - the module already knows both ids.
                    dest = module.child_id if key in opt.foreign_keys else module.parent_id
                    if dest in inboxes:
                        inboxes[dest][key] = msg

            for lid, msgs in inboxes.items():
                optimizers[lid].apply_incoming(msgs)

            for opt in optimizers.values():
                opt.variable_step()

        for opt in optimizers.values():
            opt.retract()

    for m in modules:
        apply_fn(m)
