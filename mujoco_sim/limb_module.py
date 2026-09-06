import gtsam
import numpy as np
import torch
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from gtsam_examples.gtsam_factors import make_calib_kinematics_factor, make_fixed_kinematics_factor
from gtsam_gbp import GBPParams, DistributedGBPOptimizer

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


def run_distributed_gbp(modules, n_outer=8, n_inner=8, damping=0.0) -> None:
    """
    One synchronous distributed GBP solve across all modules: every module relinearizes
    locally, then for n_inner rounds every module computes its local messages and all
    modules exchange mailboxes before updating beliefs. Mirrors FactorGraph.gbp_solve,
    just fanned out over one DistributedGBPOptimizer per module.
    """
    by_id = {m.limb_id: m for m in modules}
    optimizers = {}
    for m in modules:
        opt = m.build_optimizer(n_outer, n_inner, damping)
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
                parent_id = by_id[lid].parent_id
                for key, msg in opt.get_outgoing().items():
                    # A foreign-key message goes to whoever owns that key; a remote-link
                    # message (this module's own belief) goes up to its parent.
                    dest = key_owner(key) if key in opt.foreign_keys else parent_id
                    if dest in inboxes:
                        inboxes[dest][key] = msg

            for lid, msgs in inboxes.items():
                optimizers[lid].apply_incoming(msgs)

            for opt in optimizers.values():
                opt.variable_step()

        for opt in optimizers.values():
            opt.retract()

    for m in modules:
        m.apply_optimizer_result()
