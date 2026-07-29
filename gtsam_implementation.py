import math
import random
import json

import gtsam
from gtsam import symbol
import numpy as np
import torch

from gbp_utilities import Gaussian

# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------
KINEMATIC_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5,  0.5,  1e-4]))
ANCHOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-6]))
CALIB_NOISE        = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))
SENSOR_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 0.001]))
SENSOR_NOISE       = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0 ]))


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
# Keys are encoded as symbol(char, integer_index).
# Limb pose    per observation : symbol('L', limb_id * 10000 + step)
# Sensor pose  per observation : symbol('S', limb_id * 10000 + step)
# Joint calib  singleton       : symbol('C', limb_id)          -- shared across all steps
# Sensor calib singleton       : symbol('Z', limb_id)          -- shared across all steps

def _limb_key(limb_id: int, step: int) -> int:
    return symbol('L', limb_id * 10000 + step)

def _sensor_key(limb_id: int, step: int) -> int:
    return symbol('S', limb_id * 10000 + step)

def _calib_key(limb_id: int) -> int:
    """Singleton joint-calibration key – same key every step."""
    return symbol('C', limb_id)

def _sensor_calib_key(limb_id: int) -> int:
    """Singleton sensor-calibration key – same key every step."""
    return symbol('Z', limb_id)


# ---------------------------------------------------------------------------
# Custom kinematics factor
# ---------------------------------------------------------------------------
def make_kinematics_factor(key_n1: int, key_c: int, key_n2: int,
                            L: float, theta_joint: float,
                            noise_model) -> gtsam.CustomFactor:
    """
    3-node factor: T_pred = N1 * Pose2(L,0,0) * C * Pose2(0,0,theta_joint)
    error = N2.localCoordinates(T_pred)
    Analytic Jacobians via GTSAM Pose2 chain rule.
    """
    L_link  = gtsam.Pose2(float(L), 0.0, 0.0)
    J_joint = gtsam.Pose2(0.0, 0.0, float(theta_joint))

    def error_func(this, values, jacobians):
        n1 = values.atPose2(this.keys()[0])
        c  = values.atPose2(this.keys()[1])
        n2 = values.atPose2(this.keys()[2])

        def z():
            return np.zeros((3, 3), order='F')

        H_A_N1 = z(); H_A_L = z()
        A = n1.compose(L_link, H_A_N1, H_A_L)

        H_B_A = z(); H_B_C = z()
        B = A.compose(c, H_B_A, H_B_C)

        H_T_B = z(); H_T_J = z()
        T_pred = B.compose(J_joint, H_T_B, H_T_J)

        H_e_N2 = z(); H_e_T = z()
        error = n2.localCoordinates(T_pred, H_e_N2, H_e_T)

        if jacobians is not None:
            jacobians[0] = H_e_T @ H_T_B @ H_B_A @ H_A_N1  # de/dN1
            jacobians[1] = H_e_T @ H_T_B @ H_B_C            # de/dC
            jacobians[2] = H_e_N2                            # de/dN2

        return error

    keys = gtsam.KeyVector()
    keys.append(key_n1)
    keys.append(key_c)
    keys.append(key_n2)
    return gtsam.CustomFactor(noise_model, keys, error_func)


# ---------------------------------------------------------------------------
# GBP building blocks
# ---------------------------------------------------------------------------

class GBPNode:
    """
    Represents a single variable node in the GBP message-passing graph.

    Belief is stored as information-form Gaussian (eta, lam) in the LOCAL
    tangent-space (delta-space) relative to the current linearisation point.
    get_delta() solves lam * delta = eta to give the retract step.
    """
    DOF = 3  # all nodes are Pose2

    def __init__(self, key: int, adj_factor_indices: list):
        self.key = key
        self.adj_factor_indices = list(adj_factor_indices)  # indices into gbp_factors list
        # Belief in information form – reset each outer (linearisation) iteration
        self.belief_eta = torch.zeros(self.DOF, dtype=torch.float64)
        self.belief_lam = torch.eye(self.DOF, dtype=torch.float64) * 1e-6

    def reset_belief(self) -> None:
        self.belief_eta = torch.zeros(self.DOF, dtype=torch.float64)
        self.belief_lam = torch.eye(self.DOF, dtype=torch.float64) * 1e-6

    def update_belief(self, gbp_factors: list) -> None:
        """Accumulate messages from all adjacent factors."""
        self.reset_belief()
        for fi in self.adj_factor_indices:
            factor = gbp_factors[fi]
            msg_ix = factor.adj_keys.index(self.key)
            self.belief_eta += factor.messages[msg_ix].eta
            self.belief_lam += factor.messages[msg_ix].lam

    def get_delta(self) -> torch.Tensor:
        """Solve lam * delta = eta for the tangent-space correction."""
        return torch.linalg.solve(self.belief_lam, self.belief_eta)


class GBPFactor:
    """
    Represents a factor in the GBP message-passing graph.

    adj_keys   – ordered list of GTSAM keys, same order as the factor's keys()
                 which matches the column blocks of information() / jacobian().
    node_dofs  – DOF per adjacent node (all 3 for Pose2).
    messages   – one Gaussian per adjacent node, holding the current outgoing
                 message from this factor to that node.
    """

    def __init__(self, adj_keys: list, node_dofs: list):
        self.adj_keys  = list(adj_keys)
        self.node_dofs = list(node_dofs)
        self.messages  = [Gaussian(d, type=torch.float64) for d in node_dofs]

    def reset_messages(self) -> None:
        self.messages = [Gaussian(d, type=torch.float64) for d in self.node_dofs]

    def compute_messages(self, eta: torch.Tensor, lam: torch.Tensor,
                         gbp_nodes: dict, damping: float = 0.0) -> None:
        """
        Compute all outgoing messages from this factor given the current
        factor information (eta, lam) at the linearisation point.

        Uses the standard GBP marginalisation formula:
          lam_msg = lam_oo  - lam_on @ inv(lam_nn) @ lam_no
          eta_msg = eta_o   - lam_on @ inv(lam_nn) @ eta_n
        where 'o' = the target variable, 'n' = all other variables.

        Cavity: before marginalising we incorporate the incoming messages
        from all neighbours OTHER than the current target variable, so that
        we avoid double-counting.
        """
        messages_eta, messages_lam = [], []

        start_dim = 0  # column offset of the current target variable in lam/eta
        for v in range(len(self.adj_keys)):
            # Work with fresh copies of the full factor eta/lam
            eta_f = eta.clone().double()
            lam_f = lam.clone().double()

            # Incorporate cavity: add in the belief minus the old message for
            # every neighbour OTHER than the target variable v
            col = 0
            for var, adj_key in enumerate(self.adj_keys):
                if var != v:
                    nd = gbp_nodes[adj_key].DOF
                    eta_f[col:col + nd] += (
                        gbp_nodes[adj_key].belief_eta - self.messages[var].eta
                    )
                    lam_f[col:col + nd, col:col + nd] += (
                        gbp_nodes[adj_key].belief_lam - self.messages[var].lam
                    )
                col += gbp_nodes[adj_key].DOF

            # Partition the updated information into target (o) and rest (n) blocks
            d = gbp_nodes[self.adj_keys[v]].DOF  # DOF of target variable

            eo   = eta_f[start_dim : start_dim + d]
            eno  = torch.cat([eta_f[:start_dim], eta_f[start_dim + d:]])

            loo  = lam_f[start_dim:start_dim + d, start_dim:start_dim + d]
            lono = torch.cat([
                lam_f[start_dim:start_dim + d, :start_dim],
                lam_f[start_dim:start_dim + d, start_dim + d:]
            ], dim=1)
            lnoo = torch.cat([
                lam_f[:start_dim,         start_dim:start_dim + d],
                lam_f[start_dim + d:,     start_dim:start_dim + d]
            ], dim=0)
            lnono = torch.cat([
                torch.cat([lam_f[:start_dim,     :start_dim],
                           lam_f[:start_dim,     start_dim + d:]], dim=1),
                torch.cat([lam_f[start_dim + d:, :start_dim],
                           lam_f[start_dim + d:, start_dim + d:]], dim=1),
            ], dim=0)

            # Marginalise out the 'n' variables
            if lnono.numel() == 0:
                # Unary factor: message = factor itself
                new_lam = loo
                new_eta = eo
            else:
                lnono_inv_lnoo = torch.linalg.solve(lnono, lnoo)
                new_lam = loo  - lono @ lnono_inv_lnoo
                new_eta = eo   - lono @ torch.linalg.solve(lnono, eno)

            # Damping
            new_eta = (1 - damping) * new_eta + damping * self.messages[v].eta
            new_lam = (1 - damping) * new_lam + damping * self.messages[v].lam

            messages_eta.append(new_eta)
            messages_lam.append(new_lam)
            start_dim += d

        # Write messages back all at once (synchronous update)
        for v in range(len(self.adj_keys)):
            self.messages[v].eta = messages_eta[v]
            self.messages[v].lam = messages_lam[v]


# ---------------------------------------------------------------------------
# GTSAMSolver — wraps graph + values + GBP state + step counter
# ---------------------------------------------------------------------------
class GTSAMSolver:
    """
    Drop-in replacement for the old FactorGraph object expected by live_server.py.

    Public interface:
        fg.step_count                    int
        fg.var_nodes                     dict  (for logging)
        fg.factors                       list  (for logging)
        update_factor_graph(data, fg)
        fg.gbp_solve(n_iters, n_inner)   runs GBP outer/inner loops
        fg.energy()                      float graph error
        extract_calibrations(fg)         list of calibration dicts

    GBP state:
        fg.gbp_nodes    dict[key -> GBPNode]    one per Values entry
        fg.gbp_factors  list[GBPFactor]         parallel to graph factors
    """

    # GBP hyper-parameters (can be overridden after construction)
    N_OUTER  = 2   # re-linearisation steps per gbp_solve() call
    N_INNER  = 10   # message-passing iterations per linearisation
    DAMPING  = 0.0  # message damping (0 = no damping)

    def __init__(self):
        self.graph      = gtsam.NonlinearFactorGraph()
        self.values     = gtsam.Values()
        self.step_count = 0

        # GBP message-passing registries
        self.gbp_nodes:   dict = {}   # key (int) -> GBPNode
        self.gbp_factors: list = []   # list[GBPFactor], parallel to self.graph

        # Metadata for calibration extraction
        self._calib_ids:       list = []
        self._sensor_calib_ids: list = []

        # Logging mirrors (live_server.py reads these)
        self.var_nodes: dict = {}
        self.factors:   list = []

    # ------------------------------------------------------------------
    def _register_node(self, key: int) -> None:
        """Add a GBPNode for a newly inserted Values entry (if not already present)."""
        if key not in self.gbp_nodes:
            self.gbp_nodes[key] = GBPNode(key, adj_factor_indices=[])

    def _register_factor(self, nl_factor_keys: list) -> int:
        """
        Add a GBPFactor that mirrors the NonlinearFactor just appended to self.graph.
        Returns the index of the new GBPFactor in self.gbp_factors.
        All nodes are Pose2 (DOF=3).
        """
        node_dofs = [GBPNode.DOF] * len(nl_factor_keys)
        gbp_f = GBPFactor(adj_keys=nl_factor_keys, node_dofs=node_dofs)
        fi = len(self.gbp_factors)
        self.gbp_factors.append(gbp_f)
        # Register adjacency in each node
        for key in nl_factor_keys:
            if key in self.gbp_nodes:
                if fi not in self.gbp_nodes[key].adj_factor_indices:
                    self.gbp_nodes[key].adj_factor_indices.append(fi)
        return fi

    def _add_factor(self, gtsam_factor, logging_tag) -> None:
        """
        Append a GTSAM factor to the graph and register the matching GBPFactor.
        All graph additions go through this method so the two registries stay in sync.
        """
        self.graph.add(gtsam_factor)
        keys = list(gtsam_factor.keys())
        self._register_factor(keys)
        self.factors.append((logging_tag, *keys))

    # ------------------------------------------------------------------
    def centralised_solve(self, n_iters: int = 20) -> None:
        """Run Levenberg-Marquardt optimisation and update internal values."""
        if self.values.size() == 0:
            return
        params = gtsam.LevenbergMarquardtParams()
        # params.setMaxIterations(n_iters)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.values, params)
        self.values = optimizer.optimize()

    def gbp_solve(self, n_iters: int = None, n_inner: int = None,
                  damping: float = None) -> None:
        """
        Run GBP with re-linearisation.

        n_iters : number of outer (re-linearisation + retract) steps.
                  Defaults to self.N_OUTER.
        n_inner : number of inner message-passing iterations per linearisation.
                  Defaults to self.N_INNER.
        damping : message damping coefficient [0, 1).
                  Defaults to self.DAMPING.
        """
        if self.values.size() == 0:
            return

        n_outer = n_iters if n_iters is not None else self.N_OUTER
        n_in    = n_inner if n_inner is not None else self.N_INNER
        damp    = damping if damping is not None else self.DAMPING

        for _outer in range(n_outer):
            # ---- Step 1: Linearise at current Values ----
            gaussian_graph = self.graph.linearize(self.values)

            # ---- Step 2: Reset messages and beliefs ----
            for gf in self.gbp_factors:
                gf.reset_messages()
            for node in self.gbp_nodes.values():
                node.reset_belief()

            # ---- Step 3: Extract (eta, lam) from each linearised factor ----
            factor_eta_lam = []
            for fi in range(gaussian_graph.size()):
                gf_lin = gaussian_graph.at(fi)
                A, b = gf_lin.jacobian()
                eta  = torch.from_numpy(A.T @ b).flatten().double()
                lam  = torch.from_numpy(gf_lin.information()).double()
                factor_eta_lam.append((eta, lam))

            # ---- Step 4: Inner GBP loop on the fixed linearised graph ----
            for _inner in range(n_in):
                # Compute messages (all factors first – synchronous schedule)
                for fi, gbp_f in enumerate(self.gbp_factors):
                    eta, lam = factor_eta_lam[fi]
                    gbp_f.compute_messages(eta, lam, self.gbp_nodes, damp)
                # Update all node beliefs
                for node in self.gbp_nodes.values():
                    node.update_belief(self.gbp_factors)

            # ---- Step 5: Retract using each node's converged delta ----
            deltas = gtsam.VectorValues()
            for key, node in self.gbp_nodes.items():
                delta = node.get_delta().numpy()
                deltas.insert(key, delta)
            self.values = self.values.retract(deltas)

    # ------------------------------------------------------------------
    def energy(self) -> float:
        """Return current graph error (sum of squared whitened residuals)."""
        if self.values.size() == 0:
            return 0.0
        return self.graph.error(self.values)


# ---------------------------------------------------------------------------
# Noise helpers (unchanged from original)
# ---------------------------------------------------------------------------
def add_noise_to_measurement(angle_rad: float, noise_std: float) -> float:
    if noise_std > 0:
        return angle_rad + random.gauss(0, noise_std)
    return angle_rad


def add_noise(data, angle_noise=True, angle_noise_deg=2.0,
              distance_noise=False, dist_noise_units=2.0):
    limbs       = data["limbs"]
    connections = data["connections"]
    limbs       = sorted(limbs,       key=lambda x: x['depth'])
    connections = sorted(connections, key=lambda x: x['depth'])

    limbs_dict = {limb["id"]: limb for limb in limbs}

    if angle_noise:
        for limb in limbs_dict.values():
            limb["local_angle"] = add_noise_to_measurement(
                limb["local_angle"], angle_noise_deg)

    if distance_noise:
        for conn in connections:
            conn["distance"] = add_noise_to_measurement(
                conn["distance"], dist_noise_units)

    return limbs_dict, connections


# ---------------------------------------------------------------------------
# Forward kinematics (unchanged from original – returns torch tensors)
# ---------------------------------------------------------------------------
def forward_kinematics_step(prev_conn_pose, joint_angle_rad, limb_length):
    joint_angle_rad = torch.tensor(float(joint_angle_rad))
    current_theta   = prev_conn_pose[2] + joint_angle_rad
    current_theta   = torch.atan2(torch.sin(current_theta),
                                  torch.cos(current_theta))

    current_limb_pose = torch.tensor([
        prev_conn_pose[0].item() if hasattr(prev_conn_pose[0], 'item') else float(prev_conn_pose[0]),
        prev_conn_pose[1].item() if hasattr(prev_conn_pose[1], 'item') else float(prev_conn_pose[1]),
        current_theta.item(),
    ])

    next_x = float(current_limb_pose[0]) + limb_length * float(torch.cos(current_theta))
    next_y = float(current_limb_pose[1]) + limb_length * float(torch.sin(current_theta))

    next_conn_pose = torch.tensor([next_x, next_y, current_theta.item()])
    return current_limb_pose, next_conn_pose


# ---------------------------------------------------------------------------
# create_gbp_solver  – returns a GTSAMSolver (same name, same call site)
# ---------------------------------------------------------------------------
def create_gbp_solver() -> GTSAMSolver:
    """
    Create and return a fresh GTSAMSolver ready to receive poses.
    Call once per calibration session (e.g. on WebSocket connect or reset).
    """
    return GTSAMSolver()


# ---------------------------------------------------------------------------
# update_factor_graph
# ---------------------------------------------------------------------------
def update_factor_graph(data: dict, fg: GTSAMSolver) -> None:
    """
    Ingest one pose observation into the factor graph.

    For each step a new set of limb-pose and sensor-pose nodes is added.
    Calibration nodes (joint and sensor) are singletons: created only on the
    first step and reused (connected to) every subsequent step.

    Args:
        data: dict with keys "limbs" and "connections".
        fg:   GTSAMSolver instance (from create_gbp_solver()).
    """
    step = fg.step_count

    limbs_dict, connections = add_noise(
        data,
        angle_noise=False,  angle_noise_deg=1,
        distance_noise=False, dist_noise_units=1,
    )

    next_conn_pose = torch.tensor([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #
    # 2. Insert variable nodes into Values and add singleton factors      #
    # ------------------------------------------------------------------ #
    for limb_id, limb in limbs_dict.items():

        length = float(limb["limb_length"])
        theta_rad = math.radians(float(limb["local_angle"]))
        position_estimate, next_conn_pose = forward_kinematics_step(next_conn_pose, theta_rad, length)

        L_key  = _limb_key(limb_id, step)
        S_key  = _sensor_key(limb_id, step)
        C_key  = _calib_key(limb_id)
        Z_key  = _sensor_calib_key(limb_id)

        # Limb pose node (new every step)
        fg.values.insert(
            L_key,
            gtsam.Pose2(float(position_estimate[0]), float(position_estimate[1]), float(position_estimate[2]))
        )
        fg._register_node(L_key)

        # Sensor pose node — initially same as limb pose (refined via factors)
        fg.values.insert(
            S_key,
            gtsam.Pose2(float(next_conn_pose[0]), float(next_conn_pose[1]), float(next_conn_pose[2]))
        )
        fg._register_node(S_key)

        # Anchor: pin base limb (depth 0) to origin
        if limb["depth"] == 0:
            fg._add_factor(
                gtsam.PriorFactorPose2(L_key, gtsam.Pose2(0.0, 0.0, 0.0), ANCHOR_NOISE),
                'anchor'
            )

        # Singleton calibration nodes — created only on step 0
        if step == 0:
            if limb["depth"] != 0:
                # Joint calibration (only non-root limbs have a parent joint)
                fg.values.insert(C_key, gtsam.Pose2(0.0, 0.0, 0.0))
                fg._register_node(C_key)
                fg._add_factor(
                    gtsam.PriorFactorPose2(C_key, gtsam.Pose2(0.0, 0.0, 0.0), CALIB_NOISE),
                    'calib_prior'
                )
                if limb_id not in fg._calib_ids:
                    fg._calib_ids.append(limb_id)

            # Sensor calibration (all limbs that carry a sensor)
            fg.values.insert(Z_key, gtsam.Pose2(0.0, 0.0, 0.0))
            fg._register_node(Z_key)
            fg._add_factor(
                gtsam.PriorFactorPose2(Z_key, gtsam.Pose2(0.0, 0.0, 0.0), SENSOR_CALIB_NOISE),
                'sensor_calib_prior'
            )
            if limb_id not in fg._sensor_calib_ids:
                fg._sensor_calib_ids.append(limb_id)

        # Update var_nodes mirror for live_server.py logging
        key_str = f"L{limb_id}"
        fg.var_nodes.setdefault(key_str, []).append({'step': step, 'limb_id': limb_id})

    # ------------------------------------------------------------------ #
    # 3. Add kinematic and distance factors for each connection           #
    # ------------------------------------------------------------------ #
    for conn in connections:
        parent_id = conn["parent_id"]
        child_id  = conn["child_id"]

        parent_limb = limbs_dict[parent_id]
        child_limb  = limbs_dict[child_id]

        parent_L = float(parent_limb["limb_length"])
        angle_rad = math.radians(float(child_limb["local_angle"]))
        dist_meas = float(conn["distance"])

        parent_sensor_offset = parent_limb["sensor_offset"]
        child_sensor_offset  = child_limb["sensor_offset"]
        # Sensor offset is stored as {x, y} in the limb's local frame.
        # We model the sensor as a point at distance sqrt(x^2+y^2) along a
        # direction given by atan2(y, x) from the limb origin.
        # s_parent_dist  = math.hypot(parent_sensor_offset["x"], parent_sensor_offset["y"])
        # s_parent_angle = math.atan2(parent_sensor_offset["y"], parent_sensor_offset["x"])
        # s_child_dist   = math.hypot(child_sensor_offset["x"],  child_sensor_offset["y"])
        # s_child_angle  = math.atan2(child_sensor_offset["y"],  child_sensor_offset["x"])

        L_parent = _limb_key(parent_id, step)
        L_child  = _limb_key(child_id,  step)
        C_child  = _calib_key(child_id)           # singleton
        S_parent = _sensor_key(parent_id, step)
        S_child  = _sensor_key(child_id,  step)
        Z_parent = _sensor_calib_key(parent_id)   # singleton
        Z_child  = _sensor_calib_key(child_id)    # singleton

        # Kinematic factor: parent limb -> joint calib -> child limb
        fg._add_factor(
            make_kinematics_factor(L_parent, C_child, L_child,
                                   L=parent_L, theta_joint=angle_rad,
                                   noise_model=KINEMATIC_NOISE),
            'kinematics'
        )

        # Sensor placement factors: limb origin -> sensor calib -> sensor pose
        fg._add_factor(
            make_kinematics_factor(L_parent, Z_parent, S_parent,
                                   L=parent_sensor_offset["x"], theta_joint=0.0,
                                   noise_model=KINEMATIC_NOISE),
            'sensor_kin'
        )

        fg._add_factor(
            make_kinematics_factor(L_child, Z_child, S_child,
                                   L=child_sensor_offset["x"], theta_joint=0.0,
                                   noise_model=KINEMATIC_NOISE),
            'sensor_kin'
        )

        # Distance measurement: range between sensor origins
        fg._add_factor(
            gtsam.RangeFactorPose2(S_parent, S_child, dist_meas, SENSOR_NOISE),
            'range'
        )

    fg.step_count += 1


# ---------------------------------------------------------------------------
# extract_calibrations  – queries gtsam.Values + gtsam.Marginals
# ---------------------------------------------------------------------------
def extract_calibrations(fg: GTSAMSolver) -> list:
    """
    Extract joint calibration estimates from the factor graph.

    Returns a list of dicts, one per joint-calibration node:
      {
        "id":     "calib2",
        "mean":   [x, y],
        "cov_xy": [[cxx, cxy], [cyx, cyy]]
      }
    """
    results = []

    # Attempt to compute marginals for covariances (may fail on under-constrained graphs)
    marginals = None
    if fg.values.size() > 0:
        try:
            marginals = gtsam.Marginals(fg.graph, fg.values)
        except Exception:
            marginals = None

    for limb_id in fg._calib_ids:
        c_key = _calib_key(limb_id)
        label = f"calib{limb_id}"

        try:
            pose = fg.values.atPose2(c_key)
            mean_x = pose.x()
            mean_y = pose.y()

            if marginals is not None:
                cov = marginals.marginalCovariance(c_key)   # 3x3
                cov_xy = [
                    [float(cov[0, 0]), float(cov[0, 1])],
                    [float(cov[1, 0]), float(cov[1, 1])],
                ]
            else:
                cov_xy = [[1000.0, 0.0], [0.0, 1000.0]]

            results.append({
                "id":     label,
                "mean":   [float(mean_x), float(mean_y)],
                "cov_xy": cov_xy,
            })

        except Exception:
            results.append({
                "id":     label,
                "mean":   [0.0, 0.0],
                "cov_xy": [[1000.0, 0.0], [0.0, 1000.0]],
            })

    return results


# ---------------------------------------------------------------------------
# __main__ – smoke-test matching the original gbp_implementation.py style
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fg = create_gbp_solver()

    with open("pose_data_example.json", "r") as f:
        poses = json.load(f)

    for pose in poses:
        update_factor_graph(pose, fg)

    print(f"Graph built: {fg.values.size()} variables, {len(fg.factors)} factors")

    fg.centralised_solve()

    print("Optimisation complete")
    print(f"Energy: {fg.energy():.4f}")

    calibrations = extract_calibrations(fg)
    for c in calibrations:
        print(f"  {c['id']}: mean={[round(v,4) for v in c['mean']]}")
