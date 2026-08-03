import gtsam
import torch
from gbp_utilities import Gaussian


class GBPParams:
    def __init__(self, n_outer: int = 5, n_inner: int = 10,
                 damping: float = 0.0, dof: int = None):
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.damping = damping
        self.dof     = dof  # None = auto-detect per variable from Values


class GBPNode:
    def __init__(self, key: int, dof: int, adj_factor_indices: list = None):
        self.key               = key
        self.DOF               = dof
        self.adj_factor_indices = list(adj_factor_indices) if adj_factor_indices else []
        self.belief_eta = torch.zeros(self.DOF, dtype=torch.float64)
        self.belief_lam = torch.eye(self.DOF,   dtype=torch.float64) * 1e-6

    def reset_belief(self) -> None:
        self.belief_eta = torch.zeros(self.DOF, dtype=torch.float64)
        self.belief_lam = torch.eye(self.DOF,   dtype=torch.float64) * 1e-6

    def update_belief(self, gbp_factors: list) -> None:
        self.reset_belief()
        for fi in self.adj_factor_indices:
            factor = gbp_factors[fi]
            msg_ix = factor.adj_keys.index(self.key)
            self.belief_eta += factor.messages[msg_ix].eta
            self.belief_lam += factor.messages[msg_ix].lam

    def get_delta(self) -> torch.Tensor:
        return torch.linalg.solve(self.belief_lam, self.belief_eta)


class GBPFactor:
    def __init__(self, adj_keys: list, node_dofs: list):
        self.adj_keys  = list(adj_keys)
        self.node_dofs = list(node_dofs)
        self.messages  = [Gaussian(d, type=torch.float64) for d in node_dofs]

    def reset_messages(self) -> None:
        self.messages = [Gaussian(d, type=torch.float64) for d in self.node_dofs]

    def compute_messages(self, eta: torch.Tensor, lam: torch.Tensor,
                         gbp_nodes: dict, damping: float = 0.0) -> None:
        messages_eta, messages_lam = [], []

        start_dim = 0
        for v in range(len(self.adj_keys)):
            eta_f = eta.clone().double()
            lam_f = lam.clone().double()

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

            d    = gbp_nodes[self.adj_keys[v]].DOF
            eo   = eta_f[start_dim:start_dim + d]
            eno  = torch.cat([eta_f[:start_dim], eta_f[start_dim + d:]])
            loo  = lam_f[start_dim:start_dim + d, start_dim:start_dim + d]
            lono = torch.cat([
                lam_f[start_dim:start_dim + d, :start_dim],
                lam_f[start_dim:start_dim + d, start_dim + d:]
            ], dim=1)
            lnoo = torch.cat([
                lam_f[:start_dim,     start_dim:start_dim + d],
                lam_f[start_dim + d:, start_dim:start_dim + d]
            ], dim=0)
            lnono = torch.cat([
                torch.cat([lam_f[:start_dim,     :start_dim],
                           lam_f[:start_dim,     start_dim + d:]], dim=1),
                torch.cat([lam_f[start_dim + d:, :start_dim],
                           lam_f[start_dim + d:, start_dim + d:]], dim=1),
            ], dim=0)

            if lnono.numel() == 0:
                new_lam = loo
                new_eta = eo
            else:
                lnono_inv_lnoo = torch.linalg.solve(lnono, lnoo)
                new_lam = loo - lono @ lnono_inv_lnoo
                new_eta = eo  - lono @ torch.linalg.solve(lnono, eno)

            new_eta = (1 - damping) * new_eta + damping * self.messages[v].eta
            new_lam = (1 - damping) * new_lam + damping * self.messages[v].lam

            messages_eta.append(new_eta)
            messages_lam.append(new_lam)
            start_dim += d

        for v in range(len(self.adj_keys)):
            self.messages[v].eta = messages_eta[v]
            self.messages[v].lam = messages_lam[v]


class GBPOptimizer:
    """
    GBP solver with the same call interface as gtsam.LevenbergMarquardtOptimizer.

    Usage:
        # Uniform DOF (e.g. all Pose2):
        result = GBPOptimizer(graph, values, GBPParams(dof=3)).optimize()

        # Mixed DOF (e.g. planning graph with Pose2 + Vector variables):
        result = GBPOptimizer(graph, values, params, dof_map={key: dof, ...}).optimize()
    """

    def __init__(self, graph: gtsam.NonlinearFactorGraph,
                 values: gtsam.Values,
                 params: GBPParams = None,
                 dof_map: dict = None):
        self.graph   = graph
        self.values  = values
        self.params  = params if params is not None else GBPParams()
        self.dof_map = dof_map
        self._gbp_nodes:   dict = {}
        self._gbp_factors: list = []
        self._build_topology()

    def _build_topology(self) -> None:
        for key in self.values.keys():
            if self.dof_map is not None:
                dof = self.dof_map[key]
            else:
                dof = self.params.dof
            self._gbp_nodes[key] = GBPNode(key, dof)

        for i in range(self.graph.size()):
            f = self.graph.at(i)  # TODO: guard against None when sliding window is added
            keys      = list(f.keys())
            node_dofs = [self._gbp_nodes[k].DOF for k in keys]
            gbp_f = GBPFactor(adj_keys=keys, node_dofs=node_dofs)
            self._gbp_factors.append(gbp_f)
            for k in keys:
                if k in self._gbp_nodes:
                    self._gbp_nodes[k].adj_factor_indices.append(i)

    def optimize(self) -> gtsam.Values:
        if self.values.size() == 0:
            return self.values

        values = self.values

        for _outer in range(self.params.n_outer):
            gaussian_graph = self.graph.linearize(values)

            for gbp_f in self._gbp_factors:
                gbp_f.reset_messages()
            for node in self._gbp_nodes.values():
                node.reset_belief()

            factor_eta_lam = []
            for fi in range(gaussian_graph.size()):
                gf_lin = gaussian_graph.at(fi)
                A, b   = gf_lin.jacobian()
                eta    = torch.from_numpy(A.T @ b).flatten().double()
                lam    = torch.from_numpy(gf_lin.information()).double()
                factor_eta_lam.append((eta, lam))

            for _inner in range(self.params.n_inner):
                for fi, gbp_f in enumerate(self._gbp_factors):
                    eta, lam = factor_eta_lam[fi]
                    gbp_f.compute_messages(eta, lam, self._gbp_nodes, self.params.damping)
                for node in self._gbp_nodes.values():
                    node.update_belief(self._gbp_factors)

            deltas = gtsam.VectorValues()
            for key, node in self._gbp_nodes.items():
                deltas.insert(key, node.get_delta().numpy())
            values = values.retract(deltas)

        return values
