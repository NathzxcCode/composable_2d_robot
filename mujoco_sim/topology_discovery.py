import gtsam
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional, Tuple
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from gtsam_examples.gtsam_factors import make_calib_kinematics_factor
from gtsam_gbp import GBPOptimizer, GBPParams


_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))


class CandidateStatus(Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


@dataclass
class Observation:
    J_a:     gtsam.Pose2
    J_b:     gtsam.Pose2
    theta_b: float


@dataclass
class CandidatePair:
    child_id:              int
    parent_id:             int
    status:                CandidateStatus        = CandidateStatus.CANDIDATE
    co_occurrence_count:   int                    = 0
    absence_count:         int                    = 0
    observations:          deque                  = field(default_factory=deque)
    cj_estimate:           Optional[gtsam.Pose2]  = None   # warm start preserved across resets
    cj_cov:                Optional[np.ndarray]   = None
    last_cost_per_obs:     float                  = float('inf')
    last_cov_trace:        float                  = float('inf')
    consecutive_high_cost: int                    = 0


class TopologyDiscovery:
    """
    Per-limb topology discovery via factor graph structure learning.

    For each candidate child limb b, maintains an isolated GBP subgraph with a
    single calibration variable CJ(a→b). GPS-observed joint poses are added as
    PriorFactorPose2 variables (tightness controlled by sigma_gps) connected via
    make_calib_kinematics_factor. A truly connected pair converges to a stable,
    low-residual CJ; an unconnected pair cannot find a time-invariant CJ.

    State machine:
        CANDIDATE: accumulates observations each co-occurrence. At exactly K
                   co-occurrences a one-shot GBP solve decides the outcome:
                     - cost_per_obs < cost_thr AND cov < cov_thr → CONFIRMED
                     - otherwise → reset clock (co_occurrence=0, clear obs, keep CJ warm start)
        CONFIRMED: monitors every new observation via a sliding window. If
                   cost_per_obs > reject_thr for K_stale consecutive steps → reset to CANDIDATE.
                   TTL absence removes the pair entirely.

    K is chosen as the minimum batch size for a reliable one-shot decision,
    determined empirically from convergence properties and noise level.

    GPS source: joint pivot position from MuJoCo xanchor + body rotation matrix.
    Simplification: sensor-to-joint conversion is skipped; xanchor is used
    directly as the joint pose. Replace with S.compose(Pose2(-sx,-sy,0)) when
    physical sensor offsets are introduced.
    """

    def __init__(
        self,
        limb_id:         int,
        limbs,
        sigma_gps_pos:   float = 0.001,
        sigma_gps_theta: float = 0.005,
        n_sigma_search:  float = 5.0,
        K:               int   = 5,      # batch size: minimum observations for a reliable decision
        TTL_max:         int   = 15,
        T_max:           int   = 15,     # sliding window size for CONFIRMED monitoring (>= K)
        cost_thr:        float = 0.5,    # per-observation cost threshold for confirmation
        cov_thr:         float = 0.01,
        reject_thr:      float = 5.0,    # per-observation cost above which a pair is considered disconnected
        K_stale:         int   = 3,      # consecutive high-cost steps before CONFIRMED resets
        n_outer:         int   = 3,
        n_inner:         int   = 5,
    ):
        assert K <= T_max, "K must be <= T_max"

        self.limb_id         = limb_id
        self.limbs           = limbs
        self.L_a             = limbs[limb_id].length

        self.sigma_gps_pos   = sigma_gps_pos
        self.sigma_gps_theta = sigma_gps_theta
        self.search_radius   = n_sigma_search * sigma_gps_pos

        self.K          = K
        self.TTL_max    = TTL_max
        self.T_max      = T_max
        self.cost_thr   = cost_thr
        self.cov_thr    = cov_thr
        self.reject_thr = reject_thr
        self.K_stale    = K_stale
        self.n_outer    = n_outer
        self.n_inner    = n_inner

        self._gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([sigma_gps_pos, sigma_gps_pos, sigma_gps_theta])
        )

        self._persistence:        Dict[int, CandidatePair] = {}
        self._confirmed_children: Dict[int, Tuple]         = {}

    def update(self, gps_joint_poses: dict, encoder_angles: dict) -> None:
        """
        gps_joint_poses : {limb_id: Pose2(x, y, theta)} for all limbs in the scene
        encoder_angles  : {limb_id: float} MuJoCo qpos value per limb
        """
        J_a = gps_joint_poses[self.limb_id]

        candidates_this_step = {
            other_id
            for other_id, J_b in gps_joint_poses.items()
            if other_id != self.limb_id and self._in_search_region(J_a, J_b)
        }

        for cid in candidates_this_step:
            if cid not in self._persistence:
                self._persistence[cid] = CandidatePair(
                    child_id=cid, parent_id=self.limb_id,
                    observations=deque(maxlen=self.T_max),
                )

            pair = self._persistence[cid]
            pair.co_occurrence_count += 1
            pair.absence_count = 0
            pair.observations.append(Observation(
                J_a=J_a,
                J_b=gps_joint_poses[cid],
                theta_b=encoder_angles.get(cid, 0.0),
            ))

            if pair.status == CandidateStatus.CANDIDATE and pair.co_occurrence_count == self.K:
                # K-batch decision: first reliable opportunity to evaluate the connection
                cost_per_obs, cov_trace, cj_est, cj_cov = self._optimise_subgraph(pair)
                pair.last_cost_per_obs = cost_per_obs
                pair.last_cov_trace    = cov_trace
                pair.cj_estimate       = cj_est
                pair.cj_cov            = cj_cov

                if cost_per_obs < self.cost_thr and cov_trace < self.cov_thr:
                    pair.status = CandidateStatus.CONFIRMED
                    self._confirmed_children[cid] = (cj_est, cj_cov)
                else:
                    # Failed: remove entirely so the next attempt starts with a clean slate
                    del self._persistence[cid]

            elif pair.status == CandidateStatus.CONFIRMED:
                cost_per_obs, cov_trace, cj_est, cj_cov = self._optimise_subgraph(pair)
                pair.last_cost_per_obs = cost_per_obs
                pair.last_cov_trace    = cov_trace
                pair.cj_estimate       = cj_est
                pair.cj_cov            = cj_cov

                if cost_per_obs > self.reject_thr:
                    pair.consecutive_high_cost += 1
                    if pair.consecutive_high_cost >= self.K_stale:
                        # Sustained high cost — reset to CANDIDATE for fresh re-evaluation
                        pair.status = CandidateStatus.CANDIDATE
                        pair.co_occurrence_count = 0
                        pair.observations.clear()
                        pair.consecutive_high_cost = 0
                        self._confirmed_children.pop(cid, None)
                else:
                    pair.consecutive_high_cost = 0
                    self._confirmed_children[cid] = (cj_est, cj_cov)

        for cid in list(self._persistence):
            if cid not in candidates_this_step:
                self._persistence[cid].absence_count += 1
                if self._persistence[cid].absence_count > self.TTL_max:
                    self._confirmed_children.pop(cid, None)
                    del self._persistence[cid]

    def _in_search_region(self, J_a: gtsam.Pose2, J_b: gtsam.Pose2) -> bool:
        """2D capsule test: is J_b within search_radius of J_a's body segment?"""
        pa    = np.array([J_a.x(), J_a.y()])
        pb    = np.array([J_b.x(), J_b.y()])
        theta = J_a.theta()
        tip   = pa + np.array([np.cos(theta), np.sin(theta)]) * self.L_a
        seg   = tip - pa
        sq    = seg @ seg
        if sq < 1e-12:
            return bool(np.linalg.norm(pb - pa) < self.search_radius)
        t       = float(np.clip((pb - pa) @ seg / sq, 0.0, 1.0))
        closest = pa + t * seg
        return bool(np.linalg.norm(pb - closest) < self.search_radius)

    def _optimise_subgraph(self, pair: CandidatePair):
        """
        Build and solve the isolated calibration subgraph for one candidate pair.

        Variables : CJ (singleton), J_a(t) and J_b(t) per observation
        Factors   : PriorFactorPose2 on J_a(t) and J_b(t) (GPS measurement),
                    make_calib_kinematics_factor(J_a(t), CJ, J_b(t), L_a, theta_b),
                    weak PriorFactorPose2 on CJ.

        The graph is cycle-free; GBP converges exactly in n_inner iterations.
        Cost is normalized by observation count for scale-invariant thresholding.
        """
        graph  = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        cj_key = gtsam.Symbol('c', self.limb_id * 100 + pair.child_id).key()
        values.insert(cj_key, pair.cj_estimate if pair.cj_estimate is not None else gtsam.Pose2())
        graph.add(gtsam.PriorFactorPose2(cj_key, gtsam.Pose2(), _CALIB_NOISE))

        for t, obs in enumerate(pair.observations):
            ja_key = gtsam.Symbol('a', t).key()
            jb_key = gtsam.Symbol('b', t).key()
            values.insert(ja_key, obs.J_a)
            values.insert(jb_key, obs.J_b)
            graph.add(gtsam.PriorFactorPose2(ja_key, obs.J_a, self._gps_noise))
            graph.add(gtsam.PriorFactorPose2(jb_key, obs.J_b, self._gps_noise))
            graph.add(make_calib_kinematics_factor(
                ja_key, cj_key, jb_key,
                self.L_a, obs.theta_b, self._gps_noise,
            ))

        params = GBPParams(n_outer=self.n_outer, n_inner=self.n_inner, damping=0.0, dof=3)
        result       = GBPOptimizer(graph, values, params).optimize()
        cost_per_obs = graph.error(result) / len(pair.observations)
        cj_est       = result.atPose2(cj_key)

        try:
            marginals = gtsam.Marginals(graph, result)
            cj_cov    = marginals.marginalCovariance(cj_key)
            cov_trace = float(np.trace(cj_cov[:2, :2]))
        except Exception:
            cj_cov    = np.eye(3) * 1000.0
            cov_trace = float('inf')

        return cost_per_obs, cov_trace, cj_est, cj_cov

    def get_children(self) -> dict:
        """Returns {child_id: (CJ_pose2, cov_3x3)} for all confirmed pairs."""
        return dict(self._confirmed_children)

    def get_persistence_state(self) -> dict:
        """Returns {child_id: CandidateStatus} for all tracked pairs."""
        return {cid: pair.status for cid, pair in self._persistence.items()}

    def get_diagnostics(self) -> dict:
        """
        Returns per-observation cost and convergence info for evaluated pairs.
        Use this output to empirically tune cost_thr, reject_thr, and cov_thr.
        Only includes pairs that have been evaluated at least once (gate has fired).
        """
        return {
            cid: {
                "cost_per_obs":          pair.last_cost_per_obs,
                "cov_trace":             pair.last_cov_trace,
                "status":                pair.status.value,
                "n_obs":                 len(pair.observations),
                "consecutive_high_cost": pair.consecutive_high_cost,
                "cj_estimate":           pair.cj_estimate,
            }
            for cid, pair in self._persistence.items()
            if pair.last_cost_per_obs < float('inf')
        }


def identify_root(discoveries: list) -> set:
    """Returns limb IDs not claimed as a confirmed child by any other limb."""
    claimed = set()
    for disc in discoveries:
        claimed.update(disc.get_children().keys())
    return {disc.limb_id for disc in discoveries} - claimed
