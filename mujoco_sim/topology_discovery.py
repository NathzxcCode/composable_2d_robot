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


# Weak regularisation prior on CJ — same convention as calibration_loop.py
_CALIB_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([10.0, 10.0, 0.001]))


class CandidateStatus(Enum):
    CANDIDATE = "candidate"
    ACTIVE    = "active"
    CONFIRMED = "confirmed"
    STALE     = "stale"


@dataclass
class Observation:
    J_a:     gtsam.Pose2
    J_b:     gtsam.Pose2
    theta_b: float          # encoder angle of candidate child limb


@dataclass
class CandidatePair:
    child_id:            int
    parent_id:           int
    status:              CandidateStatus   = CandidateStatus.CANDIDATE
    co_occurrence_count: int               = 0
    absence_count:       int               = 0
    # Sliding window of observations — oldest dropped automatically at T_max
    observations:        deque             = field(default_factory=deque)
    cj_estimate:         Optional[gtsam.Pose2]  = None   # warm start
    cj_cov:              Optional[np.ndarray]   = None
    last_cost:           float             = float('inf')
    last_cov_trace:      float             = float('inf')


class TopologyDiscovery:
    """
    Per-limb topology discovery via factor graph structure learning.

    For each candidate child limb b, maintains an isolated GBP subgraph with a
    single calibration variable CJ(a→b). GPS-observed joint poses are added as
    PriorFactorPose2 variables (tightness controlled by sigma_gps) connected via
    make_calib_kinematics_factor. A truly connected pair converges to a stable,
    low-residual CJ; an unconnected pair cannot find a time-invariant CJ.

    The subgraph is cycle-free (star topology rooted at CJ with per-timestep
    leaf branches), so GBP converges exactly in ~n_inner iterations.

    GPS source: joint pivot position from MuJoCo xanchor + body rotation matrix.
    Simplification: sensor-to-joint conversion is skipped; xanchor is used
    directly as the joint pose. Replace with S.compose(Pose2(-sx,-sy,0)) when
    physical sensor offsets are introduced.
    """

    def __init__(
        self,
        limb_id:          int,
        limbs,
        sigma_gps_pos:    float = 0.001,   # GPS position noise (m) — drives prior AND search radius
        sigma_gps_theta:  float = 0.005,   # GPS orientation noise (rad)
        n_sigma_search:   float = 5.0,     # search_radius = n_sigma * sigma_gps_pos
        K:                int   = 5,       # co-occurrence count to become ACTIVE
        TTL_max:          int   = 15,      # absence steps before removal
        T_max:            int   = 15,      # max observations in sliding window
        cost_thr:         float = 1.0,     # dual discriminator cost threshold
        cov_thr:          float = 0.01,    # dual discriminator CJ XY covariance trace threshold
        n_outer:          int   = 3,
        n_inner:          int   = 5,
        stale_multiplier: float = 5.0,
    ):
        self.limb_id         = limb_id
        self.limbs           = limbs
        self.L_a             = limbs[limb_id].length

        self.sigma_gps_pos   = sigma_gps_pos
        self.sigma_gps_theta = sigma_gps_theta
        self.search_radius   = n_sigma_search * sigma_gps_pos

        self.K                = K
        self.TTL_max          = TTL_max
        self.T_max            = T_max
        self.cost_thr         = cost_thr
        self.cov_thr          = cov_thr
        self.n_outer          = n_outer
        self.n_inner          = n_inner
        self.stale_multiplier = stale_multiplier

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

        # State machine updates
        for cid in candidates_this_step:
            if cid not in self._persistence:
                self._persistence[cid] = CandidatePair(
                    child_id=cid, parent_id=self.limb_id,
                    observations=deque(maxlen=self.T_max),
                    co_occurrence_count=1,
                )
            else:
                pair = self._persistence[cid]
                pair.co_occurrence_count += 1
                pair.absence_count = 0
                if pair.status == CandidateStatus.CANDIDATE and pair.co_occurrence_count >= self.K:
                    pair.status = CandidateStatus.ACTIVE

        for cid in list(self._persistence):
            if cid not in candidates_this_step:
                self._persistence[cid].absence_count += 1
                if self._persistence[cid].absence_count > self.TTL_max:
                    del self._persistence[cid]
                    self._confirmed_children.pop(cid, None)

        # Accumulate observations and optimise for active/confirmed pairs
        for cid, pair in list(self._persistence.items()):
            if pair.status not in (CandidateStatus.ACTIVE, CandidateStatus.CONFIRMED):
                continue
            if cid not in gps_joint_poses:
                continue

            pair.observations.append(Observation(
                J_a=J_a,
                J_b=gps_joint_poses[cid],
                theta_b=encoder_angles.get(cid, 0.0),
            ))

            cost, cov_trace, cj_est, cj_cov = self._optimise_subgraph(pair)
            pair.last_cost      = cost
            pair.last_cov_trace = cov_trace
            pair.cj_estimate    = cj_est
            pair.cj_cov         = cj_cov

            if pair.status == CandidateStatus.ACTIVE:
                if cost < self.cost_thr and cov_trace < self.cov_thr:
                    pair.status = CandidateStatus.CONFIRMED
                    self._confirmed_children[cid] = (cj_est, cj_cov)

            elif pair.status == CandidateStatus.CONFIRMED:
                if cost > self.cost_thr * self.stale_multiplier:
                    pair.status = CandidateStatus.STALE
                    self._confirmed_children.pop(cid, None)
                else:
                    self._confirmed_children[cid] = (cj_est, cj_cov)

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
        result = GBPOptimizer(graph, values, params).optimize()
        cost   = graph.error(result)
        cj_est = result.atPose2(cj_key)

        try:
            marginals = gtsam.Marginals(graph, result)
            cj_cov    = marginals.marginalCovariance(cj_key)
            cov_trace = float(np.trace(cj_cov[:2, :2]))
        except Exception:
            cj_cov    = np.eye(3) * 1000.0
            cov_trace = float('inf')

        return cost, cov_trace, cj_est, cj_cov

    def get_children(self) -> dict:
        """Returns {child_id: (CJ_pose2, cov_3x3)} for all confirmed pairs."""
        return dict(self._confirmed_children)

    def get_persistence_state(self) -> dict:
        """Returns {child_id: CandidateStatus} for all tracked pairs."""
        return {cid: pair.status for cid, pair in self._persistence.items()}

    def get_diagnostics(self) -> dict:
        """
        Returns cost and cov_trace for active/confirmed pairs.
        Use this output to empirically tune cost_thr and cov_thr.
        """
        return {
            cid: {
                "cost":        pair.last_cost,
                "cov_trace":   pair.last_cov_trace,
                "status":      pair.status.value,
                "n_obs":       len(pair.observations),
                "cj_estimate": pair.cj_estimate
            }
            for cid, pair in self._persistence.items()
            if pair.status in (CandidateStatus.ACTIVE, CandidateStatus.CONFIRMED)
        }


def identify_root(discoveries: list) -> set:
    """Returns limb IDs not claimed as a confirmed child by any other limb."""
    claimed = set()
    for disc in discoveries:
        claimed.update(disc.get_children().keys())
    return {disc.limb_id for disc in discoveries} - claimed
