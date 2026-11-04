"""Algorithm 2: Min-cost action search toward the LandMark sequence.

Formulation (rolling-horizon, executed once per call):

    min_{U, tau}   sum_i D_LM(x_tau_i, M_i)
                 + l_P5 * J_P5 + l_S * sum||u_t - u_{t-1}||^2
                 + l_T * tau_K
    s.t. x0 = x_hat(Observation),  x_{t+1} = F_sim(x_t, u_t),
         u_t in [-1,1]^21,  0 < tau_1 < ... < tau_K <= H.

Executed: u0 only.  Next call re-anchors and re-solves.

v1 implementation: piecewise-constant action chunks + CEM over a
short horizon, scored by LandMark distance in the frame0 local
frame.  Arrival times are NOT optimized explicitly yet — the score
is the best distance achieved anywhere inside the window
(tau is implicit = argmin_t).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .centroidal_planner import _level_rot, _quat_wxyz
from .contracts import Context, PlanError

_CTRL_SUBSTEPS = 25          # phys steps per action (50 ms @ 2 kHz)
_HORIZON_ACTIONS = 8         # actions per rollout (0.4 s)
_CHUNK = 2                   # actions per sampled chunk -> 4 DoF*21
_CEM_ITERS = 4
_CEM_POP = 24
_CEM_ELITE = 6


def obs_to_core_state(obs: np.ndarray) -> dict:
    """96-dim policy obs -> set_core_state payload for robot_a.

    x_hat reconstruction: root x/y are NOT in the obs -> set to 0
    (frame0-local planning doesn't need the absolute offset);
    root yaw is not observable -> minimal tilt rotation only.
    """
    obs = np.asarray(obs, np.float64)
    jv = obs[21:42]
    w = obs[49:52]
    return {
        "root_pos": np.array([0.0, 0.0, obs[45]]),
        "root_rot": _quat_wxyz(_level_rot(obs[42:45])),
        "root_vel_local": obs[46:49].copy(),
        "root_angular_vel_local": np.sign(w) * 2.0 * w * w,
        "joint_pos_norm": obs[0:21].copy(),
        "joint_vel_norm": np.sign(jv) * (2.0 * jv) ** 2,
    }


class ActionComputer:
    """Algorithm 2 — stateless per-call trajectory optimization."""

    def __init__(self, robot_id: str = "robot_a"):
        self.robot_id = robot_id
        self._decoder = None
        self._zero_b = np.zeros(21, np.float32)

    # -- decoding -----------------------------------------------------------
    def _fk_decode(self, wsim, obs):
        """obs -> (world landmarks (5,3), frame origin xy, torso yaw)."""
        if self._decoder is None:
            from .centroidal_planner import ObsDecoder
            self._decoder = ObsDecoder(self.robot_id)
        # reuse the FK decode but keep world landmarks this time
        dec = self._decoder
        from point5.contracts import ankle_mid_xy
        fk, rid = dec.fk(), self.robot_id
        rng = fk.joint_ranges(rid).astype(np.float64)
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        R_tilt = _level_rot(obs[42:45])
        fk.set_pose(rid, np.array([0.0, 0.0, obs[45]]),
                    _quat_wxyz(R_tilt), obs[0:21] * sc + ref)
        lm = fk.landmark_positions(rid)
        torso = next(v for k, v in fk.body_pose(rid).items()
                     if "torso" in k)
        yaw = float(np.arctan2(torso[1][1, 0], torso[1][0, 0]))
        return lm, ankle_mid_xy(lm), yaw

    # -- landmark error -----------------------------------------------------
    def _to_frame0(self, pts_xy, origin, yaw0):
        c, s = np.cos(yaw0), np.sin(yaw0)
        R = np.array([[c, s], [-s, c]])
        return (np.asarray(pts_xy) - origin) @ R.T

    def _landmark_error(self, lm_world, origin, yaw0, target,
                        vel_w=None, obs=None) -> float:
        """D_LM between decoded rollout state and a LandMark."""
        from .com_solver import com_estimate
        err = 0.0
        com_t = com_estimate(lm_world)
        com_f = np.concatenate([self._to_frame0(com_t[:2], origin,
                                              yaw0), com_t[2:3]])
        err += 6.0 * float(np.sum((com_f - target.com) ** 2))
        if target.vel is not None and vel_w is not None:
            err += 2.0 * float(np.sum((vel_w - target.vel) ** 2))
        for side, fi in (("right", 3), ("left", 4)):
            f = target.feet[side]
            if f.foot_pos is not None:
                d = (self._to_frame0(lm_world[fi, :2], origin, yaw0)
                     - f.foot_pos)
                err += 4.0 * float(np.sum(d * d))
            # foot LOAD matching — obs[52:54] carries measured
            # force ratios; a liftoff anchor is mostly about this
            if obs is not None:
                actual = (obs[52] if side == "right" else obs[53])
                err += 3.0 * float((actual - f.force_ratio) ** 2)
        # upright/balance prior — penalize sinking + tilt away
        # from vertical (prevents CEM drifting into collapse)
        if obs is not None:
            g = np.asarray(obs[42:45])
            tilt = float(np.arccos(np.clip(-g[2]
                            / (np.linalg.norm(g) + 1e-9), -1, 1)))
            err += 2.0 * tilt * tilt
            err += 1.5 * float(max(0.0, 0.87 - obs[45]) ** 2)
        return err

    def _seq_score(self, wsim, obs_seq, origin0, yaw0, lms,
                   reach_eps=0.06):
        """Walk the rollout obs seq against the anchor list.

        Progress model: advance to the next anchor whenever the
        current state's error drops below reach_eps (implicit tau);
        residual cost = unfinished anchors' best distances +
        current-anchor residual, earlier anchors weighted higher.
        """
        from .com_solver import com_estimate
        tgt_i, best = 1, {}
        errs = []
        for o in obs_seq:
            lm_w, _, _ = self._fk_decode(wsim, o)
            e = self._landmark_error(lm_w, origin0, yaw0,
                                     lms[tgt_i], obs=o)
            errs.append(e)
            if e < reach_eps and tgt_i < len(lms) - 1:
                tgt_i += 1
        # total = last-observed err per unreached anchor + progress
        tail = sum(self._landmark_error(
                self._fk_decode(wsim, obs_seq[-1])[0],
                origin0, yaw0, lms[j], obs=obs_seq[-1])
            for j in range(tgt_i, min(len(lms), tgt_i + 2)))
        return float(min(errs) + tail - tgt_i * reach_eps)

    # -- rollout ------------------------------------------------------------
    def _rollout(self, wsim, chunks):
        """Apply piecewise-constant actions; return per-action obs."""
        obs_seq = []
        for a in chunks:
            wsim.set_action({self.robot_id: a.astype(np.float32),
                             "robot_b": self._zero_b})
            for _ in range(_CTRL_SUBSTEPS):
                wsim.physical_step()
            obs_seq.append(
                wsim.get_derived_state()[self.robot_id]
                ["observation"].copy())
        return obs_seq

    # -- solver -------------------------------------------------------------
    def compute(self, ctx: Context, landmarks: List, wsim) -> np.ndarray:
        """LandMark seq + obs -> next (21,) normalized action."""
        obs = np.asarray(ctx.observation, np.float64)
        if not landmarks or len(landmarks) < 2:
            raise PlanError("algorithm 2 needs >=2 landmarks")

        # --- inject x_hat(0) and record frame0 pose ------------------------
        wsim.set_core_state({self.robot_id: obs_to_core_state(obs)})
        lm0_w, origin0, yaw0 = self._fk_decode(wsim, obs)

        # --- snapshot to restore between rollouts --------------------------
        snap = self._sim_snapshot(wsim)

        n_chunks = _HORIZON_ACTIONS // _CHUNK
        dim = n_chunks * 21
        mean = np.zeros(dim)
        mean[:21] = obs[0:21]               # seed: hold current pose
        std = np.full(dim, 0.6)
        best_a, best_cost = obs[0:21].copy(), np.inf

        for _ in range(_CEM_ITERS):
            pop = np.clip(mean + std * np.random.randn(_CEM_POP, dim),
                          -1.0, 1.0)
            costs = np.empty(_CEM_POP)
            for i, cand in enumerate(pop):
                self._sim_restore(wsim, snap)
                seq = self._rollout(
                    wsim, cand.reshape(n_chunks, 21)
                    .repeat(_CHUNK, axis=0))
                costs[i] = self._seq_score(
                    wsim, seq, origin0, yaw0, landmarks)
                if costs[i] < best_cost:
                    best_cost = costs[i]
                    best_a = cand[:21].copy()
            elite = pop[np.argsort(costs)[:_CEM_ELITE]]
            mean, std = elite.mean(0), elite.std(0) + 0.05
        self._sim_restore(wsim, snap)
        return best_a.astype(np.float32)

    # -- wsim snapshot helpers ----------------------------------------------
    def _sim_snapshot(self, wsim):
        return {rid: {k: v.copy() for k, v in
                      wsim.get_core_state()[rid].items()}
                for rid in ("robot_a", "robot_b")}

    def _sim_restore(self, wsim, snap):
        wsim.set_core_state(snap)
