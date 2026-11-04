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
        self._pose_solver = None
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
        return err

    def _balance_terms(self, o, lm_world, origin0, yaw0):
        """Per-step stability penalty — summed over the horizon, so a
        candidate that collapses late still pays.  Capture-point exit
        is the predictive term: penalize BEFORE the fall is visible.
        """
        from .com_solver import com_estimate
        from .capability_solver import _hull2d
        from .support_planner import _inside
        b = 0.0
        g = np.asarray(o[42:45], np.float64)
        tilt = float(np.arccos(np.clip(-g[2]
                      / (np.linalg.norm(g) + 1e-9), -1, 1)))
        b += 6.0 * tilt * tilt
        z = float(o[45])
        b += 4.0 * max(0.0, 0.80 - z) ** 2        # root-height floor
        vz = float(o[48])
        b += 0.4 * max(0.0, -vz) ** 2              # sinking velocity
        # capture point xi = c + v/omega in frame0, outside support hull
        com_t = com_estimate(lm_world)
        com_f = np.concatenate(
            [self._to_frame0(com_t[:2], origin0, yaw0), com_t[2:3]])
        omega = np.sqrt(9.81 / max(com_f[2], 0.3))
        cp = com_f[:2] + np.asarray(o[46:48], np.float64) / omega
        sup = [self._to_frame0(lm_world[fi, :2], origin0, yaw0)
               for fi, k in ((3, 0), (4, 1)) if o[52 + k] > 0.05]
        if not sup:                               # airborne: no support
            b += 4.0 * float(np.sum((cp - com_f[:2]) ** 2))
        else:
            hull = _hull2d(np.asarray(sup))
            if not _inside(cp, hull):
                d = min(float(np.linalg.norm(cp - p)) for p in sup)
                b += 3.0 * d * d
        return b

    def _seq_score(self, wsim, obs_seq, origin0, yaw0, lms,
                   chunks=None, a_prev=None, reach_eps=0.06):
        """Per-candidate cost = accumulated balance + landmark progress
        + action smoothness.  No min() over time — instability is
        cumulative, not best-moment."""
        cost = 0.0
        tgt_i = 1
        for o in obs_seq:
            lm_w, _, _ = self._fk_decode(wsim, o)
            cost += self._balance_terms(o, lm_w, origin0, yaw0)
            e = self._landmark_error(lm_w, origin0, yaw0,
                                     lms[tgt_i], obs=o)
            if e < reach_eps and tgt_i < len(lms) - 1:
                tgt_i += 1
        end_lm = self._fk_decode(wsim, obs_seq[-1])[0]
        end_err = self._landmark_error(
            end_lm, origin0, yaw0, lms[tgt_i], obs=obs_seq[-1])
        cost += end_err - tgt_i * reach_eps
        if chunks is not None:
            for a, b_ in zip(chunks[:-1], chunks[1:]):
                cost += 0.05 * float(np.sum((a - b_) ** 2))
        if a_prev is not None:
            cost += 0.02 * float(np.sum((chunks[0] - a_prev) ** 2))
        return float(cost)

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

    _GROUPS = ((0, 3), (3, 9), (9, 15), (15, 18), (18, 21))

    def _rollout_feedback(self, wsim, k, a_ref, n_steps):
        """Closed-loop INSIDE the model: each step the action is
        u = clip(a_ref + k_group*(a_ref - q)) — the candidate is a
        tracking controller, not an open-loop tape.  Recovery
        directions (push back toward the ref when deviated) live in
        this space; raw action tapes almost never contain them.
        Returns per-step obs AND the first action (to execute)."""
        obs_seq, u0 = [], None
        kg = np.empty(21)
        for g, (lo, hi) in enumerate(self._GROUPS):
            kg[lo:hi] = k[g]
        for t in range(n_steps):
            o = wsim.get_derived_state()[self.robot_id]["observation"]
            q = np.asarray(o[0:21], np.float64)
            u = np.clip(a_ref + kg * (a_ref - q), -1.0, 1.0)
            if u0 is None:
                u0 = u.copy()
            wsim.set_action({self.robot_id: u.astype(np.float32),
                             "robot_b": self._zero_b})
            for _ in range(_CTRL_SUBSTEPS):
                wsim.physical_step()
            obs_seq.append(
                wsim.get_derived_state()[self.robot_id]
                ["observation"].copy())
        return obs_seq, u0

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

        # candidate = tracking-controller gains over 5 joint groups,
        # rolled out closed-loop inside wsim around the anchor-pose
        # reference.  Dim=5, every candidate inherently reactive.
        a_ref = self._anchor_ref_action(ctx, landmarks[1], obs)
        if a_ref is None:
            a_ref = obs[0:21].copy()
        NG = len(self._GROUPS)
        mean = np.ones(NG)
        std = np.full(NG, 0.7)
        inject = [np.zeros(NG), np.ones(NG), np.full(NG, 2.0)]
        best_k, best_cost = np.ones(NG), np.inf

        for _ in range(_CEM_ITERS):
            pop = np.clip(mean + std * np.random.randn(_CEM_POP, NG),
                          -0.5, 3.0)
            pop = np.vstack([pop, mean[None, :], np.asarray(inject)])
            costs = np.empty(len(pop))
            for i, cand in enumerate(pop):
                self._sim_restore(wsim, snap)
                seq, _ = self._rollout_feedback(
                    wsim, cand, a_ref, _HORIZON_ACTIONS)
                costs[i] = self._seq_score(
                    wsim, seq, origin0, yaw0, landmarks)
                if costs[i] < best_cost:
                    best_cost = costs[i]
                    best_k = cand.copy()
            elite = pop[np.argsort(costs)[:_CEM_ELITE]]
            mean, std = elite.mean(0), elite.std(0) + 0.05
        self._sim_restore(wsim, snap)

        kg = np.empty(21)
        for g, (lo, hi) in enumerate(self._GROUPS):
            kg[lo:hi] = best_k[g]
        u = np.clip(a_ref + kg * (a_ref - obs[0:21]), -1.0, 1.0)
        return u.astype(np.float32)

    def _anchor_ref_action(self, ctx, landmark, obs):
        """Pose-solve the anchor's source clip keyframe -> reference
        action.  Falls back to None (hold-pose prior) when the anchor
        has no usable frame link or the solve fails."""
        try:
            fr = getattr(landmark, "frame", None)
            if fr is None or ctx.clip is None:
                return None
            kfs = ctx.clip.keyframes
            if not (0 <= fr < len(kfs)):
                return None
            if self._pose_solver is None:
                from .pose_solver import Humanoid21PoseSolver
                self._pose_solver = Humanoid21PoseSolver()
            res = self._pose_solver.solve(np.asarray(kfs[fr].pos,
                                                     np.float64))
            return np.asarray(res.action, np.float64)
        except Exception:
            return None

    # -- wsim snapshot helpers ----------------------------------------------
    def _sim_snapshot(self, wsim):
        return {rid: {k: v.copy() for k, v in
                      wsim.get_core_state()[rid].items()}
                for rid in ("robot_a", "robot_b")}

    def _sim_restore(self, wsim, snap):
        wsim.set_core_state(snap)
