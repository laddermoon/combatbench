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
_HORIZON_ACTIONS = 16        # actions per rollout (0.8 s — slow
                             # crouch-collapse becomes visible INSIDE
                             # the window, not after it)
_CHUNK = 2                   # actions per sampled chunk -> 4 DoF*21
_CEM_ITERS = 4
_CEM_POP = 32
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
        self._ref_cache = {}                # (clip_id, frame) -> action
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
        err += 10.0 * float(np.sum((com_f - target.com) ** 2))
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
        b += 9.0 * tilt * tilt
        # hard gate: beyond ~25deg the margin for error collapses —
        # candidates must pay heavily to pass through these states
        b += 40.0 * max(0.0, tilt - 0.45) ** 2
        z = float(o[45])
        # root-height floor: crouching is "free" in tilt terms — make it
        # expensive or the score trades height for stability every step
        b += 25.0 * max(0.0, 0.84 - z) ** 2
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
        # same advance rule the rollout's ref walk uses (reach OR dwell)
        # — scoring must evaluate the candidate against the schedule it
        # was actually told to track, else ref and objective disagree.
        dwell = 0
        dwell_budget = 6
        from .com_solver import com_estimate
        for o in obs_seq:
            lm_w, _, _ = self._fk_decode(wsim, o)
            cost += self._balance_terms(o, lm_w, origin0, yaw0)
            # anti-crouch gate: sinking under the current anchor's com_z
            # is cheap in tilt terms — price it directly, one-sided.
            com_t = com_estimate(lm_w)
            cost += 12.0 * max(0.0, lms[tgt_i].com[2] - com_t[2]) ** 2
            e = self._landmark_error(lm_w, origin0, yaw0,
                                     lms[tgt_i], obs=o)
            dwell += 1
            if tgt_i < len(lms) - 1:
                if e < reach_eps or dwell >= dwell_budget:
                    tgt_i += 1
                    dwell = 0
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

    def _rollout_feedback(self, wsim, cand, a_refs, origin0, yaw0,
                          lms, n_steps, reach_eps=0.06):
        """Closed-loop INSIDE the model.  Candidate = [k(5), d(5)]:
        u = clip(a_ref + k_g*(a_ref - q) + d_g).

        The reference follows the anchor SEQUENCE: the same reach_eps
        walk used for scoring decides which anchor pose the controller
        is tracking.  A liftoff->touchdown pair becomes a real step —
        swing first, then land — which a single static ref can never
        express.
        Returns per-step obs AND the first action (to execute)."""
        obs_seq, u0 = [], None
        kg = np.empty(21)
        dg = np.empty(21)
        for g, (lo, hi) in enumerate(self._GROUPS):
            kg[lo:hi] = cand[g]
            dg[lo:hi] = cand[len(self._GROUPS) + g]
        tgt_i = 1
        # time-budgeted ref advance: a hard anchor (e.g. full unload)
        # may never satisfy reach_eps — without a dwell fallback the ref
        # is stuck on it forever and the swing/land poses are never
        # issued.  Each unfinished anchor gets at most ~H/remaining
        # steps before the next one is tried.
        dwell = 0
        # 0.3s per anchor: enough time for the tracker to actually move
        # (3-step budgets thrash — ref switches before the body reacts)
        budget = 6
        for t in range(n_steps):
            o = wsim.get_derived_state()[self.robot_id]["observation"]
            q = np.asarray(o[0:21], np.float64)
            a_ref = a_refs[min(tgt_i, len(a_refs) - 1)]
            u = np.clip(a_ref + kg * (a_ref - q) + dg, -1.0, 1.0)
            if u0 is None:
                u0 = u.copy()
            wsim.set_action({self.robot_id: u.astype(np.float32),
                             "robot_b": self._zero_b})
            for _ in range(_CTRL_SUBSTEPS):
                wsim.physical_step()
            o2 = wsim.get_derived_state()[self.robot_id]["observation"]
            obs_seq.append(o2.copy())
            # absolute viability ceiling: past ~35deg the state is
            # unrecoverable — reject the whole candidate regardless of
            # how close it got to the anchor
            g2 = np.asarray(o2[42:45])
            if float(np.arccos(np.clip(-g2[2]
                     / (np.linalg.norm(g2) + 1e-9), -1, 1))) > 0.61:
                return obs_seq, u0, True
            # advance the reference anchor when the current one is met
            # OR its dwell budget is spent (fallback keeps the
            # swing->land sequence flowing)
            dwell += 1
            if tgt_i < len(lms) - 1:
                lm_w, _, _ = self._fk_decode(wsim, o2)
                e = self._landmark_error(lm_w, origin0, yaw0,
                                         lms[tgt_i], obs=o2)
                if e < reach_eps or dwell >= budget:
                    tgt_i += 1
                    dwell = 0
        return obs_seq, u0, False

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

        # candidate = [k(5), d(5)]: tracking gains + residual offsets
        # over 5 joint groups, rolled out closed-loop inside wsim.
        # references: one solved pose action per anchor, walked in order.
        NG = len(self._GROUPS)
        a_refs = [obs[0:21].copy()]          # index 0 = start (unused)
        for lm in landmarks[1:]:
            a = self._anchor_ref_action(ctx, lm, obs)
            a_refs.append(a if a is not None else obs[0:21].copy())
        mean = np.concatenate([np.ones(NG), np.zeros(NG)])
        std = np.concatenate([np.full(NG, 0.7), np.full(NG, 0.3)])
        inj = [np.concatenate([k, np.zeros(NG)])
               for k in (np.zeros(NG), np.ones(NG), np.full(NG, 2.0))]
        best_c, best_cost = np.concatenate([np.ones(NG),
                                            np.zeros(NG)]), np.inf

        # deterministic RNG per call -> reproducible choice for a given
        # context (Executor stays stateless; callers may vary the seed
        # via ctx.meta["seed"] if they want diversity).
        rng = np.random.default_rng(int(ctx.meta.get("seed", 0)))
        for _ in range(_CEM_ITERS):
            pop = np.clip(mean + std * rng.standard_normal(
                (_CEM_POP, 2 * NG)), -1.0, 3.0)
            pop = np.vstack([pop, mean[None, :], np.asarray(inj)])
            costs = np.empty(len(pop))
            for i, cand in enumerate(pop):
                self._sim_restore(wsim, snap)
                seq, _, aborted = self._rollout_feedback(
                    wsim, cand, a_refs, origin0, yaw0, landmarks,
                    _HORIZON_ACTIONS)
                costs[i] = (np.inf if aborted else self._seq_score(
                    wsim, seq, origin0, yaw0, landmarks))
                if costs[i] < best_cost:
                    best_cost = costs[i]
                    best_c = cand.copy()
            elite = pop[np.argsort(costs)[:_CEM_ELITE]]
            mean, std = elite.mean(0), elite.std(0) + 0.05
        self._sim_restore(wsim, snap)

        kg = np.empty(21)
        dg = np.empty(21)
        for g, (lo, hi) in enumerate(self._GROUPS):
            kg[lo:hi] = best_c[g]
            dg[lo:hi] = best_c[NG + g]
        a_ref0 = a_refs[min(1, len(a_refs) - 1)]
        u = np.clip(a_ref0 + kg * (a_ref0 - obs[0:21]) + dg, -1.0, 1.0)
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
            # cache by pose content — the clip object is rebuilt every
            # control step so object identity is useless as a key.
            key = np.asarray(kfs[fr].pos, np.float64).tobytes()
            if key not in self._ref_cache:
                res = self._pose_solver.solve(
                    np.asarray(kfs[fr].pos, np.float64))
                self._ref_cache[key] = np.asarray(res.action, np.float64)
            return self._ref_cache[key]
        except Exception:
            return None

    # -- wsim snapshot helpers ----------------------------------------------
    def _sim_snapshot(self, wsim):
        return {rid: {k: v.copy() for k, v in
                      wsim.get_core_state()[rid].items()}
                for rid in ("robot_a", "robot_b")}

    def _sim_restore(self, wsim, snap):
        wsim.set_core_state(snap)
