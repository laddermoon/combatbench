"""Stride-settled movement rewards — skill_step_mbs.

STRIDE_REWARD_PLAN.md.  Replaces per-frame smoothed-velocity
``r_move``/``r_drift`` with per-stride settlement for move episodes.
A stride = two consecutive foot-strikes (left+right step = one gait
cycle).  Foot-strike boundaries are detected offline from
``feet_forces`` (obs[:,52:54], body-weight normalized) — rising edges
of ground contact — no env/observer changes.

At every foot-strike (3rd onward) the window ``(s_{i-2}, s_i]`` is
scored once at the settlement frame t_end = s_i:

  v_rad  = (xy(t_end) − xy(t_a))·d_hat / Δt   radial velocity
  v_off  = |lateral component| / Δt            tangential drift
  |Δyaw| = |yaw(t_end) − yaw(t_a)|             net heading change

  r_move_step[t_end]  = +2·v_rad·Δt            (integral-equivalent)
  r_drift_step[t_end] = −(v_off·Δt + 0.5·|Δyaw|)

Sliding windows: every landing settles the most recent two steps —
feedback on every footfall, ~16× sparser than per-frame.  Fallbacks:
a virtual strike at T−1 closes the in-flight stride; <2 strikes → one
settlement over the whole episode at T−1 (shuffling earns the
displacement verdict but no stepping bonus).

Only the REWARD is single-point: actor_weight stays the parent's
dense φ² per-frame weight so GAE can backpropagate each settlement
through the whole gait cycle that produced it (weight shift, swing,
placement) — masking non-settlement frames would zero exactly the
credit the cycle deserves.

step/idle episodes reuse the same machinery: ``r_hold`` is dropped and
each stride settles ``r_drift_step[t_end] = −(|Δxy| + 0.5·|Δyaw|)`` —
with no command direction every displacement is drift.

turn/strike identical to the parent.  Trains from the S2 checkpoint
(same lineage as repro).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData

from .exp_skill_train_mbs import SKILLS, SkillTrainMBS


class SkillStepMBS(SkillTrainMBS):
    """Skill training with stride-settled move rewards."""

    name = "skill_step_mbs"

    _channel_names = (
        "r_potential", "r_fall", "r_move_step", "r_drift_step",
        "r_turn", "r_strike", "r_std_clip",
    )
    _channel_gammas = {k: 0.99 for k in _channel_names}

    # --- Stride detection / settlement constants ---
    _FEET = slice(52, 54)      # feet_forces index in obs (weight-normalized)
    _CONTACT_TAU = 0.10        # contact threshold (×body weight)
    _DYAW_W = 0.5              # heading-change penalty per rad / stride

    # ------------------------------------------------------------------
    # Stride segmentation (shared by trajectory builder and eval)
    # ------------------------------------------------------------------

    @classmethod
    def _strike_bounds(cls, obs: np.ndarray) -> List[int]:
        """Foot-strike frame indices (rising edge of either foot's
        contact), plus a virtual strike at the last frame so the
        in-flight stride gets settled."""
        ff = np.asarray(obs[:, cls._FEET], dtype=np.float64)
        contact = ff > cls._CONTACT_TAU           # (T,2)
        prev = np.zeros_like(contact)
        prev[1:] = contact[:-1]
        rise = contact & ~prev                    # rising edges
        bounds = np.flatnonzero(rise.any(axis=1)).tolist()
        T = obs.shape[0]
        if not bounds or bounds[-1] < T - 1:
            bounds.append(T - 1)
        return bounds

    @staticmethod
    def _stride_windows(bounds: List[int], T: int) -> List[tuple]:
        """Sliding two-strike windows (s_{i-2}, s_i]."""
        wins = [(bounds[i - 2], bounds[i]) for i in
                range(2, len(bounds))]
        if not wins and len(bounds) >= 2:
            wins = [(bounds[0], bounds[1])]
        if not wins:
            wins = [(0, T - 1)]
        return wins

    # ------------------------------------------------------------------
    # Trajectory: swap dense r_move/r_drift for stride-settled channels
    # ------------------------------------------------------------------

    def _build_agent_trajectory(self, episode, agent_id, foot_key,
                                phi_key):
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key)
        for tr in trajs:
            if "r_move" in tr.channels:
                self._settle_strides(episode, agent_id, tr)
            elif "r_hold" in tr.channels:
                self._settle_idle_strides(episode, agent_id, tr)
        return trajs

    def _settle_strides(self, episode, agent_id, tr) -> None:
        T = tr.obs.shape[0]
        kin = self._kin(episode, agent_id, T)
        move_aw = tr.channels.pop("r_move").actor_weight
        drift_aw = tr.channels.pop("r_drift").actor_weight
        r_move = np.zeros(T, dtype=np.float32)
        r_drift = np.zeros(T, dtype=np.float32)
        if kin is not None:
            xy, yaw, _ = kin
            yaw_u = np.unwrap(np.asarray(yaw[:T], dtype=np.float64))
            skills = (episode.episode_options or {}).get("skills") or {}
            de = np.asarray(SKILLS[str(skills.get(
                agent_id, "step"))]["dir_ego"])
            c, s = np.cos(yaw_u[0]), np.sin(yaw_u[0])
            d_hat = np.array([c * de[0] - s * de[1],
                              s * de[0] + c * de[1]])
            for t_a, t_b in self._stride_windows(
                    self._strike_bounds(tr.obs), T):
                dt = (t_b - t_a) / 20.0
                if dt <= 0:
                    continue
                dxy = xy[t_b] - xy[t_a]
                v_rad = float(dxy @ d_hat) / dt
                lat = dxy - (dxy @ d_hat) * d_hat
                v_off = float(np.linalg.norm(lat)) / dt
                dyaw = abs(float(yaw_u[t_b] - yaw_u[t_a]))
                r_move[t_b] += 2.0 * v_rad * dt
                r_drift[t_b] += -(v_off * dt + self._DYAW_W * dyaw)
        # Reward is settled at the window's closing foot-strike only,
        # but actor_weight stays dense (parent's 3·phi²): the whole
        # gait cycle — not just the landing frame — earned the verdict.
        tr.channels["r_move_step"] = ChannelData(
            reward=r_move, is_terminated=False,
            actor_weight=move_aw)
        tr.channels["r_drift_step"] = ChannelData(
            reward=r_drift, is_terminated=False,
            actor_weight=drift_aw)

    def _settle_idle_strides(self, episode, agent_id, tr) -> None:
        """step/idle: same stride settlement, but with no commanded
        direction ALL net displacement is drift — penalize |Δxy| plus
        heading change per stride, no positive term."""
        T = tr.obs.shape[0]
        kin = self._kin(episode, agent_id, T)
        hold_aw = tr.channels.pop("r_hold").actor_weight
        r_drift = np.zeros(T, dtype=np.float32)
        if kin is not None:
            xy, yaw, _ = kin
            yaw_u = np.unwrap(np.asarray(yaw[:T], dtype=np.float64))
            for t_a, t_b in self._stride_windows(
                    self._strike_bounds(tr.obs), T):
                if t_b <= t_a:
                    continue
                d_tot = float(np.linalg.norm(xy[t_b] - xy[t_a]))
                dyaw = abs(float(yaw_u[t_b] - yaw_u[t_a]))
                r_drift[t_b] += -(d_tot + self._DYAW_W * dyaw)
        tr.channels["r_drift_step"] = ChannelData(
            reward=r_drift, is_terminated=False,
            actor_weight=hold_aw)

    # ------------------------------------------------------------------
    # Eval: parent's metrics + per-stride stats for move skills
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        per_skill: Dict[str, Dict[str, List[float]]] = {}
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            skills = (ep.episode_options or {}).get("skills") or {}
            for aid, _, _ in self._AGENT_OBS:
                skill = str(skills.get(aid, ""))
                fam = SKILLS.get(skill, {}).get("fam")
                if fam not in ("move", "idle"):
                    continue
                obs = ep.observations.get(aid)
                kin = self._kin(ep, aid, ep.num_frames)
                if obs is None or kin is None:
                    continue
                obs = np.asarray(obs, dtype=np.float32)
                xy, yaw, _ = kin
                yaw_u = np.unwrap(np.asarray(yaw, dtype=np.float64))
                m = per_skill.setdefault(
                    skill, {"strides": [], "svrad": [], "svoff": [],
                            "sdyaw": [], "nostep": [], "svtot": []})
                wins = self._stride_windows(
                    self._strike_bounds(obs), ep.num_frames)
                m["strides"].append(float(len(wins)))
                m["nostep"].append(float(len(wins) <= 1))
                d_hat = None
                if fam == "move":
                    de = np.asarray(SKILLS[skill]["dir_ego"])
                    c, s = np.cos(yaw_u[0]), np.sin(yaw_u[0])
                    d_hat = np.array([c * de[0] - s * de[1],
                                      s * de[0] + c * de[1]])
                for t_a, t_b in wins:
                    dt = (t_b - t_a) / 20.0
                    if dt <= 0:
                        continue
                    dxy = xy[t_b] - xy[t_a]
                    m["svtot"].append(float(np.linalg.norm(dxy)) / dt)
                    m["sdyaw"].append(abs(float(
                        yaw_u[t_b] - yaw_u[t_a])))
                    if fam == "move":
                        m["svrad"].append(float(dxy @ d_hat) / dt)
                        m["svoff"].append(float(np.linalg.norm(
                            dxy - (dxy @ d_hat) * d_hat)) / dt)
        for skill, m in per_skill.items():
            for k, v in m.items():
                res["info"][f"{skill}_{k}"] = round(
                    float(np.mean(v)), 3) if v else 0.0
        return res


EXPERIMENT_CLASS = SkillStepMBS
