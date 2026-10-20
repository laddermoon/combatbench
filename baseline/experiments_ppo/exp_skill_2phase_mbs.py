"""Two-phase strike reward split — skill_2phase_mbs.

SKILL_TRAINING_PLAN.md §7.  The strike segment is split at the
launch gate (boundary unchanged: every non-ankle controlled joint
within 5% of full range vs clip frame 0).  Each phase gets its own
potential-form channels, actor-weight-masked to its own steps via
the observer's per-step `gated` flag — the phases learn their
objectives separately:

  Phase 1 (approach → ready pose @ frame 0):
    r_strike_p1_mse  −MSE of non-ankle joint errors vs frame 0
    r_strike_p1_var  −variance of the per-joint squared errors
                     (penalizes one large outlier joint the mean
                     would dilute)
  Phase 2 (ready → punch through, target = landing frame k_max):
    r_strike_p2_mse / r_strike_p2_var — same potentials vs k_max
    r_strike_p2_foot — −displacement of the rear (power) foot from
                     its world anchor recorded at gate pass; the
                     power foot must not move while it drives the
                     punch

The single ``r_strike`` channel is replaced by the five above.
Everything else (warmup-generalized starts, stride-settled move
rewards, launch-gate verdict + frac curriculum, ext_only pointer
metrics) inherits SkillFullMBS unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from baseline.framework.rollout import extract_per_step_field

from .exp_skill_full_mbs import SkillFullMBS
from .exp_skill_train_mbs import SKILLS


class Skill2PhaseMBS(SkillFullMBS):
    """Phase-split strike channels on top of the full skill recipe."""

    name = "skill_2phase_mbs"

    TWO_PHASE_STRIKE: bool = True

    # Fixed 5% gate boundary — the phase transition is the strict
    # criterion from the start; the adaptive frac curriculum is
    # disabled by collapsing its range to the terminal value.
    _FRAC_LO = _FRAC_HI = 0.05

    _channel_names = tuple(
        c for c in SkillFullMBS._channel_names if c != "r_strike") + (
        "r_strike_p1_mse", "r_strike_p1_var",
        "r_strike_p2_mse", "r_strike_p2_var", "r_strike_p2_foot")
    _channel_gammas = {
        **SkillFullMBS._channel_gammas,
        **{k: 0.99 for k in _channel_names},
    }

    # Parent eval metrics + per-skill rear-foot displacement (max over
    # gated phase-2 steps) — monitors the power-foot constraint.
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        per_skill: Dict[str, List[float]] = {}
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            skills = (ep.episode_options or {}).get("skills") or {}
            for aid, _, _ in self._AGENT_OBS:
                skill = str(skills.get(aid, ""))
                if SKILLS.get(skill, {}).get("fam") != "strike":
                    continue
                sk = self._STRIKE_OBS[aid]
                fd = extract_per_step_field(
                    ep.observer_outputs, sk, "foot_disp_m",
                    ep.num_frames)
                if fd is None:
                    continue
                vals = np.asarray(fd)
                vals = vals[vals > 0]
                if vals.size:
                    per_skill.setdefault(skill, []).append(
                        float(vals.max()))
        for skill, v in per_skill.items():
            res["info"][f"{skill}_foot_disp"] = round(
                float(np.mean(v)), 3)
        return res


EXPERIMENT_CLASS = Skill2PhaseMBS
