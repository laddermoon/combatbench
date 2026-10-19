"""Launch-pose gate for strikes — skill_gate_mbs.

STRIKE_LAUNCH_GATE.md.  Under warmup-generalized starts the monotone
progress pointer can fast-forward past the clip's wind-up — the robot
never has to demonstrate it can return to the ready pose, so strikes
are not re-entrant.  This experiment injects ``launch_gate`` into
every episode: a strike segment only begins progress tracking after
the live state scores >= θ vs the clip's launch window [0,4].
Until then each step's r_strike reward IS the alignment score —
dense shaping toward the ready pose (in_strike stays 1).

θ=1.65 measured on trans u10135 exports: clean stance peaks ~1.80,
warmup-after-move ~1.69, warmup-after-turn ~1.63 (rejected).

SkillTransMBS parent keeps warmup specs; turn/step/move/strike
channels untouched otherwise.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from baseline.framework.rollout import extract_per_step_field

from .exp_skill_train_mbs import SKILLS
from .exp_skill_trans_mbs import SkillTransMBS


class SkillGateMBS(SkillTransMBS):
    """Transition training + strict launch-pose gate on strikes."""

    name = "skill_gate_mbs"

    # Diagnostic-measured threshold (see STRIKE_LAUNCH_GATE.md §2):
    # clean stance ≈1.80, mid-motion starts ≈1.6 → θ=1.65 strict.
    GATE_SPEC: Dict[str, float] = {"score": 1.65, "frames": 4}

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        jobs = super().build_jobs(policy_bp, base_seed, n_episodes,
                                  update=update, stochastic=stochastic)
        for j in jobs:
            j.episode_options["launch_gate"] = dict(self.GATE_SPEC)
        return jobs

    # Eval: parent metrics + gate diagnostics per strike skill.
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        per_skill: Dict[str, Dict[str, List[float]]] = {}
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            skills = (ep.episode_options or {}).get("skills") or {}
            for aid, _, _ in self._AGENT_OBS:
                skill = str(skills.get(aid, ""))
                if SKILLS.get(skill, {}).get("fam") != "strike":
                    continue
                sk = self._STRIKE_OBS[aid]
                T = ep.num_frames
                g = extract_per_step_field(
                    ep.observer_outputs, sk, "gated", T)
                gt = extract_per_step_field(
                    ep.observer_outputs, sk, "gate_t", T)
                m = per_skill.setdefault(
                    skill, {"gate_t": [], "ungated": []})
                if g is None:
                    continue
                gated = bool(np.asarray(g).max() > 0.5)
                m["ungated"].append(float(not gated))
                if gated and gt is not None:
                    vals = np.asarray(gt)
                    vals = vals[vals >= 0]
                    if vals.size:
                        m["gate_t"].append(float(vals.min()))
        for skill, m in per_skill.items():
            for k, v in m.items():
                res["info"][f"{skill}_{k}"] = round(
                    float(np.mean(v)), 3) if v else 0.0
        return res


EXPERIMENT_CLASS = SkillGateMBS
