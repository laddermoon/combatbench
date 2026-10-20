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

    # Launch gate (P2-1 裁决): 只匹配 clip 第 0 帧；判据改为纯姿态—
    # —所有受控关节(踝关节除外)与参考帧的误差 < 各自全量程的 5%。
    # 复合分保留为过门前的稠密 shaping，不再作门槛（权重和会掩盖
    # 关键关节偏差）。
    GATE_SPEC: Dict[str, float] = {"frames": 0, "frac": 0.05}

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
                gq = extract_per_step_field(
                    ep.observer_outputs, sk, "gate_qmax", T)
                m = per_skill.setdefault(
                    skill, {"gate_t": [], "ungated": [],
                            "gate_qmax": []})
                if g is None:
                    continue
                gated = bool(np.asarray(g).max() > 0.5)
                m["ungated"].append(float(not gated))
                if gq is not None:
                    vals = np.asarray(gq)
                    vals = vals[vals > 0]     # 0 = 非门控期帧
                    if vals.size:
                        # 最接近过门的一次：max 关节误差的 episode 最小值
                        m["gate_qmax"].append(float(vals.min()))
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
