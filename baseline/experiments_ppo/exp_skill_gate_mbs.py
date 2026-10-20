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
    # 复合分整体弃用于段 A（权重和会掩盖关键关节偏差），过门前
    # shaping = 逐关节 Lorentzian 均值 − qmax（见 observer 注释）。
    GATE_SPEC: Dict[str, float] = {"frames": 0, "frac": 0.05}

    # frac 自适应课程（奖励沙漠破解）：严格 5% 判据下 qmax 距阈
    # 0.3-0.6，post-gate 奖励从未被采样——"过门→出拳"的价值链
    # 无法建立，出拳旧习惯无消亡压力，形成死锁。frac 从 _FRAC_HI
    # 起步让门可开，按 eval 过门率每轮 ±_FRAC_STEP 收紧/放松，
    # 把 ungated 率维持在 _UNGATE_BAND 区间——policy 始终处在
    # "门努努力就能够到"的前沿，直到自然收敛回 5% 终态判据。
    _FRAC_LO, _FRAC_HI = 0.05, 0.30
    _FRAC_STEP = 0.01
    _UNGATE_BAND = (0.30, 0.80)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._gate_frac = self._FRAC_HI

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        jobs = super().build_jobs(policy_bp, base_seed, n_episodes,
                                  update=update, stochastic=stochastic)
        spec = dict(self.GATE_SPEC)
        spec["frac"] = self._gate_frac
        for j in jobs:
            j.episode_options["launch_gate"] = dict(spec)
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
        # frac 课程调度：以全体 strike 技能的平均 ungated 率调节
        rates = [np.mean(m["ungated"]) for m in per_skill.values()
                 if m["ungated"]]
        if rates:
            rate = float(np.mean(rates))
            if rate < self._UNGATE_BAND[0]:
                self._gate_frac = max(self._FRAC_LO,
                                      self._gate_frac - self._FRAC_STEP)
            elif rate > self._UNGATE_BAND[1]:
                self._gate_frac = min(self._FRAC_HI,
                                      self._gate_frac + self._FRAC_STEP)
        res["info"]["gate_frac"] = round(self._gate_frac, 3)
        return res


EXPERIMENT_CLASS = SkillGateMBS
