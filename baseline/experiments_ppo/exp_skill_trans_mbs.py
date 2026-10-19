"""Stage-2 skill training: transition coverage via skill warmup.

上半阶段（skill_train_mbs）的每个 episode 都从 WarmupSteppingPlugin
的踏步中间态开始 —— 技能只在"站立起步"分布下学习，任意技能任意
阶段 → 目标技能的衔接瞬态从未被覆盖（EXP_CURVED_MOTION 实测：
穿插/组合指令下 10-42% episode 时间处于失衡低位）。

本实验**不改** SkillTrainMBS 的任何代码，仅继承并重写 stochastic
build_jobs：对每个 robot 独立采样一个**热身技能**和**热身时长**
（均匀 [0, cap]，0 = 无热身 = 原站立起步），通过
``episode_options["warmup"]`` 交给 WarmupSteppingPlugin 的
技能热身路径执行。Eval（stochastic=False）走父类不变。

热身采样规则：
- 热身技能在全部技能上均匀采样；唯一排除：热身打击 == 目标打击
  （同 clip 相位冲突——打击中段再打同招不合理）。
- 非打击目标不排除同技能热身（move→move 连续覆盖同样有效）。
- 热身时长按技能族设上限（打击窗口短 → cap 小）。
- 若所有 robot 都采到 steps=0 则不写 "warmup" 键，走原路径。
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict

import numpy as np

from .exp_skill_train_mbs import (
    _FAMILY_SKILLS, SKILLS, SkillTrainMBS, _skill_len, _skill_rows)
from .exp_balance_step import _balance_explore_factor


class SkillTransMBS(SkillTrainMBS):
    """Transition-warmup variant of the composite skill experiment."""

    name = "skill_trans_mbs"

    # 每族技能的热身步数上限（均匀 [0,cap] 采样，0=无热身）
    WARMUP_STEP_CAP: Dict[str, int] = {
        "move": 35, "turn": 30, "idle": 40, "strike": 20,
    }
    # 衔接阶段预算独立于上半阶段：trans 从 u9861 续训起步，
    # 10000 的继承上限只够 ~140 update——本阶段单独放宽。
    max_updates: int = 20000
    # 热身技能候选 = 全技能；目标为打击时排除同名打击（相位冲突）
    warmup_exclude_same_strike: bool = True
    # step/idle 初始条件按显式概率分两类：倒地恢复（RandomFallenState
    # env）vs 其它状态起手（warmup env——技能中段或站立）。
    # 倒地覆盖只存在于 idle episode，故用显式概率而非 warmup 是否
    # 采中非空来划分（warmup 命中率 ~97%，会挤掉倒地样本）。
    IDLE_FALLEN_PROB: float = 0.5

    def _sample_warmup(self, rng, target: str) -> Dict[str, Any]:
        """Per-robot warmup spec: {'skill','steps','schedule'} or None."""
        t_fam = SKILLS[target]["fam"]
        cands = [s for s in SKILLS
                 if not (self.warmup_exclude_same_strike
                         and t_fam == "strike" and s == target)]
        w = str(rng.choice(cands))
        cap = self.WARMUP_STEP_CAP[SKILLS[w]["fam"]]
        steps = int(rng.integers(0, cap + 1))
        if steps == 0:
            return None
        return {"skill": w, "steps": steps,
                "schedule": _skill_rows(w)}

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        env_cache: Dict[Any, Any] = {}

        def _env(fam: str, max_steps: int):
            key = (fam, max_steps)
            if key not in env_cache:
                env_cache[key] = self._fam_env_pb(fam).materialize(
                    max_steps=max_steps)
            return env_cache[key]

        def _add_warmup(self_rng, skills: Dict[str, str]) -> Dict[str, Any]:
            w: Dict[str, Any] = {}
            for rid, st in skills.items():
                spec = self._sample_warmup(self_rng, st)
                if spec is not None:
                    w[rid] = spec
            return w

        if not stochastic:
            # Eval 与 train 同一分布：同样的 round-robin 技能对 +
            # 同样的 warmup 采样（确定性 rng 派生自 base_seed）。
            jobs = super().build_jobs(
                policy_bp, base_seed, n_episodes,
                update=update, stochastic=False)
            rng = np.random.default_rng(base_seed + 0x9E3779B9)
            for i, j in enumerate(jobs):
                skills = j.episode_options.get("skills") or {}
                fam = SKILLS[str(next(iter(skills.values())))]["fam"]
                if fam == "idle" and rng.random() < self.IDLE_FALLEN_PROB:
                    continue                # 倒地初始条件：无 warmup
                warmup = _add_warmup(rng, skills)
                if warmup:
                    j.episode_options["warmup"] = warmup
                if fam == "idle":
                    n = max(_skill_len(s) for s in skills.values())
                    jobs[i] = dataclasses.replace(
                        j, env_bp=_env("idle_warm", n))
            return jobs
        from baseline.framework.rollout import Job, SamplingSpec
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=_balance_explore_factor)
        fams = list(self.FAMILY_WEIGHTS)
        probs = np.array([self.FAMILY_WEIGHTS[f] for f in fams])
        probs = probs / probs.sum()

        jobs = []
        for i in range(n_episodes):
            fam = str(rng.choice(fams, p=probs))
            ks = _FAMILY_SKILLS[fam]
            kw = np.array([self.SKILL_SAMPLE_W.get(k, 1.0) for k in ks])
            kw = kw / kw.sum()
            sa = str(rng.choice(ks, p=kw))
            sb = str(rng.choice(ks, p=kw))
            n = max(_skill_len(sa), _skill_len(sb))
            if fam == "idle" and rng.random() < self.IDLE_FALLEN_PROB:
                env_fam, warmup = "idle", {}        # 倒地恢复初始条件
            else:
                warmup = _add_warmup(
                    rng, {"robot_a": sa, "robot_b": sb})
                env_fam = "idle_warm" if fam == "idle" else fam
            opts = {
                "initial_distance": float(rng.uniform(
                    self.init_distance_min, self.init_distance_max)),
                "skills": {"robot_a": sa, "robot_b": sb},
                "move_schedule": {
                    "robot_a": _skill_rows(sa),
                    "robot_b": _skill_rows(sb),
                },
                "warmup_policy_cls": policy_bp.cls,
            }
            if warmup:
                opts["warmup"] = warmup
            jobs.append(Job(
                policy_a_bp=policy_bp, policy_b_bp=policy_bp,
                env_bp=_env(env_fam, n),
                seed=int(base_seed + i),
                episode_options=opts,
                sampling_a=sampling, sampling_b=sampling,
                stochastic=True,
            ))
        return jobs


EXPERIMENT_CLASS = SkillTransMBS
