"""Stage-3 precision strikes: parameterized world-anchored targets.

阶段三（详案见 hierarchy/PRECISION_STRIKE_PLAN.md）：打击技能从
"模仿参考 clip"升级为"命中指令参数化的目标点"。本实验继承
SkillTransMBS——**技能衔接热身（初始姿态泛化）全部保留**，任意
技能任意阶段起手都要能命中指定点。

机制（对阶段一/二零侵入）：
- CommandSimulator 新增 ``tgt_anchor`` episode option：打击段的
  ``tx,ty,tz`` 语义从"选 clip 的 intent"变为"段首 ego 目标点"——
  段首烘成世界锚点（含可选 ``tgt_drift`` 时变），逐帧重投影回 ego
  写入 cmd[6:9]。option 缺省时行为与旧路径完全一致。
- StrikeTrackObserver 新增每步 ``hand_e*``（出拳手 pelvis-rel ego）
  与 ``tgt_e*``（显示给 policy 的 ego 目标）字段；``strike_clip``
  option 显式绑定拳型 clip（目标点不再参与 clip 选择）。
- 奖励（build_trajectories 事后重建，仅在 fam=="strike" 时附加）：

      伸展段曲线 = hand_ego(t), t ∈ [0, t_reach]
        t_reach = argmax ||hand_ego||（出手最远点，之后的收手不计）
      d_min   = min_t ||hand_ego(t) − tgt_ego(t)||   → 精度
      t*      = argmin_t ...                          → 到达最近点用时

      r_tgt_dist: reward[t*] = -d_min                （稀疏终点罚）
      r_tgt_time: reward[t] = -1, t ∈ [0, t*]         （稠密时间罚）

  距离在 ego 系内计算等价于世界系距离（刚体变换保距），且 tgt_e*
  直接读显示值——与 policy 所见严格一致。

- 课程（锥形放宽）：目标 ~ c_skill + Uniform(ball(r_curr))，c_skill
  为 P0 实测命中分布均值（hit_distribution_p0.json，站立起手条件）。
  eval 按当前半径采样测 hit_rate(d_min ≤ ε=0.08m)，连续
  PROMOTE_PATIENCE 次 eval ≥ θ_hit 晋级一档半径。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData
from baseline.framework.rollout import extract_per_step_field
from .exp_skill_train_mbs import SKILLS, SkillTrainMBS
from .exp_skill_trans_mbs import SkillTransMBS


def _clip_for(skill: str) -> str:
    """Resolve a strike skill to its reference clip (nominal land)."""
    from baseline.humanoid21.hierarchy.strike_track_observer import (
        resolve_clip)
    s = SKILLS[skill]
    land = s["land"]
    return resolve_clip(
        s["side"], (land[0], land[1] * (1 if s["side"] > 0 else -1),
                    land[2]))


class SkillPreciseMBS(SkillTransMBS):
    """Precision-target variant of the transition experiment."""

    name = "skill_precise_mbs"

    # --- channels: parent's eight + two precision penalties ----------
    _channel_names = SkillTrainMBS._channel_names + (
        "r_tgt_dist", "r_tgt_time")
    _channel_gammas = {**SkillTrainMBS._channel_gammas,
                       "r_tgt_dist": 0.99, "r_tgt_time": 0.99}

    # --- precision reward constants -----------------------------------
    r_tgt_dist_actor_weight: float = 3.0
    r_tgt_time_actor_weight: float = 2.0
    tgt_time_step_penalty: float = 0.1      # −0.1/step until t*
    tgt_dist_clip: float = 1.5              # m — cap sparse penalty
    reach_min: float = 0.35                 # m — below = "never punched"
    no_reach_dist: float = 0.5              # m — d_min floor when so

    # --- cone curriculum (ego frame at segment start, pelvis-rel) ----
    # P0 measured hit means (standing-conditioned, hit_distribution_p0)
    TGT_CENTER: Dict[str, tuple] = {
        "jab_l":   (0.19, 0.33, 0.54),
        "jab_r":   (0.40, -0.31, 0.53),
        "heavy_l": (0.51, -0.13, 0.52),
        "heavy_r": (0.42, 0.14, 0.48),
    }
    TGT_R_INIT: float = 0.10   # m — initial cone radius
    TGT_R_STEP: float = 0.05   # m — per-level widening
    TGT_R_MAX: float = 0.45    # m
    HIT_EPS: float = 0.08      # m — hit threshold (~fist face)
    PROMOTE_HIT_RATE: float = 0.6
    PROMOTE_PATIENCE: int = 6  # consecutive evals at/above threshold

    def __init__(self):
        super().__init__()
        self._tgt_radius = self.TGT_R_INIT
        self._promote_count = 0
        self._eval_rng = np.random.default_rng(20261019)

    # ------------------------------------------------------------------
    # Target sampling — cone around the skill's measured hit center
    # ------------------------------------------------------------------
    def _sample_tgt(self, rng, skill: str) -> np.ndarray:
        c = np.asarray(self.TGT_CENTER[skill], dtype=np.float64)
        d = rng.normal(size=3)
        d = d / max(np.linalg.norm(d), 1e-9)
        r = self._tgt_radius * rng.random() ** (1.0 / 3.0)  # ball-uniform
        tgt = c + d * r
        # Reachability guard: keep inside the forward punchable half-dome
        tgt[0] = float(np.clip(tgt[0], -0.05, 0.85))
        tgt[2] = float(np.clip(tgt[2], 0.10, 0.80))
        tgt[1] = float(np.clip(tgt[1], -0.75, 0.75))
        return tgt

    def _apply_tgt(self, opts: Dict[str, Any], rng) -> None:
        """Rewrite strike rows' tgt to a sampled cone point; enable
        anchored display + explicit clip binding."""
        skills = opts.get("skills") or {}
        sched = opts.get("move_schedule") or {}
        touched = False
        for rid in ("robot_a", "robot_b"):
            skill = str(skills.get(rid, ""))
            if skill not in SKILLS or SKILLS[skill]["fam"] != "strike":
                continue
            rows = sched.get(rid)
            if not rows:
                continue
            tgt = self._sample_tgt(rng, skill)
            rows[0][7:10] = [float(tgt[0]), float(tgt[1]),
                             float(tgt[2])]
            opts.setdefault("strike_clip", {})[rid] = _clip_for(skill)
            opts.setdefault("tgt_spec", {})[rid] = {
                "skill": skill, "tgt": [float(x) for x in tgt],
                "radius": self._tgt_radius}
            touched = True
        if touched:
            opts["tgt_anchor"] = True

    # ------------------------------------------------------------------
    # Jobs — parent builds warmup/skill episodes; we rewrite targets
    # ------------------------------------------------------------------
    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        jobs = super().build_jobs(
            policy_bp, base_seed, n_episodes,
            update=update, stochastic=stochastic)
        rng = (self._eval_rng if not stochastic
               else np.random.default_rng(base_seed + 7919))
        for job in jobs:
            self._apply_tgt(job.episode_options, rng)
        return jobs

    # ------------------------------------------------------------------
    # Trajectories — attach precision channels on strike episodes
    # ------------------------------------------------------------------
    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> list:
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key)
        if not trajs:
            return trajs
        skills = (episode.episode_options or {}).get("skills") or {}
        skill = str(skills.get(agent_id, ""))
        if skill not in SKILLS or SKILLS[skill]["fam"] != "strike":
            return trajs

        T_full = episode.num_frames
        T = trajs[0].obs.shape[0]
        sk = self._STRIKE_OBS[agent_id]
        oo = episode.observer_outputs

        def _f(name):
            a = extract_per_step_field(oo, sk, name, T_full)
            return (np.asarray(a[:T], dtype=np.float64)
                    if a is not None else np.zeros(T))

        ins = np.asarray(_f("in_strike")) > 0.5
        hand = np.stack([_f("hand_ex"), _f("hand_ey"), _f("hand_ez")],
                        axis=1)
        tgt = np.stack([_f("tgt_ex"), _f("tgt_ey"), _f("tgt_ez")],
                       axis=1)
        idx = np.flatnonzero(ins)
        if idx.size < 3 or not bool(
                (episode.episode_options or {}).get("tgt_anchor")):
            # no anchored target recorded → zero channels (critics still
            # exist; actor weight 0)
            z = np.zeros(T, dtype=np.float32)
            for t in trajs:
                t.channels["r_tgt_dist"] = ChannelData(
                    reward=z.copy(), is_terminated=False,
                    actor_weight=z.copy())
                t.channels["r_tgt_time"] = ChannelData(
                    reward=z.copy(), is_terminated=False,
                    actor_weight=z.copy())
            return trajs

        lo, hi = int(idx[0]), int(idx[-1])
        seg_hand = hand[lo:hi + 1]
        reach = np.linalg.norm(seg_hand, axis=1)
        t_reach = lo + int(np.argmax(reach))
        if float(reach.max()) < self.reach_min:
            d_min, t_star = self.no_reach_dist, hi
        else:
            d = np.linalg.norm(hand[lo:t_reach + 1] - tgt[lo:t_reach + 1],
                               axis=1)
            j = int(np.argmin(d))
            d_min, t_star = float(d[j]), lo + j

        phi = extract_per_step_field(oo, phi_key, "potential", T_full)
        phi = (np.clip(np.asarray(phi[:T], dtype=np.float32), 0.0, 1.0)
               if phi is not None else np.ones(T, dtype=np.float32))
        aw_dist = np.zeros(T, dtype=np.float32)
        aw_dist[t_star] = self.r_tgt_dist_actor_weight \
            * float(phi[t_star]) ** 2
        aw_time = np.zeros(T, dtype=np.float32)
        aw_time[lo:t_star + 1] = self.r_tgt_time_actor_weight

        r_dist = np.zeros(T, dtype=np.float32)
        r_dist[t_star] = -min(d_min, self.tgt_dist_clip)
        r_time = np.zeros(T, dtype=np.float32)
        r_time[lo:t_star + 1] = -self.tgt_time_step_penalty

        for t in trajs:
            t.channels["r_tgt_dist"] = ChannelData(
                reward=r_dist.copy(), is_terminated=False,
                actor_weight=aw_dist.copy())
            t.channels["r_tgt_time"] = ChannelData(
                reward=r_time.copy(), is_terminated=False,
                actor_weight=aw_time.copy())
        return trajs

    # ------------------------------------------------------------------
    # Eval — per-skill precision metrics + radius promotion
    # ------------------------------------------------------------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        out = super().on_eval(episodes, update)
        hits, tms = [], []
        per_skill: Dict[str, Dict[str, list]] = {}
        for ep in episodes:
            opts = ep.episode_options or {}
            spec = opts.get("tgt_spec") or {}
            if not opts.get("tgt_anchor") or not spec:
                continue
            skills = opts.get("skills") or {}
            for aid, sk_key in (("robot_a", "strike_track_a"),
                                ("robot_b", "strike_track_b")):
                sp = spec.get(aid)
                if sp is None:
                    continue
                T = ep.num_frames
                oo = ep.observer_outputs

                def _f(name, T=T, oo=oo, sk_key=sk_key):
                    a = extract_per_step_field(oo, sk_key, name, T)
                    return (np.asarray(a[:T], dtype=np.float64)
                            if a is not None else np.zeros(T))

                ins = np.asarray(_f("in_strike")) > 0.5
                hand = np.stack([_f("hand_ex"), _f("hand_ey"),
                                 _f("hand_ez")], axis=1)
                tgt = np.stack([_f("tgt_ex"), _f("tgt_ey"),
                                _f("tgt_ez")], axis=1)
                idx = np.flatnonzero(ins)
                if idx.size < 3:
                    continue
                lo, hi = int(idx[0]), int(idx[-1])
                reach = np.linalg.norm(hand[lo:hi + 1], axis=1)
                t_reach = lo + int(np.argmax(reach))
                if float(reach.max()) < self.reach_min:
                    d_min, t_s = self.no_reach_dist, hi
                else:
                    d = np.linalg.norm(
                        hand[lo:t_reach + 1] - tgt[lo:t_reach + 1],
                        axis=1)
                    j = int(np.argmin(d))
                    d_min, t_s = float(d[j]), lo + j
                m = per_skill.setdefault(sp["skill"], {})
                m.setdefault("dmin", []).append(d_min)
                m.setdefault("tt", []).append(float(t_s - lo))
                m.setdefault("hit", []).append(
                    float(d_min <= self.HIT_EPS))
                hits.append(d_min <= self.HIT_EPS)
                tms.append(t_s - lo)

        info = out.setdefault("info", {})
        for skill, m in per_skill.items():
            info[f"{skill}_tgt_dmin"] = round(float(np.mean(
                m["dmin"])), 3)
            info[f"{skill}_tgt_t"] = round(float(np.mean(m["tt"])), 2)
            info[f"{skill}_tgt_hit"] = round(float(np.mean(
                m["hit"])), 3)
        if hits:
            hit_rate = float(np.mean(hits))
            info["tgt_hit_rate"] = round(hit_rate, 3)
            info["tgt_radius"] = round(self._tgt_radius, 3)
            if hit_rate >= self.PROMOTE_HIT_RATE:
                self._promote_count += 1
                if (self._promote_count >= self.PROMOTE_PATIENCE
                        and self._tgt_radius < self.TGT_R_MAX):
                    self._tgt_radius = min(
                        self.TGT_R_MAX,
                        self._tgt_radius + self.TGT_R_STEP)
                    self._promote_count = 0
                    info["tgt_promoted"] = round(self._tgt_radius, 3)
            else:
                self._promote_count = 0
        return out

    # ------------------------------------------------------------------
    # State persistence — curriculum radius survives resume
    # ------------------------------------------------------------------
    def state(self) -> dict:
        s = super().state()
        s["tgt_radius"] = self._tgt_radius
        s["promote_count"] = self._promote_count
        return s

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._tgt_radius = float(
            state.get("tgt_radius", self.TGT_R_INIT))
        self._promote_count = int(state.get("promote_count", 0))


EXPERIMENT_CLASS = SkillPreciseMBS
