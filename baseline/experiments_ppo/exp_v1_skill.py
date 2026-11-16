"""V1 技能训练实验（DECISION_LOG D3 简化版）。

每轮迭代的闭环：

1. ``build_jobs``：各 agent 独立采样技能，从该技能**候选起点集**
   （READY + ∪ 前序技能终止集）取一个快照起点，经
   ``v1_snapshot``/``v1_cmd`` 注入 episode_options。
2. episode 内 ``V1SkillSimulator`` 重置到快照并持续下发
   cmd[0:6]=技能码的 9 维指令；失衡提前终止。
3. ``build_trajectories``：V1ClipObserver 逐帧世界系特征 → Clip →
   三个整段 score → 末帧结算（r_com/r_step/r_posture，
   actor_weight 全程放开，见 D3 奖励传递约定）。
4. 合格执行（未失衡终止 且 min(scores) ≥ QUALIFY_MIN）的末帧
   core_state 快照写回该技能的**终止状态集**，成为其后继技能
   下一轮迭代的候选起点。

warm-start（D2：s42i_b u01680，66-dim 指令槽只喂 6 维）::

    PYTHONPATH=. python3 baseline/framework/train.py \
        --experiment v1_skill --background \
        --resume-from baseline/runs/hier_s2_balance_s42i_b/\
checkpoints/checkpoint_u01680.pt --reset-update
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory)
from baseline.humanoid21.v1.pool import StartPool
from baseline.humanoid21.v1.skills import SKILLS, Clip
from baseline.humanoid21.v1.state import ready_core_state, to_jsonable
from .base import CombatExperimentPPOBase

_ENV_YAML = (Path(__file__).resolve().parent.parent
             / "humanoid21" / "v1" / "v1_skill_env.yaml")
_CLIP_OBS = {"robot_a": "v1_clip_a", "robot_b": "v1_clip_b"}
_CLIP_FIELDS = ("com", "ankle_l", "ankle_r", "hand_l", "hand_r",
                "root_pos", "root_pry", "root_vel", "contact")
_SNAP_FIELDS = ("root_pos", "root_rot", "root_vel_local",
                "root_angular_vel_local", "joint_pos_norm",
                "joint_vel_norm")


class ExpV1Skill(CombatExperimentPPOBase):
    """第一阶段：4 技能（step_l/r、jab_l/r）前序链式起点池训练。"""

    name = "v1_skill"
    obs_dim = 66
    action_dim = 21
    actor_blueprint = "init_policy_mbs_66d.yaml"

    # v0：短 episode（≤40 步）→ 每 update 少量 episode 即可
    episodes_per_update = 256
    eval_episodes = 16

    # 合格线：min(三个 score) ≥ 才记为可用执行（v0 单一阈值）
    QUALIFY_MIN = 0.40

    _CLIP_KEYS = ("r_com", "r_step", "r_posture")

    # ------------------------------------------------------------------
    # 内部状态
    # ------------------------------------------------------------------

    def _pool(self) -> StartPool:
        p = getattr(self, "_pool_inst", None)
        if p is None:
            p = StartPool(to_jsonable(ready_core_state()))
            self._pool_inst = p
        return p

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint)
        return ParameterizedEnvBlueprint.load(_ENV_YAML)

    # ------------------------------------------------------------------
    # 奖励通道：整段评分 → 末帧结算（D3）
    # ------------------------------------------------------------------

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(RewardChannel(k, gamma=0.99, gae_lambda=0.95)
                     for k in self._CLIP_KEYS)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job

        self._update = update
        if getattr(self, "_iter_update", None) != update:
            self._pool().begin_iteration(SKILLS, update)
            self._iter_update = update
            self._rollout_stat = {
                k: {"n": 0, "q": 0, "scores": []} for k in SKILLS}

        rng = np.random.default_rng(base_seed)
        sampling = self._sampling_spec()
        max_steps = max(sk.step_budget for sk in SKILLS.values())
        env_bp = self._env_pb().materialize(max_steps=max_steps)
        names = list(SKILLS)

        jobs = []
        for i in range(n_episodes):
            ep_opts: Dict[str, Any] = {
                "skills": {}, "v1_cmd": {}, "v1_snapshot": {},
                "v1_start_key": {},
            }
            for j, rid in enumerate(("robot_a", "robot_b")):
                if stochastic:
                    sk = str(rng.choice(names))
                else:
                    sk = names[(i + j) % len(names)]   # eval 轮转覆盖
                entry = self._pool().sample(sk, rng)
                ep_opts["skills"][rid] = sk
                ep_opts["v1_start_key"][rid] = entry.key
                ep_opts["v1_cmd"][rid] = np.concatenate(
                    [SKILLS[sk].instruction, np.zeros(3)]).tolist()
                ep_opts["v1_snapshot"][rid] = entry.state
            jobs.append(Job(
                policy_a_bp=policy_bp, policy_b_bp=policy_bp,
                env_bp=env_bp, seed=int(base_seed + i),
                episode_options=ep_opts,
                sampling_a=sampling, sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Trajectories：Clip 评分 → 末帧结算 + 终止状态回收
    # ------------------------------------------------------------------

    def build_trajectories(self, episodes) -> List[Trajectory]:
        trs: List[Trajectory] = []
        for ep in episodes:
            for aid in ("robot_a", "robot_b"):
                trs.extend(self._agent_traj(ep, aid))
        return trs

    def _clip_fields(self, ep, aid: str, T: int) -> Optional[Clip]:
        node = (ep.observer_outputs or {}).get(_CLIP_OBS[aid]) or {}
        fields = {}
        for f in _CLIP_FIELDS:
            arr = node.get(f)
            if arr is None:
                return None
            arr = np.asarray(arr, np.float64)
            if arr.shape[0] < T:
                return None
            fields[f] = arr[:T]
        return Clip(**fields)

    def _terminal_snapshot(self, ep, aid: str, T: int) -> Dict[str, Any]:
        node = (ep.observer_outputs or {}).get(_CLIP_OBS[aid]) or {}
        snap = {}
        for f in _SNAP_FIELDS:
            arr = node.get("snap_" + f)
            if arr is None or len(arr) < T:
                return {}
            snap[f] = np.asarray(arr)[T - 1].tolist()
        return snap

    def _agent_traj(self, ep, aid: str) -> List[Trajectory]:
        T = int(ep.agent_frame_boundary.get(aid, ep.num_frames))
        if T <= 0:
            return []
        obs = np.asarray(ep.observations[aid][:T], np.float32)
        acts = np.asarray(ep.actions[aid][:T], np.float32)
        last = np.asarray(ep.final_observation[aid], np.float32)

        opts = ep.episode_options or {}
        sk_name = str((opts.get("skills") or {}).get(aid, ""))
        sk = SKILLS.get(sk_name)
        clip = self._clip_fields(ep, aid, T)
        if sk is None or clip is None or len(clip) == 0:
            return []

        scores = {
            "r_com": float(sk.score_com(clip)),
            "r_step": float(sk.score_step(clip)),
            "r_posture": float(sk.score_posture(clip)),
        }

        channels = {}
        for k, v in scores.items():
            r = np.zeros(T, np.float32)
            r[-1] = v
            channels[k] = ChannelData(
                reward=r, is_terminated=True,
                actor_weight=np.ones(T, np.float32))

        # 合格判定 + 终止状态回收（D3：未失衡 且 min(scores) ≥ 阈值）
        reason = str(ep.agent_termination_reason.get(aid, ""))
        fell = reason.startswith("imbalance")
        qualified = (not fell) and min(scores.values()) >= self.QUALIFY_MIN
        if qualified:
            snap = self._terminal_snapshot(ep, aid, T)
            if snap:
                self._pool().record_terminal(sk_name, snap,
                                             getattr(self, "_update", 0))

        st = getattr(self, "_rollout_stat", None)
        if st is not None and sk_name in st:
            d = st[sk_name]
            d["n"] += 1
            d["q"] += int(qualified)
            d["scores"].append(tuple(scores.values()))

        return [Trajectory(
            obs=obs, actions=acts, last_obs=last,
            channels=channels, importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(ep, aid, T),
        )]

    # ------------------------------------------------------------------
    # Eval / metrics
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        m: Dict[str, Any] = {}
        per_skill: Dict[str, Dict[str, Any]] = {}
        n_all, q_all = 0, 0
        for ep in episodes:
            for aid in ("robot_a", "robot_b"):
                opts = ep.episode_options or {}
                sk_name = str((opts.get("skills") or {}).get(aid, ""))
                sk = SKILLS.get(sk_name)
                T = int(ep.agent_frame_boundary.get(aid, ep.num_frames))
                if sk is None or T <= 0:
                    continue
                clip = self._clip_fields(ep, aid, T)
                if clip is None or len(clip) == 0:
                    continue
                s3 = (float(sk.score_com(clip)),
                      float(sk.score_step(clip)),
                      float(sk.score_posture(clip)))
                reason = str(ep.agent_termination_reason.get(aid, ""))
                qual = int((not reason.startswith("imbalance"))
                           and min(s3) >= self.QUALIFY_MIN)
                d = per_skill.setdefault(sk_name, {"n": 0, "q": 0, "s": []})
                d["n"] += 1
                d["q"] += qual
                d["s"].append(s3)
                n_all += 1
                q_all += qual

        for k, d in per_skill.items():
            arr = np.asarray(d["s"], np.float64)
            m[f"{k}_qual"] = round(d["q"] / max(1, d["n"]), 3)
            for i, c in enumerate(("com", "step", "posture")):
                m[f"{k}_{c}"] = round(float(arr[:, i].mean()), 3)

        st = getattr(self, "_rollout_stat", None) or {}
        for k, d in st.items():
            if d["n"]:
                m[f"tr/{k}_qual"] = round(d["q"] / d["n"], 3)
        for grp, dd in self._pool().stats().items():
            for k, v in dd.items():
                m[f"pool_{grp[:4]}_{k}"] = v

        qual_rate = q_all / max(1, n_all)
        m["qual_rate"] = round(qual_rate, 3)
        is_new_best = qual_rate > getattr(self, "_best_qual", -1.0)
        if is_new_best:
            self._best_qual = qual_rate
        return {"is_new_best": is_new_best, "info": m}

    # ------------------------------------------------------------------
    # checkpoint 持久化：起点池随实验状态保存
    # ------------------------------------------------------------------

    def state(self) -> dict:
        s = super().state()
        s["v1_pool"] = self._pool().save()
        s["best_qual"] = getattr(self, "_best_qual", -1.0)
        return s

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        if "v1_pool" in state:
            self._pool().load(state["v1_pool"])
        self._best_qual = float(state.get("best_qual", -1.0))


EXPERIMENT_CLASS = ExpV1Skill
