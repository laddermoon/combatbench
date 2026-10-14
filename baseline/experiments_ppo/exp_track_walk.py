"""M2 运动跟踪实验（DeepMimic 式）——先用 cmu_07_01 走路片段。

环境：``track_ref_env.yaml``（MotionTrackingSimulator + RSI 随机相位初始化 +
MotionTrackGuardPlugin 摔倒/片段结束终止 + MotionTrackRewarder 模仿奖励）。

- robot_a：训练中的跟踪策略（obs 142 = 96 基础 + 46 参考特征）。
- robot_b：固定 StandingCombatPolicy（已训练站立策略），仅作场景；
  其观测仍是基础 96 维（扩展维只加给被跟踪机器人）。

奖励：单通道 r_track = DeepMimic 加权模仿奖励（pose/vel/keypoint/height），
已按步输出标量，轨迹构建只做字段提取。

终止语义：track_fall / track_clip_end → is_terminated=True；
max_steps timeout 视为不终止（不可能触及：clip ≤ 200 控制帧，max_steps 200，
仅长 clip 片段可能打到——以 track_clip_end 为主）。

Eval 指标：survive_frac（episode 步数 / clip 剩余控制帧）、kp_err_m 均值、
q_rmse_deg 均值、fall 率；best = 最大 survived 分数。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from envs.framework.policy import PolicyBlueprint
from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.framework.rollout.job import Job, SamplingSpec

from .base import CombatExperimentPPOBase


class TrackWalk(CombatExperimentPPOBase):

    name = "track_walk"
    actor_blueprint = "init_policy_truncated_normal.yaml"

    # obs = 96 基础 + 46 参考跟踪特征
    obs_dim: int = 142
    action_dim: int = 21

    env_blueprint = "track_ref_env.yaml"
    agent_used = "robot_a"
    max_steps: int = 200
    init_distance_min: float = 6.0
    init_distance_max: float = 6.0

    # --- Reward ---
    _channel_name = "r_track"
    _gamma: float = 0.99
    _gae_lambda: float = 0.95

    # --- Exploration ---
    explore_factor: float = 0.0

    # --- PPO ---
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096

    episodes_per_update: int = 256
    max_updates: int = 2000
    eval_interval: int = 5
    eval_episodes: int = 16
    video_eval_interval: int = 10

    _OBS_KEY = "track_a"
    _best_score: float = -1.0

    # robot_b 用已训练的站立策略，96 维 obs
    @staticmethod
    def _standing_bp() -> PolicyBlueprint:
        return PolicyBlueprint.load(
            Path(__file__).resolve().parents[2]
            / "policy" / "blueprints" / "humanoid21" / "standing.yaml")

    # ------------------------------------------------------------------
    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return (RewardChannel(
            name=self._channel_name, gamma=self._gamma,
            gae_lambda=self._gae_lambda),)

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True) -> List[Job]:
        env_bp = self._env_pb().materialize(max_steps=self.max_steps)
        standing = self._standing_bp()
        sampling_a = SamplingSpec(explore_factor=self.explore_factor)
        jobs = []
        for i in range(n_episodes):
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=standing,
                env_bp=env_bp,
                seed=int(base_seed + i),
                episode_options={"initial_distance": 6.0},
                sampling_a=sampling_a,
                sampling_b=SamplingSpec(),
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    def build_trajectories(self, episodes) -> List[Trajectory]:
        out: List[Trajectory] = []
        for ep in episodes:
            T_full = ep.num_frames
            if T_full == 0:
                continue
            T = ep.agent_frame_boundary.get("robot_a", T_full)
            if T == 0:
                continue
            obs_all = ep.observations.get("robot_a")
            acts_all = ep.actions.get("robot_a")
            fin_obs = ep.final_observation.get("robot_a")
            if obs_all is None or acts_all is None or fin_obs is None:
                continue
            rew = extract_per_step_field(
                ep.observer_outputs, self._OBS_KEY, "reward", T_full)
            if rew is None:
                rew = np.zeros(T_full, dtype=np.float32)
            rew = np.asarray(rew[:T], dtype=np.float32)

            # 摔倒/片段结束 → 真终止（不 bootstrap）；timeout → 未终止
            reason = ep.agent_termination_reason.get("robot_a", "")
            is_terminated = bool(reason) and reason != "timeout"

            out.append(Trajectory(
                obs=np.asarray(obs_all, dtype=np.float32)[:T],
                actions=np.asarray(acts_all, dtype=np.float32)[:T],
                last_obs=np.asarray(fin_obs, dtype=np.float32),
                channels={self._channel_name: ChannelData(
                    reward=rew, is_terminated=is_terminated,
                    actor_weight=np.ones(T, dtype=np.float32))},
                importance=1.0,
                sampling_ctx=self.extract_sampling_ctx(ep, "robot_a", T),
            ))
        return out

    # ------------------------------------------------------------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survives, kp_errs, q_rmses, falls, completes = [], [], [], 0, 0
        for ep in episodes:
            T = ep.agent_frame_boundary.get("robot_a", ep.num_frames)
            if T == 0:
                continue
            reason = ep.agent_termination_reason.get("robot_a", "")
            if reason == "track_fall":
                falls += 1
            if reason == "track_clip_end":
                completes += 1
            # 存活比：终止帧对应的 ref 相位即完成度；用 reward 输出的 frame
            fr = extract_per_step_field(
                ep.observer_outputs, self._OBS_KEY, "frame", ep.num_frames)
            kp = extract_per_step_field(
                ep.observer_outputs, self._OBS_KEY, "kp_err_m", ep.num_frames)
            qr = extract_per_step_field(
                ep.observer_outputs, self._OBS_KEY, "q_rmse_deg", ep.num_frames)
            n_ctrl = None
            if fr is not None and len(fr):
                # frame 达到的最后控制帧号 / 片段总控制帧数不可知——用 phase
                ph = extract_per_step_field(
                    ep.observer_outputs, self._OBS_KEY, "phase", ep.num_frames)
                if ph is None:
                    survives.append(float(T) / 200.0)
                else:
                    survives.append(float(np.max(ph)))
            if kp is not None and len(kp):
                kp_errs.append(float(np.mean(kp[:T])))
            if qr is not None and len(qr):
                q_rmses.append(float(np.mean(qr[:T])))

        n = max(len(survives), 1)
        survive = float(np.mean(survives)) if survives else 0.0
        score = survive - 0.5 * (falls / max(len(episodes), 1))
        is_new_best = score > self._best_score
        if is_new_best:
            self._best_score = score
        return {
            "is_new_best": is_new_best,
            "info": {
                "survive": round(survive, 3),
                "complete_rate": round(completes / max(len(episodes), 1), 3),
                "fall_rate": round(falls / max(len(episodes), 1), 3),
                "kp_err": round(float(np.mean(kp_errs)), 3) if kp_errs else 0.0,
                "q_rmse": round(float(np.mean(q_rmses)), 1) if q_rmses else 0.0,
            },
        }

    def state(self) -> dict:
        return {**super().state(), "best_score": self._best_score}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._best_score = float(state.get("best_score", -1.0))


EXPERIMENT_CLASS = TrackWalk
