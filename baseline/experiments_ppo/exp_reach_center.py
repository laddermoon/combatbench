"""M4 先验受控 A/B 实验：走到场地中心（reach-center）。

任务：robot_a 站立出生在 -x 侧（初始距离 d → 离中心 d/2，面向中心），
走到 |xy| < 0.8m 记 reach_done；跌倒记 reach_fall；timeout 未到达。
每 episode initial_distance ∈ [dist_min, dist_max] 均匀采样→走 2-5m。

A/B 对照（README §9 M4 / §10）：
  - 两臂共用同一 env、observer、网络结构、episode 预算；
  - 唯一差异 = r_style 通道的 actor_weight：
      ``--set style_weight=0.0``（无先验）vs 正值（有先验）。
  - r_style 恒被记录（两臂均可比较风格分轨迹），critic 也在，
    仅 actor 优势中该项被 0 权重消去——干净的双臂消融。

指标：reach_rate / fall_rate / timeout_rate / mean_steps /
D_mean（在线风格分）/ final_dist。best = reach_rate - 0.5·fall_rate。
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


class ReachCenter(CombatExperimentPPOBase):

    name = "reach_center"
    actor_blueprint = "init_policy_truncated_normal.yaml"

    obs_dim: int = 96
    action_dim: int = 21

    env_blueprint = "reach_center_env.yaml"
    agent_used = "robot_a"
    max_steps: int = 200
    # A 出生半径 = dist/2 ∈ [1.5, 2.5]m（围墙 r≈3.38，留安全边距）；
    # RotateSpawnPlugin 把方向随机化到整圆 → 走 1.5-2.5m 到中心。
    dist_min: float = 3.0
    dist_max: float = 5.0

    # --- Reward channels ---
    _channel_names = ("r_task", "r_style")
    _channel_gammas = {"r_task": 0.99, "r_style": 0.99}
    _gae_lambda: float = 0.95

    # A/B 旋钮：r_style 进 actor 优势的权重（0=无先验臂）
    style_weight: float = 0.5

    # --- Exploration / PPO ---
    explore_factor: float = 0.0
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096

    episodes_per_update: int = 256
    max_updates: int = 800
    eval_interval: int = 5
    eval_episodes: int = 32
    video_eval_interval: int = 10

    _REWARD_KEY = "reach_reward_a"
    _STYLE_KEY = "amp_style_a"
    _best_score: float = -1.0

    def build_actor(self, device):
        blueprint_dir = Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / self.actor_blueprint)
        return bp.build(obs_dim=self.obs_dim).to(device)

    @staticmethod
    def _standing_bp() -> PolicyBlueprint:
        return PolicyBlueprint.load(
            Path(__file__).resolve().parents[2]
            / "policy" / "blueprints" / "humanoid21" / "standing.yaml")

    # ------------------------------------------------------------------
    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(RewardChannel(
            name=k, gamma=self._channel_gammas[k],
            gae_lambda=self._gae_lambda) for k in self._channel_names)

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True) -> List[Job]:
        env_bp = self._env_pb().materialize(max_steps=self.max_steps)
        standing = self._standing_bp()
        sampling_a = SamplingSpec(explore_factor=self.explore_factor)
        rng = np.random.RandomState(base_seed)
        jobs = []
        for i in range(n_episodes):
            dist = float(rng.uniform(self.dist_min, self.dist_max))
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=standing,
                env_bp=env_bp,
                seed=int(base_seed + i),
                episode_options={"initial_distance": dist},
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

            r_task = extract_per_step_field(
                ep.observer_outputs, self._REWARD_KEY, "reward", T_full)
            r_style = extract_per_step_field(
                ep.observer_outputs, self._STYLE_KEY, "reward", T_full)
            r_task = np.zeros(T_full) if r_task is None else np.asarray(r_task)
            r_style = np.zeros(T_full) if r_style is None else np.asarray(r_style)

            reason = ep.agent_termination_reason.get("robot_a", "")
            is_terminated = bool(reason) and reason != "timeout"

            out.append(Trajectory(
                obs=np.asarray(obs_all, dtype=np.float32)[:T],
                actions=np.asarray(acts_all, dtype=np.float32)[:T],
                last_obs=np.asarray(fin_obs, dtype=np.float32),
                channels={
                    "r_task": ChannelData(
                        reward=r_task[:T].astype(np.float32),
                        is_terminated=is_terminated,
                        actor_weight=np.ones(T, dtype=np.float32)),
                    "r_style": ChannelData(
                        reward=r_style[:T].astype(np.float32),
                        is_terminated=is_terminated,
                        actor_weight=np.full(
                            T, self.style_weight, dtype=np.float32)),
                },
                importance=1.0,
                sampling_ctx=self.extract_sampling_ctx(ep, "robot_a", T),
            ))
        return out

    # ------------------------------------------------------------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        n = max(len(episodes), 1)
        reach = fall = timeout = 0
        steps, ds, dists = [], [], []
        for ep in episodes:
            T = ep.agent_frame_boundary.get("robot_a", ep.num_frames)
            if T == 0:
                continue
            reason = ep.agent_termination_reason.get("robot_a", "")
            if reason == "reach_done":
                reach += 1
            elif reason == "reach_fall":
                fall += 1
            else:
                timeout += 1
            steps.append(T)
            d = extract_per_step_field(
                ep.observer_outputs, self._REWARD_KEY, "dist", ep.num_frames)
            if d is not None and len(d):
                dists.append(float(d[min(T - 1, len(d) - 1)]))
            st = extract_per_step_field(
                ep.observer_outputs, self._STYLE_KEY, "D", ep.num_frames)
            if st is not None and len(st):
                ds.append(float(np.mean(st[:T])))

        reach_rate = reach / n
        score = reach_rate - 0.5 * (fall / n)
        is_new_best = score > self._best_score
        if is_new_best:
            self._best_score = score
        return {
            "is_new_best": is_new_best,
            "info": {
                "reach_rate": round(reach_rate, 3),
                "fall_rate": round(fall / n, 3),
                "timeout_rate": round(timeout / n, 3),
                "mean_steps": round(float(np.mean(steps)), 1) if steps else 0.0,
                "D_mean": round(float(np.mean(ds)), 3) if ds else 0.0,
                "final_dist": round(float(np.mean(dists)), 2) if dists else 0.0,
            },
        }

    def state(self) -> dict:
        return {**super().state(), "best_score": self._best_score}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._best_score = float(state.get("best_score", -1.0))


EXPERIMENT_CLASS = ReachCenter
