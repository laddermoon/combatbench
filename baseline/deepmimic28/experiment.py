"""MimicKit DeepMimic 复现实验（humanoid28 + spinkick）。

方法全部对齐 MimicKit deepmimic_humanoid_ppo_agent.yaml：
  - obs 464 = char_obs(140) + tar_obs(3×108)，全局系
  - 奖励配方 pose.5/vel.1/root_pose.15/root_vel.1/key_pos.15
    尺度 .25/.01/5/1/10，per-joint 权重
  - 终止：接触摔倒（双脚 whitelist）+ pose 偏差 1m + 10s 超时
  - RSI 均匀相位；td_lambda 0.95；clip 0.2；discount 0.99
  - 策略：GaussianMLPPolicy 无界对角高斯（归一化动作空间），
    FIXED std=0.05（log_std 冻结），init_output_scale 0.01，FC 2×1024

框架差异（诚实记录，见 DESIGN.md）：
  - 优化器 Adam（框架硬编码）而非 SGD，lr 1e-4 保留
  - 无 obs running 归一化（框架不支持）
  - update_epochs 统一 5（框架无 actor/critic epochs 分离）
  - 无 action_bound_weight 惩罚项（用户决策：不为框架加扩展）
  - adv_winsorize_sigma=4.0 对应他的 norm_adv_clip=4.0

两种 critic 结构（用户实验）：
  multi_critic=False — 单通道 r_track = 加权和（他的方式）
  multi_critic=True  — 五通道独立 critic，actor_weight = 他的权重
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.framework.rollout.job import Job, SamplingSpec

from baseline.experiments_ppo.base import CombatExperimentPPOBase

_PKG_DIR = Path(__file__).resolve().parent

# MimicKit 权重
W = dict(pose=0.5, vel=0.1, root_pose=0.15, root_vel=0.1, key_pos=0.15)
MULTI_CHANNELS = ('r_pose', 'r_vel', 'r_root_pose', 'r_root_vel', 'r_key_pos')
CH_WEIGHT = {'r_pose': 0.5, 'r_vel': 0.1, 'r_root_pose': 0.15,
             'r_root_vel': 0.1, 'r_key_pos': 0.15}


class DeepMimic28Track(CombatExperimentPPOBase):

    name = "deepmimic28"

    obs_dim: int = 464
    action_dim: int = 28

    env_blueprint = "track_env.yaml"
    agent_used = "robot_a"
    motion: str = "humanoid_spinkick"
    rand_reset: bool = True
    episode_steps: int = 200
    multi_critic: bool = False        # 实验变量：单 critic 加权和 vs 多 critic
    freeze_std: bool = True           # MimicKit FIXED std=0.05
    init_log_std: float = float(np.log(0.05))
    init_distance: float = 6.0

    # --- PPO（MimicKit 对齐）---
    learning_rate: float = 1e-4       # Adam（框架硬编码；他是 SGD——差异记录）
    critic_learning_rate: float = 1e-4
    target_kl: float = 0.0            # 他无 KL early-stop
    update_epochs: int = 5            # actor_epochs=5（框架无分离，critic 同步 5）
    minibatch_size: int = 4096
    adv_winsorize_sigma: float = 4.0  # = 他的 norm_adv_clip

    episodes_per_update: int = 256
    max_updates: int = 2000
    eval_interval: int = 5
    eval_episodes: int = 16
    video_eval_interval: int = 10

    _OBS_KEY = "track_a"
    _best_score: float = -1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    def _env_pb(self) -> ParameterizedEnvBlueprint:
        return ParameterizedEnvBlueprint.load(
            _PKG_DIR / 'envs' / self.env_blueprint)

    def build_actor(self, device):
        from baseline.framework.ppo.policies.gaussian_mlp import GaussianMLPPolicy
        pol = GaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=1024,
            device=device, init_log_std=self.init_log_std,
            init_output_scale=0.01)
        if self.freeze_std:
            # MimicKit actor_std_type=FIXED：log_std 移出可训练参数
            # （register_buffer 保留在 state_dict，导出/加载不丢）
            import torch
            val = pol.log_std.detach().clone()
            del pol._parameters['log_std']
            pol.register_buffer('log_std', val)
        return pol

    # ------------------------------------------------------------------
    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        if self.multi_critic:
            return tuple(RewardChannel(name=c, gamma=0.99, gae_lambda=0.95)
                         for c in MULTI_CHANNELS)
        return (RewardChannel(name='r_track', gamma=0.99, gae_lambda=0.95),)

    def _materialize_kwargs(self) -> Dict[str, Any]:
        return dict(motion=self.motion, rand_reset=self.rand_reset,
                    episode_steps=self.episode_steps,
                    initial_distance=self.init_distance)

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True) -> List[Job]:
        env_bp = self._env_pb().materialize(**self._materialize_kwargs())
        # robot_b 不训练：用 random 陪衬（其动作不影响评分，A/B 相距 6m）
        zero_bp = PolicyBlueprint.load(_PKG_DIR / 'blueprints' / 'zero.yaml')
        sampling_a = SamplingSpec(explore_factor=0.0)
        return [Job(
            policy_a_bp=policy_bp,
            policy_b_bp=zero_bp,
            env_bp=env_bp,
            seed=int(base_seed + i),
            sampling_a=sampling_a,
            sampling_b=SamplingSpec(),
            stochastic=stochastic,
        ) for i in range(n_episodes)]

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

            reason = ep.agent_termination_reason.get("robot_a", "")
            is_terminated = bool(reason) and reason != "timeout"

            if self.multi_critic:
                channels = {}
                for c in MULTI_CHANNELS:
                    rew = extract_per_step_field(
                        ep.observer_outputs, self._OBS_KEY, c, T_full)
                    if rew is None:
                        rew = np.zeros(T_full, dtype=np.float32)
                    rew = np.asarray(rew[:T], dtype=np.float32)
                    channels[c] = ChannelData(
                        reward=rew, is_terminated=is_terminated,
                        actor_weight=np.full(T, CH_WEIGHT[c], dtype=np.float32))
            else:
                rew = extract_per_step_field(
                    ep.observer_outputs, self._OBS_KEY, 'reward', T_full)
                if rew is None:
                    rew = np.zeros(T_full, dtype=np.float32)
                rew = np.asarray(rew[:T], dtype=np.float32)
                channels = {'r_track': ChannelData(
                    reward=rew, is_terminated=is_terminated,
                    actor_weight=np.ones(T, dtype=np.float32))}

            out.append(Trajectory(
                obs=np.asarray(obs_all, dtype=np.float32)[:T],
                actions=np.asarray(acts_all, dtype=np.float32)[:T],
                last_obs=np.asarray(fin_obs, dtype=np.float32),
                channels=channels,
                importance=1.0,
                sampling_ctx=self.extract_sampling_ctx(ep, "robot_a", T),
            ))
        return out

    # ------------------------------------------------------------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survives, root_devs, key_errs = [], [], []
        falls = devs = timeouts = 0
        rewards = []
        for ep in episodes:
            T = ep.agent_frame_boundary.get("robot_a", ep.num_frames)
            if T == 0:
                continue
            reason = ep.agent_termination_reason.get("robot_a", "")
            if reason == "track_fall":
                falls += 1
            elif reason == "track_deviation":
                devs += 1
            elif reason == "timeout" or not reason:
                timeouts += 1
            survives.append(float(T) / self.episode_steps)
            rew = extract_per_step_field(
                ep.observer_outputs, self._OBS_KEY, 'reward', ep.num_frames)
            if rew is not None and len(rew):
                rewards.append(float(np.mean(rew[:T])))
            for key, acc in (('root_dev', root_devs), ('key_err', key_errs)):
                v = extract_per_step_field(
                    ep.observer_outputs, self._OBS_KEY, key, ep.num_frames)
                if v is not None and len(v):
                    acc.append(float(np.mean(v[:T])))

        n_eps = max(len(episodes), 1)
        survive = float(np.mean(survives)) if survives else 0.0
        fail_rate = (falls + devs) / n_eps
        score = survive - 0.5 * fail_rate
        is_new_best = score > self._best_score
        if is_new_best:
            self._best_score = score

        def _m(xs):
            return round(float(np.mean(xs)), 4) if xs else 0.0

        return {
            "is_new_best": is_new_best,
            "info": {
                "survive": round(survive, 3),
                "complete": round(timeouts / n_eps, 3),
                "fall": round(falls / n_eps, 3),
                "dev": round(devs / n_eps, 3),
                "reward": _m(rewards),
                "root_dev": _m(root_devs),
                "key_err": _m(key_errs),
            },
        }

    def state(self) -> dict:
        return {**super().state(), "best_score": self._best_score}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._best_score = float(state.get("best_score", -1.0))


EXPERIMENT_CLASS = DeepMimic28Track
