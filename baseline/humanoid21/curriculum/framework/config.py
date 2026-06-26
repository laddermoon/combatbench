"""ExperimentConfig — 课程学习 (Curriculum Learning) 的实验抽象基类.

这个框架服务于**课程学习范式**：将一个复杂的格斗任务拆解为多个阶段（stage），
从简单技能开始逐步叠加难度，让 agent 循序渐进地掌握完整能力。

核心思路
--------
以人形机器人格斗为例，完整能力 = 保持平衡 + 接近对手 + 造成伤害。
直接用全部奖励信号训练往往失败（credit assignment 困难、初期信号噪声大），
因此采用课程式分阶段训练：

  1. Stage 1 — 只练平衡（r_fall + r_cross），不倒地是第一优先级
  2. Stage 2 — 加入接近（+ r_relation 或 r_hold/r_radial），学会走向对手
  3. Stage 3 — 加入战斗（+ r_damage），在平衡和接近的基础上练出拳

每个阶段通过 ``stage_weights`` 控制各 reward 分量的权重，实现"先易后难"。
``next_weights()`` 根据评估指标（episode 平均长度、是否到达对手区域）自动
决定何时升级/降级阶段。

为什么要抽象
------------
V1 用 4 个 reward（r_relation 做接近信号），V2 用 6 个 reward
（r_hold/r_radial/r_tangential 做接近信号）。奖励方案不同，但 PPO 训练
循环、checkpoint、logging 完全相同。将差异封装在 ExperimentConfig 里，
框架代码只需写一次，新增实验只需一个文件。

每个实验（如 v1_relation、v2_follow）通过实现本 ABC 来指定：
  * reward keys 和 discount factors（每个 critic 独立 γ）
  * 环境蓝图文件名
  * 权重调度策略（initial_weights + next_weights）
  * 从 rollout observer 输出中提取 reward 的逻辑
  * episode 级别的评估指标（用于课程阶段判定）
  * 调度器状态持久化（支持有状态调度器如 v2_follow）
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class ExperimentConfig(ABC):
    """Abstract per-experiment configuration.

    Subclass this for each reward scheme / curriculum strategy.
    """

    # --- Identity ---
    name: str = ""
    reward_keys: Tuple[str, ...] = ()
    gammas: Dict[str, float] = {}

    # --- Network shape ---
    obs_dim: int = 96
    action_dim: int = 21
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    log_std_min: float = -4.0
    log_std_max: float = 0.0

    # --- GAE ---
    gae_lambda: float = 0.95

    # --- PPO knobs ---
    learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 8

    # --- Rollout schedule ---
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Parallelism ---
    rollout_workers: int = max(1, (os.cpu_count() or 1) // 2)
    eval_workers: int = max(1, (os.cpu_count() or 1) // 4)

    seed: int = 42

    # --- Free-form experiment-specific parameters ---
    #
    # Knobs that vary per-experiment (not framework-level) live here as a plain
    # key-value dict.  No fixed schema — each experiment declares only what it
    # needs.  The defaults below cover the parameters used by the built-in
    # ``build_rollout_jobs`` implementation.  Subclasses override specific
    # entries by merging::
    #
    #     custom_config = {**ExperimentConfig.DEFAULT_CUSTOM_CONFIG, "max_steps": 400}
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        # Rollout initial-distance sampling range
        "rollout_distance_min": 1.5,
        "rollout_distance_max": 3.5,
        # Runtime horizon (20 Hz × 10 s)
        "max_steps": 200,
        # Terminal fall penalty
        "terminal_fall_penalty": 1.0,
    }

    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # --- Abstract methods ---

    @abstractmethod
    def initial_weights(self) -> Tuple[float, ...]:
        """Return the initial stage-weight tuple (one entry per reward_key)."""
        ...

    @abstractmethod
    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Return the next stage-weight tuple given eval metrics.

        Parameters
        ----------
        eval_metrics : dict
            Aggregated eval metrics. Always contains ``mean_length`` and
            ``final_in_zone_ratio``. May contain additional experiment-specific
            keys from ``compute_episode_metrics``.
        current_weights : tuple of float
            Current stage weights (same length as ``reward_keys``).
        """
        ...

    @abstractmethod
    def extract_rewards(self, episode: "Episode") -> Dict[str, np.ndarray]:
        """Extract per-step reward arrays from an episode.

        Parameters
        ----------
        episode : Episode
            The completed episode object.  Use
            ``episode.observer_outputs``, ``episode.num_frames`` (T),
            and ``episode.termination_proposals`` as needed.

        Returns
        -------
        dict mapping reward key (str) -> np.ndarray of shape (T,).
        """
        ...

    @abstractmethod
    def compute_episode_metrics(self, episode: "Episode") -> Dict[str, float]:
        """Compute aggregate metrics for one episode (used for eval & logging).

        Parameters
        ----------
        episode : Episode
            The completed episode object.  Use
            ``episode.termination_proposals`` to distinguish why the episode
            ended (e.g. ``"imbalance"`` for a fall vs ``"timeout"``).
            Empty tuple means the episode was truncated externally without a
            specific reason.
        """
        ...

    def segment_episode(self, episode: "Episode") -> List[Tuple[int, int]]:
        """Return ``(start, end)`` index pairs delimiting training segments.

        The PPO buffer splits each episode into sub-episodes along these
        boundaries and computes GAE independently per segment.  This is
        essential when a mixed policy is used: steps where the fallback
        policy was active must be excluded so the actor is not trained to
        imitate fallback actions.

        Default: return the full episode as a single segment ``[(0, T)]``.
        Override to exclude fallback steps (e.g. via ``action_extras``).
        Return an empty list to skip the episode entirely.
        """
        return [(0, episode.num_frames)]


    @abstractmethod
    def scheduler_info(self) -> Dict[str, Any]:
        """Return extra info dict for logging (phase, consecutive_pass, etc.)."""
        ...

    @abstractmethod
    def compare_eval(self, esum: Dict[str, float], best_esum: Dict[str, float]) -> bool:
        """Return True if esum is better than best_esum.

        Parameters
        ----------
        esum, best_esum:
            Batch summary dicts returned by ``compute_episode_metrics()``.
            Typical keys: ``mean_length``, ``in_zone``, ``survival_rate``, etc.
        """
        ...

    # --- Blueprint ownership ---
    #
    # The experiment OWNS the env blueprint lifecycle. The training loop never
    # touches blueprints — it only calls ``build_rollout_jobs``.

    def _make_video_blueprint(self, env_pb: ParameterizedEnvBlueprint) -> EnvBlueprint:
        """Materialize *env_pb* with video-appropriate defaults.

        Uses the experiment's ``max_steps`` from ``custom_config`` and fixes
        ``agent_id`` to ``"robot_a"``.  Subclasses may call this from their
        ``video_env_blueprint()`` implementation.
        """
        return env_pb.materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id="robot_a",
        )

    @abstractmethod
    def video_env_blueprint(self) -> EnvBlueprint:
        """Return the env blueprint to use for video rendering.

        Implementations should load a :class:`ParameterizedEnvBlueprint` and
        call ``self._make_video_blueprint(env_pb)`` to obtain a concrete,
        serialisable :class:`EnvBlueprint`.
        """
        ...

    # --- Rollout job construction ---

    @staticmethod
    def _agent_from_rollout_seed(seed: int) -> str:
        rng = np.random.default_rng(int(seed) + 937)
        return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"

    def _build_selfplay_jobs(
        self,
        env_pb: ParameterizedEnvBlueprint,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Self-play helper: random agent_id + random initial distance.

        Subclasses that want standard self-play behaviour call this from their
        ``build_rollout_jobs`` implementation, passing the loaded blueprint.
        """
        max_steps = self.custom_config["max_steps"]
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=max_steps, agent_id=aid)
            for aid in ("robot_a", "robot_b")
        }

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(
                    self.custom_config["rollout_distance_min"],
                    self.custom_config["rollout_distance_max"],
                )
            )
            jobs.append((
                policy_bp, policy_bp,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    @abstractmethod
    def build_rollout_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Build rollout jobs for one training update (uses self.episodes_per_update)."""
        ...

    @abstractmethod
    def build_eval_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Build rollout jobs for evaluation (uses self.eval_episodes)."""
        ...

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize experiment config to a JSON-friendly dict.

        Subclasses with extra config fields should override this and
        call ``super().to_dict()`` then add their own keys.
        """
        return {
            "name": self.name,
            "reward_keys": list(self.reward_keys),
            "gammas": self.gammas,
            "initial_weights": list(self.initial_weights()),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_hidden_dim": self.actor_hidden_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
            "gae_lambda": self.gae_lambda,
            "custom_config": dict(self.custom_config),
            "learning_rate": self.learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "clip_eps": self.clip_eps,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "grad_clip_norm": self.grad_clip_norm,
            "target_kl": self.target_kl,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "episodes_per_update": self.episodes_per_update,
            "max_updates": self.max_updates,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "video_eval_interval": self.video_eval_interval,
            "video_env_blueprint": self.video_env_blueprint(),
            "rollout_workers": self.rollout_workers,
            "eval_workers": self.eval_workers,
            "seed": self.seed,
        }

    def save_run_config(self, run_dir: Path, *, smoke: bool = False) -> None:
        """Save a full config snapshot to ``run_dir/config.json``."""
        payload = {
            "experiment": self.to_dict(),
            "smoke": smoke,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    # --- Optional state persistence ---

    def scheduler_state(self) -> dict:
        """Serialize mutable scheduler state for checkpointing."""
        return {}

    def load_scheduler_state(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        pass

    def training_state(self) -> dict:
        """Serialize training hyperparameters for checkpointing."""
        return {
            "learning_rate": self.learning_rate,
        }

    def load_training_state(self, state: dict) -> None:
        """Restore training hyperparameters from a checkpoint.
        Disabled loading learning_rate to ensure configuration-specified LR takes precedence.
        """
        pass
