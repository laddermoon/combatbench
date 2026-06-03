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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class ExperimentConfig(ABC):
    """Abstract per-experiment configuration.

    Subclass this for each reward scheme / curriculum strategy.
    """

    # --- Class-level attributes (set by subclass) ---
    name: str = ""
    reward_keys: Tuple[str, ...] = ()
    gammas: Dict[str, float] = {}
    env_blueprint: str = ""  # filename relative to blueprints/
    ppo_overrides: Dict[str, Any] = {}  # optional overrides for TrainConfig

    # Rollout distance range (can be overridden per experiment)
    rollout_distance_min: float = 1.5
    rollout_distance_max: float = 3.5

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
    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
    ) -> Dict[str, np.ndarray]:
        """Extract per-step reward arrays from observer outputs.

        Parameters
        ----------
        observer_outputs : dict
            The ``Episode.observer_outputs`` dict.
        T : int
            Episode length (number of steps).

        Returns
        -------
        dict mapping reward key (str) -> np.ndarray of shape (T,).
        Do NOT include ``r_fall`` — the framework injects it automatically.
        """
        ...

    @abstractmethod
    def compute_episode_metrics(
        self,
        observer_outputs: dict,
        T: int,
        terminated: bool,
    ) -> Dict[str, float]:
        """Compute aggregate metrics for one episode (used for eval & logging)."""
        ...

    @abstractmethod
    def scheduler_info(self) -> Dict[str, Any]:
        """Return extra info dict for logging (phase, consecutive_pass, etc.)."""
        ...

    # --- Blueprint ownership ---
    #
    # The experiment OWNS the env blueprint lifecycle: which file to use,
    # loading, caching, and switching across curriculum stages. The training
    # loop never touches blueprints — it only calls ``build_rollout_jobs``.

    @staticmethod
    def blueprint_dir() -> Path:
        """Directory containing env blueprint YAML files."""
        return Path(__file__).resolve().parent.parent.parent / "blueprints"

    def current_env_blueprint(self) -> str:
        """Return the active env blueprint filename for the current stage.

        Default returns the static ``env_blueprint``. Stateful experiments
        (e.g. multi-stage curricula) override this to return a different
        blueprint depending on their internal scheduler state.
        """
        return self.env_blueprint

    def _get_env_pb(self) -> ParameterizedEnvBlueprint:
        """Load (and cache) the ParameterizedEnvBlueprint for the active stage.

        Cached per blueprint filename, so switching back and forth between
        stages does not re-read from disk.
        """
        name = self.current_env_blueprint()
        cache: Dict[str, ParameterizedEnvBlueprint] = self.__dict__.setdefault(
            "_env_pb_cache", {}
        )
        if name not in cache:
            cache[name] = ParameterizedEnvBlueprint.load(self.blueprint_dir() / name)
        return cache[name]

    # --- Rollout job construction ---

    @staticmethod
    def _agent_from_rollout_seed(seed: int) -> str:
        rng = np.random.default_rng(int(seed) + 937)
        return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"

    def build_rollout_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        max_steps: int,
        *,
        policy_bp_b: PolicyBlueprint | None = None,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Build rollout job list.

        Default implementation: self-play with random agent_id assignment
        and random initial distance. Override for asymmetric policies,
        fixed opponent, or other rollout strategies.

        The env blueprint is resolved internally via ``_get_env_pb()`` so the
        correct (possibly stage-dependent) blueprint is used each call.

        Parameters
        ----------
        policy_bp : PolicyBlueprint
            Policy blueprint for agent A (and agent B if ``policy_bp_b`` is None).
        base_seed : int
            Base RNG seed for this batch.
        n_episodes : int
            Number of episodes to prepare.
        max_steps : int
            Episode horizon.
        policy_bp_b : PolicyBlueprint or None
            Policy for agent B. If None, uses ``policy_bp`` (self-play).
        """
        env_pb = self._get_env_pb()
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=max_steps, agent_id=aid)
            for aid in ("robot_a", "robot_b")
        }

        bp_b = policy_bp_b if policy_bp_b is not None else policy_bp

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(self.rollout_distance_min, self.rollout_distance_max)
            )
            jobs.append((
                policy_bp, bp_b,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

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
            "env_blueprint": self.env_blueprint,
            "ppo_overrides": self.ppo_overrides,
            "initial_weights": list(self.initial_weights()),
            "rollout_distance_min": self.rollout_distance_min,
            "rollout_distance_max": self.rollout_distance_max,
        }

    # --- Optional state persistence ---

    def scheduler_state(self) -> dict:
        """Serialize mutable scheduler state for checkpointing."""
        return {}

    def load_scheduler_state(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        pass
