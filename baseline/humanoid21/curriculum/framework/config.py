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
from typing import Any, Dict, Tuple

import numpy as np


class ExperimentConfig(ABC):
    """Abstract per-experiment configuration.

    Subclass this for each reward scheme / curriculum strategy.
    """

    # --- Class-level attributes (set by subclass) ---
    name: str = ""
    reward_keys: Tuple[str, ...] = ()
    gammas: Dict[str, float] = {}
    env_blueprint: str = ""  # filename relative to blueprints/
    ppo_overrides: Dict[str, Any] = {}  # optional overrides for CurriculumConfig

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

    # --- Optional state persistence ---

    def scheduler_state(self) -> dict:
        """Serialize mutable scheduler state for checkpointing."""
        return {}

    def load_scheduler_state(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        pass
