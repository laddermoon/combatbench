"""Reusable building blocks for ``humanoid21`` standing/balance experiments.

Layout (kept deliberately flat):

  * **Hyperparameters** — ``StandingConfig`` dataclass bundles every
    knob a standing trainer typically wants to override per run. Env
    constants that affect the runtime wiring itself (``MAX_STEPS``,
    ``CONTROL_FREQUENCY``, fall thresholds, ...) live as module-level
    constants because :func:`make_standing_runtime` reads them.

  * **Observer plugins (rewards)**:
      - :class:`StandingPostureRewarder` — instantaneous posture score
        (height + uprightness + drift + joint pose/vel penalties).
      - :class:`StandingPostureDeltaRewarder` — score *delta* vs. the
        previous step, which is the per-step reward used by GRPO-RTG.
      - :class:`BalanceValueRewarder` / :class:`BalanceValueDeltaRewarder`
        — same pair built on the support-polygon balance analysis.

  * **Termination plugins**:
      - :class:`StandingTerminationPlugin` — fall detection (height +
        uprightness streak).
      - :class:`BalanceScoreTerminationPlugin` — persistently low
        balance score.

  * **Top-level factories** (picklable for ``RolloutCollector`` / the
    parallel rollout pool under ``spawn``):
      - :func:`make_standing_runtime`
      - :func:`make_standing_policy`
      - :func:`make_standing_options_fn`
      - :func:`set_seed`

Hook conventions
----------------
Every observer here uses the framework's *current* dispatch hooks:
``on_pre_episode`` / ``on_post_action_step`` / ``on_post_episode``.
Earlier revisions of this file used legacy hook names (``on_reset`` /
``on_post_step``) — those are NOT dispatched by
:class:`envs.framework.observer_plugin._ObserverDispatcherPlugin`, so
observers wired with them silently returned their initial output for
the entire episode (see the long bug-fix block in
``standing_grpo_rtg_tune_v2.py`` for the diagnosis). When in doubt,
grep ``observer_plugin.py`` for the canonical hook names.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from baseline.common.policies import (
    DEFAULT_LOG_STD_MAX,
    DEFAULT_LOG_STD_MIN,
    TanhGaussianMLPPolicy,
)
from envs.framework import (
    BaseObserverPlugin,
    BasePlugin,
    EnvRuntime,
    ReadOnlySimContext,
    SimContext,
    TerminationReason,
)
from envs.humanoid21 import MujocoCombatSimulator
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver
from envs.humanoid21.plugins import CombatScoringPlugin



class ImbalanceTerminationPlugin(BasePlugin):
    """检测机器人是否失衡的终止插件

    失衡判定规则：当机器人除了双脚之外的第三点与地面接触时，判定为失衡。
    这是课程学习第一阶段的终止条件。

    参数：
        agent_id: 监控的机器人ID ('robot_a' 或 'robot_b')
        force_threshold: 接触力阈值（牛顿），低于此值的接触不计数，避免误判
        grace_steps: 宽容步数，连续 N 步失衡才触发终止
    """

    # 双脚身体名称后缀
    FOOT_BODY_NAMES = {'foot_left', 'foot_right'}

    def __init__(
        self,
        agent_id: str,
        force_threshold: float = 5.0,
        grace_steps: int = 2,
    ) -> None:
        self.agent_id = str(agent_id)
        self.force_threshold = float(force_threshold)
        self.grace_steps = max(1, int(grace_steps))
        self._streak = 0
        self._ground_geom_name: Optional[str] = None

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "force_threshold": self.force_threshold,
            "grace_steps": self.grace_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ImbalanceTerminationPlugin":
        return cls(**config)

    @property
    def name(self) -> str:
        return f"{self.agent_id}_imbalance_termination"

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._streak = 0
        # 获取地面 geom 名称
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get('ground_geom_name', 'ground')

    def on_post_action_step(self, ctx: SimContext) -> None:
        derived_state = ctx.accessor.get_derived_state()
        contacts = derived_state.get('contacts', [])

        # 统计该机器人与地面的接触
        ground_contact_bodies = set()
        for contact in contacts:
            geom_a = contact.get('geom_a_name', '')
            geom_b = contact.get('geom_b_name', '')
            force = contact.get('force_magnitude', 0.0)

            # 跳过力太小的接触
            if force < self.force_threshold:
                continue

            # 检查是否是与地面的接触
            if self._ground_geom_name not in (geom_a, geom_b):
                continue

            # 获取接触的身体名称
            body_a = contact.get('body_a_name', '')
            body_b = contact.get('body_b_name', '')

            # 判断哪个身体属于该机器人
            robot_suffix = '_red' if self.agent_id == 'robot_a' else '_blue'
            for body_name in (body_a, body_b):
                if body_name and body_name.endswith(robot_suffix):
                    # 提取基础名称（去掉后缀）
                    base_name = body_name[:-len(robot_suffix)] if robot_suffix else body_name
                    ground_contact_bodies.add(base_name)

        # 统计非脚部的接触点数量
        non_foot_contacts = 0
        for body_name in ground_contact_bodies:
            # 检查是否是脚部
            if not any(foot_name in body_name for foot_name in self.FOOT_BODY_NAMES):
                non_foot_contacts += 1

        # 如果有第三个点接触地面，判定为失衡
        is_imbalanced = non_foot_contacts > 0
        self._streak = 0 if not is_imbalanced else self._streak + 1

        if self._streak >= self.grace_steps:
            ctx.request_termination(TerminationReason.CUSTOM)

