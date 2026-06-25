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
)
from envs.humanoid21 import Humanoid21Simulator
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver
from envs.humanoid21.plugins import CombatScoringPlugin


'''
class ImbalanceTerminationPlugin(BasePlugin):
    """检测机器人是否失衡的终止插件（物理步粒度计数）。

    失衡判定规则：在一个 action step 内的多个物理子步中，统计 "机器人除了双
    脚之外的身体部位与地面接触" 的次数。若一个 action step 内 ≥
    ``phy_hits_threshold`` 个物理步检测到此情况，则在该 action step 结束时
    请求终止。

    这种粒度的计数能抓住高频抖动场景：即使在 action step 末尾采样时身体刚好
    离地，只要中途有物理步发生了接触也会被记录。

    参数：
        agent_id: 监控的机器人ID ('robot_a' 或 'robot_b')
        force_threshold: 接触力阈值（牛顿），低于此值的接触不计数，避免误判
        phy_hits_threshold: 一个 action step 内触发终止所需的物理步命中次数
    """

    # 双脚身体名称后缀
    FOOT_BODY_NAMES = {'foot_left', 'foot_right'}

    def __init__(
        self,
        agent_id: str,
        force_threshold: float = 5.0,
        phy_hits_threshold: int = 2,
    ) -> None:
        self.agent_id = str(agent_id)
        self.force_threshold = float(force_threshold)
        self.phy_hits_threshold = max(1, int(phy_hits_threshold))
        self._phy_hits_in_action: int = 0
        self._ground_geom_name: Optional[str] = None

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "force_threshold": self.force_threshold,
            "phy_hits_threshold": self.phy_hits_threshold,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ImbalanceTerminationPlugin":
        return cls(**config)

    @property
    def name(self) -> str:
        return f"{self.agent_id}_imbalance_termination"

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._phy_hits_in_action = 0
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get('ground_geom_name', 'ground')

    def on_pre_action_step(self, ctx: SimContext) -> None:
        # 每个 action step 开始时清零物理步命中计数。
        self._phy_hits_in_action = 0

    def on_post_phy_step(self, ctx: SimContext) -> None:
        # 在每个物理子步后采样一次接触，若机器人非脚部部位接触地面则计数 +1。
        if self._is_non_foot_grounded(ctx):
            self._phy_hits_in_action += 1

    def on_post_action_step(self, ctx: SimContext) -> None:
        if self._phy_hits_in_action >= self.phy_hits_threshold:
            ctx.request_termination("imbalance")

    def _is_non_foot_grounded(self, ctx: SimContext) -> bool:
        """检查当前物理步快照下，机器人是否有非脚部部位与地面接触。

        使用 ``robot_environment_contacts``，其中每条记录已预过滤为
        "机器人身体 ↔ 环境几何体" 的接触，字段为：
          robot              : 'robot_a' / 'robot_b'
          body               : 机器人侧 body 名，如 'torso_red'
          environment_geom   : 环境侧 geom 名，如 'ground'
          force              : 接触力标量（牛顿）
        """
        derived_state = ctx.accessor.get_derived_state()
        env_contacts = derived_state.get('robot_environment_contacts', [])
        ground_geom = self._ground_geom_name or 'ground'

        for contact in env_contacts:
            if contact.get('robot') != self.agent_id:
                continue
            if contact.get('environment_geom') != ground_geom:
                continue
            if contact.get('force', 0.0) < self.force_threshold:
                continue
            body_name = contact.get('body', '')
            if not any(foot in body_name for foot in self.FOOT_BODY_NAMES):
                return True

        return False
'''

class ImbalanceTerminationPlugin(BasePlugin):

    # 双脚身体名称后缀
    FOOT_BODY_NAMES = {'foot_left', 'foot_right'}

    def __init__(
        self,
        agent_id: str,
        force_threshold: float = 1.0,
        tolerance: int = 1,
    ) -> None:
        self.agent_id = str(agent_id)
        self.force_threshold = float(force_threshold)
        self.tolerance = int(tolerance)
        self._ground_geom_name: Optional[str] = None

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "force_threshold": self.force_threshold,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ImbalanceTerminationPlugin":
        return cls(**config)

    @property
    def name(self) -> str:
        return f"{self.agent_id}_imbalance_termination"

    def on_pre_episode(self, ctx: SimContext) -> None:
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get('ground_geom_name', 'ground')
        self._imbalance_counter = {"robot_a": 0, "robot_b": 0}

    def on_post_action_step(self, ctx: SimContext) -> None:
        a_fell = self._is_non_foot_grounded(ctx, "robot_a")
        b_fell = self._is_non_foot_grounded(ctx, "robot_b")

        if a_fell:
            self._imbalance_counter["robot_a"] += 1
        else:
            self._imbalance_counter["robot_a"] = max(0, self._imbalance_counter["robot_a"] - 1)

        if b_fell:
            self._imbalance_counter["robot_b"] += 1
        else:
            self._imbalance_counter["robot_b"] = max(0, self._imbalance_counter["robot_b"] - 1)

        if self.agent_id == "both":
            a_term = self._imbalance_counter["robot_a"] >= self.tolerance
            b_term = self._imbalance_counter["robot_b"] >= self.tolerance
            if a_term and b_term:
                ctx.request_termination("imbalance_both")
            elif a_term:
                ctx.request_termination("imbalance_robot_a")
            elif b_term:
                ctx.request_termination("imbalance_robot_b")
        else:
            if self._imbalance_counter[self.agent_id] >= self.tolerance:
                ctx.request_termination("imbalance")

    def _is_non_foot_grounded(self, ctx: SimContext, robot_id: str) -> bool:
        """检查当前物理步快照下，指定机器人是否有非脚部部位与地面接触 — 向量化版本。

        使用 ``contacts_vec``，通过 aff 筛选机器人↔环境接触，
        通过 geom_id_to_name 确认地面 geom，通过 body_id_to_name 获取 body 名。
        """
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')
        if cv is None or cv['ncon'] == 0:
            return False

        static_data = ctx.accessor.get_static_data()
        body_id_to_name = static_data.get('body_id_to_name', {})
        geom_id_to_name = static_data.get('geom_id_to_name', {})
        ground_geom = self._ground_geom_name or 'ground'

        # robot_id → aff code
        robot_aff = 1 if robot_id == 'robot_a' else 2

        aff1 = cv['aff1']
        aff2 = cv['aff2']
        geom1 = cv['geom1']
        geom2 = cv['geom2']
        body1 = cv['body1']
        body2 = cv['body2']
        force_mag = cv['force_mag']

        for i in range(cv['ncon']):
            # One side is env (aff=0), other side is the target robot
            if aff1[i] == 0 and aff2[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom1[i]), '')
                body_robot = body_id_to_name.get(int(body2[i]), '')
            elif aff2[i] == 0 and aff1[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom2[i]), '')
                body_robot = body_id_to_name.get(int(body1[i]), '')
            else:
                continue

            if geom_env != ground_geom:
                continue
            if float(force_mag[i]) < self.force_threshold:
                continue
            if not any(foot in body_robot for foot in self.FOOT_BODY_NAMES):
                return True

        return False

