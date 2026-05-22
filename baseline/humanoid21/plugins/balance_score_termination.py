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
from envs.humanoid21 import Humanoid21Observer, MujocoCombatSimulator
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver
from envs.humanoid21.plugins import CombatScoringPlugin


class BalanceScoreTerminationPlugin(BasePlugin):
    """Terminate when the balance score stays below threshold for N steps.

    Designed for tasks where the height/upright termination would
    conflict with optimization (e.g. a slightly lower stance can still
    be perfectly balanced). Watches the same absolute score that
    :class:`BalanceValueRewarder` produces.
    """

    def __init__(
        self,
        agent_id: str,
        score_threshold: float = 0.3,
        grace_steps: int = 3,
    ) -> None:
        self.agent_id = str(agent_id)
        self.score_threshold = float(score_threshold)
        self.grace_steps = max(1, int(grace_steps))
        self._inner = Humanoid21BalanceAnalysisObserver(agent_id)
        self._streak = 0

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "score_threshold": self.score_threshold,
            "grace_steps": self.grace_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "BalanceScoreTerminationPlugin":
        return cls(**config)

    @property
    def name(self) -> str:
        return f"{self.agent_id}_balance_score_termination"

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._inner.on_pre_episode(ctx)
        self._streak = 0

    def on_post_action_step(self, ctx: SimContext) -> None:
        self._inner.on_post_action_step(ctx)
        out = self._inner.get_output()
        score = (
            float(_compute_balance_value_terms(out)["absolute_score"])
            if isinstance(out, dict) else -1
        )
        self._streak = self._streak + 1 if score < self.score_threshold else 0
        if self._streak >= self.grace_steps:
            ctx.request_termination(TerminationReason.CUSTOM)

