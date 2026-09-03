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

from baseline.framework.ppo.policies import (
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
from envs.humanoid21 import Humanoid21Simulator
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver
from envs.humanoid21.plugins import CombatScoringPlugin

class StandingTerminationPlugin(BasePlugin):
    """Terminate when the agent has fallen for ``fall_grace_steps`` in a row.

    "Fallen" = below ``fall_height_threshold`` OR below
    ``fall_upright_threshold`` (cosine of tilt). The grace window
    avoids spurious triggers from physics jitter.
    """

    def __init__(
        self,
        agent_id: str,
        fall_height_threshold: float = 1.10,
        fall_upright_threshold: float = 0.8,
        fall_grace_steps: int = 3,
    ) -> None:
        self.agent_id = str(agent_id)
        self.fall_height_threshold = float(fall_height_threshold)
        self.fall_upright_threshold = float(fall_upright_threshold)
        self.fall_grace_steps = max(1, int(fall_grace_steps))
        self._streak = 0

    @property
    def name(self) -> str:
        return f"{self.agent_id}_standing_termination"

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "fall_height_threshold": self.fall_height_threshold,
            "fall_upright_threshold": self.fall_upright_threshold,
            "fall_grace_steps": self.fall_grace_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandingTerminationPlugin":
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._streak = 0

    def on_post_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )
        is_standing = (
            height >= self.fall_height_threshold
            and uprightness >= self.fall_upright_threshold
        )
        self._streak = 0 if is_standing else self._streak + 1
        if self._streak >= self.fall_grace_steps:
            ctx.request_termination(TerminationReason.CUSTOM)

