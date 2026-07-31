"""Generic composite policy: primary policy + frozen standup fallback.

A two-state state machine that switches between an arbitrary primary policy
and a frozen standup policy based on robot height:

    primary (any policy)  ←→  standup (frozen)

Transitions:
    any      + height < fall_height      -> standup
    standup  + height > stand_height     -> primary

No Gating MLP, no recover mode, no timeout — just height-based switching.

gating_mode values in action_extras:
    1.0  -> primary  (trainable, this is what PPO trains on)
    -2.0 -> standup  (frozen fallback, excluded from training)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.framework.policy import Policy, PolicyBlueprint

_DEBUG = os.environ.get("STANDUP_FALLBACK_POLICY_DEBUG", "0") == "1"

_OBS_HEIGHT = 48  # root Z height in humanoid21 96-dim observation (after 42 proprio + 6 orientation)


class StandupFallbackPolicy(Policy):
    """Two-way switching composite: primary + standup fallback."""

    def __init__(
        self,
        primary_policy_bp: str | Dict[str, Any] | PolicyBlueprint = "/data1/mono/things/combatbench/policy/blueprints/random.yaml",
        standup_policy_bp: str | Dict[str, Any] | PolicyBlueprint = "/data1/mono/things/combatbench/baseline/runs/train_standing_balance_4stage_dense_ppo_resume5k_20260730_211100/policy_exports/u04935/policy_blueprint.yaml",
        fall_height: float = 0.5,
        stand_height: float = 1.25,
        **kwargs: Any,
    ) -> None:
        self.primary_policy_bp = self._resolve_bp(primary_policy_bp)
        self.primary_policy = self.primary_policy_bp.build()

        self.standup_policy_bp = self._resolve_bp(standup_policy_bp)
        self.standup_policy = self.standup_policy_bp.build()

        self.fall_height = float(fall_height)
        self.stand_height = float(stand_height)

        self.active_mode = "primary"
        self._step_count = 0

    @staticmethod
    def _resolve_bp(bp: str | Dict[str, Any] | PolicyBlueprint) -> PolicyBlueprint:
        if isinstance(bp, PolicyBlueprint):
            return bp
        if isinstance(bp, str):
            return PolicyBlueprint.load(bp)
        return PolicyBlueprint.from_dict(bp)

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        height = float(observation[_OBS_HEIGHT])

        prev_mode = self.active_mode

        if height < self.fall_height:
            new_mode = "standup"
        elif self.active_mode == "standup":
            if height > self.stand_height:
                new_mode = "primary"
            else:
                new_mode = "standup"
        else:
            new_mode = "primary"

        if new_mode != self.active_mode:
            if new_mode == "standup":
                if hasattr(self.standup_policy, "reset"):
                    self.standup_policy.reset()
            elif new_mode == "primary":
                if hasattr(self.primary_policy, "reset"):
                    self.primary_policy.reset()
            self.active_mode = new_mode

        self._step_count += 1
        if self.active_mode != prev_mode or self._step_count % 20 == 0:
            print(
                f"[StandupFallbackPolicy] step={self._step_count} "
                f"h={height:.3f}m fall_h={self.fall_height} stand_h={self.stand_height} "
                f"mode={prev_mode}->{self.active_mode}",
                flush=True,
            )

        if self.active_mode == "primary":
            action, extra = self.primary_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = 1.0
        else:
            action, extra = self.standup_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = -1.0

        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        if hasattr(self.primary_policy, "reset"):
            self.primary_policy.reset(seed)
        if hasattr(self.standup_policy, "reset"):
            self.standup_policy.reset(seed)
        self.active_mode = "primary"
        self._step_count = 0

    def close(self) -> None:
        if hasattr(self.primary_policy, "close"):
            self.primary_policy.close()
        if hasattr(self.standup_policy, "close"):
            self.standup_policy.close()
