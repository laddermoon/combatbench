"""Reusable posture-score building blocks for ``humanoid21`` standing experiments.

Provides:

  * :class:`StandingPostureRewarder` — instantaneous posture score
    (height + uprightness + drift + joint pose/vel penalties).
  * :class:`StandingPostureDeltaRewarder` — score *delta* vs. the
    previous step, which is the per-step reward used by GRPO-RTG / PPO.

Hook conventions
----------------
Observers use the framework's canonical dispatch hooks:
``on_pre_episode`` / ``on_post_action_step``.  Earlier revisions used
legacy names (``on_reset`` / ``on_post_step``) — those are NOT
dispatched by the current
:class:`envs.framework.observer_plugin._ObserverDispatcherPlugin`, so
observers wired with them silently returned their initial output for
the entire episode.  When in doubt, grep ``observer_plugin.py`` for
the canonical hook names.
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


# Posture-score shaping (full-penalty = 1.0 normalization scale).
STANDING_SCORE_MAX = 1.0
TARGET_HEIGHT = 1.28
HEIGHT_FULL_PENALTY_DELTA = 0.20
UPRIGHT_TILT_FULL_PENALTY_DEGREES = 30.0
ROOT_XY_FULL_PENALTY_DISTANCE = 1.5
JOINT_POSE_FULL_PENALTY_MEAN_ABS = 0.2
JOINT_VEL_FULL_PENALTY_MEAN_ABS = 1.0


# ---------------------------------------------------------------------------
# Observer plugins — posture rewards
# ---------------------------------------------------------------------------
def _read_posture_state(
    ctx: ReadOnlySimContext, agent_id: str,
) -> Dict[str, float]:
    """Pull the four scalars/arrays the posture score depends on."""
    core_state = ctx.accessor.get_core_state()[agent_id]
    derived_state = ctx.accessor.get_derived_state()[agent_id]
    return {
        "height": float(core_state["root_pos"][2]),
        "uprightness": float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        ),
        "root_xy": np.asarray(core_state["root_pos"][:2], dtype=np.float32),
        "joint_pos": np.asarray(core_state["joint_pos_norm"], dtype=np.float32),
        "joint_vel": np.asarray(core_state["joint_vel_norm"], dtype=np.float32),
    }


def _posture_score(
    state: Dict[str, Any],
    *,
    reference_root_xy: np.ndarray,
    reference_joint_pos: np.ndarray,
) -> float:
    """Bit-identical to v1/v2's ``_compute_posture_terms['total_score']``.

    Returns a scalar in roughly ``[-N, 1.0]`` — 1.0 means perfect match
    of the standing reference; each penalty term squares the normalized
    deviation, so penalties grow quickly past their full-penalty scale.
    """
    root_xy_distance = float(
        np.linalg.norm(state["root_xy"] - reference_root_xy)
    )
    joint_pose_mean_abs = float(
        np.mean(np.abs(state["joint_pos"] - reference_joint_pos))
    )
    joint_velocity_mean_abs = float(np.mean(np.abs(state["joint_vel"])))
    height_deficit = max(0.0, TARGET_HEIGHT - state["height"])
    tilt_angle_degrees = float(
        np.degrees(np.arccos(np.clip(state["uprightness"], -1.0, 1.0)))
    )
    height_penalty = (height_deficit / HEIGHT_FULL_PENALTY_DELTA) ** 2
    upright_penalty = (tilt_angle_degrees / UPRIGHT_TILT_FULL_PENALTY_DEGREES) ** 2
    drift_penalty = (root_xy_distance / ROOT_XY_FULL_PENALTY_DISTANCE) ** 2
    pose_penalty = (joint_pose_mean_abs / JOINT_POSE_FULL_PENALTY_MEAN_ABS) ** 2
    vel_penalty = (joint_velocity_mean_abs / JOINT_VEL_FULL_PENALTY_MEAN_ABS) ** 2
    total_penalty = (
        height_penalty + upright_penalty + drift_penalty
        + pose_penalty + vel_penalty
    )
    return float(STANDING_SCORE_MAX - total_penalty)


class StandingPostureRewarder(BaseObserverPlugin):
    """Per-step **absolute** posture score (1.0 = perfect standing).

    Output is ``_posture_score`` evaluated against the per-episode
    reference root-XY / joint-pose snapshot taken at episode start.
    Use :class:`StandingPostureDeltaRewarder` instead if you want a
    per-step reward signal (which is what GRPO-RTG and PPO want).
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self._reference_root_xy: Optional[np.ndarray] = None
        self._reference_joint_pos: Optional[np.ndarray] = None
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        state = _read_posture_state(ctx, self.agent_id)
        self._reference_root_xy = state["root_xy"].copy()
        self._reference_joint_pos = state["joint_pos"].copy()
        self._output = _posture_score(
            state,
            reference_root_xy=self._reference_root_xy,
            reference_joint_pos=self._reference_joint_pos,
        )

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        # Should not happen but guard for safety: if the framework ever
        # dispatches post-step before pre-episode, fall back to the
        # current step's snapshot as the reference.
        state = _read_posture_state(ctx, self.agent_id)
        if self._reference_root_xy is None:
            self._reference_root_xy = state["root_xy"].copy()
            self._reference_joint_pos = state["joint_pos"].copy()
        self._output = _posture_score(
            state,
            reference_root_xy=self._reference_root_xy,
            reference_joint_pos=self._reference_joint_pos,
        )

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandingPostureRewarder":
        return cls(**config)


class StandingPostureDeltaRewarder(BaseObserverPlugin):
    """Per-step posture-score *delta* — the GRPO-RTG / PPO reward signal.

    Reward at step ``t`` is ``score_t - score_{t-1}``; the first step
    after reset returns 0.0 by convention (no prior score to diff
    against).
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self._reference_root_xy: Optional[np.ndarray] = None
        self._reference_joint_pos: Optional[np.ndarray] = None
        self._previous_score: float = 0.0
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        state = _read_posture_state(ctx, self.agent_id)
        self._reference_root_xy = state["root_xy"].copy()
        self._reference_joint_pos = state["joint_pos"].copy()
        self._previous_score = _posture_score(
            state,
            reference_root_xy=self._reference_root_xy,
            reference_joint_pos=self._reference_joint_pos,
        )
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        state = _read_posture_state(ctx, self.agent_id)
        score = _posture_score(
            state,
            reference_root_xy=self._reference_root_xy,
            reference_joint_pos=self._reference_joint_pos,
        )
        self._output = float(score - self._previous_score)
        self._previous_score = score

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandingPostureDeltaRewarder":
        return cls(**config)

