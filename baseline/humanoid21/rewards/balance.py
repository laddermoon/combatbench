"""Balance analysis reward plugins for humanoid21.

Provides:
  * :class:`BalanceValueRewarder` — Per-step absolute balance score from
    support-polygon analysis.
  * :class:`BalanceValueDeltaRewarder` — Per-step balance-score delta.

Hook conventions
----------------
Observers use the framework's canonical dispatch hooks:
``on_pre_episode`` / ``on_post_action_step``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver


# ---------------------------------------------------------------------------
# Balance-score shaping constants
# ---------------------------------------------------------------------------
BALANCE_INVALID_SCORE = -1.0
BALANCE_FRONT_PENALTY_COEF = 2.0
BALANCE_BACK_PENALTY_COEF = 6.0
# Safe-zone is along the support-lateral axis (positive = forward of the
# ankle-to-ankle line toward the toes; negative = behind toward the heels).
# BACK margin can be negative: once the CoM is at/behind the ankle line a
# tiny backward perturbation already tips the robot, so full reward
# requires CoM to be at least |BACK_MARGIN| forward of the ankle line
# when BACK_MARGIN < 0.
BALANCE_SAFE_FRONT_MARGIN = 0.10
BALANCE_SAFE_BACK_MARGIN = -0.02
BALANCE_CENTER_OFFSET_PENALTY_COEF = 1.0
BALANCE_SUPPORT_AXIS_VELOCITY_COEF = 0.25
BALANCE_SUPPORT_LATERAL_VELOCITY_COEF = 0.5
BALANCE_VELOCITY_CLIP = 1.5
BALANCE_SCORE_CLIP_MIN = -4.0
BALANCE_SCORE_CLIP_MAX = 1.0
BALANCE_TERMINATION_SCORE_THRESHOLD = 0.3
BALANCE_TERMINATION_GRACE_STEPS = 3


# ---------------------------------------------------------------------------
# Observer plugins — balance-analysis rewards
# ---------------------------------------------------------------------------
def _compute_balance_value_terms(
    balance_output: Dict[str, Any],
) -> Dict[str, float]:
    """Score a balance-analysis snapshot in roughly ``[-4, 1]``.

    Returns ``BALANCE_INVALID_SCORE`` when the support-polygon frame
    isn't well-defined (e.g. one foot in the air), otherwise a clipped
    affine combination of penalty / velocity-toward-center terms.
    """
    if not bool(balance_output.get("ground_support_frame_defined", False)):
        return _balance_invalid_terms()
    support_span = float(balance_output["support_span"])
    support_axis_proj = float(balance_output["support_axis_projection_coordinate"])
    support_lat_dist = float(balance_output["support_lateral_signed_distance"])
    support_axis_vel = float(balance_output["center_of_mass_velocity_along_support_axis"])
    support_lat_vel = float(balance_output["center_of_mass_velocity_along_support_lateral_axis"])
    required = np.asarray(
        [support_span, support_axis_proj, support_lat_dist,
         support_axis_vel, support_lat_vel],
        dtype=np.float64,
    )
    if support_span <= 0.0 or not np.all(np.isfinite(required)):
        return _balance_invalid_terms()
    axis_center_offset = support_axis_proj - 0.5 * support_span
    front_distance = max(support_lat_dist - BALANCE_SAFE_FRONT_MARGIN, 0.0)
    back_distance = max(-support_lat_dist - BALANCE_SAFE_BACK_MARGIN, 0.0)
    center_offset_distance = abs(axis_center_offset)
    if center_offset_distance > 1e-6:
        axis_vel_toward_center = (
            -np.sign(axis_center_offset) * support_axis_vel
        )
    else:
        axis_vel_toward_center = -abs(support_axis_vel)
    if abs(support_lat_dist) > 1e-6:
        lat_vel_toward_center = -np.sign(support_lat_dist) * support_lat_vel
    else:
        lat_vel_toward_center = -abs(support_lat_vel)
    axis_vel_toward_center = float(np.clip(
        axis_vel_toward_center, -BALANCE_VELOCITY_CLIP, BALANCE_VELOCITY_CLIP,
    ))
    lat_vel_toward_center = float(np.clip(
        lat_vel_toward_center, -BALANCE_VELOCITY_CLIP, BALANCE_VELOCITY_CLIP,
    ))
    score = 1.0
    score -= BALANCE_FRONT_PENALTY_COEF * front_distance
    score -= BALANCE_BACK_PENALTY_COEF * back_distance
    score -= BALANCE_CENTER_OFFSET_PENALTY_COEF * center_offset_distance
    score += BALANCE_SUPPORT_AXIS_VELOCITY_COEF * axis_vel_toward_center
    score += BALANCE_SUPPORT_LATERAL_VELOCITY_COEF * lat_vel_toward_center
    return {
        "absolute_score": float(np.clip(
            score, BALANCE_SCORE_CLIP_MIN, BALANCE_SCORE_CLIP_MAX,
        )),
        "front_distance": float(front_distance),
        "back_distance": float(back_distance),
        "center_offset_distance": float(center_offset_distance),
        "support_axis_velocity_toward_center": axis_vel_toward_center,
        "support_lateral_velocity_toward_center": lat_vel_toward_center,
    }


def _balance_invalid_terms() -> Dict[str, float]:
    return {
        "absolute_score": float(BALANCE_INVALID_SCORE),
        "front_distance": 0.0,
        "back_distance": 0.0,
        "center_offset_distance": 0.0,
        "support_axis_velocity_toward_center": 0.0,
        "support_lateral_velocity_toward_center": 0.0,
    }


class BalanceValueRewarder(BaseObserverPlugin):
    """Per-step *absolute* balance score from the support-polygon analysis."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self._inner = Humanoid21BalanceAnalysisObserver(agent_id)
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._inner.on_pre_episode(ctx)
        self._output = self._score()

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._inner.on_post_action_step(ctx)
        self._output = self._score()

    def get_output(self) -> float:
        return float(self._output)

    def _score(self) -> float:
        out = self._inner.get_output()
        if not isinstance(out, dict):
            return float(BALANCE_INVALID_SCORE)
        return float(_compute_balance_value_terms(out)["absolute_score"])

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "BalanceValueRewarder":
        return cls(**config)


class BalanceValueDeltaRewarder(BaseObserverPlugin):
    """Per-step balance-score *delta*, mirror of :class:`StandingPostureDeltaRewarder`."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self._inner = BalanceValueRewarder(agent_id)
        self._previous: float = 0.0
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._inner.on_pre_episode(ctx)
        self._previous = self._inner.get_output()
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._inner.on_post_action_step(ctx)
        current = self._inner.get_output()
        self._output = float(current - self._previous)
        self._previous = current

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "BalanceValueDeltaRewarder":
        return cls(**config)

