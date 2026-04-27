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
      - :func:`make_standing_adapter`
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
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from baseline.common.policies import (
    DEFAULT_LOG_STD_MAX,
    DEFAULT_LOG_STD_MIN,
    TanhGaussianMLPPolicy,
    TorchPolicyAdapter,
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
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver


# ---------------------------------------------------------------------------
# Env-side constants (read by ``make_standing_runtime``)
# ---------------------------------------------------------------------------
CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 10.0
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)
INITIAL_DISTANCE = 3.0
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5

# Posture-score shaping (full-penalty = 1.0 normalization scale).
STANDING_SCORE_MAX = 1.0
TARGET_HEIGHT = 1.28
HEIGHT_FULL_PENALTY_DELTA = 0.20
UPRIGHT_TILT_FULL_PENALTY_DEGREES = 30.0
UPRIGHT_FULL_PENALTY_COSINE = float(
    np.cos(np.deg2rad(UPRIGHT_TILT_FULL_PENALTY_DEGREES))
)
ROOT_XY_FULL_PENALTY_DISTANCE = 1.5
JOINT_POSE_FULL_PENALTY_MEAN_ABS = 0.2
JOINT_VEL_FULL_PENALTY_MEAN_ABS = 1.0

# Fall-detection (height + uprightness with grace steps).
FALL_HEIGHT_THRESHOLD = 1.10
FALL_UPRIGHT_THRESHOLD = 0.8
FALL_GRACE_STEPS = 3

# Balance-score shaping (used by ``BalanceValueRewarder`` family).
BALANCE_INVALID_SCORE = float(os.environ.get("STANDING_BALANCE_INVALID_SCORE", "-1.0"))
BALANCE_FRONT_PENALTY_COEF = float(os.environ.get("STANDING_BALANCE_FRONT_PENALTY_COEF", "2.0"))
BALANCE_BACK_PENALTY_COEF = float(os.environ.get("STANDING_BALANCE_BACK_PENALTY_COEF", "6.0"))
# Safe-zone is along the support-lateral axis (positive = forward of the
# ankle-to-ankle line toward the toes; negative = behind toward the heels).
# BACK margin can be negative: once the CoM is at/behind the ankle line a
# tiny backward perturbation already tips the robot, so full reward
# requires CoM to be at least |BACK_MARGIN| forward of the ankle line
# when BACK_MARGIN < 0.
BALANCE_SAFE_FRONT_MARGIN = float(os.environ.get("STANDING_BALANCE_SAFE_FRONT_MARGIN", "0.10"))
BALANCE_SAFE_BACK_MARGIN = float(os.environ.get("STANDING_BALANCE_SAFE_BACK_MARGIN", "-0.02"))
BALANCE_CENTER_OFFSET_PENALTY_COEF = float(os.environ.get("STANDING_BALANCE_CENTER_OFFSET_PENALTY_COEF", "1.0"))
BALANCE_SUPPORT_AXIS_VELOCITY_COEF = float(os.environ.get("STANDING_BALANCE_SUPPORT_AXIS_VELOCITY_COEF", "0.25"))
BALANCE_SUPPORT_LATERAL_VELOCITY_COEF = float(os.environ.get("STANDING_BALANCE_SUPPORT_LATERAL_VELOCITY_COEF", "0.5"))
BALANCE_VELOCITY_CLIP = float(os.environ.get("STANDING_BALANCE_VELOCITY_CLIP", "1.5"))
BALANCE_SCORE_CLIP_MIN = float(os.environ.get("STANDING_BALANCE_SCORE_CLIP_MIN", "-4.0"))
BALANCE_SCORE_CLIP_MAX = float(os.environ.get("STANDING_BALANCE_SCORE_CLIP_MAX", "1.0"))
BALANCE_TERMINATION_SCORE_THRESHOLD = float(os.environ.get("STANDING_BALANCE_TERMINATION_SCORE_THRESHOLD", "0.3"))
BALANCE_TERMINATION_GRACE_STEPS = int(os.environ.get("STANDING_BALANCE_TERMINATION_GRACE_STEPS", "3"))


# ---------------------------------------------------------------------------
# Hyperparameter bundle
# ---------------------------------------------------------------------------
@dataclass
class StandingConfig:
    """Hyperparameters that vary per training run.

    Defaults match ``standing_grpo_rtg_tune_v2.py`` / v1 so any trainer
    dropping in :class:`StandingConfig` reproduces the known-good
    settings without touching the env wiring.
    """

    # Network shape — keep aligned with ``Humanoid21Observer``.
    obs_dim: int = Humanoid21Observer.OBS_DIM
    action_dim: int = Humanoid21Observer.ACTION_DIM
    actor_hidden_dim: int = 256
    log_std_min: float = DEFAULT_LOG_STD_MIN
    log_std_max: float = DEFAULT_LOG_STD_MAX

    # PPO/GRPO knobs.
    learning_rate: float = 1e-4
    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096

    # GRPO-RTG-specific.
    rtg_gamma: float = 0.9
    group_size: int = 8

    # Rollout schedule.
    episodes_per_update: int = 256
    max_updates: int = 10000

    # Eval schedule.
    eval_interval: int = 5
    eval_episodes: int = 16

    # Parallelism.
    rollout_workers: int = field(default_factory=lambda: max(
        1, min(64, max(1, (os.cpu_count() or 1) // 2))
    ))
    eval_workers: int = field(default_factory=lambda: max(
        1, min(16, max(1, (os.cpu_count() or 1) // 4))
    ))

    seed: int = 42


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set numpy + torch (CPU/CUDA) seeds in one call."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


# ---------------------------------------------------------------------------
# Termination plugins
# ---------------------------------------------------------------------------
class StandingTerminationPlugin(BasePlugin):
    """Terminate when the agent has fallen for ``fall_grace_steps`` in a row.

    "Fallen" = below ``fall_height_threshold`` OR below
    ``fall_upright_threshold`` (cosine of tilt). The grace window
    avoids spurious triggers from physics jitter.
    """

    def __init__(
        self,
        agent_id: str,
        fall_height_threshold: float = FALL_HEIGHT_THRESHOLD,
        fall_upright_threshold: float = FALL_UPRIGHT_THRESHOLD,
        fall_grace_steps: int = FALL_GRACE_STEPS,
    ) -> None:
        self.agent_id = str(agent_id)
        self.fall_height_threshold = float(fall_height_threshold)
        self.fall_upright_threshold = float(fall_upright_threshold)
        self.fall_grace_steps = max(1, int(fall_grace_steps))
        self._streak = 0

    @property
    def name(self) -> str:
        return f"{self.agent_id}_standing_termination"

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
        score_threshold: float = BALANCE_TERMINATION_SCORE_THRESHOLD,
        grace_steps: int = BALANCE_TERMINATION_GRACE_STEPS,
    ) -> None:
        self.agent_id = str(agent_id)
        self.score_threshold = float(score_threshold)
        self.grace_steps = max(1, int(grace_steps))
        self._inner = Humanoid21BalanceAnalysisObserver(agent_id)
        self._streak = 0

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
            if isinstance(out, dict) else float(BALANCE_INVALID_SCORE)
        )
        self._streak = self._streak + 1 if score < self.score_threshold else 0
        if self._streak >= self.grace_steps:
            ctx.request_termination(TerminationReason.CUSTOM)


# ---------------------------------------------------------------------------
# Top-level factories (picklable for parallel rollout under ``spawn``)
# ---------------------------------------------------------------------------
def make_standing_runtime() -> EnvRuntime:
    """Build a fresh :class:`EnvRuntime` for the standing task.

    Wires :class:`Humanoid21Observer` (96-D obs) + per-agent
    :class:`StandingPostureDeltaRewarder` reward observers + per-agent
    :class:`StandingTerminationPlugin` fall-detection plugins. The
    per-episode ``initial_distance`` is set on the simulator and may be
    overridden each episode via ``options={"initial_distance": ...}``;
    see :func:`make_standing_options_fn` for the canonical options
    callable.

    Top-level (no closures) so ``RolloutCollector`` can pickle and ship
    it to ``spawn``-mode worker processes.
    """
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))
    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins={
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
            "robot_a_reward": StandingPostureDeltaRewarder("robot_a"),
            "robot_b_reward": StandingPostureDeltaRewarder("robot_b"),
        },
        plugins=[
            StandingTerminationPlugin("robot_a"),
            StandingTerminationPlugin("robot_b"),
        ],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=MAX_STEPS,
    )
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.action_space = Humanoid21Observer.get_action_space()
    return runtime


def make_standing_adapter() -> TorchPolicyAdapter:
    """Picklable factory for the *worker-side* shared-architecture adapter.

    Each worker gets its own :class:`TanhGaussianMLPPolicy` with the
    standing-task default shape; the trainer's main-process actor is
    pushed in via ``RolloutCollector.collect(state_dicts=...)`` before
    every rollout, so worker weights are always synchronized.

    ``deterministic=False`` because rollout collection wants stochastic
    actions; eval flips it on via ``deterministic=True`` on the
    collector / evaluator side.
    """
    actor = TanhGaussianMLPPolicy(
        obs_dim=Humanoid21Observer.OBS_DIM,
        action_dim=Humanoid21Observer.ACTION_DIM,
        hidden_dim=256,
        log_std_min=DEFAULT_LOG_STD_MIN,
        log_std_max=DEFAULT_LOG_STD_MAX,
    )
    return TorchPolicyAdapter(actor=actor, device="cpu", deterministic=False)


def make_standing_options_fn(
    *,
    distance_min: float = ROLLOUT_INITIAL_DISTANCE_MIN,
    distance_max: float = ROLLOUT_INITIAL_DISTANCE_MAX,
    salt: int = 2024,
) -> Callable[[int], Dict[str, Any]]:
    """Build an ``options_fn(episode_index) -> options`` for ``RolloutCollector``.

    Returned callable is invoked on the **main process** to produce a
    deterministic per-episode ``options`` dict — sequential and
    parallel rollouts therefore see identical options for the same
    episode index, which keeps trajectories bit-equal across worker
    counts (modulo simulator non-determinism).

    Closures ARE allowed here because :func:`RolloutCollector.collect`
    evaluates ``options_fn`` on the main process before shipping the
    resolved options dict to workers.
    """

    def _fn(episode_index: int) -> Dict[str, Any]:
        rng = np.random.default_rng(int(episode_index) + int(salt))
        return {
            "initial_distance": float(rng.uniform(distance_min, distance_max)),
        }

    return _fn


__all__ = [
    # Hyperparameters
    "StandingConfig",
    # Constants (commonly imported)
    "CONTROL_FREQUENCY",
    "MATCH_DURATION_SECONDS",
    "MAX_STEPS",
    "INITIAL_DISTANCE",
    "ROLLOUT_INITIAL_DISTANCE_MIN",
    "ROLLOUT_INITIAL_DISTANCE_MAX",
    "TARGET_HEIGHT",
    "FALL_HEIGHT_THRESHOLD",
    "FALL_UPRIGHT_THRESHOLD",
    "FALL_GRACE_STEPS",
    "BALANCE_SAFE_FRONT_MARGIN",
    "BALANCE_SAFE_BACK_MARGIN",
    "BALANCE_TERMINATION_SCORE_THRESHOLD",
    "BALANCE_TERMINATION_GRACE_STEPS",
    # Observers
    "StandingPostureRewarder",
    "StandingPostureDeltaRewarder",
    "BalanceValueRewarder",
    "BalanceValueDeltaRewarder",
    # Termination plugins
    "StandingTerminationPlugin",
    "BalanceScoreTerminationPlugin",
    # Factories / helpers
    "make_standing_runtime",
    "make_standing_adapter",
    "make_standing_options_fn",
    "set_seed",
]
