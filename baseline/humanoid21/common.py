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
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin
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

# Initial-state perturbation defaults — verbatim from
# ``standing_turbulence_dense_reward_ppo.py`` and ``standing_balance_ppo.py``.
# These are the perturbation ranges applied once at episode start to the
# robot(s) tagged with :class:`InitialStatePerturbationPlugin`. Keep them
# small enough that a stable standing policy survives the perturbation
# on most seeds, big enough that the learned policy has to actively
# stabilize instead of coasting on a perfectly symmetric reset.
PERTURBATION_JOINT_POS_DELTA_MAX = 0.05
PERTURBATION_JOINT_VEL_DELTA_MAX = 0.05
PERTURBATION_ROOT_XY_OFFSET_MAX = 0.05
PERTURBATION_ROOT_TILT_DEG_MAX = 10.0
PERTURBATION_ROOT_LINEAR_VELOCITY_DELTA_MAX = (0.5, 0.5, 0.0)
PERTURBATION_ROOT_ANGULAR_VELOCITY_DELTA_MAX = (0.5, 0.5, 0.2)

# ``MATCH_DURATION_SECONDS`` above is the standing-task default (10 s).
# Perturbed-balance training historically used a shorter horizon because
# episodes that survive 3 s of perturbation already prove robustness.
PERTURBED_MATCH_DURATION_SECONDS = 3.0
PERTURBED_MAX_STEPS = int(CONTROL_FREQUENCY * PERTURBED_MATCH_DURATION_SECONDS)

# Cross-support balance (交替支撑平衡) 训练参数
# 足底接触：从 derived_state["robot_environment_contacts"] 读取，与 ground geom
# 有接触即视为着地（无力阈值）。
#
# 以下默认步数按本模块 ``CONTROL_FREQUENCY``（当前 20 Hz，约 50 ms/步）设计；
# 若改控制频率，建议按比例缩放各 *_STEPS 环境变量。
#
# CROSS_SUPPORT_INITIAL_GRACE_STEPS（默认 30）
#   复位后允许「尚未出现第一次单脚支撑」的等待步数上限（可双脚着地/双脚离地），
#   再长则按 initial 惩罚。约 1.5 s：给接触与姿态稳定留余量。
# CROSS_SUPPORT_INITIAL_PENALTY_COEF（默认 0.25）
#   第一次单脚支撑前等待过长时的惩罚系数；按超时比例线性增加，封顶 1 倍系数。
# CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS（默认 4）
#   每次单脚支撑中，支撑脚着地时长最小值（约 0.2 s）。
# CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF（默认 0.45）
#   支撑脚着地时长过短时的惩罚强度。
# CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS（默认 18）
#   换支撑脚间隔（A->B）的最大步数（约 0.9 s）。
# CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF（默认 0.25）
#   换支撑脚间隔不在区间内时的惩罚强度。
CROSS_SUPPORT_INITIAL_GRACE_STEPS = int(os.environ.get(
    "CROSS_SUPPORT_INITIAL_GRACE_STEPS", "30"
))
CROSS_SUPPORT_INITIAL_PENALTY_COEF = float(os.environ.get(
    "CROSS_SUPPORT_INITIAL_PENALTY_COEF", "0.25"
))
CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS = int(os.environ.get(
    "CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS", "4"
))
CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF = float(os.environ.get(
    "CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF", "0.45"
))
CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS = int(os.environ.get(
    "CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS", "18"
))
CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF = float(os.environ.get(
    "CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF", "0.25"
))

# Opponent-relation reward (课程二：接近并朝向对手)
# 两个损失项都采用“容忍区间内无损失，超出后线性增长”：
# 1) 距离损失：距离在 [min, max] 内为 0，超出则线性惩罚
# 2) 朝向损失：朝向误差角 <= max_angle 时为 0，超出则线性惩罚
OPP_REL_DIST_MIN = float(os.environ.get("OPP_REL_DIST_MIN", "1.0"))
OPP_REL_DIST_MAX = float(os.environ.get("OPP_REL_DIST_MAX", "2.2"))
OPP_REL_DIST_LINEAR_RANGE = float(os.environ.get("OPP_REL_DIST_LINEAR_RANGE", "1.0"))
OPP_REL_HEADING_MAX_ANGLE_DEG = float(os.environ.get("OPP_REL_HEADING_MAX_ANGLE_DEG", "25.0"))
OPP_REL_HEADING_LINEAR_RANGE_DEG = float(os.environ.get("OPP_REL_HEADING_LINEAR_RANGE_DEG", "45.0"))
OPP_REL_DIST_PENALTY_COEF = float(os.environ.get("OPP_REL_DIST_PENALTY_COEF", "1.0"))
OPP_REL_HEADING_PENALTY_COEF = float(os.environ.get("OPP_REL_HEADING_PENALTY_COEF", "1.0"))


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


@dataclass
class PerturbedBalanceConfig:
    """Hyperparameters for the perturbed-balance PPO trainer.

    Defaults mirror ``standing_balance_ppo.py`` so this dataclass is a
    drop-in replacement for its module-level constants. Kept separate
    from :class:`StandingConfig` because the algorithm is genuinely
    different (PPO with critic + GAE vs. GRPO-RTG) and the env horizon
    is shorter.
    """

    # Network shape.
    obs_dim: int = Humanoid21Observer.OBS_DIM
    action_dim: int = Humanoid21Observer.ACTION_DIM
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    log_std_min: float = DEFAULT_LOG_STD_MIN
    log_std_max: float = DEFAULT_LOG_STD_MAX

    # PPO knobs.
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 32

    # GAE.
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Rollout schedule.
    episodes_per_update: int = 256 * 32
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


class CrossSupportBalanceRewarder(BaseObserverPlugin):
    """交叉支撑平衡奖励插件（语义归约版）。

    保留初始逻辑：开局到第一次单脚支撑前，超过 ``initial_grace_steps`` 开始惩罚。

    进入单脚支撑后，仅关注两项原子指标：
      1) 单脚支撑时长（单次段落）：只惩罚过短，不惩罚过长
      2) 换支撑脚间隔（A -> B）：超过 ``switch_interval_max_steps`` 则惩罚

    其中 A -> B 间隔从 A 脚本轮第一次单脚支撑开始计时，中间允许出现 A 脚再次单脚支撑，
    直到第一次出现 B 脚单脚支撑。
    """

    STATE_WAIT_FIRST_SINGLE_SUPPORT = "wait_first_single_support"
    STATE_TRACKING = "tracking"

    def __init__(
        self,
        agent_id: str,
        initial_grace_steps: int = CROSS_SUPPORT_INITIAL_GRACE_STEPS,
        initial_penalty_coef: float = CROSS_SUPPORT_INITIAL_PENALTY_COEF,
        foot_lift_min_steps: int = CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS,
        foot_lift_penalty_coef: float = CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF,
        switch_interval_max_steps: int = CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS,
        switch_interval_penalty_coef: float = CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF,
    ) -> None:
        self.agent_id = str(agent_id)
        self.initial_grace_steps = int(initial_grace_steps)
        self.initial_penalty_coef = float(initial_penalty_coef)
        self.foot_lift_min_steps = int(foot_lift_min_steps)
        self.foot_lift_penalty_coef = float(foot_lift_penalty_coef)
        self.switch_interval_max_steps = max(0, int(switch_interval_max_steps))
        self.switch_interval_penalty_coef = float(switch_interval_penalty_coef)

        # 状态变量
        self._state: str = self.STATE_WAIT_FIRST_SINGLE_SUPPORT
        self._state_timer: int = 0
        self._current_support_foot: Optional[str] = None  # 'left' or 'right'
        self._current_support_steps: int = 0
        self._switch_anchor_foot: Optional[str] = None  # 'left' or 'right'
        self._switch_interval_steps: int = 0
        self._output: float = 0.0
        self._ground_geom_name: Optional[str] = None

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        """重置状态"""
        self._state = self.STATE_WAIT_FIRST_SINGLE_SUPPORT
        self._state_timer = 0
        self._current_support_foot = None
        self._current_support_steps = 0
        self._switch_anchor_foot = None
        self._switch_interval_steps = 0
        self._output = 0.0
        # 获取地面 geom 名称
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get('ground_geom_name', 'ground')

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        """计算每步奖励"""
        reward = 0.0

        # 检测双脚接触状态
        left_foot_contact, right_foot_contact = self._get_foot_contact_state(ctx)

        if self._state == self.STATE_WAIT_FIRST_SINGLE_SUPPORT:
            reward = self._handle_wait_first_single_support(left_foot_contact, right_foot_contact)
        elif self._state == self.STATE_TRACKING:
            reward = self._handle_tracking(left_foot_contact, right_foot_contact)

        self._output = reward

    def get_output(self) -> float:
        return float(self._output)

    def _get_foot_contact_state(self, ctx: ReadOnlySimContext) -> tuple[bool, bool]:
        """检测双脚是否与地面接触（无力阈值：有接触条目即算）。

        仅使用 ``derived_state['robot_environment_contacts']``，与
        ``get_static_data()['ground_geom_name']`` 中的地面 geom 名匹配。

        Returns:
            (left_foot_contact, right_foot_contact)
        """
        derived_state = ctx.accessor.get_derived_state()
        env_contacts = derived_state.get("robot_environment_contacts", [])

        robot_suffix = '_red' if self.agent_id == 'robot_a' else '_blue'
        left_foot_body = f"foot_left{robot_suffix}"
        right_foot_body = f"foot_right{robot_suffix}"
        ground_geom = self._ground_geom_name or "ground"

        left_foot_contact = False
        right_foot_contact = False

        for contact in env_contacts:
            if contact.get("robot") != self.agent_id:
                continue
            env_geom = contact.get("environment_geom", "") or ""
            if env_geom != ground_geom:
                continue
            body = contact.get("body", "") or ""
            if body == left_foot_body:
                left_foot_contact = True
            elif body == right_foot_body:
                right_foot_contact = True

        return left_foot_contact, right_foot_contact

    def _single_support_foot(self, left_foot_contact: bool, right_foot_contact: bool) -> Optional[str]:
        """返回当前是否为单脚支撑：'left' / 'right' / None。"""
        if left_foot_contact and not right_foot_contact:
            return "left"
        if right_foot_contact and not left_foot_contact:
            return "right"
        return None

    def _begin_tracking(self, support_foot: str) -> None:
        """第一次进入单脚支撑后，初始化追踪器。"""
        self._state = self.STATE_TRACKING
        self._current_support_foot = support_foot
        self._current_support_steps = 1
        self._switch_anchor_foot = support_foot
        self._switch_interval_steps = 0

    def _handle_wait_first_single_support(
        self, left_foot_contact: bool, right_foot_contact: bool
    ) -> float:
        """从任意初始接触状态，等待第一次单脚支撑。"""
        reward = 0.0
        support_foot = self._single_support_foot(left_foot_contact, right_foot_contact)
        if support_foot is not None:
            self._begin_tracking(support_foot)
            return reward

        self._state_timer += 1
        if self._state_timer > self.initial_grace_steps:
            excess = self._state_timer - self.initial_grace_steps
            denom = max(1, self.initial_grace_steps)
            reward -= self.initial_penalty_coef * min(excess / denom, 1.0)
        return reward

    def _handle_tracking(
        self, left_foot_contact: bool, right_foot_contact: bool
    ) -> float:
        """追踪单脚支撑短时惩罚与换脚间隔区间惩罚。"""
        reward = 0.0
        current_single_support = self._single_support_foot(left_foot_contact, right_foot_contact)

        # A -> B 换脚间隔从 A 脚本轮首次单脚开始计时，期间允许 A 再次单脚。
        self._switch_interval_steps += 1

        # 1) 单脚支撑时长：仅惩罚过短（段落结束时结算）
        if self._current_support_foot is None:
            if current_single_support is not None:
                self._current_support_foot = current_single_support
                self._current_support_steps = 1
        elif current_single_support == self._current_support_foot:
            self._current_support_steps += 1
        else:
            if self._current_support_steps < self.foot_lift_min_steps:
                deficit = self.foot_lift_min_steps - self._current_support_steps
                reward -= self.foot_lift_penalty_coef * (deficit / max(1, self.foot_lift_min_steps))
            if current_single_support is None:
                self._current_support_foot = None
                self._current_support_steps = 0
            else:
                self._current_support_foot = current_single_support
                self._current_support_steps = 1

        # 2) 换支撑脚间隔：当首次出现 opposite single support 时结算并重置锚点
        if (
            current_single_support is not None
            and self._switch_anchor_foot is not None
            and current_single_support != self._switch_anchor_foot
        ):
            if self._switch_interval_steps > self.switch_interval_max_steps:
                excess = self._switch_interval_steps - self.switch_interval_max_steps
                denom = max(1, self.switch_interval_max_steps)
                reward -= self.switch_interval_penalty_coef * min(excess / denom, 1.0)
            self._switch_anchor_foot = current_single_support
            self._switch_interval_steps = 0

        return reward


class OpponentRelationRewarder(BaseObserverPlugin):
    """与对手相对关系奖励（距离 + 朝向）。

    设计目标：课程二中鼓励“接近并朝向对手”，同时保持一个容忍范围：
    - 距离在 [dist_min, dist_max] 内：无距离惩罚
    - 朝向误差角 <= heading_max_angle_deg：无朝向惩罚
    - 超出后按线性范围增长惩罚，并按系数加权
    """

    def __init__(
        self,
        agent_id: str,
        dist_min: float = OPP_REL_DIST_MIN,
        dist_max: float = OPP_REL_DIST_MAX,
        dist_linear_range: float = OPP_REL_DIST_LINEAR_RANGE,
        heading_max_angle_deg: float = OPP_REL_HEADING_MAX_ANGLE_DEG,
        heading_linear_range_deg: float = OPP_REL_HEADING_LINEAR_RANGE_DEG,
        dist_penalty_coef: float = OPP_REL_DIST_PENALTY_COEF,
        heading_penalty_coef: float = OPP_REL_HEADING_PENALTY_COEF,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self.dist_min = float(dist_min)
        self.dist_max = float(dist_max)
        self.dist_linear_range = max(1e-6, float(dist_linear_range))
        self.heading_max_angle_deg = float(heading_max_angle_deg)
        self.heading_linear_range_deg = max(1e-6, float(heading_linear_range_deg))
        self.dist_penalty_coef = float(dist_penalty_coef)
        self.heading_penalty_coef = float(heading_penalty_coef)
        self._output: float = 0.0

    @staticmethod
    def _robot_forward_xy_from_root_rot(root_rot_wxyz: np.ndarray) -> np.ndarray:
        """从四元数 [w, x, y, z] 计算机体前向在地平面的单位向量。"""
        q = np.asarray(root_rot_wxyz, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-8:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        w, x, y, z = (q / norm).tolist()
        # 旋转矩阵第一列（本地 x 轴在世界系中的方向）
        fx = 1.0 - 2.0 * (y * y + z * z)
        fy = 2.0 * (x * y + w * z)
        fxy = np.asarray([fx, fy], dtype=np.float64)
        f_norm = float(np.linalg.norm(fxy))
        if f_norm < 1e-8:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        return fxy / f_norm

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        self_state = core_state[self.agent_id]
        opp_state = core_state[self.opponent_id]

        self_xy = np.asarray(self_state["root_pos"][:2], dtype=np.float64)
        opp_xy = np.asarray(opp_state["root_pos"][:2], dtype=np.float64)
        delta_xy = opp_xy - self_xy
        distance = float(np.linalg.norm(delta_xy))

        # 1) 距离区间惩罚
        if distance < self.dist_min:
            dist_excess = self.dist_min - distance
        elif distance > self.dist_max:
            dist_excess = distance - self.dist_max
        else:
            dist_excess = 0.0
        dist_penalty = min(dist_excess / self.dist_linear_range, 1.0)

        # 2) 朝向惩罚（面向对手角度）
        if distance < 1e-6:
            heading_penalty = 0.0
        else:
            to_opp_unit = delta_xy / distance
            forward_unit = self._robot_forward_xy_from_root_rot(
                np.asarray(self_state["root_rot"], dtype=np.float64)
            )
            cosang = float(np.clip(np.dot(forward_unit, to_opp_unit), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cosang)))
            angle_excess = max(0.0, angle_deg - self.heading_max_angle_deg)
            heading_penalty = min(angle_excess / self.heading_linear_range_deg, 1.0)

        total_penalty = (
            self.dist_penalty_coef * dist_penalty
            + self.heading_penalty_coef * heading_penalty
        )
        self._output = -float(total_penalty)

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


def make_perturbed_balance_runtime() -> EnvRuntime:
    """Build a fresh :class:`EnvRuntime` for the perturbed-balance task.

    Differs from :func:`make_standing_runtime` in three ways:

      * **Reward** — :class:`BalanceValueRewarder` (support-polygon
        balance score, bounded in ``[-4, 1]``) instead of the posture
        delta. This is the absolute score; PPO post-processes it
        downstream (critic values, GAE, bootstrap on truncation).
      * **Termination** — :class:`BalanceScoreTerminationPlugin`
        (persistently-low balance score) instead of height/uprightness
        fall detection. A slightly lower stance can still be balanced,
        and the height-based rule would fight the balance objective.
      * **Initial-state perturbation** — an
        :class:`InitialStatePerturbationPlugin` is attached per agent
        so every reset nudges joints / root pose / velocities within
        the :data:`PERTURBATION_*` ranges. Worker RNG drives the
        perturbation (``random_seed=None``), so episode seeds from
        ``RolloutCollector`` still produce deterministic trajectories
        downstream of whatever the simulator does with them.

    Uses the shorter :data:`PERTURBED_MAX_STEPS` horizon (3 s @ 20 Hz)
    — surviving perturbation for 3 s is a strong robustness signal
    already, and the shorter horizon keeps rollout cost sane.

    Top-level (no closures) so the collector can pickle & ship it to
    spawn-mode worker processes unchanged.
    """
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))
    perturbations = {
        agent: InitialStatePerturbationPlugin(
            target_robot=agent,
            joint_pos_delta_max=PERTURBATION_JOINT_POS_DELTA_MAX,
            joint_vel_delta_max=PERTURBATION_JOINT_VEL_DELTA_MAX,
            root_xy_offset_max=PERTURBATION_ROOT_XY_OFFSET_MAX,
            root_tilt_deg_max=PERTURBATION_ROOT_TILT_DEG_MAX,
            root_linear_velocity_delta_max=list(PERTURBATION_ROOT_LINEAR_VELOCITY_DELTA_MAX),
            root_angular_velocity_delta_max=list(PERTURBATION_ROOT_ANGULAR_VELOCITY_DELTA_MAX),
            random_seed=None,
        )
        for agent in ("robot_a", "robot_b")
    }
    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins={
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
            "robot_a_reward": BalanceValueRewarder("robot_a"),
            "robot_b_reward": BalanceValueRewarder("robot_b"),
        },
        plugins=[
            BalanceScoreTerminationPlugin("robot_a"),
            BalanceScoreTerminationPlugin("robot_b"),
            perturbations["robot_a"],
            perturbations["robot_b"],
        ],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=PERTURBED_MAX_STEPS,
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
    "PerturbedBalanceConfig",
    # Constants (commonly imported)
    "CONTROL_FREQUENCY",
    "MATCH_DURATION_SECONDS",
    "MAX_STEPS",
    "PERTURBED_MATCH_DURATION_SECONDS",
    "PERTURBED_MAX_STEPS",
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
    "PERTURBATION_JOINT_POS_DELTA_MAX",
    "PERTURBATION_JOINT_VEL_DELTA_MAX",
    "PERTURBATION_ROOT_XY_OFFSET_MAX",
    "PERTURBATION_ROOT_TILT_DEG_MAX",
    "PERTURBATION_ROOT_LINEAR_VELOCITY_DELTA_MAX",
    "PERTURBATION_ROOT_ANGULAR_VELOCITY_DELTA_MAX",
    # Cross-support balance constants
    "CROSS_SUPPORT_INITIAL_GRACE_STEPS",
    "CROSS_SUPPORT_INITIAL_PENALTY_COEF",
    "CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS",
    "CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF",
    "CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS",
    "CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF",
    # Opponent-relation constants
    "OPP_REL_DIST_MIN",
    "OPP_REL_DIST_MAX",
    "OPP_REL_DIST_LINEAR_RANGE",
    "OPP_REL_HEADING_MAX_ANGLE_DEG",
    "OPP_REL_HEADING_LINEAR_RANGE_DEG",
    "OPP_REL_DIST_PENALTY_COEF",
    "OPP_REL_HEADING_PENALTY_COEF",
    # Observers
    "StandingPostureRewarder",
    "StandingPostureDeltaRewarder",
    "BalanceValueRewarder",
    "BalanceValueDeltaRewarder",
    "CrossSupportBalanceRewarder",
    "OpponentRelationRewarder",
    # Termination plugins
    "StandingTerminationPlugin",
    "BalanceScoreTerminationPlugin",
    "ImbalanceTerminationPlugin",
    # Factories / helpers
    "make_standing_runtime",
    "make_perturbed_balance_runtime",
    "make_standing_adapter",
    "make_standing_options_fn",
    "set_seed",
]
