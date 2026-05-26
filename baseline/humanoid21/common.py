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

# ---------------------------------------------------------------------------
# Curriculum (multi-stage combat) constants
# ---------------------------------------------------------------------------
# Curriculum match horizon — long enough that approach + a few exchanges are
# observable; ``MATCH_DURATION_SECONDS`` (10 s) is the existing canonical value.
CURRICULUM_MATCH_DURATION_SECONDS = MATCH_DURATION_SECONDS
CURRICULUM_MAX_STEPS = MAX_STEPS
# Damage scaling — keep the env's default (100.0). Per-step net damage values
# stay roughly in [-0.5, +0.5] under typical hits, so the scale is comparable
# to r1/r2 once a small reward weight is applied.
CURRICULUM_DAMAGE_SCALE = 100.0
# Per-component reward scales (applied INSIDE :class:`MultiSignalRewardObserver`
# before the curriculum stage weights). The scales are chosen so all three
# components contribute the SAME order of magnitude per episode once the
# corresponding stage is active:
#
#   * r1 (cross-support balance) — exactly matches ``stage1.py``'s
#     ``cross_support_reward_scale=0.02``. Per-step raw r1 is in roughly
#     [-1, 0]; per-200-step episode sum lands in [-5, 0]; scaled = [-0.1, 0].
#     Combined with the sparse -1 terminal fall penalty (applied at
#     ``_inject_terminal_fall_penalty`` time), the stage-1 episodic
#     reward is in [-1.1, 0] — bit-identical to stage1.py's recipe.
#
#   * r2 (opponent relation) — per-step raw r2 in [-2, +1]; once the
#     policy starts approaching, sums of +30 to +200 are typical. Scale
#     0.02 keeps the per-episode contribution in [0, +4], same order
#     of magnitude as r1 + terminal penalty.
#
#   * r3 (net damage) — per-step raw r3 is the per-step delta of
#     ``damage_taken_*`` (already pre-multiplied by the env's
#     ``damage_scale=100``). Typical episode net-damage sums are tens
#     of points. Scale 0.05 keeps the per-episode contribution comparable
#     to r1 + r2.
#
# All three constants can be overridden via env var for ablations.
CURRICULUM_R1_SCALE = float(os.environ.get("CURRICULUM_R1_SCALE", "0.02"))
CURRICULUM_R2_SCALE = float(os.environ.get("CURRICULUM_R2_SCALE", "0.02"))
CURRICULUM_R3_SCALE = float(os.environ.get("CURRICULUM_R3_SCALE", "0.05"))
# Terminal fall penalty — subtracted from the LAST step reward of every
# imbalance-terminated trajectory, in EVERY stage (falling is bad
# regardless of which stage we're in). Mirrors stage1.py's
# ``terminal_fall_penalty=1.0`` default.
CURRICULUM_TERMINAL_FALL_PENALTY = float(
    os.environ.get("CURRICULUM_TERMINAL_FALL_PENALTY", "1.0")
)
# CombatScoring HP — set very high so KO never terminates curriculum
# episodes. The damage stream is what we want, not the KO event.
CURRICULUM_NO_KO_HEALTH = float(os.environ.get("CURRICULUM_NO_KO_HEALTH", "1.0e9"))

# Curriculum gate thresholds (eval-driven classifier; see
# :class:`CurriculumStageGate`). Decision rule (single-shot, no
# hysteresis):
#   * eval mean_length < pass_len_ratio * max_steps  -> stage 1
#   * else if final_in_zone_ratio < pass_final_in_zone -> stage 2
#   * else                                           -> stage 3
# ``pass_len_ratio = 0.98`` accepts mean eval length >= 196 / 200 as
# "Stage 1 mastered". The user's spec is "保持平衡满200步"; with
# ``eval_episodes=16`` and per-episode opponent-position randomness,
# strictly requiring 200/200 means a single unlucky episode (mean = 199.x)
# trivially demotes the gate one update after promotion — observed in
# run ``curriculum_20260510_225402``: 4 separate Stage 1->2 promotions
# all bounced back within 5 updates because eval came in at 199.3 / 184 / etc.
# 0.98 leaves a one-episode tolerance for stochastic opponent variance
# while still demanding effective full-horizon survival.
#
# Stage 2 "passes" when, in addition, at least half of the eval episodes
# end with the agent inside the OpponentRelationRewarder non-penalty
# zone (distance band AND heading within max angle).
CURRICULUM_STAGE1_PASS_LEN_RATIO = 0.98      # eval mean_length / max_steps
CURRICULUM_STAGE2_PASS_FINAL_IN_ZONE = 0.5   # eval final_in_zone_ratio

# Stage-3 stickiness floor for ``len_ratio``. In Stage 3, the opponent
# is actively attacking, so episodes that end short of the full horizon
# may reflect SUCCESSFUL combat (opponent knocked the target over), not
# regressed balance. Demotion from Stage 3 therefore requires EITHER
# catastrophic length collapse (len_ratio < this floor) OR a drop in
# ``final_in_zone_ratio`` below ``pass_final_in_zone`` (combat skill
# regression). Without this floor, run ``curriculum_20260511_143835``
# repeatedly bounced Stage 3 -> Stage 1 on evals with length=190.6 and
# final_in_zone=1.0 (perfect combat) — wasting updates on (1,0,0)
# weights and slowly degrading the live actor.
CURRICULUM_STAGE3_STICKY_LEN_RATIO = 0.70    # < 140 / 200 steps == catastrophic

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
    obs_dim: int = 96
    action_dim: int = 21
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
    obs_dim: int = 96
    action_dim: int = 21
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


@dataclass
class CurriculumConfig:
    """Hyperparameters for the unified curriculum-learning PPO trainer.

    Mirrors :class:`Stage1Config` (the single-stage cross-support PPO
    trainer in ``stage1.py``) for the algorithm-side knobs and adds
    the curriculum-specific knobs that drive the data-driven stage
    gate (see :class:`CurriculumStageGate`).

    Defaults intentionally match ``stage1.py`` where they overlap so a
    user can swap the trainers without retuning PPO.
    """

    # Network shape.
    obs_dim: int = 96
    action_dim: int = 21
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    log_std_min: float = DEFAULT_LOG_STD_MIN
    # Curriculum-specific override: cap policy std at e^0 = 1.0 instead
    # of the global default e^1 ≈ 2.72. Empirically the broader cap
    # leaves stochastic-rollout actions saturated at tanh(±)=±1 (= near
    # random) for a long time after the underlying mean policy has
    # converged, which causes a large train/eval gap and strands the
    # eval-driven gate at stage 1 forever waiting for stochastic
    # mastery. Tighter cap shrinks that gap and lets the gate actually
    # see deterministic progress in the train batch too.
    log_std_max: float = 0.0

    # PPO knobs.
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 8

    # GAE.
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Rollout schedule.
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # Critic-warmup phase (after ``--resume-from``). For the FIRST
    # ``critic_warmup_updates`` rollouts, only the critic receives a
    # gradient — the actor is held fixed. This gives the freshly-init
    # critic a chance to fit the value function of the loaded policy
    # before PPO uses (initially garbage) advantages to move the
    # actor and collapse it. Set to 0 to disable (default for
    # train-from-scratch runs); the CLI auto-bumps this to 20 when
    # ``--resume-from`` is provided. See ``_ppo_update`` and
    # ``_load_actor_checkpoint`` docstrings for context.
    critic_warmup_updates: int = 0

    # Runtime horizon.
    max_steps: int = CURRICULUM_MAX_STEPS


    # Curriculum component scales (applied inside the multi-signal observer).
    r1_scale: float = CURRICULUM_R1_SCALE
    r2_scale: float = CURRICULUM_R2_SCALE
    r3_scale: float = CURRICULUM_R3_SCALE
    # Terminal fall penalty (subtracted from last step of terminated
    # trajectories, in every stage). 0.0 disables.
    terminal_fall_penalty: float = CURRICULUM_TERMINAL_FALL_PENALTY

    # Stage-gate pass thresholds (eval-driven classifier; see
    # :class:`CurriculumStageGate`).
    stage1_pass_len_ratio: float = CURRICULUM_STAGE1_PASS_LEN_RATIO
    stage2_pass_final_in_zone: float = CURRICULUM_STAGE2_PASS_FINAL_IN_ZONE

    # Initial-state perturbation toggle (always on for curriculum stage 1).
    enable_perturbation: bool = True

    # Parallelism.
    rollout_workers: int = field(default_factory=lambda: max(
        1, (os.cpu_count() or 1) // 2
    ))
    eval_workers: int = field(default_factory=lambda: max(
        1, (os.cpu_count() or 1) // 4
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
# Curriculum stage gate
# ---------------------------------------------------------------------------
class CurriculumStageGate:
    """Eval-driven curriculum stage classifier (rewritten 2026-05-09).

    **Design philosophy** (per user request):

      The gate is a *stateless classifier* over the latest evaluation
      result. On every eval cycle it picks which stage the next batch
      of training should be in — purely from the eval summary, no
      hysteresis, no dwell, no rolling window, no ordered transition
      graph.

      A single eval can move stage from 1 directly to 3 (or 3 back to
      1) — whatever the deterministic policy's current capability says
      it deserves.

      Why eval and not train rollouts? Train rollouts use stochastic
      ``tanh(N(mu, sigma))`` actions; until ``log_std`` shrinks, the
      sampled policy is much weaker than the underlying mean policy.
      Gating on train metrics indefinitely strands the curriculum at
      stage 1 even after the deterministic policy can clear stage 1
      perfectly. Eval rollouts (deterministic, mean action) are the
      faithful capability measurement.

    **Decision rule** (single-pass classification on eval summary)::

        len_ratio          = eval_mean_length / max_steps
        final_in_zone_rate = eval_final_in_zone_ratio

        if len_ratio < pass_len_ratio:
            stage = 1   # haven't mastered balance yet ("不满200步")
        elif final_in_zone_rate < pass_final_in_zone:
            stage = 2   # balance OK, work on getting INTO the zone
        else:
            stage = 3   # balance + final zone both OK, do combat

    "final_in_zone" means: at the LAST step of the eval episode the
    agent is BOTH within the OpponentRelationRewarder distance band
    AND its heading angle to the opponent is within ``heading_max_angle``.
    This is the user's explicit Stage 3 admission criterion: "保持
    平衡满200步并且最终的距离和朝向进入非惩罚区"

    **Weights**::

        stage 1 -> (1, 0, 0)   r1 only
        stage 2 -> (1, 1, 0)   r1 + r2
        stage 3 -> (1, 1, 1)   r1 + r2 + r3

    Lower-stage rewards stay active in higher stages — that's the
    catastrophic-forgetting safeguard. Re-classification happens
    every eval; if the policy regresses (e.g. forgets balance after
    chasing damage), the next eval pulls us back to stage 1
    immediately.

    Pure Python, picklable, no numpy/torch deps.
    """

    STAGE_WEIGHTS: Dict[int, tuple] = {
        1: (1.0, 0.0, 0.0),
        2: (1.0, 1.0, 0.0),
        3: (1.0, 1.0, 1.0),
    }

    def __init__(
        self,
        *,
        max_steps: int,
        pass_len_ratio: float = CURRICULUM_STAGE1_PASS_LEN_RATIO,
        pass_final_in_zone: float = CURRICULUM_STAGE2_PASS_FINAL_IN_ZONE,
        stage3_sticky_len_ratio: float = CURRICULUM_STAGE3_STICKY_LEN_RATIO,
        initial_stage: int = 1,
    ) -> None:
        if initial_stage not in self.STAGE_WEIGHTS:
            raise ValueError(f"initial_stage must be 1/2/3; got {initial_stage}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0; got {max_steps}")
        if not 0.0 <= stage3_sticky_len_ratio <= pass_len_ratio:
            raise ValueError(
                f"stage3_sticky_len_ratio must be in [0, pass_len_ratio]; "
                f"got {stage3_sticky_len_ratio} (pass_len_ratio={pass_len_ratio})"
            )
        self.max_steps = int(max_steps)
        self.pass_len_ratio = float(pass_len_ratio)
        self.pass_final_in_zone = float(pass_final_in_zone)
        self.stage3_sticky_len_ratio = float(stage3_sticky_len_ratio)
        self.stage = int(initial_stage)
        # Last eval-summary metrics (or None if no eval yet). Kept only
        # for logging; not consulted by the next decision.
        self._last_eval_len_ratio: Optional[float] = None
        self._last_eval_final_in_zone: Optional[float] = None
        self._last_decision_reason: str = "init"

    @property
    def weights(self) -> tuple:
        return self.STAGE_WEIGHTS[self.stage]

    def assign_from_eval(self, eval_summary: Dict[str, float]) -> Dict[str, Any]:
        """Pick the next training stage from a single eval summary.

        ``eval_summary`` must carry:
          * ``mean_length`` — mean episode length over the eval batch
            (deterministic policy).
          * ``final_in_zone_ratio`` — fraction of eval episodes whose
            LAST step has both ``in_range`` and heading-in-tolerance
            (the OpponentRelationRewarder ``in_non_penalty_zone`` flag).

        Returns a dict with the new stage, weights, previous stage,
        the two derived ratios, and a human-readable reason string.
        """
        len_ratio = float(eval_summary.get("mean_length", 0.0)) / float(self.max_steps)
        final_in_zone = float(eval_summary.get("final_in_zone_ratio", 0.0))

        prev_stage = self.stage
        # Stage-3 stickiness: when we're already in combat training and
        # the actor is still demonstrating combat skill (final_in_zone
        # high), accept shorter eval episodes as "opponent landed hits"
        # rather than "balance regression". This stops the spurious
        # Stage 3 -> Stage 1 demotions observed when the opponent
        # successfully attacks (e.g. eval_length=190 with
        # final_in_zone=1.0 — clearly still combat-capable).
        sticky_stage3 = (
            prev_stage == 3
            and len_ratio >= self.stage3_sticky_len_ratio
            and final_in_zone >= self.pass_final_in_zone
        )
        if sticky_stage3:
            new_stage = 3
            reason = (
                f"eval len_ratio={len_ratio:.3f}>={self.stage3_sticky_len_ratio:.2f}"
                f" (stage-3 sticky), final_in_zone={final_in_zone:.3f}>=pass"
                " -> stage 3 (combat)"
            )
        elif len_ratio < self.pass_len_ratio:
            new_stage = 1
            reason = (
                f"eval len_ratio={len_ratio:.3f}<{self.pass_len_ratio:.2f}"
                " -> stage 1 (balance)"
            )
        elif final_in_zone < self.pass_final_in_zone:
            new_stage = 2
            reason = (
                f"eval len_ratio={len_ratio:.3f}>=pass,"
                f" final_in_zone={final_in_zone:.3f}<{self.pass_final_in_zone:.2f}"
                " -> stage 2 (approach)"
            )
        else:
            new_stage = 3
            reason = (
                f"eval len_ratio={len_ratio:.3f}>=pass,"
                f" final_in_zone={final_in_zone:.3f}>=pass"
                " -> stage 3 (combat)"
            )

        self.stage = new_stage
        self._last_eval_len_ratio = len_ratio
        self._last_eval_final_in_zone = final_in_zone
        self._last_decision_reason = reason

        return {
            "stage": self.stage,
            "weights": self.weights,
            "prev_stage": prev_stage,
            "eval_len_ratio": len_ratio,
            "eval_final_in_zone_ratio": final_in_zone,
            "reason": reason,
        }

    def current_state(self) -> Dict[str, Any]:
        """Read-only snapshot for logging on non-eval updates."""
        return {
            "stage": self.stage,
            "weights": self.weights,
            "prev_stage": self.stage,
            "eval_len_ratio": self._last_eval_len_ratio,
            "eval_final_in_zone_ratio": self._last_eval_final_in_zone,
            "reason": self._last_decision_reason,
        }

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
    return runtime


def make_curriculum_runtime_for(target_agent: str) -> EnvRuntime:
    """Build a fresh :class:`EnvRuntime` for curriculum training, target-aware.

    Mirrors ``stage1.py``'s :func:`make_stage1_runtime_for` pattern: only
    the **target agent** has an :class:`ImbalanceTerminationPlugin`, so
    the episode terminates iff THAT agent falls. The non-target agent
    continues to act (its trajectories are discarded by the trainer)
    — this gives the target an honest opponent without contaminating
    the target's terminal-fall signal with the opponent's falls.

    The reward observer is :class:`MultiSignalRewardObserver` for both
    agents so the trainer can swap weights via ``options_fn`` without
    ever rebuilding the runtime. With weights ``(1, 0, 0)``, the
    per-step reward is exactly ``r1_scale * r1_cross_support`` =
    stage1.py's ``cross_support_reward_scale * cross_support_reward``,
    so stage-1 training signal is bit-identical to stage1.py.

    Other wiring:
      * :class:`CombatScoringPlugin` with very high HP (default
        :data:`CURRICULUM_NO_KO_HEALTH`) to expose the damage stream
        on ``ctx.metrics`` *without* triggering KO termination.
      * No initial-state perturbation — the user prefers "learn combat
        fast" over "robustness training" for this run.

    Top-level (no closures) so :class:`RolloutCollector` can pickle
    ``functools.partial(make_curriculum_runtime_for, "robot_a")`` and
    ship it to spawn-mode worker processes unchanged.
    """
    target = str(target_agent)
    if target not in ("robot_a", "robot_b"):
        raise ValueError(f"Unsupported agent_id: {target_agent!r}")
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))

    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins={
            "robot_a_reward": MultiSignalRewardObserver("robot_a"),
            "robot_b_reward": MultiSignalRewardObserver("robot_b"),
        },
        plugins=[
            CombatScoringPlugin(
                initial_health_a=CURRICULUM_NO_KO_HEALTH,
                initial_health_b=CURRICULUM_NO_KO_HEALTH,
                damage_scale=CURRICULUM_DAMAGE_SCALE,
            ),
            ImbalanceTerminationPlugin(target),
        ],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=CURRICULUM_MAX_STEPS,
    )
    return runtime


def make_standing_policy() -> TanhGaussianMLPPolicy:
    """Picklable factory for the *worker-side* shared-architecture policy.

    Each worker gets its own :class:`TanhGaussianMLPPolicy` with the
    standing-task default shape; the trainer's main-process actor is
    pushed in via ``RolloutCollector.collect(state_dicts=...)`` before
    every rollout, so worker weights are always synchronized.

    ``deterministic=False`` because rollout collection wants stochastic
    actions; eval flips it on via ``deterministic=True`` on the
    collector / evaluator side.
    """
    return TanhGaussianMLPPolicy(
        obs_dim=96,
        action_dim=21,
        hidden_dim=256,
        log_std_min=DEFAULT_LOG_STD_MIN,
        log_std_max=DEFAULT_LOG_STD_MAX,
        device="cpu",
        deterministic=False,
    )


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
    "CurriculumConfig",
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
    # Curriculum constants
    "CURRICULUM_MATCH_DURATION_SECONDS",
    "CURRICULUM_MAX_STEPS",
    "CURRICULUM_DAMAGE_SCALE",
    "CURRICULUM_R1_SCALE",
    "CURRICULUM_R2_SCALE",
    "CURRICULUM_R3_SCALE",
    "CURRICULUM_TERMINAL_FALL_PENALTY",
    "CURRICULUM_NO_KO_HEALTH",
    "CURRICULUM_STAGE1_PASS_LEN_RATIO",
    "CURRICULUM_STAGE2_PASS_FINAL_IN_ZONE",
    "CURRICULUM_STAGE3_STICKY_LEN_RATIO",
    # Observers
    "StandingPostureRewarder",
    "StandingPostureDeltaRewarder",
    "BalanceValueRewarder",
    "BalanceValueDeltaRewarder",
    "CrossSupportBalanceRewarder",
    "OpponentRelationRewarder",
    "NetDamageRewarder",
    "MultiSignalRewardObserver",
    # Curriculum gate
    "CurriculumStageGate",
    # Termination plugins
    "StandingTerminationPlugin",
    "BalanceScoreTerminationPlugin",
    "ImbalanceTerminationPlugin",
    # Factories / helpers
    "make_standing_runtime",
    "make_perturbed_balance_runtime",
    "make_curriculum_runtime_for",
    "make_standing_policy",
    "make_standing_options_fn",
    "set_seed",
]
