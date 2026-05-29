"""Curriculum training constants and configuration for ``humanoid21`` combat experiments.

This module contains only the constants and config classes needed for
the multi-stage curriculum PPO trainer.
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


# ---------------------------------------------------------------------------
# Env-side constants (read by ``make_standing_runtime``)
# ---------------------------------------------------------------------------
CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 10.0
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)
INITIAL_DISTANCE = 3.0
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5

# ---------------------------------------------------------------------------
# Curriculum (multi-stage combat) constants
# ---------------------------------------------------------------------------
# Curriculum match horizon — long enough that approach + a few exchanges are
# observable; ``MATCH_DURATION_SECONDS`` (10 s) is the existing canonical value.
CURRICULUM_MATCH_DURATION_SECONDS = MATCH_DURATION_SECONDS
CURRICULUM_MAX_STEPS = MAX_STEPS
# Damage scaling — keep the env's default (100.0). Per-step net damage values
# stay roughly in [-0.5, +0.5] under typical hits, so the scale is comparable
# to r_cross/r_relation once a small reward weight is applied.
CURRICULUM_DAMAGE_SCALE = 100.0
# Per-component reward scales (applied INSIDE :class:`MultiSignalRewardObserver`
# before the curriculum stage weights). The scales are chosen so all three
# components contribute the SAME order of magnitude per episode once the
# corresponding stage is active:
#
#   * r_cross (cross-support balance) — exactly matches ``stage1.py``'s
#     ``cross_support_reward_scale=0.02``. Per-step raw r_cross is in roughly
#     [-1, 0]; per-200-step episode sum lands in [-5, 0]; scaled = [-0.1, 0].
#     Combined with the sparse -1 terminal fall penalty (applied at
#     ``_inject_terminal_fall_penalty`` time), the stage-1 episodic
#     reward is in [-1.1, 0] — bit-identical to stage1.py's recipe.
#
#   * r_relation (opponent relation) — per-step raw r_relation in [-2, +1]; once the
#     policy starts approaching, sums of +30 to +200 are typical. Scale
#     0.02 keeps the per-episode contribution in [0, +4], same order
#     of magnitude as r_cross + terminal penalty.
#
#   * r_damage (net damage) — per-step raw r_damage is the per-step delta of
#     ``damage_taken_*`` (already pre-multiplied by the env's
#     ``damage_scale=100``). Typical episode net-damage sums are tens
#     of points. Scale 0.05 keeps the per-episode contribution comparable
#     to r_cross + r_relation.
#
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
    critic_learning_rate: float = 3e-4  # Separate LR for critics (can be different from actor)
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 8

    # GAE.
    # ---- Per-reward γ (discount = credit-assignment horizon) ----
    # Env runs at 20 Hz, so effective horizon in steps ≈ 1/(1-γ),
    # in seconds ≈ horizon_steps / 20. Choose γ from the *physical
    # causal time-scale* of each reward, not from agent speed:
    #   r_fall     : fall is foreshadowed ~1-2s ahead → γ≈0.97 (~30 steps, ~1.5s)
    #   r_cross    : balance/support drifts ~1-2s     → γ≈0.97
    #   r_relation : a single body turn can flip the
    #                relative-position signal → shorter
    #                horizon, ~0.7s             → γ≈0.93 (~14 steps)
    #   r_damage   : impact is near-instantaneous,
    #                only the last few frames cause it → γ≈0.80 (~5 steps)
    # λ is the Critic-trust knob (bias-variance), not a physical
    # time-scale, so we keep ONE shared λ across all critics.
    gammas: Dict[str, float] = field(default_factory=lambda: {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_relation": 0.85,
        "r_damage": 0.80,
    })
    gae_lambda: float = 0.95

    # Rollout schedule.
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # Video recording: every ``video_eval_interval`` evals, render a single
    # match using ``round_runner`` and the latest exported policy. Set to 0
    # to disable. Output goes to ``<run_dir>/videos/u{update:05d}.mp4``.
    video_eval_interval: int = 5
    # Env blueprint used for video rendering. If None, defaults to
    # ``envs/humanoid21/blueprint.yaml`` (resolved relative to the cwd
    # where the trainer was launched, typically the combatbench repo root).
    video_env_blueprint: Optional[str] = None

    # Runtime horizon.
    max_steps: int = CURRICULUM_MAX_STEPS

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
    """Eval-driven curriculum stage classifier.

    **Design philosophy** (per user request):

      The gate is a *stateless classifier* over the latest evaluation
      result. On every eval cycle it picks which stage the next batch
      of training should be in — purely from the eval summary, no
      hysteresis, no dwell, no rolling window, no ordered transition
      graph.

      A single eval can move stage from 1 directly to 3 (or 3 back to
      1) — whatever the deterministic policy's current capability says
      it deserves.

    **Decision rule** (single-pass classification on eval summary)::

        len_ratio          = eval_mean_length / max_steps
        final_in_zone_rate = eval_final_in_zone_ratio

        if len_ratio < pass_len_ratio:
            stage = 1   # haven't mastered balance yet ("不满200步")
        elif final_in_zone_rate < pass_final_in_zone:
            stage = 2   # balance OK, work on getting INTO the zone
        else:
            stage = 3   # balance + final zone both OK, do combat

    **Reward components** (4 separate rewards, each with its own critic)::

        r_fall  - terminal fall penalty (sparse, terminal-only)
        r_cross      - cross_support balance
        r_relation      - opponent_relation (distance + heading)
        r_damage      - damage

    **Stage weights** (active-set, normalized to sum to 1)::

        stage 1 -> (0.5,  0.5,  0.0,  0.0)        r_fall + r_cross
        stage 2 -> (1/3,  1/3,  1/3,  0.0)        + r_relation
        stage 3 -> (0.25, 0.25, 0.25, 0.25)       + r_damage

    Each higher stage simply turns on one additional reward; the
    weights are an even split across the active components. Lower-stage
    rewards stay active in higher stages (catastrophic-forgetting
    safeguard).

    Pure Python, picklable, no numpy/torch deps.
    """

    # Active flags per stage; the ``weights`` property normalizes
    # the active components to sum to 1. Order: (r_fall, r_cross, r_relation, r_damage).
    STAGE_WEIGHTS: Dict[int, tuple] = {
        1: (3.0, 1.0, 0.0, 0.0), # 不倒地是第一优化级
        2: (3.0, 1.0, 1.0, 0.0), 
        3: (3.0, 1.0, 1.0, 1.0),
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
        """Active-set weights normalized to sum=1 (no warmup ramp)."""
        base = self.STAGE_WEIGHTS[self.stage]
        total = sum(base)
        if total <= 0.0:
            # Defensive fallback: should never happen for any valid stage.
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(w / total for w in base)

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
