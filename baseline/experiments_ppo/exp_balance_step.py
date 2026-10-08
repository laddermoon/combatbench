"""Balance reinforcement on top of the converged stepping policy.

Warm-start from a ``step``/``step_mbs`` checkpoint.  The env adds
``StandingTriggeredForcePlugin`` (``balance_step_env.yaml``): once the
robot stands steadily (h > 1.15 for 20 steps) it takes random-direction
torso pushes; a non-foot ground contact during PUSHING+OBSERVE counts
as a push-induced fall.

The reward recipe is IDENTICAL to ``exp_step.py`` — no new channels, no
phase switch.  The push introduces no reward term; robustness is
learned through the existing structure:

  * pushed mid-swing → φ collapses → fall-onset penalty (r_fall) and
    the full r_potential veto teach swing-margin and recovery stepping;
  * pushed while standing → stepping income keeps paying only if the
    gait survives, so robustness is directly on the gradient;
  * pushed down → standup potential still pays for getting back up.

Curriculum (same 12 levels as archived exp_balance_v2):
  level 0-3   40N,  duration windows 1-10 / 11-20 / 21-30 / 31-40 steps
  level 4-11  100N, duration windows 1-5 … 36-40 steps
  Promotion: eval recovery_rate (1 - falls/pushes) >= 0.8 for
  PROMOTE_PATIENCE consecutive evals.  Eval jobs carry the same
  impulse_params as rollout, so eval metrics are measured under the
  current perturbation level.

Best-of-run / early stop use a composite quality — curriculum level
dominates, recovery_rate breaks ties, gait quality (the exp_step
composite, reconstructed from eval info) refines:
    quality = 10·level + recovery_rate + 0.05·gait_quality
with the same EMA ratchet as exp_step so slow climbs don't get killed
by a lucky raw spike.

Blueprint: baseline/humanoid21/end2end/balance_step_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .exp_step import Step, _phase_explore_factor


class BalanceStep(Step):
    """Stepping + standing-triggered push perturbation curriculum."""

    name = "balance_step"

    # --- Curriculum: 12 levels (same as exp_balance_v2) ---
    LEVEL_FORCES: Tuple[float, ...] = (
        40.0, 40.0, 40.0, 40.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    )
    LEVEL_DURATION_RANGES: Tuple[Tuple[int, int], ...] = (
        (1, 10), (11, 20), (21, 30), (31, 40),
        (1, 5), (6, 10), (11, 15), (16, 20),
        (21, 25), (26, 30), (31, 35), (36, 40),
    )
    PROMOTE_RECOVERY_RATE: float = 0.8
    PROMOTE_PATIENCE: int = 2

    # --- Curriculum / tracking state ---
    _level: int = 0
    _consecutive_pass: int = 0
    _recovery_rate: float = 1.0
    # Separate best/EMA/anchor for the composite quality — the parent's
    # gait-only bookkeeping keeps running but its stop/export decision
    # is overridden by ours.
    _b_best_quality: float = -1.0
    _b_quality_ema: float | None = None
    _b_last_best_update: int = -1

    # ------------------------------------------------------------------
    # Blueprint — step env + StandingTriggeredForcePlugin
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "balance_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    @property
    def current_force(self) -> float:
        idx = max(0, min(self._level, len(self.LEVEL_FORCES) - 1))
        return float(self.LEVEL_FORCES[idx])

    @property
    def current_duration_range(self) -> Tuple[int, int]:
        idx = max(0, min(self._level, len(self.LEVEL_DURATION_RANGES) - 1))
        return self.LEVEL_DURATION_RANGES[idx]

    # ------------------------------------------------------------------
    # Jobs — Step's (initial_distance + phase ef) + impulse_params
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        env_bp = self._env_pb().materialize(max_steps=self.max_steps)
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=_phase_explore_factor)
        force = self.current_force
        dur_min, dur_max = self.current_duration_range
        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            impulse_params = {
                rid: {
                    "force": force,
                    "duration_min": dur_min,
                    "duration_max": dur_max,
                    "body": "torso",
                    "seed": seed,
                }
                for rid in self._AGENT_IDS
            }
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bp,
                seed=seed,
                episode_options={
                    "initial_distance": initial_distance,
                    "impulse_params": impulse_params,
                },
                sampling_a=sampling,
                sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Eval — Step's gait metrics + push recovery + curriculum
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        info = result["info"]

        # --- Push accounting from plugin metrics ---
        total_push = 0
        total_fall = 0
        for ep in episodes:
            em = dict(getattr(ep, "episode_metrics", None) or {})
            for rid in self._AGENT_IDS:
                total_push += int(em.get(f"{rid}_push_count", 0))
                total_fall += int(em.get(f"{rid}_fall_count", 0))
        recovery_rate = (
            float(1.0 - total_fall / total_push) if total_push > 0 else 1.0
        )
        self._recovery_rate = recovery_rate

        # --- Promotion: recovery >= 0.8 for PROMOTE_PATIENCE evals.
        # total_push>0 guard prevents a push-less eval (nobody stood
        # long enough) from counting as a pass.
        prev_level = self._level
        if self._level < len(self.LEVEL_FORCES) - 1:
            if recovery_rate >= self.PROMOTE_RECOVERY_RATE and total_push > 0:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0
        promoted = self._level > prev_level

        # --- Composite quality: level dominates, recovery breaks ties,
        # gait quality refines.  Reconstruct the parent's gait quality
        # from its (rounded) info fields — same formula, same role.
        gait_q = (
            info.get("step", 0.0)
            - 0.1 * (info.get("falls") or 0.0)
            + 0.1 * (info.get("alt") or 0.0)
            * min(1.0, info.get("cycles", 0.0) / 10.0)
            + min(info.get("solepk") or 0.0, 0.10)
            + 0.005 * min(info.get("cycles", 0.0), 20.0)
        )
        quality = 10.0 * self._level + recovery_rate + 0.05 * gait_q

        if self._b_quality_ema is None:
            self._b_quality_ema = quality
        else:
            self._b_quality_ema = (
                0.85 * self._b_quality_ema + 0.15 * quality
            )
        improved = (
            info.get("success", 0.0) >= 0.9
            and quality > self._b_best_quality
        )
        if improved:
            self._b_best_quality = quality
        trend_improved = (
            info.get("success", 0.0) >= 0.9
            and self._b_quality_ema > self._b_best_quality
        )
        if trend_improved:
            self._b_best_quality = self._b_quality_ema
        if improved or trend_improved or promoted:
            self._b_last_best_update = update
        if self._b_last_best_update < 0:
            # Fresh resume anchor — see Step.load_state.
            self._b_last_best_update = update

        no_improvement = update - self._b_last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )
        is_new_best = improved or promoted

        info.update({
            "recovery": round(recovery_rate, 3),
            "pushes": total_push,
            "push_falls": total_fall,
            "level": float(self._level),
            "force": round(self.current_force, 1),
        })
        return {
            "is_new_best": is_new_best,
            "stop_training": stop_training,
            "info": info,
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            **super().state(),
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "recovery_rate": self._recovery_rate,
            "b_best_quality": self._b_best_quality,
            "b_quality_ema": (
                self._b_quality_ema if self._b_quality_ema is not None
                else -1.0
            ),
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._recovery_rate = float(state.get("recovery_rate", 1.0))
        self._b_best_quality = float(state.get("b_best_quality", -1.0))
        _ema = float(state.get("b_quality_ema", -1.0))
        self._b_quality_ema = _ema if _ema >= 0 else None
        self._b_last_best_update = -1


EXPERIMENT_CLASS = BalanceStep
