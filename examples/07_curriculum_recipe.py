"""Example 07 — Curriculum learning as an ``options_fn`` recipe.

Why this is an example, not a library
-------------------------------------
Curriculum schedules are 5-line closures over ``episode_index``. Wrapping
them in a generic ``OptionsSchedule(kind="linear" | "step" | "cosine" |
"adaptive")`` would just recreate ``Callable[[int], dict]`` with worse
ergonomics — ``baseline/DESIGN.md`` §4 explicitly drops this as a point
because it is not deep enough to earn a slot.

What this file demonstrates
---------------------------
The same plugin (``CurriculumPushPlugin`` from ``examples/03``) can be
driven by **arbitrary** curriculum logic *without changing the plugin*,
by writing four ``options_fn`` closures that produce a per-episode
``{"push_force": magnitude}`` dict:

  1. ``linear_ramp``    — push grows linearly over ``ramp_episodes``.
  2. ``step_schedule``  — discrete jumps at episode milestones.
  3. ``cosine_schedule`` — smooth ramp with plateau via cosine warmup.
  4. ``adaptive_schedule`` — push grows only if recent return >= threshold
     ("getting too easy → bump it up").

We then plug each closure into :class:`baseline.common.eval.PolicyEvaluator`,
collect a small batch of episodes per schedule, and print mean returns +
the actually-applied push magnitudes side-by-side.

The takeaway: curriculum is just a closure over a counter. The
:class:`RolloutCollector` / :class:`PolicyEvaluator` ``options_fn``
parameter is the recommended pluggable seam (see
``envs/framework/RESET.md`` §4 for the ``ctx.episode_options`` channel
specification).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from _common import build_humanoid21_runtime
from baseline.common.eval import PolicyEvaluator
from envs.framework.policy import Policy

# Reuse the curriculum push plugin from example 03 — same code, no changes.
import importlib.util
import sys
from pathlib import Path

_EX03 = Path(__file__).resolve().parent / "03_training_aids.py"
_spec = importlib.util.spec_from_file_location("_ex03", _EX03)
assert _spec is not None and _spec.loader is not None
_ex03 = importlib.util.module_from_spec(_spec)
sys.modules["_ex03"] = _ex03
_spec.loader.exec_module(_ex03)
CurriculumPushPlugin = _ex03.CurriculumPushPlugin


# ---------------------------------------------------------------------------
# Curriculum schedules — pure closures over ``episode_index``.
# ---------------------------------------------------------------------------
def linear_ramp(max_force: float, ramp_episodes: int) -> Callable[[int], Dict[str, Any]]:
    def schedule(episode_index: int) -> Dict[str, Any]:
        progress = min(1.0, episode_index / max(1, ramp_episodes))
        return {"push_force": progress * max_force}
    return schedule


def step_schedule(steps: List[tuple[int, float]]) -> Callable[[int], Dict[str, Any]]:
    """``steps`` is sorted [(start_episode, force), ...] — most-recent applies."""
    def schedule(episode_index: int) -> Dict[str, Any]:
        force = 0.0
        for start, value in steps:
            if episode_index >= start:
                force = value
        return {"push_force": force}
    return schedule


def cosine_schedule(max_force: float, ramp_episodes: int) -> Callable[[int], Dict[str, Any]]:
    def schedule(episode_index: int) -> Dict[str, Any]:
        progress = min(1.0, episode_index / max(1, ramp_episodes))
        # Cosine warmup: smooth S-curve from 0 to 1.
        warmup = 0.5 * (1.0 - math.cos(math.pi * progress))
        return {"push_force": warmup * max_force}
    return schedule


@dataclass
class _AdaptiveState:
    current_force: float = 0.0
    recent_returns: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.recent_returns is None:
            self.recent_returns = []


def adaptive_schedule(
    state: _AdaptiveState,
    *,
    threshold: float,
    bump: float,
    window: int,
) -> Callable[[int], Dict[str, Any]]:
    """Push force grows only when the trailing-``window`` return clears ``threshold``.

    The caller updates ``state.recent_returns`` between collects; the
    closure reads it on the next call. Standard "automatic curriculum"
    pattern (no domain knowledge needed beyond a return threshold).
    """
    def schedule(episode_index: int) -> Dict[str, Any]:
        if len(state.recent_returns) >= window:
            avg = float(np.mean(state.recent_returns[-window:]))
            if avg >= threshold:
                state.current_force += bump
                state.recent_returns.clear()
        return {"push_force": state.current_force}
    return schedule


# ---------------------------------------------------------------------------
# A trivial policy — random small actions. The point is the curriculum
# wiring, not the policy.
# ---------------------------------------------------------------------------
class _RandomPolicy(Policy):
    def __init__(self, action_dim: int = 21, seed: int = 0) -> None:
        self._action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Any) -> np.ndarray:
        return (self._rng.standard_normal(self._action_dim) * 0.1).astype(np.float32)


def _policy_factory() -> Policy:
    return _RandomPolicy()


# ---------------------------------------------------------------------------
# Run one schedule, return per-episode (push_force, return_a) tuples.
# ---------------------------------------------------------------------------
def _evaluate_schedule(
    name: str,
    options_fn: Callable[[int], Dict[str, Any]],
    *,
    n_episodes: int,
    base_seed: int,
) -> Dict[str, Any]:
    def runtime_factory():
        return build_humanoid21_runtime(
            match_duration=2.0,
            extra_plugins=[CurriculumPushPlugin()],
        )

    # No reward observer registered → tell the collector to skip the
    # reward channel (PolicyEvaluator will report return=0 for every
    # episode, which is fine here — we only care about the curriculum
    # wiring + that the schedule fires per-episode as expected).
    evaluator = PolicyEvaluator(
        runtime_factory=runtime_factory,
        policy_factories={
            "robot_a": _policy_factory,
            "robot_b": _policy_factory,
        },
        reward_observer_template=None,
    )
    try:
        report = evaluator.evaluate(
            n=n_episodes,
            base_seed=base_seed,
            options_fn=options_fn,
        )
    finally:
        evaluator.close()

    return {
        "name": name,
        "mean_return": float(report.per_agent["robot_a"]["return"].mean),
        "raw_returns": report.per_agent["robot_a"]["return"].raw.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Example 07 — Curriculum recipes via options_fn")
    print("=" * 70)

    # Tiny budget on purpose — this is a wiring demo, not a benchmark.
    n_episodes = 4
    base_seed = 100

    schedules = {
        "linear_ramp(max=200, ramp=3)": linear_ramp(max_force=200.0, ramp_episodes=3),
        "step([(0,0), (2,150), (3,300)])": step_schedule(
            [(0, 0.0), (2, 150.0), (3, 300.0)]
        ),
        "cosine(max=200, ramp=4)": cosine_schedule(max_force=200.0, ramp_episodes=4),
    }

    print(f"\nRunning {n_episodes} episodes for each of {len(schedules)} schedules.\n")
    print("Each row shows the per-episode push_force the schedule emits to")
    print("CurriculumPushPlugin via ctx.episode_options['push_force'].\n")
    print(f"{'schedule':<35} | per-episode push_force (N)")
    print("-" * 90)
    for name, fn in schedules.items():
        forecast = [float(fn(i)["push_force"]) for i in range(n_episodes)]
        forecast_str = " ".join(f"{v:>7.2f}" for v in forecast)
        print(f"{name:<35} | {forecast_str}")
        # And actually run them — confirms the closure plumbs through
        # RolloutCollector / EpisodeRunner without errors. Returns are 0
        # because we did not register a reward observer; that's fine, the
        # point is the curriculum dispatch.
        _evaluate_schedule(
            name=name, options_fn=fn,
            n_episodes=n_episodes, base_seed=base_seed,
        )

    # Adaptive curriculum: state lives outside the closure, so subsequent
    # invocations see updated force. We simulate "got rewards" by feeding
    # synthetic returns into the state — in real training, the trainer
    # appends the previous batch's mean return between collect() calls.
    print("\nAdaptive (push_force += 50 when last 2 returns avg >= 1.0):")
    state = _AdaptiveState()
    fn = adaptive_schedule(state, threshold=1.0, bump=50.0, window=2)
    forecast: List[float] = []
    for i in range(n_episodes):
        forecast.append(float(fn(i)["push_force"]))
        # Pretend the trainer just collected a batch with mean return = 1.5
        # (above threshold) so the schedule will bump on the next tick.
        state.recent_returns.append(1.5)
    forecast_str = " ".join(f"{v:>7.2f}" for v in forecast)
    print(f"  per-episode push_force = [{forecast_str}]")
    print(f"  final adaptive force   = {state.current_force:.1f}")

    print("\n" + "=" * 70)
    print("Takeaway: curriculum is a Callable[[int], dict] closure — the")
    print("RolloutCollector / PolicyEvaluator options_fn parameter is the")
    print("recommended seam. No 'OptionsSchedule' library required.")


if __name__ == "__main__":
    main()
