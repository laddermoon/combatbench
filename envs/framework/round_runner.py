"""Combat-specific round runner — thin subclass of :class:`EpisodeRunner`.

Historically this file held a standalone ``RoundRunner`` that re-implemented
the policy-step-collect loop and hard-coded combat interpretation (HP /
winner / damage tallies / hit-event printing). It has been refactored into
a thin adapter over :class:`envs.framework.episode_runner.EpisodeRunner`
so the generic loop (seed handling, observation/reward pulling, rollout
capture, hooks) lives in one place.

Public surface:

* ``RoundRunner(policy_a, policy_b, runtime, verbose=True)``
* ``runner.run(seed=None, options=None, videosave_path=None) -> dict`` where
  the dict has the historical keys: ``steps`` / ``winner`` / ``final_health``
  / ``damage_taken`` / ``termination_reasons``.

Lifecycle: ``RoundRunner.run`` does NOT close the runtime. Callers own
the runtime and are expected to call ``runtime.close()`` themselves once
they are done with it. This is consistent with
:meth:`EpisodeRunner.close`'s docstring ("Runtime lifecycle is owned by
the caller") and lets :class:`MatchRunner` re-use a single runtime across
multiple rounds via ``runtime.reset(options=...)`` instead of rebuilding.

New code should prefer :class:`EpisodeRunner` directly plus a post-hoc
reducer over recorder data — combat-specific fields (``health`` /
``damage_taken`` / ``winner``) are published by the humanoid21
``CombatScoringPlugin``, not by this runner.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .common_plugins import VideoRecorderPlugin
from .episode_runner import (
    AGENT_IDS,
    EpisodeResult,
    EpisodeRunner,
    ObserverBinding,
    RolloutConfig,
    StepContext,
)


class RoundRunner(EpisodeRunner):
    """Run a single combat round and surface the legacy result dict.

    Subclasses :class:`EpisodeRunner` with:
    - legacy positional ``(policy_a, policy_b, runtime, verbose)`` signature,
    - combat-specific ``on_step`` / ``on_episode_end`` hooks that print hit
      events and a round summary when ``verbose=True``,
    - a ``.run()`` method that returns the historical result dict.
      The runtime is **not** closed here — callers (e.g. :class:`MatchRunner`)
      own the runtime lifecycle and recycle it across rounds via
      ``runtime.reset(options=...)``.
    """

    def __init__(
        self,
        policy_a: Any,
        policy_b: Any,
        runtime: Any,
        verbose: bool = True,
    ) -> None:
        self.verbose = bool(verbose)
        # No rollout capture — match the old RoundRunner which only returned a
        # summary dict. Callers that want trajectories should use
        # ``EpisodeRunner`` directly.
        # Combat rounds judge winner/HP via ``shared_info_final`` (published
        # by humanoid21's ``CombatScoringPlugin``), NOT via per-step reward
        # observers, so we bind ``reward_name=None`` explicitly. This also
        # removes RoundRunner's implicit requirement that callers register
        # ``robot_{a,b}_reward`` observers — historically that coupling was
        # unstated and broke ``make_env`` users who only needed obs observers.
        super().__init__(
            runtime=runtime,
            policies={"robot_a": policy_a, "robot_b": policy_b},
            rollout=RolloutConfig(capture_a=False, capture_b=False),
            observer_bindings={
                agent: ObserverBinding(obs_name=f"{agent}_obs", reward_name=None)
                for agent in AGENT_IDS
            },
            on_step=self._verbose_on_step if self.verbose else None,
            on_episode_end=self._verbose_on_episode_end if self.verbose else None,
        )
        if self.verbose:
            self._print_header()

    # ------------------------------------------------------------------
    # Public: legacy ``run`` surface
    # ------------------------------------------------------------------
    def run(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        videosave_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one round. Returns the legacy result dict.

        ``options`` is forwarded to :meth:`EnvRuntime.reset` and visible to
        plugins / observers via ``ctx.episode_options`` — this is how
        :class:`MatchRunner` injects per-round HP carry-over
        (``initial_health_a`` / ``initial_health_b``). See RESET.md §4.

        ``videosave_path`` retargets any :class:`VideoRecorderPlugin`
        instances already attached to the runtime — historical behavior
        preserved for ``MatchRunner``. Implemented by merging
        ``{VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY: videosave_path}``
        into ``options``; the plugin picks it up from
        ``ctx.episode_options`` in its ``on_pre_episode``. options-level
        keys provided by the caller win over ``videosave_path``.

        The runtime is intentionally NOT closed here; the caller owns its
        lifecycle (see module docstring & RESET.md §7-G3).
        """
        merged_options = self._merge_video_path_into_options(options, videosave_path)
        result = self.run_episode(seed=seed, options=merged_options)
        return self._build_legacy_result(result)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_video_path_into_options(
        options: Optional[Dict[str, Any]],
        videosave_path: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if videosave_path is None:
            return options
        merged: Dict[str, Any] = dict(options or {})
        merged.setdefault(
            VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY, videosave_path
        )
        return merged

    def _build_legacy_result(self, result: EpisodeResult) -> Dict[str, Any]:
        shared = result.shared_info_final
        final_health = self._extract_dict(shared, "health")
        damage_taken = self._extract_dict(shared, "damage_taken")
        return {
            "steps": result.num_steps,
            "winner": self._resolve_winner(shared, final_health),
            "final_health": final_health,
            "damage_taken": damage_taken,
            "termination_reasons": list(result.termination_reasons),
        }

    @staticmethod
    def _extract_dict(shared: Dict[str, Any], key: str) -> Dict[str, float]:
        raw = shared.get(key) or shared.get("metrics", {}).get(key) or {}
        return {agent: float(raw.get(agent, 100.0)) for agent in AGENT_IDS}

    @staticmethod
    def _resolve_winner(
        shared: Dict[str, Any],
        final_health: Dict[str, float],
    ) -> str:
        declared = shared.get("winner")
        if isinstance(declared, str):
            return declared
        ha = final_health.get("robot_a", 0.0)
        hb = final_health.get("robot_b", 0.0)
        if ha <= 0.0 and hb <= 0.0:
            return "draw"
        if ha <= 0.0:
            return "robot_b"
        if hb <= 0.0:
            return "robot_a"
        if ha > hb:
            return "robot_a"
        if hb > ha:
            return "robot_b"
        return "draw"

    # ------------------------------------------------------------------
    # Verbose hooks (combat-specific printing lives here, not in the core)
    # ------------------------------------------------------------------
    def _print_header(self) -> None:
        print("=" * 60)
        print("CombatBench Round Started")
        print("=" * 60)

    def _verbose_on_step(self, ctx: StepContext) -> None:
        # Print hit events (combat-specific metric published by humanoid21
        # CombatScoringPlugin into shared_info["events"]).
        for event in ctx.shared_info.get("events", []) or []:
            if isinstance(event, dict) and event.get("type") == "hit":
                print(
                    f"[Step {ctx.step_index}] {event.get('attacker')} hit "
                    f"{event.get('defender')} at {event.get('part')} for "
                    f"{float(event.get('damage', 0.0)):.2f} damage!"
                )
        if ctx.step_index % 100 == 0:
            health = self._extract_dict(ctx.shared_info, "health")
            print(
                f"Step {ctx.step_index:03d} - HP: "
                f"robot_a={health['robot_a']:.1f}, robot_b={health['robot_b']:.1f}"
            )

    def _verbose_on_episode_end(self, result: EpisodeResult) -> None:
        legacy = self._build_legacy_result(result)
        print("-" * 60)
        print(f"Round ended. Total steps: {legacy['steps']}")
        print(f"Reason: {legacy['termination_reasons']}")
        print(f"Winner: {legacy['winner']}")
        print(
            f"Final HP: robot_a={legacy['final_health']['robot_a']:.1f}, "
            f"robot_b={legacy['final_health']['robot_b']:.1f}"
        )
        print("-" * 60)


# Preferred new name for clarity — use in new code; ``RoundRunner`` kept as
# the canonical class for backward compatibility with MatchRunner and the
# humanoid21 run scripts.
CombatRoundRunner = RoundRunner


if __name__ == "__main__":
    from envs.humanoid21 import make_env

    runtime = make_env(
        plugins=[VideoRecorderPlugin(fps=30, output_path="match.mp4")],
        match_duration=30.0,
    )

    class DummyPolicy:
        def act(self, obs):
            return np.zeros(21, dtype=np.float32)

    result = RoundRunner(DummyPolicy(), DummyPolicy(), runtime).run(seed=42)
    print(f"Result: {result}")
