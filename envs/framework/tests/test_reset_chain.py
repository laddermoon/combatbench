"""Invariant tests for the reset chain — pin RESET.md §8 (I1–I6) and §7-G4.

Each test maps to one numbered invariant in ``envs/framework/RESET.md``.
If you change the reset call chain, run these first; if you need to break
one, update RESET.md §8 in the same patch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.framework import EnvRuntime
from envs.framework.episode_runner import (
    AGENT_IDS,
    EpisodeRunner,
    ObserverBinding,
    RolloutConfig,
)
from envs.framework.plugin import BasePlugin
from envs.framework.policy import Policy
from envs.framework.recorder import PostActionRecorder
from envs.framework.runtime_plugin import BaseObserverPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _StaticActionPolicy(Policy):
    """Deterministic zero-action policy — keeps trajectories bit-equal across
    runs at the same seed (no policy RNG involvement)."""

    def act(self, obs: Any) -> np.ndarray:
        return np.zeros(21, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> None:
        return None


def _bindings() -> Dict[str, ObserverBinding]:
    return {
        agent: ObserverBinding(obs_name=f"{agent}_obs", reward_name=None)
        for agent in AGENT_IDS
    }


class _ZeroObserver(BaseObserverPlugin):
    """Minimal observer; just satisfies obs binding requirements."""

    def __init__(self) -> None:
        self._output = np.zeros(1, dtype=np.float32)

    def on_pre_episode(self, ctx) -> None:  # noqa: D401
        pass

    def on_post_action_step(self, ctx) -> None:  # noqa: D401
        pass

    def get_output(self) -> Any:
        return self._output


# ---------------------------------------------------------------------------
# I1 — episode_step / physics_step are 0 after reset
# ---------------------------------------------------------------------------
class TestI1_StepCountersZeroedOnReset:
    def test_episode_and_physics_step_are_zero_after_reset(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator)
        runtime.reset()
        assert runtime.ctx.episode_step == 0
        assert runtime.ctx.physics_step == 0

    def test_counters_reset_on_consecutive_episodes(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator, max_steps=2)
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        assert runtime.ctx.episode_step == 1
        runtime.reset()  # explicit second reset
        assert runtime.ctx.episode_step == 0
        assert runtime.ctx.physics_step == 0


# ---------------------------------------------------------------------------
# I2 — termination_proposals empty after reset (unless on_pre_episode raised
# one, in which case the episode is already terminated and is_episode_active
# is False).
# ---------------------------------------------------------------------------
class _RequestTerminationOnPreEpisode(BasePlugin):
    """Test plugin: requests termination during on_pre_episode."""

    @property
    def name(self) -> str:
        return "early_killer"

    def on_pre_episode(self, ctx) -> None:
        ctx.request_termination("test_pre_episode_termination")


class TestI2_TerminationProposalsCleanAfterReset:
    def test_clean_when_no_plugin_terminates(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator)
        runtime.reset()
        assert runtime.ctx.termination_proposals == []
        assert runtime.is_episode_active is True

    def test_pre_episode_termination_yields_inactive_runtime(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_RequestTerminationOnPreEpisode()],
        )
        runtime.reset()
        # Plugin requested termination ⇒ episode immediately terminated.
        assert "test_pre_episode_termination" in runtime.ctx.termination_proposals
        assert runtime.is_episode_active is False


# ---------------------------------------------------------------------------
# I3 — same base_seed → bit-equal trajectories
# (smaller and faster than the existing test_seed.py determinism tests; this
# one specifically exercises the reset chain with options.)
# ---------------------------------------------------------------------------
class TestI3_SameSeedSameTrajectory:
    def test_same_seed_produces_identical_observations_and_options(self):
        from .conftest import MockSimulator

        def _build_runner():
            runtime = EnvRuntime(
                simulator=MockSimulator(),
                observer_plugins={
                    "robot_a_obs": _ZeroObserver(),
                    "robot_b_obs": _ZeroObserver(),
                },
                max_steps=3,
            )
            return EpisodeRunner(
                runtime=runtime,
                policies={a: _StaticActionPolicy() for a in AGENT_IDS},
                observer_bindings=_bindings(),
                rollout=RolloutConfig(capture_a=True, capture_b=False),
            )

        opts = {"initial_distance": 1.5, "push_force": 42.0}
        r1 = _build_runner().run_episode(seed=12345, options=opts)
        r2 = _build_runner().run_episode(seed=12345, options=opts)
        assert r1.seed == r2.seed
        assert r1.num_steps == r2.num_steps
        traj1 = r1.trajectories["robot_a"]
        traj2 = r2.trajectories["robot_a"]
        assert len(traj1.observations) == len(traj2.observations)
        for o1, o2 in zip(traj1.observations, traj2.observations):
            assert np.array_equal(o1, o2)


# ---------------------------------------------------------------------------
# I4 — options keys are visible to plugins AND observers via
# ctx.episode_options on the on_pre_episode hook (and stay visible on
# subsequent on_post_action_step hooks).
# ---------------------------------------------------------------------------
class _OptionsCapturingPlugin(BasePlugin):
    """Records options seen at on_pre_episode + on_post_action_step."""

    def __init__(self) -> None:
        self.pre_episode_options: Dict[str, Any] = {}
        self.post_step_options: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "options_capture"

    def on_pre_episode(self, ctx) -> None:
        self.pre_episode_options = dict(ctx.episode_options)

    def on_post_action_step(self, ctx) -> None:
        self.post_step_options = dict(ctx.episode_options)


class _OptionsCapturingObserver(BaseObserverPlugin):
    """Same idea, but on the observer side (ReadOnlySimContext)."""

    def __init__(self) -> None:
        self.pre_episode_options: Dict[str, Any] = {}
        self.post_step_options: Dict[str, Any] = {}
        self._output = 0.0

    def on_pre_episode(self, ctx) -> None:
        self.pre_episode_options = dict(ctx.episode_options)

    def on_post_action_step(self, ctx) -> None:
        self.post_step_options = dict(ctx.episode_options)

    def get_output(self) -> float:
        return self._output


class TestI4_OptionsVisibleToPluginsAndObservers:
    def test_plugin_sees_options_on_pre_episode_and_post_step(self, mock_simulator):
        plugin = _OptionsCapturingPlugin()
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])
        runtime.reset(options={"push_force": 42.0, "opponent": "snapshot_7"})
        assert plugin.pre_episode_options == {"push_force": 42.0, "opponent": "snapshot_7"}

        runtime.step(np.zeros(21), np.zeros(21))
        assert plugin.post_step_options == {"push_force": 42.0, "opponent": "snapshot_7"}

    def test_observer_sees_options_on_pre_episode_and_post_step(self, mock_simulator):
        observer = _OptionsCapturingObserver()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"capture": observer},
        )
        runtime.reset(options={"initial_distance": 1.25})
        assert observer.pre_episode_options == {"initial_distance": 1.25}

        runtime.step(np.zeros(21), np.zeros(21))
        assert observer.post_step_options == {"initial_distance": 1.25}

    def test_options_cleared_between_episodes(self, mock_simulator):
        plugin = _OptionsCapturingPlugin()
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])
        runtime.reset(options={"push_force": 100.0})
        assert plugin.pre_episode_options == {"push_force": 100.0}
        runtime.reset(options=None)
        assert plugin.pre_episode_options == {}

    def test_episode_runner_threads_options_through(self, mock_simulator):
        plugin = _OptionsCapturingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            observer_plugins={
                "robot_a_obs": _ZeroObserver(),
                "robot_b_obs": _ZeroObserver(),
            },
            max_steps=1,
        )
        runner = EpisodeRunner(
            runtime=runtime,
            policies={a: _StaticActionPolicy() for a in AGENT_IDS},
            observer_bindings=_bindings(),
        )
        runner.run_episode(seed=1, options={"hp_a": 75.0, "hp_b": 50.0})
        assert plugin.pre_episode_options == {"hp_a": 75.0, "hp_b": 50.0}

    def test_run_n_episodes_options_fn_called_per_index(self):
        from .conftest import MockSimulator

        captured_indices: List[int] = []

        def options_fn(idx: int) -> Dict[str, Any]:
            captured_indices.append(idx)
            return {"epoch_index": idx, "push": idx * 10.0}

        plugin = _OptionsCapturingPlugin()
        runtime = EnvRuntime(
            simulator=MockSimulator(),
            plugins=[plugin],
            observer_plugins={
                "robot_a_obs": _ZeroObserver(),
                "robot_b_obs": _ZeroObserver(),
            },
            max_steps=1,
        )
        runner = EpisodeRunner(
            runtime=runtime,
            policies={a: _StaticActionPolicy() for a in AGENT_IDS},
            observer_bindings=_bindings(),
        )
        runner.run_n_episodes(3, base_seed=42, options_fn=options_fn)
        # options_fn must have been called once per episode with the right index
        assert captured_indices == [0, 1, 2]
        # And the LAST episode's options must be the last ones the plugin saw
        assert plugin.pre_episode_options == {"epoch_index": 2, "push": 20.0}


# ---------------------------------------------------------------------------
# I5 — observer.on_pre_episode runs BEFORE all non-dispatcher plugin
# on_pre_episode hooks.
# ---------------------------------------------------------------------------
class _OrderRecorderPlugin(BasePlugin):
    def __init__(self, log: List[str], label: str, priority_value: int = 0) -> None:
        self._log = log
        self._label = label
        self._priority = priority_value

    @property
    def name(self) -> str:
        return f"order_plugin:{self._label}"

    @property
    def priority(self) -> int:
        return self._priority

    def on_pre_episode(self, ctx) -> None:
        self._log.append(f"plugin:{self._label}")


class _OrderRecorderObserver(BaseObserverPlugin):
    def __init__(self, log: List[str], label: str) -> None:
        self._log = log
        self._label = label
        self._output = 0.0

    def on_pre_episode(self, ctx) -> None:
        self._log.append(f"observer:{self._label}")

    def get_output(self) -> float:
        return self._output


class TestI5_ObserverPrecedesPluginsOnPreEpisode:
    def test_observer_on_pre_episode_runs_before_plugins(self, mock_simulator):
        log: List[str] = []
        # Default-priority plugins — observer dispatcher (priority=1e6) must
        # still beat them.
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[
                _OrderRecorderPlugin(log, "low", priority_value=0),
                _OrderRecorderPlugin(log, "high", priority_value=10),
            ],
            observer_plugins={
                "obs1": _OrderRecorderObserver(log, "first"),
                "obs2": _OrderRecorderObserver(log, "second"),
            },
        )
        runtime.reset()

        # All observer entries should appear before any plugin entry.
        first_plugin_idx = next(i for i, e in enumerate(log) if e.startswith("plugin:"))
        observer_indices = [i for i, e in enumerate(log) if e.startswith("observer:")]
        assert observer_indices, "no observer hooks observed"
        assert max(observer_indices) < first_plugin_idx, (
            f"observer.on_pre_episode must precede plugins; got log={log}"
        )

    def test_high_priority_plugin_does_not_overtake_observer(self, mock_simulator):
        """Even a plugin with priority > 1_000_000 ... wait, actually the
        dispatcher's priority is fixed at 1_000_000 and a sufficiently high
        plugin priority CAN run earlier.  RESET.md §3.4 only guarantees the
        ordering for "normal" priorities.  So this test pins the realistic
        guarantee: at default and reasonable priorities, observers win."""
        log: List[str] = []
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_OrderRecorderPlugin(log, "p", priority_value=999_999)],
            observer_plugins={"o": _OrderRecorderObserver(log, "o")},
        )
        runtime.reset()
        # Observer (priority 1_000_000) > plugin (999_999) → observer first.
        observer_idx = log.index("observer:o")
        plugin_idx = log.index("plugin:p")
        assert observer_idx < plugin_idx


# ---------------------------------------------------------------------------
# I6 — recorder.on_pre_episode runs AFTER all plugin on_pre_episode hooks.
# ---------------------------------------------------------------------------
class _OrderRecorder(PostActionRecorder):
    def __init__(self, log: List[str]) -> None:
        self._log = log

    def on_pre_episode(self, ctx, observer_outputs) -> None:
        self._log.append("recorder:pre_episode")

    def on_post_action_step(self, ctx, observer_outputs, action_extras=None) -> None:  # noqa: D401
        pass

    def on_post_episode(self, ctx, observer_outputs) -> None:
        self._log.append("recorder:post_episode")


class TestI6_RecorderRunsAfterPlugins:
    def test_recorder_pre_episode_after_plugin_pre_episode(self, mock_simulator):
        log: List[str] = []
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_OrderRecorderPlugin(log, "p1"), _OrderRecorderPlugin(log, "p2")],
            recorders=[_OrderRecorder(log)],
        )
        runtime.reset()

        plugin_indices = [i for i, e in enumerate(log) if e.startswith("plugin:")]
        recorder_idx = log.index("recorder:pre_episode")
        assert plugin_indices, "no plugin hooks observed"
        assert recorder_idx > max(plugin_indices), (
            f"recorder must come after plugins; got log={log}"
        )


# ---------------------------------------------------------------------------
# G4 — mid-episode reset gracefully terminates the in-flight episode
# (on_post_episode fires for the abandoned episode with reason "abandoned").
# ---------------------------------------------------------------------------
class _PostEpisodeRecordingPlugin(BasePlugin):
    def __init__(self) -> None:
        self.post_episode_calls: List[Tuple[int, Tuple[str, ...]]] = []

    @property
    def name(self) -> str:
        return "post_episode_observer"

    def on_post_episode(self, ctx) -> None:
        self.post_episode_calls.append(
            (ctx.episode_step, tuple(ctx.termination_proposals))
        )


class TestG4_MidEpisodeResetGracefulTermination:
    def test_mid_episode_reset_fires_post_episode_with_abandoned_reason(
        self, mock_simulator
    ):
        plugin = _PostEpisodeRecordingPlugin()
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        # is_episode_active is True; reset mid-episode now → graceful terminate.
        assert runtime.is_episode_active is True
        runtime.reset()
        # Post-episode fired exactly once for the abandoned run, with reason.
        assert len(plugin.post_episode_calls) == 1
        _step, reasons = plugin.post_episode_calls[0]
        assert "abandoned" in reasons
        # And the new episode is fresh.
        assert runtime.is_episode_active is True
        assert runtime.ctx.episode_step == 0
        assert runtime.ctx.termination_proposals == []

    def test_normal_reset_does_not_fire_extra_post_episode(self, mock_simulator):
        plugin = _PostEpisodeRecordingPlugin()
        runtime = EnvRuntime(
            simulator=mock_simulator, plugins=[plugin], max_steps=1
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))  # episode ends naturally
        assert runtime.is_episode_active is False
        # First post_episode = natural timeout, no "abandoned" reason.
        assert len(plugin.post_episode_calls) == 1
        runtime.reset()  # Now is_episode_active was False → no extra call.
        assert len(plugin.post_episode_calls) == 1


# ---------------------------------------------------------------------------
# G5 — base_seed ownership: clear_episode_state wipes prior value, runtime
# writes new one.
# ---------------------------------------------------------------------------
class TestG5_BaseSeedOwnership:
    def test_base_seed_set_via_runtime_reset_arg(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator)
        runtime.reset(base_seed=12345)
        assert runtime.ctx.base_seed == 12345

    def test_base_seed_cleared_when_not_provided_to_next_reset(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator)
        runtime.reset(base_seed=99)
        assert runtime.ctx.base_seed == 99
        runtime.reset()  # caller deliberately passes nothing
        # Not leaked from the previous episode — see RESET.md §7-G5.
        assert runtime.ctx.base_seed is None

    def test_episode_runner_publishes_base_seed_on_ctx(self, mock_simulator):
        plugin = _OptionsCapturingPlugin()  # captures via on_pre_episode
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[plugin],
            observer_plugins={
                "robot_a_obs": _ZeroObserver(),
                "robot_b_obs": _ZeroObserver(),
            },
            max_steps=1,
        )
        # Snapshot ctx.base_seed seen by the plugin during on_pre_episode.
        seen_base_seeds: List[Optional[int]] = []

        original_on_pre_episode = plugin.on_pre_episode

        def _capture(ctx) -> None:
            seen_base_seeds.append(ctx.base_seed)
            original_on_pre_episode(ctx)

        plugin.on_pre_episode = _capture  # type: ignore[assignment]

        runner = EpisodeRunner(
            runtime=runtime,
            policies={a: _StaticActionPolicy() for a in AGENT_IDS},
            observer_bindings=_bindings(),
        )
        runner.run_episode(seed=4242)
        # EpisodeRunner _resolve_seed → 4242 → published on ctx.base_seed.
        assert seen_base_seeds == [4242]
