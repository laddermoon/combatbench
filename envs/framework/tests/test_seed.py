"""Seed-management tests.

Covers the contract spelled out in ``envs/framework/SEED.md``:
  * ``_resolve_seed`` never returns ``None``
  * ``_derive_batch_seeds`` is deterministic and matches between
    :class:`EpisodeRunner` and :class:`ParallelRunner`
  * ``EpisodeRunner._derive_seeds`` uses ``SeedSequence.spawn`` and gives
    every consumer (runtime, policies, seedable plugins) a unique,
    reproducible ``int`` seed
  * ``_reset_all`` order is plugin-seed → ``runtime.reset`` → policy-seed,
    and ``ctx.base_seed`` is published before ``runtime.reset`` fires
  * :class:`BasePlugin` default ``set_episode_seed`` is a no-op and such
    plugins are NOT routed to seed allocation
  * :class:`BaseFrameRecorder` persists ``base_seed`` in the episode
    ``manifest.json``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytest

from envs.framework.context import SimContext
from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import (
    AGENT_IDS,
    EpisodeRunner,
    EpisodeSeeds,
    _derive_batch_seeds,
    _resolve_seed,
)
from envs.framework.parallel_runner import _derive_seeds as _parallel_derive_seeds
from envs.framework.plugin import BasePlugin
from envs.framework.policy import Policy
from envs.framework.recorder import BaseFrameRecorder
from envs.framework.runtime_plugin import BaseObserverPlugin


# ---------------------------------------------------------------------------
# Helpers: a minimal Policy + seedable plugin + identity observer plugin.
# ---------------------------------------------------------------------------
class _ZeroPolicy(Policy):
    def __init__(self, action_dim: int = 21):
        self.action_dim = action_dim
        self.reset_seeds: List[Optional[int]] = []

    def act(self, observation: Any) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> None:
        self.reset_seeds.append(seed)


class _SeedableTrackerPlugin(BasePlugin):
    """Plugin that records every seed it is handed. Overrides
    :meth:`BasePlugin.set_episode_seed` so the runner enumerates it."""

    def __init__(self, plugin_name: str = "seed_tracker"):
        self._name = plugin_name
        self.seeds_received: List[int] = []

    @property
    def name(self) -> str:
        return self._name

    def set_episode_seed(self, seed: int) -> None:
        # Matches the production pattern: one line, rebuild RNG immediately.
        self.seeds_received.append(int(seed))


class _NoRngPlugin(BasePlugin):
    """Plugin with no RNG; keeps default ``set_episode_seed`` (no-op). The
    runner must NOT enumerate it for seed allocation."""

    @property
    def name(self) -> str:
        return "no_rng"


class _IdentityObserver(BaseObserverPlugin):
    """Returns a fixed float so reward extraction is deterministic."""

    def __init__(self, value: float = 0.0):
        self._value = float(value)

    def on_reset(self, ctx): ...
    def on_post_step(self, ctx): ...
    def on_post_episode(self, ctx): ...
    def on_manual_refresh(self, ctx): ...

    def get_output(self):
        return self._value


def _build_runner(
    mock_simulator,
    extra_plugins: Optional[List[BasePlugin]] = None,
    max_steps: int = 3,
) -> EpisodeRunner:
    from envs.framework.common_plugins import TimeoutPlugin
    from envs.framework.episode_runner import ObserverBinding

    plugins: List[BasePlugin] = [TimeoutPlugin(max_steps=max_steps)]
    plugins.extend(extra_plugins or [])
    runtime = EnvRuntime(
        simulator=mock_simulator,
        plugins=plugins,
        observer_plugins={
            "robot_a_obs": _IdentityObserver(0.1),
            "robot_a_reward": _IdentityObserver(0.0),
            "robot_b_obs": _IdentityObserver(0.2),
            "robot_b_reward": _IdentityObserver(0.0),
        },
        phy_steps_per_action=1,
        max_steps=max_steps,
    )
    return EpisodeRunner(
        runtime=runtime,
        policies={"robot_a": _ZeroPolicy(), "robot_b": _ZeroPolicy()},
        observer_bindings={
            "robot_a": ObserverBinding(obs_name="robot_a_obs", reward_name="robot_a_reward"),
            "robot_b": ObserverBinding(obs_name="robot_b_obs", reward_name="robot_b_reward"),
        },
    )


# ---------------------------------------------------------------------------
# _resolve_seed
# ---------------------------------------------------------------------------
class TestResolveSeed:
    def test_none_resolves_to_int(self):
        seed = _resolve_seed(None)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32

    def test_none_resolutions_are_not_all_equal(self):
        """``secrets.randbits`` is nondeterministic between calls."""
        seeds = {_resolve_seed(None) for _ in range(10)}
        assert len(seeds) > 1, (
            "10 independent _resolve_seed(None) calls collapsed to one "
            "value; secrets.randbits should give enough entropy."
        )

    def test_int_passthrough(self):
        assert _resolve_seed(42) == 42
        assert _resolve_seed(0) == 0
        assert _resolve_seed(2**32 - 1) == 2**32 - 1

    def test_returns_python_int_not_numpy(self):
        # Downstream json.dump hates numpy ints; guard against regression.
        value = _resolve_seed(np.uint32(7))
        assert type(value) is int


# ---------------------------------------------------------------------------
# _derive_batch_seeds — shared between EpisodeRunner and ParallelRunner
# ---------------------------------------------------------------------------
class TestDeriveBatchSeeds:
    def test_deterministic_for_same_base(self):
        a = _derive_batch_seeds(42, 8)
        b = _derive_batch_seeds(42, 8)
        np.testing.assert_array_equal(a, b)

    def test_different_base_gives_different_batch(self):
        a = _derive_batch_seeds(42, 8)
        b = _derive_batch_seeds(43, 8)
        assert not np.array_equal(a, b)

    def test_parallel_runner_uses_same_derivation(self):
        """ParallelRunner._derive_seeds must call the shared helper so
        sequential and parallel paths run the same episodes."""
        direct = _derive_batch_seeds(123, 5)
        via_parallel = _parallel_derive_seeds(123, 5)
        np.testing.assert_array_equal(direct, via_parallel)

    def test_parallel_none_resolved_at_entry(self):
        """``base_seed=None`` in ParallelRunner must yield a concrete
        batch (not crash) and two independent calls typically differ."""
        a = _parallel_derive_seeds(None, 4)
        b = _parallel_derive_seeds(None, 4)
        assert a.shape == (4,)
        assert b.shape == (4,)
        # Overwhelmingly likely to differ; if this flakes, entropy source broke.
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# EpisodeRunner._derive_seeds — via SeedSequence.spawn
# ---------------------------------------------------------------------------
class TestEpisodeSeedsDerivation:
    def test_same_base_same_bundle(self, mock_simulator):
        tracker = _SeedableTrackerPlugin()
        runner = _build_runner(mock_simulator, extra_plugins=[tracker])

        s1 = runner._derive_seeds(42)
        s2 = runner._derive_seeds(42)

        assert s1 == s2
        assert isinstance(s1, EpisodeSeeds)
        assert s1.base == 42
        assert set(s1.policies.keys()) == set(AGENT_IDS)
        assert id(tracker) in s1.plugins

    def test_different_base_different_bundle(self, mock_simulator):
        runner = _build_runner(mock_simulator, extra_plugins=[_SeedableTrackerPlugin()])

        s1 = runner._derive_seeds(42)
        s2 = runner._derive_seeds(43)

        assert s1.runtime != s2.runtime
        assert s1.policies != s2.policies
        assert s1.plugins != s2.plugins

    def test_all_derived_seeds_are_int(self, mock_simulator):
        runner = _build_runner(mock_simulator, extra_plugins=[_SeedableTrackerPlugin()])
        seeds = runner._derive_seeds(42)
        assert type(seeds.runtime) is int
        assert all(type(v) is int for v in seeds.policies.values())
        assert all(type(v) is int for v in seeds.plugins.values())

    def test_per_consumer_seeds_are_distinct(self, mock_simulator):
        runner = _build_runner(
            mock_simulator,
            extra_plugins=[
                _SeedableTrackerPlugin("tracker_a"),
                _SeedableTrackerPlugin("tracker_b"),
            ],
        )
        seeds = runner._derive_seeds(42)
        all_values = {seeds.runtime, *seeds.policies.values(), *seeds.plugins.values()}
        total = 1 + len(seeds.policies) + len(seeds.plugins)
        assert len(all_values) == total, (
            f"Expected {total} distinct seeds, got {len(all_values)}. "
            f"SeedSequence.spawn children collided."
        )

    def test_plugin_seed_keyed_by_object_id(self, mock_simulator):
        t1 = _SeedableTrackerPlugin("dup_name")
        t2 = _SeedableTrackerPlugin("dup_name")  # same .name, distinct instances
        runner = _build_runner(mock_simulator, extra_plugins=[t1, t2])
        seeds = runner._derive_seeds(42)
        # Both must be allocated independently despite name collision.
        assert id(t1) in seeds.plugins
        assert id(t2) in seeds.plugins
        assert seeds.plugins[id(t1)] != seeds.plugins[id(t2)]


# ---------------------------------------------------------------------------
# Seedable plugin enumeration
# ---------------------------------------------------------------------------
class TestSeedablePluginEnumeration:
    def test_default_no_op_plugin_is_not_enumerated(self, mock_simulator):
        no_rng = _NoRngPlugin()
        seeded = _SeedableTrackerPlugin("seeded")
        runner = _build_runner(mock_simulator, extra_plugins=[no_rng, seeded])

        enumerated = runner._seedable_plugins()
        assert seeded in enumerated
        assert no_rng not in enumerated

    def test_baseplugin_default_set_episode_seed_is_noop(self):
        p = _NoRngPlugin()
        # Must not raise and must not mutate anything.
        p.set_episode_seed(7)
        assert p.set_episode_seed(7) is None


# ---------------------------------------------------------------------------
# _reset_all order + ctx.base_seed publication
# ---------------------------------------------------------------------------
class _OrderProbePlugin(BasePlugin):
    """Watches set_episode_seed AND on_pre_episode to verify ordering vs
    runtime.reset. on_pre_episode fires from inside runtime.reset, so we
    can assert the plugin's RNG was rebuilt BEFORE that hook."""

    def __init__(self):
        self.order: List[str] = []
        self._rng: Optional[np.random.RandomState] = None

    @property
    def name(self) -> str:
        return "order_probe"

    def set_episode_seed(self, seed: int) -> None:
        self.order.append("set_episode_seed")
        self._rng = np.random.RandomState(int(seed))

    def on_pre_episode(self, ctx) -> None:
        self.order.append("on_pre_episode")
        # The runner must have published ctx.base_seed BEFORE runtime.reset.
        assert ctx.base_seed is not None, (
            "ctx.base_seed must be set before runtime.reset fires plugin "
            "on_pre_episode hooks."
        )
        # Our RNG must have been rebuilt already.
        assert self._rng is not None


class TestResetAllOrdering:
    def test_plugin_seed_then_runtime_then_policy(self, mock_simulator):
        probe = _OrderProbePlugin()
        runner = _build_runner(mock_simulator, extra_plugins=[probe])

        runner.run_episode(seed=99)

        # set_episode_seed must strictly precede on_pre_episode.
        assert probe.order[0] == "set_episode_seed"
        assert probe.order[1] == "on_pre_episode"

    def test_policy_reset_receives_derived_seed(self, mock_simulator):
        runner = _build_runner(mock_simulator)
        runner.run_episode(seed=42)
        a = runner.policies["robot_a"]
        b = runner.policies["robot_b"]
        assert isinstance(a, _ZeroPolicy) and isinstance(b, _ZeroPolicy)
        assert len(a.reset_seeds) == 1 and len(b.reset_seeds) == 1
        # Derived seeds are concrete ints (never None).
        assert isinstance(a.reset_seeds[0], int)
        assert isinstance(b.reset_seeds[0], int)
        # Two policies get DIFFERENT seeds (spawn, not shared).
        assert a.reset_seeds[0] != b.reset_seeds[0]

    def test_none_seed_resolves_and_writes_back_to_result(self, mock_simulator):
        runner = _build_runner(mock_simulator)
        result = runner.run_episode(seed=None)
        assert isinstance(result.seed, int)
        # Record matches what the runner used to derive the rest.
        seeds_rederived = runner._derive_seeds(result.seed)
        assert seeds_rederived.base == result.seed


# ---------------------------------------------------------------------------
# Recorder manifest — base_seed
# ---------------------------------------------------------------------------
class TestRecorderManifestBaseSeed:
    def test_manifest_contains_resolved_base_seed(self, mock_simulator, tmp_path: Path):
        recorder = BaseFrameRecorder(
            output_dir=tmp_path / "rec",
            save_image=False,
            save_core_state=True,
            save_static_data=False,
            quiet=True,
        )
        runner = _build_runner(mock_simulator)
        runner.runtime.attach_recorder(recorder)

        runner.run_episode(seed=1234)

        manifest_path = tmp_path / "rec" / "episode_00000" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["base_seed"] == 1234

    def test_manifest_records_none_seed_resolution(self, mock_simulator, tmp_path: Path):
        recorder = BaseFrameRecorder(
            output_dir=tmp_path / "rec",
            save_image=False,
            save_core_state=True,
            save_static_data=False,
            quiet=True,
        )
        runner = _build_runner(mock_simulator)
        runner.runtime.attach_recorder(recorder)

        result = runner.run_episode(seed=None)

        manifest = json.loads(
            (tmp_path / "rec" / "episode_00000" / "manifest.json").read_text()
        )
        # Manifest captures the RESOLVED int — same value as result.seed.
        assert manifest["base_seed"] == result.seed
        assert isinstance(manifest["base_seed"], int)


# ---------------------------------------------------------------------------
# SimContext.base_seed plumbing
# ---------------------------------------------------------------------------
class TestSimContextBaseSeed:
    def test_default_base_seed_is_none(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        assert ctx.base_seed is None

    def test_readonly_context_exposes_base_seed(self, mock_simulator):
        from envs.framework.context import ReadOnlySimContext

        ctx = SimContext(mock_simulator)
        ctx.base_seed = 7
        readonly = ReadOnlySimContext.from_sim_context(ctx)
        assert readonly.base_seed == 7
