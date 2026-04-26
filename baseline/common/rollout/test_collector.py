"""Integration tests for ``RolloutCollector``.

These tests pin the contract documented in ``baseline/DESIGN.md`` §3.3:

  * ``EpisodeRunner`` is the actual episode driver — collector never
    writes a ``while is_episode_active: step(...)`` loop.
  * Multi-controlled-agent works out of the box (both ``robot_a`` and
    ``robot_b`` returning lists of ``RolloutBatch``).
  * ``state_dicts`` hot-reloads weights without rebuilding the runtime.
  * ``capture_agents`` filters which agents are returned.
  * ``RolloutBatch`` invariants hold post-conversion (``obs.shape[0] ==
    actions.shape[0] + 1``, ``log_probs`` aligned with actions).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch

# Make project root importable (mirrors envs/framework/tests/conftest.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from envs.framework.env_runtime import EnvRuntime
from envs.framework.policy import Policy
from envs.framework.runtime_plugin import BaseObserverPlugin

from baseline.common.policies import (
    CriticMLP,
    TanhGaussianMLPPolicy,
    TorchPolicyAdapter,
)
from baseline.common.rollout import RolloutBatch, RolloutCollector

# Re-use the framework conftest's MockSimulator: import from the
# framework tests package by side effect (path injected above).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "envs" / "framework" / "tests"))
from conftest import MockSimulator  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Test observers
# ---------------------------------------------------------------------------
OBS_DIM = 5
ACTION_DIM = 21  # MockSimulator action shape


class _QposObserver(BaseObserverPlugin):
    def __init__(self) -> None:
        self._output = np.zeros(OBS_DIM, dtype=np.float32)

    def on_pre_episode(self, ctx) -> None:
        self._output = ctx.accessor.get_core_state()["qpos"][:OBS_DIM].astype(np.float32)

    def on_post_action_step(self, ctx) -> None:
        self._output = ctx.accessor.get_core_state()["qpos"][:OBS_DIM].astype(np.float32)

    def get_output(self) -> np.ndarray:
        return self._output.copy()


class _StepRewardObserver(BaseObserverPlugin):
    def __init__(self) -> None:
        self._step = 0

    def on_pre_episode(self, ctx) -> None:
        self._step = 0

    def on_post_action_step(self, ctx) -> None:
        self._step += 1

    def get_output(self) -> float:
        return float(self._step)


def _make_runtime(max_steps: int = 3) -> EnvRuntime:
    return EnvRuntime(
        simulator=MockSimulator(),
        observer_plugins={
            "robot_a_obs": _QposObserver(),
            "robot_a_reward": _StepRewardObserver(),
            "robot_b_obs": _QposObserver(),
            "robot_b_reward": _StepRewardObserver(),
        },
        max_steps=max_steps,
        phy_steps_per_action=1,
    )


def _make_adapter(deterministic: bool = True, with_critic: bool = True) -> TorchPolicyAdapter:
    torch.manual_seed(0)
    actor = TanhGaussianMLPPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=16)
    critic = CriticMLP(obs_dim=OBS_DIM, hidden_dim=16) if with_critic else None
    return TorchPolicyAdapter(actor=actor, critic=critic, deterministic=deterministic)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSingleEpisodeShape:
    def test_default_capture_yields_both_agents(self):
        collector = RolloutCollector(
            runtime_factory=lambda: _make_runtime(max_steps=3),
            policy_factories={
                "robot_a": _make_adapter,
                "robot_b": _make_adapter,
            },
        )
        try:
            out = collector.collect(n=2, base_seed=42)
        finally:
            collector.close()

        assert set(out.keys()) == {"robot_a", "robot_b"}
        assert len(out["robot_a"]) == 2
        assert len(out["robot_b"]) == 2

        for batch in out["robot_a"] + out["robot_b"]:
            assert isinstance(batch, RolloutBatch)
            batch.validate()  # pins obs.shape[0] == actions.shape[0] + 1
            assert batch.obs.shape == (4, OBS_DIM)        # T=3 → T+1=4
            assert batch.actions.shape == (3, ACTION_DIM)
            assert batch.rewards.shape == (3,)
            assert batch.log_probs is None or batch.log_probs.shape == (3,)
            assert batch.values is not None and batch.values.shape == (3,)

    def test_capture_filter_drops_one_side(self):
        collector = RolloutCollector(
            runtime_factory=lambda: _make_runtime(max_steps=2),
            policy_factories={
                "robot_a": _make_adapter,
                "robot_b": _make_adapter,
            },
            capture_agents=("robot_a",),
        )
        try:
            out = collector.collect(n=1, base_seed=7)
        finally:
            collector.close()
        assert set(out.keys()) == {"robot_a"}
        assert len(out["robot_a"]) == 1


class TestSeedReproducibility:
    def test_same_base_seed_produces_bit_equal_actions(self):
        # Two independent collectors, same base_seed → same actions
        # (relies on EpisodeRunner's seed plumbing; collector adds no
        # non-determinism of its own).
        #
        # MockSimulator.physical_step uses *legacy* np.random.randn, which
        # depends on global numpy RNG state. Reset it before each batch so
        # the obs trajectories are bit-equal between the two collects —
        # this is a test-fixture quirk, not a collector behavior.
        def _collect():
            np.random.seed(0)
            c = RolloutCollector(
                runtime_factory=lambda: _make_runtime(max_steps=3),
                policy_factories={
                    "robot_a": _make_adapter,
                    "robot_b": _make_adapter,
                },
            )
            try:
                return c.collect(n=2, base_seed=2026)
            finally:
                c.close()

        a = _collect()
        b = _collect()
        for batches_a, batches_b in zip(a["robot_a"], b["robot_a"]):
            np.testing.assert_array_equal(batches_a.actions, batches_b.actions)
            np.testing.assert_array_equal(batches_a.obs, batches_b.obs)


class TestStateDictHotReload:
    def test_state_dicts_change_subsequent_actions(self):
        # Build a collector, run once to capture a baseline action.
        # Then push a fresh state_dict and verify the next collect()
        # produces different actions for the same seed.
        torch.manual_seed(0)
        baseline_actor_state = TanhGaussianMLPPolicy(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=16,
        ).state_dict()

        collector = RolloutCollector(
            runtime_factory=lambda: _make_runtime(max_steps=2),
            policy_factories={
                "robot_a": _make_adapter,  # ctor with seed 0
                "robot_b": _make_adapter,
            },
        )
        try:
            out_before = collector.collect(n=1, base_seed=2026)
            # Forge a clearly different actor by reseeding init.
            torch.manual_seed(999)
            new_actor = TanhGaussianMLPPolicy(
                obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=16,
            )
            out_after = collector.collect(
                n=1,
                base_seed=2026,
                state_dicts={"robot_a": new_actor.state_dict()},
            )
        finally:
            collector.close()

        a_before = out_before["robot_a"][0].actions
        a_after = out_after["robot_a"][0].actions
        assert not np.allclose(a_before, a_after), (
            "state_dicts hot-reload had no observable effect on actions"
        )

    def test_state_dicts_unknown_agent_raises(self):
        collector = RolloutCollector(
            runtime_factory=lambda: _make_runtime(max_steps=1),
            policy_factories={
                "robot_a": _make_adapter,
                "robot_b": _make_adapter,
            },
        )
        try:
            with pytest.raises(KeyError, match="robot_x"):
                collector.collect(
                    n=1, base_seed=0, state_dicts={"robot_x": {}},
                )
        finally:
            collector.close()


class TestNoEpisodeLoop:
    """Negative test: the collector module must not contain a reimplemented
    episode loop. This is a structural assertion enforcing DESIGN.md §3.3.
    """

    def test_collector_does_not_reimplement_episode_loop(self):
        from baseline.common.rollout import collector as collector_module

        src = Path(collector_module.__file__).read_text(encoding="utf-8")
        assert "is_episode_active" not in src, (
            "RolloutCollector must delegate the episode loop to EpisodeRunner; "
            "found 'is_episode_active' in collector.py — that's the smell of "
            "a reimplemented while-loop."
        )
        assert "runtime.step(" not in src, (
            "RolloutCollector must not call runtime.step directly; use "
            "EpisodeRunner.run_episode / run_n_episodes."
        )


class TestExplicitSeeds:
    def test_seeds_argument_runs_one_episode_per_seed(self):
        collector = RolloutCollector(
            runtime_factory=lambda: _make_runtime(max_steps=2),
            policy_factories={
                "robot_a": _make_adapter,
                "robot_b": _make_adapter,
            },
        )
        try:
            out = collector.collect(seeds=[1, 2, 3])
        finally:
            collector.close()
        assert len(out["robot_a"]) == 3
        seeds_seen = [b.info["seed"] for b in out["robot_a"]]
        assert seeds_seen == [1, 2, 3]


class TestOptionsFn:
    def test_options_fn_threaded_through(self):
        captured: list = []

        class _OptionsCheckObserver(BaseObserverPlugin):
            def on_pre_episode(self, ctx) -> None:
                captured.append(dict(ctx.episode_options))

            def on_post_action_step(self, ctx) -> None:
                pass

            def get_output(self) -> float:
                return 0.0

        # Build a runtime that registers our spy as one of the standard
        # observer slots so the framework wires it in.
        def _factory() -> EnvRuntime:
            return EnvRuntime(
                simulator=MockSimulator(),
                observer_plugins={
                    "robot_a_obs": _QposObserver(),
                    "robot_a_reward": _StepRewardObserver(),
                    "robot_b_obs": _QposObserver(),
                    "robot_b_reward": _StepRewardObserver(),
                    "spy": _OptionsCheckObserver(),
                },
                max_steps=1,
                phy_steps_per_action=1,
            )

        collector = RolloutCollector(
            runtime_factory=_factory,
            policy_factories={
                "robot_a": _make_adapter,
                "robot_b": _make_adapter,
            },
        )
        try:
            collector.collect(
                n=2,
                base_seed=11,
                options_fn=lambda i: {"k": i * 10},
            )
        finally:
            collector.close()

        assert captured == [{"k": 0}, {"k": 10}]
