"""EpisodeRunner tests.

Cover: rollout shape, seed determinism, one-sided capture, extras capture,
reward extractor (scalar / dict / custom), binding validation at
construction, recorder attachment, on_step / on_episode_end hooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import (
    AgentTrajectory,
    EpisodeResult,
    EpisodeRunner,
    ObserverBinding,
    Policy,
    RolloutConfig,
    StepContext,
    default_bindings,
    default_reward_extractor,
)
from envs.framework.recorder import BaseFrameRecorder
from envs.framework.runtime_plugin import BaseObserverPlugin


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
class _QposObserver(BaseObserverPlugin):
    """Observation plugin — returns the first 5 values of qpos as the obs."""

    def __init__(self) -> None:
        self._output: np.ndarray = np.zeros(5, dtype=np.float32)

    def on_pre_episode(self, ctx):
        self._output = ctx.accessor.get_core_state()["qpos"][:5].astype(np.float32)

    def on_post_action_step(self, ctx):
        self._output = ctx.accessor.get_core_state()["qpos"][:5].astype(np.float32)

    def get_output(self) -> np.ndarray:
        return self._output.copy()


class _ScalarRewardObserver(BaseObserverPlugin):
    """Reward plugin — returns a monotonically increasing float (step index)."""

    def __init__(self) -> None:
        self._step = 0

    def on_pre_episode(self, ctx):
        self._step = 0

    def on_post_action_step(self, ctx):
        self._step += 1

    def get_output(self) -> float:
        return float(self._step)


class _DictRewardObserver(BaseObserverPlugin):
    """Reward plugin — returns a dict payload following the 'reward' key
    convention from the updated BaseRuntimeUnit docstring."""

    def __init__(self) -> None:
        self._step = 0

    def on_pre_episode(self, ctx):
        self._step = 0

    def on_post_action_step(self, ctx):
        self._step += 1

    def get_output(self) -> Dict[str, Any]:
        return {"reward": float(self._step), "step_count": self._step}


class _SeededPolicy(Policy):
    """Deterministic policy: action depends only on (seed, obs)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._rng: Optional[np.random.Generator] = None
        self.reset_calls: List[Optional[int]] = []

    def reset(self, seed: Optional[int] = None) -> None:
        self.reset_calls.append(seed)
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Any) -> np.ndarray:
        # Depend on observation so different obs → different actions; uses
        # the seeded RNG so same (seed, obs-sequence) → same actions.
        base = np.asarray(observation, dtype=np.float32).reshape(-1)
        noise = self._rng.standard_normal(21).astype(np.float32) * 0.01
        a = np.zeros(21, dtype=np.float32)
        a[: base.size] += base
        return a + noise


class _ExtrasPolicy(Policy):
    """Policy that implements act_with_extras so we can assert extras flow."""

    def __init__(self) -> None:
        self.extras_calls = 0

    def act(self, observation: Any) -> np.ndarray:  # pragma: no cover
        raise AssertionError("act should not be called when store_extras=True")

    def act_with_extras(self, observation: Any):
        self.extras_calls += 1
        return np.zeros(21, dtype=np.float32), {"log_prob": -0.5, "value": 1.0}


def _build_runtime(mock_simulator, *, reward_cls=_ScalarRewardObserver, max_steps=3) -> EnvRuntime:
    return EnvRuntime(
        simulator=mock_simulator,
        observer_plugins={
            "robot_a_obs": _QposObserver(),
            "robot_a_reward": reward_cls(),
            "robot_b_obs": _QposObserver(),
            "robot_b_reward": reward_cls(),
        },
        max_steps=max_steps,
        phy_steps_per_action=1,
    )


# ---------------------------------------------------------------------------
# default_reward_extractor
# ---------------------------------------------------------------------------
class TestDefaultRewardExtractor:
    def test_scalar_passthrough(self):
        assert default_reward_extractor(3.5) == pytest.approx(3.5)
        assert default_reward_extractor(2) == pytest.approx(2.0)
        assert default_reward_extractor(True) == pytest.approx(1.0)

    def test_numpy_scalar_and_zero_d(self):
        assert default_reward_extractor(np.float32(0.25)) == pytest.approx(0.25)
        assert default_reward_extractor(np.asarray(1.5)) == pytest.approx(1.5)
        assert default_reward_extractor(np.asarray([2.0])) == pytest.approx(2.0)

    def test_dict_with_reward_key(self):
        assert default_reward_extractor({"reward": 0.7, "other": 9}) == pytest.approx(0.7)
        assert default_reward_extractor({"total_reward": 1.2}) == pytest.approx(1.2)
        assert default_reward_extractor({"r": 3.0}) == pytest.approx(3.0)

    def test_none_is_error(self):
        with pytest.raises(ValueError, match="returned None"):
            default_reward_extractor(None)

    def test_dict_without_known_key_lists_available(self):
        with pytest.raises(KeyError, match="survival"):
            default_reward_extractor({"survival": 1.0, "alive": 0.5})

    def test_non_scalar_ndarray_rejected(self):
        with pytest.raises(TypeError, match="scalar"):
            default_reward_extractor(np.zeros(3))


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_missing_observer_plugin_raises_at_construction(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"robot_a_obs": _QposObserver()},  # no reward, no B
            max_steps=1,
        )
        with pytest.raises(KeyError, match="robot_a_reward"):
            EpisodeRunner(runtime=runtime, policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            })

    def test_missing_policy_raises(self, mock_simulator):
        runtime = _build_runtime(mock_simulator)
        with pytest.raises(ValueError, match="missing"):
            EpisodeRunner(runtime=runtime, policies={"robot_a": _SeededPolicy("a")})

    def test_extra_policy_raises(self, mock_simulator):
        runtime = _build_runtime(mock_simulator)
        with pytest.raises(ValueError, match="extra"):
            EpisodeRunner(runtime=runtime, policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
                "robot_c": _SeededPolicy("c"),
            })

    def test_non_policy_subclass_rejected(self, mock_simulator):
        runtime = _build_runtime(mock_simulator)
        with pytest.raises(TypeError, match="Policy"):
            EpisodeRunner(runtime=runtime, policies={
                "robot_a": object(),
                "robot_b": _SeededPolicy("b"),
            })


# ---------------------------------------------------------------------------
# Rollout shapes & capture toggles
# ---------------------------------------------------------------------------
class TestRolloutShapes:
    def test_full_capture_default(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, max_steps=3)
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
        )
        result = runner.run_episode(seed=7)
        assert isinstance(result, EpisodeResult)
        assert result.num_steps == 3
        assert set(result.trajectories) == {"robot_a", "robot_b"}
        for traj in result.trajectories.values():
            assert traj is not None
            # store_initial_observation=True → obs has T+1 entries
            assert len(traj.observations) == result.num_steps + 1
            assert len(traj.actions) == result.num_steps
            assert len(traj.rewards) == result.num_steps
            # _ScalarRewardObserver produces 1, 2, 3, ...
            assert traj.rewards == [1.0, 2.0, 3.0]
            assert traj.extras == []
            assert traj.truncated is True  # max_steps hit

    def test_capture_b_off_yields_none_for_b(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, max_steps=2)
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
            rollout=RolloutConfig(capture_a=True, capture_b=False),
        )
        result = runner.run_episode(seed=0)
        assert result.trajectories["robot_a"] is not None
        assert result.trajectories["robot_b"] is None

    def test_store_initial_observation_off(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, max_steps=2)
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
            rollout=RolloutConfig(store_initial_observation=False),
        )
        result = runner.run_episode(seed=0)
        for traj in result.trajectories.values():
            assert traj is not None
            # Without initial obs, T obs / T actions / T rewards.
            assert len(traj.observations) == result.num_steps
            assert len(traj.actions) == result.num_steps


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------
class _DeterministicSim:
    """MockSimulator variant whose ``physical_step`` uses a seeded RNG
    derived from the ``reset`` seed, so the whole environment is
    reproducible. Used to isolate seed determinism of EpisodeRunner from
    the global-RNG footgun in the shared conftest MockSimulator.
    """

    def __init__(self) -> None:
        from envs.framework.tests.conftest import MockSimulator
        self._inner = MockSimulator()
        self._rng: np.random.Generator = np.random.default_rng(0)

    # IDataAccessor / IDataMutator — delegate to inner.
    def __getattr__(self, name):
        return getattr(self._inner, name)

    # Lifecycle overrides
    def reset(self, seed: Optional[int] = None, options=None) -> None:
        self._inner.reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    def physical_step(self) -> None:
        if self._inner._is_closed:
            return
        self._inner._state["qpos"] += self._rng.standard_normal(100) * 0.001


class TestSeedDeterminism:
    def test_same_seed_reproduces_actions_and_rewards(self):
        """Two runs with the same seed produce bit-equal trajectories."""
        def run_once(seed: int) -> EpisodeResult:
            runtime = _build_runtime(_DeterministicSim(), max_steps=4)
            runner = EpisodeRunner(
                runtime=runtime,
                policies={
                    "robot_a": _SeededPolicy("a"),
                    "robot_b": _SeededPolicy("b"),
                },
            )
            return runner.run_episode(seed=seed)

        result_a = run_once(seed=1234)
        result_b = run_once(seed=1234)

        for agent in ("robot_a", "robot_b"):
            ta = result_a.trajectories[agent]
            tb = result_b.trajectories[agent]
            assert ta is not None and tb is not None
            assert len(ta.actions) == len(tb.actions)
            for xa, xb in zip(ta.actions, tb.actions):
                np.testing.assert_array_equal(xa, xb)
            assert ta.rewards == tb.rewards

    def test_different_seed_produces_different_actions(self):
        def run_once(seed: int):
            runtime = _build_runtime(_DeterministicSim(), max_steps=3)
            runner = EpisodeRunner(
                runtime=runtime,
                policies={
                    "robot_a": _SeededPolicy("a"),
                    "robot_b": _SeededPolicy("b"),
                },
            )
            return runner.run_episode(seed=seed)

        result_1 = run_once(seed=1)
        result_2 = run_once(seed=9999)
        actions_1 = np.stack(result_1.trajectories["robot_a"].actions)
        actions_2 = np.stack(result_2.trajectories["robot_a"].actions)
        assert not np.array_equal(actions_1, actions_2)

    def test_policy_reset_receives_distinct_seeds(self, mock_simulator):
        """robot_a and robot_b policies must NOT receive the same seed."""
        runtime = _build_runtime(mock_simulator, max_steps=1)
        pa, pb = _SeededPolicy("a"), _SeededPolicy("b")
        runner = EpisodeRunner(
            runtime=runtime, policies={"robot_a": pa, "robot_b": pb},
        )
        runner.run_episode(seed=42)
        assert len(pa.reset_calls) == 1
        assert len(pb.reset_calls) == 1
        assert pa.reset_calls[0] != pb.reset_calls[0]

    def test_run_n_episodes_reproducible_from_base_seed(self):
        def run_batch():
            runtime = _build_runtime(_DeterministicSim(), max_steps=2)
            runner = EpisodeRunner(
                runtime=runtime,
                policies={
                    "robot_a": _SeededPolicy("a"),
                    "robot_b": _SeededPolicy("b"),
                },
            )
            return runner.run_n_episodes(3, base_seed=77)

        batch_a = run_batch()
        batch_b = run_batch()
        assert [r.seed for r in batch_a] == [r.seed for r in batch_b]
        for ra, rb in zip(batch_a, batch_b):
            for agent in ("robot_a", "robot_b"):
                for aa, ab in zip(
                    ra.trajectories[agent].actions, rb.trajectories[agent].actions
                ):
                    np.testing.assert_array_equal(aa, ab)

    def test_run_n_zero(self, mock_simulator):
        runtime = _build_runtime(mock_simulator)
        runner = EpisodeRunner(
            runtime=runtime, policies={
                "robot_a": _SeededPolicy("a"), "robot_b": _SeededPolicy("b"),
            },
        )
        assert runner.run_n_episodes(0, base_seed=1) == []


# ---------------------------------------------------------------------------
# Reward plumbing: dict reward + custom extractor + None reward
# ---------------------------------------------------------------------------
class TestRewardPlumbing:
    def test_dict_reward_extracted(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, reward_cls=_DictRewardObserver, max_steps=2)
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
        )
        result = runner.run_episode(seed=0)
        # _DictRewardObserver → {"reward": step_index, ...}; runner
        # extracts "reward" via default_reward_extractor.
        assert result.trajectories["robot_a"].rewards == [1.0, 2.0]

    def test_custom_reward_extractor(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, reward_cls=_DictRewardObserver, max_steps=2)
        # Pull step_count instead of reward; flips signs to distinguish.
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
            observer_bindings={
                "robot_a": ObserverBinding(
                    obs_name="robot_a_obs",
                    reward_name="robot_a_reward",
                    reward_extractor=lambda raw: -float(raw["step_count"]),
                ),
                "robot_b": ObserverBinding(
                    obs_name="robot_b_obs", reward_name="robot_b_reward",
                ),
            },
        )
        result = runner.run_episode(seed=0)
        assert result.trajectories["robot_a"].rewards == [-1.0, -2.0]
        # Untouched binding for B still uses default extractor.
        assert result.trajectories["robot_b"].rewards == [1.0, 2.0]

    def test_no_reward_binding_fills_default(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, max_steps=2)
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
            observer_bindings={
                "robot_a": ObserverBinding(
                    obs_name="robot_a_obs", reward_name=None, default_reward=-1.0,
                ),
                "robot_b": ObserverBinding(
                    obs_name="robot_b_obs", reward_name="robot_b_reward",
                ),
            },
        )
        result = runner.run_episode(seed=0)
        assert result.trajectories["robot_a"].rewards == [-1.0, -1.0]


# ---------------------------------------------------------------------------
# Extras capture
# ---------------------------------------------------------------------------
class TestExtrasCapture:
    def test_store_extras_uses_act_with_extras(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, max_steps=2)
        pa = _ExtrasPolicy()
        runner = EpisodeRunner(
            runtime=runtime,
            policies={"robot_a": pa, "robot_b": _SeededPolicy("b")},
            rollout=RolloutConfig(store_extras=True),
        )
        result = runner.run_episode(seed=0)
        assert pa.extras_calls == 2
        traj_a = result.trajectories["robot_a"]
        assert len(traj_a.extras) == 2
        assert traj_a.extras[0] == {"log_prob": -0.5, "value": 1.0}

    def test_store_extras_off_means_no_extras(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, max_steps=2)
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
        )
        result = runner.run_episode(seed=0)
        for traj in result.trajectories.values():
            assert traj.extras == []


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
class TestHooks:
    def test_on_step_and_on_episode_end_fire(self, mock_simulator):
        runtime = _build_runtime(mock_simulator, max_steps=3)
        seen_steps: List[StepContext] = []
        seen_ends: List[EpisodeResult] = []
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
            on_step=seen_steps.append,
            on_episode_end=seen_ends.append,
        )
        result = runner.run_episode(seed=0)
        assert len(seen_steps) == 3
        assert [c.step_index for c in seen_steps] == [1, 2, 3]
        # on_step gets per-agent dicts for obs/actions/rewards.
        for ctx in seen_steps:
            assert set(ctx.actions) == {"robot_a", "robot_b"}
            assert set(ctx.rewards) == {"robot_a", "robot_b"}
        assert len(seen_ends) == 1
        assert seen_ends[0] is result


# ---------------------------------------------------------------------------
# Recorder wiring
# ---------------------------------------------------------------------------
class TestRecorderWiring:
    def test_recorder_attached_and_produces_files(self, mock_simulator, tmp_path: Path):
        runtime = _build_runtime(mock_simulator, max_steps=2)
        recorder = BaseFrameRecorder(
            output_dir=tmp_path, save_image=False, save_accessor_state=True,
        )
        runner = EpisodeRunner(
            runtime=runtime,
            policies={
                "robot_a": _SeededPolicy("a"),
                "robot_b": _SeededPolicy("b"),
            },
            recorders=[recorder],
        )
        runner.run_episode(seed=0)
        episode_dir = tmp_path / "episode_00000"
        assert (episode_dir / "manifest.json").exists()
        assert (episode_dir / "static.json").exists()
        assert sorted(episode_dir.glob("step_*.json"))  # non-empty


# ---------------------------------------------------------------------------
# default_bindings convenience
# ---------------------------------------------------------------------------
def test_default_bindings_follow_convention():
    bindings = default_bindings()
    assert bindings["robot_a"].obs_name == "robot_a_obs"
    assert bindings["robot_a"].reward_name == "robot_a_reward"
    assert bindings["robot_b"].obs_name == "robot_b_obs"
    assert bindings["robot_b"].reward_name == "robot_b_reward"


# ---------------------------------------------------------------------------
# AgentTrajectory.as_rollout_batch / to_gymnasium_style
# ---------------------------------------------------------------------------
def _make_traj(
    *,
    n_steps: int = 3,
    obs_dim: int = 4,
    action_dim: int = 2,
    terminated: bool = True,
    truncated: bool = False,
    with_extras: bool = True,
) -> AgentTrajectory:
    rng = np.random.default_rng(0)
    traj = AgentTrajectory(agent_id="robot_a")
    # T+1 observations, T actions, T rewards.
    for t in range(n_steps + 1):
        traj.observations.append(rng.normal(size=obs_dim).astype(np.float32))
    for t in range(n_steps):
        traj.actions.append(rng.normal(size=action_dim).astype(np.float32))
        traj.rewards.append(float(t + 1))
        if with_extras:
            traj.extras.append(
                {"log_prob": float(-t), "value": float(t * 0.5)}
            )
    traj.terminated = terminated
    traj.truncated = truncated
    return traj


class TestAgentTrajectoryGymnasiumStyle:
    def test_terminated_only_is_passthrough(self):
        traj = _make_traj(terminated=True, truncated=False)
        assert traj.to_gymnasium_style() == (True, False)

    def test_truncated_only_is_passthrough(self):
        traj = _make_traj(terminated=False, truncated=True)
        assert traj.to_gymnasium_style() == (False, True)

    def test_both_true_collapses_to_terminated(self):
        # Framework allows both; the Gymnasium view must pick exactly one.
        traj = _make_traj(terminated=True, truncated=True)
        assert traj.to_gymnasium_style() == (True, False)


class TestAgentTrajectoryAsRolloutBatch:
    def test_shapes_and_extras_alignment(self):
        traj = _make_traj(n_steps=4, obs_dim=5, action_dim=3)
        batch = traj.as_rollout_batch()
        assert batch.agent_id == "robot_a"
        assert batch.obs.shape == (5, 5)
        assert batch.actions.shape == (4, 3)
        assert batch.rewards.shape == (4,)
        assert batch.log_probs is not None and batch.log_probs.shape == (4,)
        assert batch.values is not None and batch.values.shape == (4,)
        # final_obs property aliases obs[-1].
        np.testing.assert_array_equal(batch.final_obs, batch.obs[-1])

    def test_missing_extras_returns_none(self):
        traj = _make_traj(with_extras=False)
        batch = traj.as_rollout_batch()
        assert batch.log_probs is None
        assert batch.values is None

    def test_episode_result_metadata_lands_in_info(self):
        traj = _make_traj()
        result = EpisodeResult(
            seed=42,
            num_steps=3,
            wall_time_sec=0.1,
            terminated=True,
            truncated=False,
            termination_reasons=["ko"],
            shared_info_final={},
            trajectories={"robot_a": traj, "robot_b": None},
        )
        batch = traj.as_rollout_batch(result)
        assert batch.info["seed"] == 42
        assert batch.info["num_steps"] == 3
        assert batch.info["termination_reasons"] == ["ko"]

    def test_caller_info_overrides_episode_result_keys(self):
        traj = _make_traj()
        result = EpisodeResult(
            seed=1, num_steps=3, wall_time_sec=0.0,
            terminated=True, truncated=False,
            termination_reasons=["ko"], shared_info_final={},
            trajectories={"robot_a": traj, "robot_b": None},
        )
        batch = traj.as_rollout_batch(result, info={"seed": 999, "tag": "x"})
        assert batch.info["seed"] == 999     # caller wins
        assert batch.info["tag"] == "x"       # extra fields pass through

    def test_zero_steps_raises(self):
        empty = AgentTrajectory(agent_id="robot_a")
        empty.observations.append(np.zeros(3, dtype=np.float32))
        with pytest.raises(ValueError, match="zero steps"):
            empty.as_rollout_batch()

    def test_terminated_truncated_coerced_in_batch(self):
        # Both flags true on the trajectory → batch sees terminated=True only.
        traj = _make_traj(terminated=True, truncated=True)
        batch = traj.as_rollout_batch()
        assert batch.terminated is True
        assert batch.truncated is False
        # validate() would otherwise crash on simultaneous flags.
        batch.validate()
