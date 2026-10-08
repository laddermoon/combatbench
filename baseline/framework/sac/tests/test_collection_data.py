"""Permanent tests for SAC collection and ``sac_transition_v2``."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import numpy as np
import pytest

from envs.framework.backend import BaseSimulator
from envs.framework.blueprint import ClassSpec, EnvBlueprint
from envs.framework.observer_plugin import BaseObserverPlugin
from envs.framework.plugin import BasePlugin
from envs.framework.policy import PolicyBlueprint

from baseline.experiments_sac.exp_sac_balance import SacBalance
from baseline.framework.sac.collected_episode import CollectedEpisode
from baseline.framework.sac.collection import SACFactSpec, SACJob
from baseline.framework.sac.collection_rollouter import (
    SACCollectionError,
    SACParallelRollouter,
)
from baseline.framework.sac.transition import (
    build_agent_transition_slice,
    validate_transition_slice,
)


class FakeSimulator(BaseSimulator):
    def __init__(self) -> None:
        self.t = 0
        self._actions: Dict[str, np.ndarray] = {
            "robot_a": np.zeros(2, dtype=np.float32),
            "robot_b": np.zeros(2, dtype=np.float32),
        }

    def reset(self, seed=None, options=None) -> None:
        self.t = 0

    def physical_step(self) -> None:
        self.t += 1

    def get_physical_frequency(self) -> float:
        return 100.0

    def get_static_data(self) -> Dict[str, Any]:
        return {}

    def get_core_state(self) -> Dict[str, Any]:
        return {
            aid: {"root_pos": np.array([0.0, 0.0, 1.28 + 0.001 * self.t])}
            for aid in ("robot_a", "robot_b")
        }

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        return {
            aid: {"uprightness": np.array([1.0], dtype=np.float32)}
            for aid in ("robot_a", "robot_b")
        }

    def get_sensor_data(self) -> Dict[str, Any]:
        return {}

    def get_action(self) -> Dict[str, Any]:
        return {aid: np.array(v, copy=True) for aid, v in self._actions.items()}

    def get_broadcastview_image(self):
        return None

    def get_observation(self) -> Dict[str, np.ndarray]:
        return {
            "robot_a": np.array([self.t, 1.0, 0.0], dtype=np.float32),
            "robot_b": np.array([self.t, -1.0, 0.0], dtype=np.float32),
        }

    def set_core_state(self, state: Dict[str, Any]) -> None:
        pass

    def set_action(self, action: Dict[str, Any]) -> None:
        self._actions = {
            aid: np.asarray(value, dtype=np.float32)
            for aid, value in action.items()
        }


class ConstantPolicy:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = float(offset)

    def reset(self, seed=None) -> None:
        self.seed = seed

    def act(self, observation, *, want_extra: bool = False):
        action = np.full(2, self.offset, dtype=np.float32)
        extra = {"dummy": np.array([0.0], dtype=np.float32)} if want_extra else None
        return action, extra


class SeededPolicy:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed=None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation, *, want_extra: bool = False):
        return (
            self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32),
            {} if want_extra else None,
        )


class FailingPolicy:
    def act(self, observation, *, want_extra: bool = False):
        raise RuntimeError("injected collection failure")


class FakeObserver(BaseObserverPlugin):
    def __init__(self, agent_id: str = "robot_a", phi: bool = False) -> None:
        self.agent_id = agent_id
        self.phi = bool(phi)
        self._step = 0

    def on_pre_episode(self, ctx) -> None:
        self._step = 0

    def on_post_action_step(self, ctx) -> None:
        self._step = int(ctx.episode_step)

    def get_output(self):
        phi_value = 0.5 + 0.01 * self._step
        if self.phi:
            return {
                "phi": phi_value,
                "height": 1.28,
                "uprightness": 1.0,
                "initial_phi": 0.5,
            }
        return {"reward": 0.1 * self._step}

    def to_blueprint(self):
        return {"agent_id": self.agent_id, "phi": self.phi}

    @classmethod
    def from_blueprint(cls, config):
        return cls(**config)


class EarlyTerminatePlugin(BasePlugin):
    def __init__(self, agent_id: str = "robot_a", at_step: int = 2) -> None:
        self.agent_id = agent_id
        self.at_step = int(at_step)

    @property
    def name(self) -> str:
        return "early_terminate"

    def on_post_action_step(self, ctx) -> None:
        if int(ctx.episode_step) == self.at_step:
            ctx.request_termination("imbalance", agent_id=self.agent_id)

    def to_blueprint(self):
        return {"agent_id": self.agent_id, "at_step": self.at_step}

    @classmethod
    def from_blueprint(cls, config):
        return cls(**config)


class TestFactProvider:
    def compute(self, accessor, agent_id: str) -> float:
        height = float(accessor.get_core_state()[agent_id]["root_pos"][2])
        uprightness = float(
            accessor.get_derived_state([agent_id])[agent_id]["uprightness"][0]
        )
        return height * uprightness / 1.28


class BadFactProvider:
    def compute(self, accessor, agent_id: str) -> float:
        return float("nan")


def _module(cls) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _env_bp(*, terminate_a_at: Optional[int] = None, balance_observers: bool = False):
    plugins = ()
    if terminate_a_at is not None:
        plugins = (
            ClassSpec(
                cls=_module(EarlyTerminatePlugin),
                config={"agent_id": "robot_a", "at_step": terminate_a_at},
            ),
        )
    observers = {
        "cross_support_a": ClassSpec(
            cls=_module(FakeObserver),
            config={"agent_id": "robot_a", "phi": False},
        ),
        "cross_support_b": ClassSpec(
            cls=_module(FakeObserver),
            config={"agent_id": "robot_b", "phi": False},
        ),
    }
    if balance_observers:
        observers["height_phi_a"] = ClassSpec(
            cls=_module(FakeObserver),
            config={"agent_id": "robot_a", "phi": True},
        )
        observers["height_phi_b"] = ClassSpec(
            cls=_module(FakeObserver),
            config={"agent_id": "robot_b", "phi": True},
        )
    return EnvBlueprint(
        simulator=ClassSpec(cls=_module(FakeSimulator), config={}),
        plugins=plugins,
        observer_plugins=observers,
        phy_steps_per_action=2,
        max_steps=4,
        strict=True,
    )


def _job(
    index: int = 0,
    *,
    seed: int = 100,
    policy_cls=ConstantPolicy,
    env_bp=None,
    fact_specs=(),
) -> SACJob:
    policy_bp = PolicyBlueprint(cls=_module(policy_cls), config={})
    return SACJob(
        policy_a_bp=policy_bp,
        policy_b_bp=policy_bp,
        env_bp=env_bp or _env_bp(),
        seed=seed,
        episode_options={"case": index},
        fact_specs=tuple(fact_specs),
        run_id="test_run",
        collection_round=7,
        job_index=index,
    )


def _phi_specs() -> tuple:
    return tuple(
        SACFactSpec(
            name="phi_pre",
            agent_id=aid,
            provider=_module(TestFactProvider),
        )
        for aid in ("robot_a", "robot_b")
    )


def test_collection_provenance_ordered_and_pre_action_facts():
    rollouter = SACParallelRollouter(num_workers=1)
    jobs = [_job(0, seed=11), _job(1, seed=22)]
    episodes = rollouter.collect(jobs)

    assert [ep.job_index for ep in episodes] == [0, 1]
    assert [ep.episode_seed for ep in episodes] == [11, 22]
    assert all(ep.run_id == "test_run" for ep in episodes)
    assert all(ep.collection_round == 7 for ep in episodes)
    assert all(ep.job_key for ep in episodes)
    assert all(ep.num_frames == 4 for ep in episodes)
    assert all(
        ep.policy_fingerprints["robot_a"] == ep.policy_fingerprints["robot_b"]
        for ep in episodes
    )
    assert episodes[0].job_key != episodes[1].job_key

    ep = rollouter.collect([_job(0, seed=33, fact_specs=_phi_specs())])[0]
    assert set(ep.pre_action_facts) == {"robot_a", "robot_b"}
    assert len(ep.pre_action_facts["robot_a"]["phi_pre"]) == ep.num_frames
    assert ep.action_extras["robot_a"]["policy_action"].shape == (4, 2)
    assert np.allclose(ep.physics_steps, [2, 4, 6, 8])


def test_behavior_seed_streams_are_independent_and_reproducible():
    job = _job(0, seed=44, policy_cls=SeededPolicy)
    rollouter = SACParallelRollouter(num_workers=1)
    ep1 = rollouter.collect([job])[0]
    ep2 = rollouter.collect([job])[0]

    np.testing.assert_allclose(ep1.actions["robot_a"], ep2.actions["robot_a"])
    np.testing.assert_allclose(ep1.actions["robot_b"], ep2.actions["robot_b"])
    assert not np.allclose(ep1.actions["robot_a"], ep1.actions["robot_b"])


def test_parallel_collection_preserves_input_order():
    rollouter = SACParallelRollouter(num_workers=2)
    jobs = [
        _job(0, seed=101),
        _job(1, seed=102, env_bp=_env_bp(terminate_a_at=2)),
        _job(2, seed=103),
    ]
    episodes = rollouter.collect(jobs)
    assert [ep.job_index for ep in episodes] == [0, 1, 2]
    assert [ep.episode_seed for ep in episodes] == [101, 102, 103]


def test_collection_failure_aborts_round():
    rollouter = SACParallelRollouter(num_workers=1)
    with pytest.raises(SACCollectionError, match="injected collection failure"):
        rollouter.collect([_job(0, policy_cls=FailingPolicy), _job(1)])


def test_missing_or_nonfinite_pre_action_fact_fails():
    bad_specs = (
        SACFactSpec(
            name="phi_pre",
            agent_id="robot_a",
            provider=_module(BadFactProvider),
        ),
    )
    with pytest.raises(SACCollectionError, match="non-finite"):
        SACParallelRollouter(num_workers=1).collect(
            [_job(0, fact_specs=bad_specs)]
        )


def _generic_slice(episode: CollectedEpisode, agent_id: str):
    T = episode.agent_frame_boundary[agent_id]
    return build_agent_transition_slice(
        episode,
        agent_id,
        channel_names=("r_main",),
        rewards={"r_main": np.ones(T, dtype=np.float32)},
        actor_gate={"r_main": np.ones(T, dtype=np.float32)},
        actor_gate_next={"r_main": np.ones(T, dtype=np.float32)},
        task_facts={
            "phi_pre": np.asarray(
                episode.pre_action_facts[agent_id]["phi_pre"], dtype=np.float32
            )
        },
        versions={
            "reward_semantics": "test",
            "objective_mode": "shannon",
            "regularizer_mode": "entropy",
            "policy_arch": "test_policy",
        },
    )


def test_timeout_boundary_next_obs_and_source_identity():
    ep = SACParallelRollouter(num_workers=1).collect(
        [_job(0, seed=55, fact_specs=_phi_specs())]
    )[0]
    sl = _generic_slice(ep, "robot_a")
    assert sl.num_transitions == 4
    assert not sl.terminated.any()
    assert sl.truncated[-1]
    assert sl.bootstrap[-1] == 1.0
    np.testing.assert_allclose(sl.next_obs[-1], ep.final_observation["robot_a"])
    np.testing.assert_allclose(sl.obs[1:], ep.observations["robot_a"][1:4])
    np.testing.assert_allclose(
        sl.task_facts["phi_pre"],
        ep.pre_action_facts["robot_a"]["phi_pre"],
    )
    assert len(set(sl.source_keys)) == 4
    assert "r000007/j000000/robot_a" in sl.source_keys[0]


def test_agent_early_termination_boundary_and_bootstrap():
    ep = SACParallelRollouter(num_workers=1).collect(
        [
            _job(
                0,
                seed=66,
                env_bp=_env_bp(terminate_a_at=2),
                fact_specs=_phi_specs(),
            )
        ]
    )[0]
    assert ep.num_frames == 4
    a = _generic_slice(ep, "robot_a")
    b = _generic_slice(ep, "robot_b")

    assert a.num_transitions == 2
    assert a.terminated[-1] and not a.truncated[-1]
    assert a.bootstrap[-1] == 0.0
    np.testing.assert_allclose(a.next_obs[-1], ep.observations["robot_a"][2])

    assert b.num_transitions == 4
    assert b.truncated[-1] and not b.terminated[-1]
    assert b.bootstrap[-1] == 1.0


def test_balance_build_slices_requires_phi_pre_and_builds_gates():
    ep = SACParallelRollouter(num_workers=1).collect(
        [
            _job(
                0,
                seed=77,
                env_bp=_env_bp(balance_observers=True),
                fact_specs=_phi_specs(),
            )
        ]
    )[0]
    exp = SacBalance()
    slices = exp.build_slices([ep])
    assert len(slices) == 2
    sl = slices[0]
    assert "phi_pre" in sl.task_facts
    assert "phi_post_reference" in sl.task_facts
    np.testing.assert_allclose(
        sl.actor_gate[:, 1], np.clip(sl.task_facts["phi_pre"], 0, 1) ** 2
    )
    np.testing.assert_allclose(
        sl.actor_gate_next[:, 1],
        np.clip(sl.task_facts["phi_post_reference"], 0, 1) ** 2,
    )
    np.testing.assert_allclose(sl.actor_weight.sum(axis=1), np.ones(4))
    np.testing.assert_allclose(sl.actor_weight_next.sum(axis=1), np.ones(4))

    missing = replace(ep, pre_action_facts={})
    with pytest.raises(KeyError, match="phi_pre"):
        exp.build_slices([missing])


def test_validator_rejects_nan_bad_gates_and_bad_source_keys():
    ep = SACParallelRollouter(num_workers=1).collect(
        [_job(0, seed=88, fact_specs=_phi_specs())]
    )[0]
    good = _generic_slice(ep, "robot_a")

    bad_reward = replace(good, rewards=good.rewards.copy())
    bad_reward.rewards[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_transition_slice(bad_reward)

    bad_gate = replace(good, actor_gate=np.zeros_like(good.actor_gate))
    with pytest.raises(ValueError, match="row sums"):
        validate_transition_slice(bad_gate)

    bad_next_gate = replace(
        good, actor_gate_next=np.zeros_like(good.actor_gate_next),
    )
    with pytest.raises(ValueError, match="actor_gate_next.*row sums"):
        validate_transition_slice(bad_next_gate)

    bad_keys = replace(good, source_keys=tuple(["same"] * good.num_transitions))
    with pytest.raises(ValueError, match="unique"):
        validate_transition_slice(bad_keys)
