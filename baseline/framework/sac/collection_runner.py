"""SAC-owned episode runner.

Copied and adapted from ``envs.framework.episode_runner`` so SAC collection
can capture pre-action facts without changing the neutral runner API.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from envs.framework.context import AGENT_IDS
from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin

from .collection_recorder import SACEpisodeRecorder
from .fact_providers import PreActionFactProvider


def _resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(secrets.randbits(32))
    return int(seed)


@dataclass(frozen=True)
class _SACEpisodeSeeds:
    base: int
    runtime: int
    policies: Dict[str, int]
    plugins: Dict[int, int]


class SACEpisodeRunner:
    """Drive one episode and attach SAC pre-action facts to each frame."""

    AGENT_IDS: Tuple[str, str] = AGENT_IDS

    def __init__(
        self,
        runtime: EnvRuntime,
        policy_a: Any,
        policy_b: Any,
        *,
        recorder: Optional[SACEpisodeRecorder] = None,
        fact_providers: Optional[
            Mapping[str, Tuple[Tuple[str, PreActionFactProvider], ...]]
        ] = None,
        post_termination_action: str = "policy",
    ) -> None:
        if not hasattr(policy_a, "act"):
            raise TypeError(
                f"policy_a must have an 'act' method; got {type(policy_a).__name__}"
            )
        if not hasattr(policy_b, "act"):
            raise TypeError(
                f"policy_b must have an 'act' method; got {type(policy_b).__name__}"
            )
        if post_termination_action not in ("policy", "hold"):
            raise ValueError(
                f"post_termination_action must be 'policy' or 'hold'; "
                f"got {post_termination_action!r}"
            )
        self.runtime = runtime
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.recorder = recorder
        self.fact_providers = {
            str(agent_id): tuple(providers)
            for agent_id, providers in (fact_providers or {}).items()
        }
        self.post_termination_action = post_termination_action

    def run_episode(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        want_extras: bool = True,
    ) -> None:
        base_seed = _resolve_seed(seed)
        episode_seeds = self._derive_seeds(base_seed)
        self._reset_all(episode_seeds, options=options)

        obs_a, obs_b = self.runtime.get_observation()
        a_active = True
        b_active = True
        last_action_a: Optional[np.ndarray] = None
        last_action_b: Optional[np.ndarray] = None

        while not self.runtime.is_episode_over():
            if a_active or self.post_termination_action == "policy":
                action_a, extra_a = self.policy_a.act(
                    obs_a,
                    want_extra=want_extras,
                )
                last_action_a = action_a
            else:
                if last_action_a is None:
                    raise RuntimeError("hold strategy requires a prior action")
                action_a, extra_a = last_action_a, None

            if b_active or self.post_termination_action == "policy":
                action_b, extra_b = self.policy_b.act(
                    obs_b,
                    want_extra=want_extras,
                )
                last_action_b = action_b
            else:
                if last_action_b is None:
                    raise RuntimeError("hold strategy requires a prior action")
                action_b, extra_b = last_action_b, None

            self._record_pre_action_facts()
            self.runtime.step(
                action_a,
                action_b,
                action_a_extra=self._with_policy_action(action_a, extra_a),
                action_b_extra=self._with_policy_action(action_b, extra_b),
            )

            a_active = a_active and self.runtime.is_agent_active("robot_a")
            b_active = b_active and self.runtime.is_agent_active("robot_b")
            obs_a, obs_b = self.runtime.get_observation()

    def _with_policy_action(self, action: Any, extra: Any) -> Dict[str, Any]:
        payload = dict(extra or {})
        payload["policy_action"] = np.asarray(action, copy=True)
        return payload

    def _record_pre_action_facts(self) -> None:
        if not self.fact_providers:
            return
        accessor = self.runtime.ctx.accessor
        facts: Dict[str, Dict[str, Any]] = {}
        for agent_id, providers in self.fact_providers.items():
            values: Dict[str, Any] = {}
            for name, provider in providers:
                value = provider.compute(accessor, agent_id)
                arr = np.asarray(value)
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"SAC pre-action fact {name!r} for {agent_id!r} "
                        f"is non-finite: {value!r}"
                    )
                values[str(name)] = arr if arr.ndim else float(arr)
            facts[agent_id] = values
        if self.recorder is None:
            raise RuntimeError(
                "SAC fact providers configured but no SACEpisodeRecorder was "
                "attached to the runner"
            )
        self.recorder.set_pending_pre_action_facts(facts)

    def set_policy_a(self, policy: Any) -> None:
        if not hasattr(policy, "act"):
            raise TypeError(
                f"policy must have an 'act' method; got {type(policy).__name__}"
            )
        self.policy_a = policy

    def set_policy_b(self, policy: Any) -> None:
        if not hasattr(policy, "act"):
            raise TypeError(
                f"policy must have an 'act' method; got {type(policy).__name__}"
            )
        self.policy_b = policy

    def set_runtime(self, runtime: EnvRuntime) -> None:
        self.runtime = runtime

    def set_recorder(self, recorder: SACEpisodeRecorder) -> None:
        self.recorder = recorder

    def close(self) -> None:
        seen = set()
        for policy in (self.policy_a, self.policy_b):
            if id(policy) in seen:
                continue
            seen.add(id(policy))
            close_fn = getattr(policy, "close", None)
            if callable(close_fn):
                close_fn()

    def _seedable_plugins(self) -> Tuple[BasePlugin, ...]:
        return tuple(
            plugin for plugin in self.runtime.plugins
            if type(plugin).set_episode_seed is not BasePlugin.set_episode_seed
        )

    def _derive_seeds(self, base_seed: int) -> _SACEpisodeSeeds:
        seedable_plugins = self._seedable_plugins()
        children = np.random.SeedSequence(int(base_seed)).spawn(
            1 + len(self.AGENT_IDS) + len(seedable_plugins)
        )
        runtime_ss, *rest = children
        policy_sss = rest[: len(self.AGENT_IDS)]
        plugin_sss = rest[len(self.AGENT_IDS):]

        def _leaf(ss: np.random.SeedSequence) -> int:
            return int(ss.generate_state(1, dtype=np.uint32)[0])

        return _SACEpisodeSeeds(
            base=int(base_seed),
            runtime=_leaf(runtime_ss),
            policies={
                agent: _leaf(ss)
                for agent, ss in zip(self.AGENT_IDS, policy_sss)
            },
            plugins={
                id(plugin): _leaf(ss)
                for plugin, ss in zip(seedable_plugins, plugin_sss)
            },
        )

    def _reset_all(
        self,
        seeds: _SACEpisodeSeeds,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        for plugin in self._seedable_plugins():
            plugin.set_episode_seed(seeds.plugins[id(plugin)])
        self.runtime.reset(
            seed=seeds.runtime,
            options=options,
            base_seed=seeds.base,
        )
        for agent_id, policy in (
            ("robot_a", self.policy_a),
            ("robot_b", self.policy_b),
        ):
            reset_fn = getattr(policy, "reset", None)
            if callable(reset_fn):
                reset_fn(seeds.policies[agent_id])


__all__ = ["SACEpisodeRunner"]
