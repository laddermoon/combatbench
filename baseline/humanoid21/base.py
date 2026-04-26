from __future__ import annotations

import copy
import multiprocessing as mp
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from baseline.common.policies import (
    CriticMLP,
    DEFAULT_EXPORT_ACTOR_HIDDEN_DIM,
    DEFAULT_LOG_STD_MAX,
    DEFAULT_LOG_STD_MIN,
    TanhGaussianMLPPolicy,
    build_actor_export_payload,
    build_export_policy_code,
    export_actor_policy_artifacts as export_actor_policy_artifacts_common,
    export_policy_artifacts_from_checkpoint as export_policy_artifacts_from_checkpoint_common,
)

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from envs.framework import BasePlugin, EnvRuntime, SimContext, TerminationReason


class Actor(TanhGaussianMLPPolicy):
    """Backward-compatible alias for humanoid21 scripts."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    @classmethod
    def build_export_policy_code(cls) -> str:
        return build_export_policy_code()

    @classmethod
    def build_export_payload_from_actor(
        cls,
        actor: "Actor",
        extra_payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_actor_export_payload(actor=actor, extra_payload=extra_payload)

    def export_policy_artifacts(
        self,
        policy_dir: Path,
        extra_payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        type(self).export_actor_policy_artifacts(
            actor=self,
            policy_dir=policy_dir,
            extra_payload=extra_payload,
        )

    @classmethod
    def export_actor_policy_artifacts(
        cls,
        actor: "Actor",
        policy_dir: Path,
        extra_payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        export_actor_policy_artifacts_common(
            actor=actor,
            policy_dir=policy_dir,
            extra_payload=extra_payload,
        )

    @classmethod
    def export_policy_artifacts_from_checkpoint(cls, model_path: Path, policy_dir: Path) -> None:
        export_policy_artifacts_from_checkpoint_common(
            model_path=model_path,
            policy_dir=policy_dir,
            default_hidden_dim=DEFAULT_EXPORT_ACTOR_HIDDEN_DIM,
        )


def export_policy_artifacts(model_path: Path, policy_dir: Path) -> None:
    Actor.export_policy_artifacts_from_checkpoint(model_path=model_path, policy_dir=policy_dir)


class Critic(CriticMLP):
    """Backward-compatible alias for humanoid21 scripts.

    The implementation now lives in
    :class:`baseline.common.policies.CriticMLP`. New code should import
    ``CriticMLP`` directly; this subclass exists only so existing
    ``baseline/humanoid21/standing_*.py`` scripts keep working unchanged.
    """


class StandingTerminationPlugin(BasePlugin):
    def __init__(
        self,
        agent_id: str,
        fall_height_threshold: float,
        fall_upright_threshold: float,
        fall_grace_steps: int,
    ):
        self.agent_id = agent_id
        self.fall_height_threshold = fall_height_threshold
        self.fall_upright_threshold = fall_upright_threshold
        self.fall_grace_steps = fall_grace_steps
        self._fall_streak = 0

    @property
    def name(self) -> str:
        return f"{self.agent_id}_standing_termination"

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._fall_streak = 0

    def on_post_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])
        is_standing = bool(height >= self.fall_height_threshold and uprightness >= self.fall_upright_threshold)
        self._fall_streak = 0 if is_standing else self._fall_streak + 1
        if self._fall_streak >= self.fall_grace_steps:
            ctx.request_termination(TerminationReason.CUSTOM)


def snapshot_module_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def snapshot_module_dict(modules: Mapping[str, nn.Module]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        name: snapshot_module_state_dict(module)
        for name, module in modules.items()
    }


def split_sequence(values: Sequence[int], parts: int) -> List[List[int]]:
    values_list = list(values)
    if not values_list:
        return []
    bounded_parts = max(1, min(int(parts), len(values_list)))
    return [list(chunk) for chunk in np.array_split(np.asarray(values_list, dtype=np.int64), bounded_parts) if len(chunk) > 0]


def build_sequential_groups(values: Sequence[int], group_size: int, drop_last: bool = False) -> List[List[int]]:
    values_list = [int(value) for value in values]
    if not values_list:
        return []
    bounded_group_size = max(1, int(group_size))
    groups = [values_list[start:start + bounded_group_size] for start in range(0, len(values_list), bounded_group_size)]
    if drop_last and groups and len(groups[-1]) < bounded_group_size:
        groups = groups[:-1]
    return [group for group in groups if group]


def flatten_groups(groups: Sequence[Sequence[Any]]) -> List[Any]:
    flattened: List[Any] = []
    for group in groups:
        flattened.extend(list(group))
    return flattened


def set_episode_seed_on_plugins(
    runtime: EnvRuntime,
    episode_seed: int,
    plugin_attr: str = "initial_state_perturbation_plugins",
    robot_names: Sequence[str] = ("robot_a", "robot_b"),
) -> None:
    plugins = getattr(runtime, plugin_attr, None)
    if not isinstance(plugins, Mapping):
        return
    for index, robot_name in enumerate(robot_names):
        plugin = plugins.get(robot_name)
        if plugin is not None and hasattr(plugin, "set_episode_seed"):
            plugin.set_episode_seed(int(episode_seed) * len(robot_names) + index)


def limit_worker_threads() -> None:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


@dataclass
class RolloutWorkerSpec:
    runtime_builder: Callable[[int], EnvRuntime]
    actor: nn.Module
    device: str = "cpu"


@dataclass
class RolloutTask:
    seeds: List[int]
    actor_state_dict: Dict[str, torch.Tensor]
    deterministic: bool = False


_ROLLOUT_RUNTIME_BUILDER: Optional[Callable[[int], EnvRuntime]] = None
_ROLLOUT_ACTOR: Optional[nn.Module] = None
_ROLLOUT_DEVICE = torch.device("cpu")


def _init_rollout_worker(spec: RolloutWorkerSpec) -> None:
    global _ROLLOUT_RUNTIME_BUILDER, _ROLLOUT_ACTOR, _ROLLOUT_DEVICE
    limit_worker_threads()
    _ROLLOUT_DEVICE = torch.device(spec.device)
    _ROLLOUT_RUNTIME_BUILDER = copy.deepcopy(spec.runtime_builder)
    _ROLLOUT_ACTOR = copy.deepcopy(spec.actor).to(_ROLLOUT_DEVICE)
    _ROLLOUT_ACTOR.eval()


def _extract_rollout_reward(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Mapping):
        for key in ("reward", "value", "score"):
            if key in value:
                return float(value[key])
        return 0.0
    return float(value)


def _collect_actor_episode(
    runtime: EnvRuntime,
    actor: nn.Module,
    device: torch.device,
    seed: int,
    deterministic: bool = False,
) -> Dict[str, Any]:
    set_episode_seed_on_plugins(runtime, int(seed))
    runtime.reset(seed=int(seed))
    obs_a_value = runtime.get_observer_output("robot_a_obs")
    obs_b_value = runtime.get_observer_output("robot_b_obs")
    if obs_a_value is None:
        raise RuntimeError("rollout runtime missing observer output: robot_a_obs")
    if obs_b_value is None:
        controlled_agent = "robot_a"
        opponent_agent = None
    else:
        controlled_agent = "robot_a" if int(np.random.default_rng(int(seed)).integers(0, 2)) == 0 else "robot_b"
        opponent_agent = "robot_b" if controlled_agent == "robot_a" else "robot_a"
    obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[float] = []
    rewards: List[float] = []
    final_obs: Optional[np.ndarray] = None
    truncated_flag = False
    terminated_flag = False
    while runtime.is_episode_active:
        controlled_action, log_prob = actor.act_numpy(obs, device=device, deterministic=deterministic)
        if opponent_agent is None:
            action_a = controlled_action
            action_b = None
        else:
            opponent_obs = np.asarray(runtime.get_observer_output(f"{opponent_agent}_obs"), dtype=np.float32)
            opponent_action, _ = actor.act_numpy(opponent_obs, device=device, deterministic=deterministic)
            if controlled_agent == "robot_a":
                action_a = controlled_action
                action_b = opponent_action
            else:
                action_a = opponent_action
                action_b = controlled_action
        observations.append(obs.copy())
        actions.append(controlled_action.copy())
        if log_prob is not None:
            log_probs.append(log_prob)
        runtime.step(action_a, action_b)
        rewards.append(_extract_rollout_reward(runtime.get_observer_output(f"{controlled_agent}_reward")))
        terminated_flag, truncated_flag = runtime.get_termination_flags()
        if terminated_flag or truncated_flag or not runtime.is_episode_active:
            next_obs_value = runtime.get_observer_output(f"{controlled_agent}_obs")
            if next_obs_value is not None:
                final_obs = np.asarray(next_obs_value, dtype=np.float32)
            break
        next_obs_value = runtime.get_observer_output(f"{controlled_agent}_obs")
        if next_obs_value is None:
            break
        obs = np.asarray(next_obs_value, dtype=np.float32)
    observations_array = np.asarray(observations, dtype=np.float32)
    actions_array = np.asarray(actions, dtype=np.float32)
    log_probs_array = np.asarray(log_probs, dtype=np.float32)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    return {
        "seed": int(seed),
        "controlled_agent": controlled_agent,
        "observations": observations_array,
        "actions": actions_array,
        "log_probs": log_probs_array,
        "rewards": rewards_array,
        "steps": int(len(observations)),
        "episode_reward": float(np.sum(rewards_array, dtype=np.float32)) if len(rewards_array) > 0 else 0.0,
        "final_obs": final_obs,
        "terminated": bool(terminated_flag),
        "truncated": bool(truncated_flag),
    }


def _collect_episode_chunk(task: RolloutTask) -> List[Dict[str, Any]]:
    if _ROLLOUT_RUNTIME_BUILDER is None or _ROLLOUT_ACTOR is None:
        raise RuntimeError("Rollout worker is not initialized")
    _ROLLOUT_ACTOR.load_state_dict(task.actor_state_dict)
    _ROLLOUT_ACTOR.eval()
    episodes: List[Dict[str, Any]] = []
    for seed in task.seeds:
        runtime = _ROLLOUT_RUNTIME_BUILDER(int(seed))
        try:
            episodes.append(
                _collect_actor_episode(
                    runtime,
                    _ROLLOUT_ACTOR,
                    _ROLLOUT_DEVICE,
                    int(seed),
                    deterministic=bool(task.deterministic),
                )
            )
        finally:
            if hasattr(runtime, "close"):
                runtime.close()
    return episodes


class RolloutCollector:
    def __init__(
        self,
        runtime_builder: Callable[[int], EnvRuntime],
        actor: nn.Module,
        max_workers: int = 1,
        worker_device: str = "cpu",
        mp_start_method: str = "spawn",
    ):
        self.runtime_builder = runtime_builder
        self.actor = actor
        self.max_workers = max(1, int(max_workers))
        self.worker_device = str(worker_device)
        self.mp_start_method = str(mp_start_method)
        self.rollout_executor = None
        if self.max_workers > 1:
            spec = RolloutWorkerSpec(
                runtime_builder=self.runtime_builder,
                actor=copy.deepcopy(self.actor).cpu(),
                device=self.worker_device,
            )
            self.rollout_executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context(self.mp_start_method),
                initializer=_init_rollout_worker,
                initargs=(spec,),
            )

    def close(self) -> None:
        if self.rollout_executor is not None:
            self.rollout_executor.shutdown(wait=True, cancel_futures=False)
            self.rollout_executor = None

    def __enter__(self) -> "RolloutCollector":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def collect_episodes(
        self,
        seeds: Sequence[int],
        worker_limit: Optional[int] = None,
        deterministic: bool = False,
    ) -> List[Dict[str, Any]]:
        seeds_list = [int(seed) for seed in seeds]
        if not seeds_list:
            return []
        actor_state_dict = snapshot_module_state_dict(self.actor)
        if self.rollout_executor is None:
            episodes: List[Dict[str, Any]] = []
            local_actor = copy.deepcopy(self.actor).cpu()
            local_actor.load_state_dict(actor_state_dict)
            local_actor.eval()
            for seed in seeds_list:
                runtime = self.runtime_builder(int(seed))
                try:
                    episodes.append(
                        _collect_actor_episode(
                            runtime,
                            local_actor,
                            torch.device("cpu"),
                            int(seed),
                            deterministic=bool(deterministic),
                        )
                    )
                finally:
                    if hasattr(runtime, "close"):
                        runtime.close()
            return episodes
        effective_workers = max(1, min(int(worker_limit or self.max_workers), self.max_workers))
        seed_chunks = split_sequence(seeds_list, effective_workers)
        tasks = [
            RolloutTask(
                seeds=seed_chunk,
                actor_state_dict=actor_state_dict,
                deterministic=bool(deterministic),
            )
            for seed_chunk in seed_chunks
            if seed_chunk
        ]
        episodes: List[Dict[str, Any]] = []
        for chunk_episodes in self.rollout_executor.map(_collect_episode_chunk, tasks):
            episodes.extend(chunk_episodes)
        return episodes

    def collect_episode_groups(
        self,
        seed_groups: Sequence[Sequence[int]],
        worker_limit: Optional[int] = None,
        deterministic: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        normalized_groups = [
            [int(seed) for seed in seed_group]
            for seed_group in seed_groups
            if seed_group
        ]
        if not normalized_groups:
            return []
        actor_state_dict = snapshot_module_state_dict(self.actor)
        if self.rollout_executor is None:
            grouped_results: List[List[Dict[str, Any]]] = []
            local_actor = copy.deepcopy(self.actor).cpu()
            local_actor.load_state_dict(actor_state_dict)
            local_actor.eval()
            for seed_group in normalized_groups:
                group_episodes: List[Dict[str, Any]] = []
                for seed in seed_group:
                    runtime = self.runtime_builder(int(seed))
                    try:
                        group_episodes.append(
                            _collect_actor_episode(
                                runtime,
                                local_actor,
                                torch.device("cpu"),
                                int(seed),
                                deterministic=bool(deterministic),
                            )
                        )
                    finally:
                        if hasattr(runtime, "close"):
                            runtime.close()
                grouped_results.append(group_episodes)
            return grouped_results
        effective_workers = max(1, min(int(worker_limit or self.max_workers), self.max_workers))
        grouped_seed_indices = split_sequence(list(range(len(normalized_groups))), effective_workers)
        tasks = [
            RolloutTask(
                seeds=flatten_groups([normalized_groups[index] for index in seed_group]),
                actor_state_dict=actor_state_dict,
                deterministic=bool(deterministic),
            )
            for seed_group in grouped_seed_indices
            if seed_group
        ]
        grouped_episodes = list(self.rollout_executor.map(_collect_episode_chunk, tasks))
        episodes = flatten_groups(grouped_episodes)
        grouped_results: List[List[Dict[str, Any]]] = []
        cursor = 0
        for normalized_group in normalized_groups:
            group_length = len(normalized_group)
            grouped_results.append(episodes[cursor:cursor + group_length])
            cursor += group_length
        return grouped_results

    def collect_grouped_episodes(
        self,
        seeds: Sequence[int],
        group_size: int,
        drop_last: bool = False,
        worker_limit: Optional[int] = None,
        deterministic: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        seed_groups = build_sequential_groups(seeds, group_size=group_size, drop_last=drop_last)
        return self.collect_episode_groups(
            seed_groups=seed_groups,
            worker_limit=worker_limit,
            deterministic=deterministic,
        )

