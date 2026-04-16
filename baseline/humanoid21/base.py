from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from envs.framework import BasePlugin, EnvRuntime, SimContext, TerminationReason


DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 1.0


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
    ):
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return mean, log_std.expand_as(mean)

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(clipped_actions)
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw_actions) - torch.log(1.0 - clipped_actions.pow(2) + 1e-6)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob.sum(dim=-1), entropy

    def act_numpy(self, obs: np.ndarray, device: torch.device, deterministic: bool) -> tuple[np.ndarray, Optional[float]]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.deterministic_action(obs_tensor)
                log_prob = None
            else:
                action, log_prob = self.sample_action(obs_tensor)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if log_prob is None:
            return action_np, None
        return action_np, float(log_prob.item())


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


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
    runtime_builder: Callable[[], Any]
    module_builders: Dict[str, Callable[[], nn.Module]]
    collect_episode_fn: Callable[..., Dict[str, Any]]
    device: str = "cpu"


@dataclass
class RolloutTask:
    seeds: List[int]
    module_state_dicts: Dict[str, Dict[str, torch.Tensor]]
    deterministic: bool
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


_ROLLOUT_RUNTIME: Optional[Any] = None
_ROLLOUT_MODULES: Dict[str, nn.Module] = {}
_ROLLOUT_COLLECT_EPISODE_FN: Optional[Callable[..., Dict[str, Any]]] = None
_ROLLOUT_DEVICE = torch.device("cpu")


def _build_worker_modules(
    module_builders: Mapping[str, Callable[[], nn.Module]],
    device: torch.device,
) -> Dict[str, nn.Module]:
    modules: Dict[str, nn.Module] = {}
    for name, builder in module_builders.items():
        module = builder()
        module = module.to(device)
        module.eval()
        modules[str(name)] = module
    return modules


def _init_rollout_worker(spec: RolloutWorkerSpec) -> None:
    global _ROLLOUT_RUNTIME, _ROLLOUT_MODULES, _ROLLOUT_COLLECT_EPISODE_FN, _ROLLOUT_DEVICE
    limit_worker_threads()
    _ROLLOUT_DEVICE = torch.device(spec.device)
    _ROLLOUT_RUNTIME = spec.runtime_builder()
    _ROLLOUT_MODULES = _build_worker_modules(spec.module_builders, _ROLLOUT_DEVICE)
    _ROLLOUT_COLLECT_EPISODE_FN = spec.collect_episode_fn


def _collect_episode_chunk(task: RolloutTask) -> List[Dict[str, Any]]:
    if _ROLLOUT_RUNTIME is None or _ROLLOUT_COLLECT_EPISODE_FN is None:
        raise RuntimeError("Rollout worker is not initialized")
    for module_name, state_dict in task.module_state_dicts.items():
        if module_name not in _ROLLOUT_MODULES:
            raise KeyError(f"Unknown rollout module in worker: {module_name}")
        module = _ROLLOUT_MODULES[module_name]
        module.load_state_dict(state_dict)
        module.eval()
    return [
        _ROLLOUT_COLLECT_EPISODE_FN(
            _ROLLOUT_RUNTIME,
            _ROLLOUT_MODULES,
            _ROLLOUT_DEVICE,
            deterministic=bool(task.deterministic),
            seed=int(seed),
            **task.extra_kwargs,
        )
        for seed in task.seeds
    ]


class RolloutCollector:
    def __init__(
        self,
        runtime_builder: Callable[[], Any],
        collect_episode_fn: Callable[..., Dict[str, Any]],
        module_builders: Optional[Dict[str, Callable[[], nn.Module]]] = None,
        max_workers: int = 1,
        worker_device: str = "cpu",
        mp_start_method: str = "spawn",
    ):
        self.runtime_builder = runtime_builder
        self.collect_episode_fn = collect_episode_fn
        self.module_builders = dict(module_builders or {})
        self.max_workers = max(1, int(max_workers))
        self.worker_device = str(worker_device)
        self.mp_start_method = str(mp_start_method)
        self.train_runtime = runtime_builder() if self.max_workers == 1 else None
        self.rollout_executor = None
        if self.max_workers > 1:
            spec = RolloutWorkerSpec(
                runtime_builder=self.runtime_builder,
                module_builders=self.module_builders,
                collect_episode_fn=self.collect_episode_fn,
                device=self.worker_device,
            )
            self.rollout_executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context(self.mp_start_method),
                initializer=_init_rollout_worker,
                initargs=(spec,),
            )

    def close(self) -> None:
        if self.train_runtime is not None and hasattr(self.train_runtime, "close"):
            self.train_runtime.close()
            self.train_runtime = None
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
        modules: Mapping[str, nn.Module],
        device: torch.device,
        deterministic: bool,
        worker_limit: Optional[int] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        seeds_list = [int(seed) for seed in seeds]
        if not seeds_list:
            return []
        rollout_kwargs = dict(extra_kwargs or {})
        if self.rollout_executor is None:
            if self.train_runtime is None:
                self.train_runtime = self.runtime_builder()
            return [
                self.collect_episode_fn(
                    self.train_runtime,
                    modules,
                    device,
                    deterministic=bool(deterministic),
                    seed=int(seed),
                    **rollout_kwargs,
                )
                for seed in seeds_list
            ]
        effective_workers = max(1, min(int(worker_limit or self.max_workers), self.max_workers))
        seed_chunks = split_sequence(seeds_list, effective_workers)
        module_state_dicts = snapshot_module_dict(modules)
        tasks = [
            RolloutTask(
                seeds=seed_chunk,
                module_state_dicts=module_state_dicts,
                deterministic=bool(deterministic),
                extra_kwargs=rollout_kwargs,
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
        modules: Mapping[str, nn.Module],
        device: torch.device,
        deterministic: bool,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        normalized_groups = [
            [int(seed) for seed in seed_group]
            for seed_group in seed_groups
            if seed_group
        ]
        if not normalized_groups:
            return []
        rollout_kwargs = dict(extra_kwargs or {})
        if self.rollout_executor is None:
            return [
                [
                    self.collect_episode_fn(
                        self.train_runtime if self.train_runtime is not None else self.runtime_builder(),
                        modules,
                        device,
                        deterministic=bool(deterministic),
                        seed=int(seed),
                        **rollout_kwargs,
                    )
                    for seed in seed_group
                ]
                for seed_group in normalized_groups
            ]
        module_state_dicts = snapshot_module_dict(modules)
        tasks = [
            RolloutTask(
                seeds=list(seed_group),
                module_state_dicts=module_state_dicts,
                deterministic=bool(deterministic),
                extra_kwargs=rollout_kwargs,
            )
            for seed_group in normalized_groups
        ]
        return list(self.rollout_executor.map(_collect_episode_chunk, tasks))

    def collect_grouped_episodes(
        self,
        seeds: Sequence[int],
        group_size: int,
        modules: Mapping[str, nn.Module],
        device: torch.device,
        deterministic: bool,
        drop_last: bool = False,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        seed_groups = build_sequential_groups(seeds, group_size=group_size, drop_last=drop_last)
        return self.collect_episode_groups(
            seed_groups=seed_groups,
            modules=modules,
            device=device,
            deterministic=deterministic,
            extra_kwargs=extra_kwargs,
        )
