from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from envs.framework import BasePlugin, EnvRuntime, SimContext, TerminationReason
from envs.humanoid21 import Humanoid21Observer, MujocoCombatSimulator
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin

CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 3.0
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)
INITIAL_DISTANCE = 3.0
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5
ACTION_DIM = Humanoid21Observer.ACTION_DIM
OBS_DIM = Humanoid21Observer.OBS_DIM
GROUP_SIZE = max(1, int(os.environ.get("STANDING_GROUP_SIZE", "32")))
EPISODES_PER_UPDATE = 256 * 32
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 4096 * 32
MAX_UPDATES = 10000
EVAL_INTERVAL = 5
EVAL_EPISODES = 16
LEARNING_RATE = 3e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 1e-3
GRAD_CLIP_NORM = 1.0
TARGET_KL = float(os.environ.get("STANDING_TARGET_KL", "0.05"))
ACTOR_HIDDEN_DIM = 256
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0
FALL_HEIGHT_THRESHOLD = 1.10
FALL_UPRIGHT_THRESHOLD = 0.8
FALL_GRACE_STEPS = 3
SEED = 42
RUNS_DIR = Path(__file__).resolve().parent / "runs"
ROLLOUT_WORKERS = max(1, int(os.environ.get("STANDING_ROLLOUT_WORKERS", str(min(64, max(1, (os.cpu_count() or 1) // 2))))))
EVAL_WORKERS = max(1, int(os.environ.get("STANDING_EVAL_WORKERS", str(min(ROLLOUT_WORKERS, EVAL_EPISODES)))))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        is_standing = bool(
            height >= self.fall_height_threshold and uprightness >= self.fall_upright_threshold
        )
        self._fall_streak = 0 if is_standing else self._fall_streak + 1
        if self._fall_streak >= self.fall_grace_steps:
            ctx.request_termination(TerminationReason.CUSTOM)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
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
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
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


def _snapshot_module_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _split_sequence(values: Sequence[Any], parts: int) -> List[List[Any]]:
    values_list = list(values)
    if not values_list:
        return []
    bounded_parts = max(1, min(parts, len(values_list)))
    chunk_size = int(math.ceil(len(values_list) / bounded_parts))
    return [values_list[start:start + chunk_size] for start in range(0, len(values_list), chunk_size)]


def _sample_rollout_setup(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    controlled_agent = "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"
    opponent_agent = "robot_b" if controlled_agent == "robot_a" else "robot_a"
    initial_distance = float(rng.uniform(ROLLOUT_INITIAL_DISTANCE_MIN, ROLLOUT_INITIAL_DISTANCE_MAX))
    return {
        "controlled_agent": controlled_agent,
        "opponent_agent": opponent_agent,
        "initial_distance": initial_distance,
    }


def _act(
    actor: Actor,
    obs: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> tuple[np.ndarray, Optional[float]]:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            action = actor.deterministic_action(obs_tensor)
            log_prob = None
        else:
            action, log_prob = actor.sample_action(obs_tensor)
    action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
    if log_prob is None:
        return action_np, None
    return action_np, float(log_prob.item())


_ROLLOUT_RUNTIME: Optional[EnvRuntime] = None
_ROLLOUT_ACTOR: Optional[Actor] = None
_CPU_DEVICE = torch.device("cpu")


def _set_group_seed(runtime: EnvRuntime, group_seed: int) -> None:
    plugins = getattr(runtime, "initial_state_perturbation_plugins", None)
    if not isinstance(plugins, dict):
        return
    robot_a_plugin = plugins.get("robot_a")
    robot_b_plugin = plugins.get("robot_b")
    if robot_a_plugin is not None:
        robot_a_plugin.set_episode_seed(group_seed * 2)
    if robot_b_plugin is not None:
        robot_b_plugin.set_episode_seed(group_seed * 2 + 1)


def _build_group_requests(total_episodes: int, group_size: int, base_group_seed: int) -> List[Dict[str, int]]:
    requests: List[Dict[str, int]] = []
    group_index = 0
    episodes_remaining = int(total_episodes)
    while episodes_remaining > 0:
        episodes_in_group = min(int(group_size), episodes_remaining)
        requests.append(
            {
                "group_seed": int(base_group_seed + group_index),
                "episodes_in_group": int(episodes_in_group),
            }
        )
        episodes_remaining -= episodes_in_group
        group_index += 1
    return requests


def _limit_worker_threads() -> None:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _init_rollout_worker() -> None:
    global _ROLLOUT_RUNTIME, _ROLLOUT_ACTOR
    _limit_worker_threads()
    _ROLLOUT_RUNTIME = build_runtime()
    _ROLLOUT_ACTOR = Actor(OBS_DIM, ACTION_DIM, ACTOR_HIDDEN_DIM).to(_CPU_DEVICE)
    _ROLLOUT_ACTOR.eval()


def _collect_episode_chunk(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    global _ROLLOUT_RUNTIME, _ROLLOUT_ACTOR
    if _ROLLOUT_RUNTIME is None or _ROLLOUT_ACTOR is None:
        _init_rollout_worker()
    _ROLLOUT_ACTOR.load_state_dict(task["actor_state_dict"])
    _ROLLOUT_ACTOR.eval()
    episodes: List[Dict[str, Any]] = []
    for group_request in task["group_requests"]:
        group_seed = int(group_request["group_seed"])
        episodes_in_group = int(group_request["episodes_in_group"])
        for _ in range(episodes_in_group):
            episodes.append(
                collect_episode(
                    _ROLLOUT_RUNTIME,
                    _ROLLOUT_ACTOR,
                    _CPU_DEVICE,
                    deterministic=bool(task["deterministic"]),
                    group_seed=group_seed,
                )
            )
    return episodes


class GRPOTrainer:
    def __init__(self, device: torch.device, resume_from: Optional[Path] = None):
        self.device = device
        self.actor = Actor(OBS_DIM, ACTION_DIM, ACTOR_HIDDEN_DIM).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.best_eval_score = (-1.0, -1.0)
        self.history: List[Dict[str, Any]] = []
        self.train_runtime = build_runtime() if ROLLOUT_WORKERS == 1 else None
        self.rollout_executor = ProcessPoolExecutor(
            max_workers=ROLLOUT_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_init_rollout_worker,
        ) if ROLLOUT_WORKERS > 1 else None
        self.run_dir = self._build_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from.resolve() if resume_from is not None else None
        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)
        self._save_config()

    def train(self) -> None:
        try:
            for update_index in range(1, MAX_UPDATES + 1):
                groups_per_update = int(math.ceil(EPISODES_PER_UPDATE / GROUP_SIZE))
                base_group_seed = SEED + update_index * groups_per_update
                group_requests = _build_group_requests(
                    total_episodes=EPISODES_PER_UPDATE,
                    group_size=GROUP_SIZE,
                    base_group_seed=base_group_seed,
                )
                episodes = self._collect_episodes(
                    group_requests=group_requests,
                    deterministic=False,
                    worker_limit=ROLLOUT_WORKERS,
                )
                update_stats = self._update_actor(episodes)
                mean_survival_steps = float(np.mean([episode["survival_steps"] for episode in episodes]))
                mean_survival_seconds = float(np.mean([episode["survival_seconds"] for episode in episodes]))
                success_rate = float(np.mean([episode["success"] for episode in episodes]))
                record = {
                    "update": update_index,
                    "train_mean_survival_steps": mean_survival_steps,
                    "train_mean_survival_seconds": mean_survival_seconds,
                    "train_success_rate": success_rate,
                    **update_stats,
                }
                if update_index % EVAL_INTERVAL == 0:
                    eval_stats = self.evaluate_actor()
                    record.update({f"eval_{key}": value for key, value in eval_stats.items()})
                    eval_score = (
                        float(eval_stats["success_rate"]),
                        float(eval_stats["mean_survival_steps"]),
                    )
                    if eval_score > self.best_eval_score:
                        self.best_eval_score = eval_score
                        self._save_checkpoint(self.run_dir / "best_model.pt")
                self.history.append(record)
                self._print_record(record)
                if update_index % EVAL_INTERVAL == 0:
                    self._write_history()
                if update_index % 25 == 0:
                    self._save_checkpoint(self.checkpoint_dir / f"update_{update_index}.pt")
            final_model_path = self.run_dir / "final_model.pt"
            self._save_checkpoint(final_model_path)
            self._write_history()
        finally:
            self.close()

    def close(self) -> None:
        if self.train_runtime is not None:
            self.train_runtime.close()
            self.train_runtime = None
        if self.rollout_executor is not None:
            self.rollout_executor.shutdown(wait=True, cancel_futures=False)
            self.rollout_executor = None

    def _collect_episodes(
        self,
        group_requests: Sequence[Dict[str, int]],
        deterministic: bool,
        worker_limit: int,
    ) -> List[Dict[str, Any]]:
        if self.rollout_executor is None:
            if self.train_runtime is None:
                self.train_runtime = build_runtime()
            episodes: List[Dict[str, Any]] = []
            for group_request in group_requests:
                group_seed = int(group_request["group_seed"])
                episodes_in_group = int(group_request["episodes_in_group"])
                for _ in range(episodes_in_group):
                    episodes.append(
                        collect_episode(
                            self.train_runtime,
                            self.actor,
                            self.device,
                            deterministic=deterministic,
                            group_seed=group_seed,
                        )
                    )
            return episodes
        actor_state_dict = _snapshot_module_state_dict(self.actor)
        group_chunks = _split_sequence(list(group_requests), max(1, min(worker_limit, ROLLOUT_WORKERS)))
        tasks = [
            {
                "actor_state_dict": actor_state_dict,
                "deterministic": deterministic,
                "group_requests": group_chunk,
            }
            for group_chunk in group_chunks
            if group_chunk
        ]
        episodes: List[Dict[str, Any]] = []
        for chunk_episodes in self.rollout_executor.map(_collect_episode_chunk, tasks):
            episodes.extend(chunk_episodes)
        return episodes

    def evaluate_actor(self) -> Dict[str, float]:
        group_requests = _build_group_requests(
            total_episodes=EVAL_EPISODES,
            group_size=1,
            base_group_seed=SEED + 100000,
        )
        episodes = self._collect_episodes(
            group_requests=group_requests,
            deterministic=True,
            worker_limit=EVAL_WORKERS,
        )
        return {
            "mean_survival_steps": float(np.mean([episode["survival_steps"] for episode in episodes])),
            "mean_survival_seconds": float(np.mean([episode["survival_seconds"] for episode in episodes])),
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        }

    def _update_actor(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        episode_returns = np.asarray([episode["episode_return"] for episode in episodes], dtype=np.float32)
        advantages_per_episode = normalize_group_returns(episode_returns, GROUP_SIZE)
        obs_batch = np.concatenate([episode["observations"] for episode in episodes], axis=0)
        action_batch = np.concatenate([episode["actions"] for episode in episodes], axis=0)
        old_log_prob_batch = np.concatenate([episode["log_probs"] for episode in episodes], axis=0)
        advantage_batch = np.concatenate([
            np.full((episode["steps"],), advantages_per_episode[episode_index], dtype=np.float32)
            for episode_index, episode in enumerate(episodes)
        ])
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        old_log_prob_tensor = torch.as_tensor(old_log_prob_batch, dtype=torch.float32, device=self.device)
        advantage_tensor = torch.as_tensor(advantage_batch, dtype=torch.float32, device=self.device)
        total_steps = obs_tensor.shape[0]
        policy_losses: List[float] = []
        entropies: List[float] = []
        ratios: List[float] = []
        approx_kls: List[float] = []
        optimizer_steps = 0
        early_stop = False
        early_stop_kl = 0.0
        for _ in range(UPDATE_EPOCHS):
            permutation = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, MINIBATCH_SIZE):
                batch_indices = permutation[start:start + MINIBATCH_SIZE]
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_prob = old_log_prob_tensor[batch_indices]
                batch_advantage = advantage_tensor[batch_indices]
                new_log_prob, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                ratio = torch.exp(new_log_prob - batch_old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                objective = torch.min(ratio * batch_advantage, clipped_ratio * batch_advantage)
                policy_loss = -objective.mean()
                approx_kl = float((batch_old_log_prob - new_log_prob).mean().item())
                approx_kls.append(approx_kl)
                if TARGET_KL > 0.0 and approx_kl > TARGET_KL:
                    early_stop = True
                    early_stop_kl = approx_kl
                    break
                loss = policy_loss - ENTROPY_COEF * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()
                optimizer_steps += 1
                policy_losses.append(float(policy_loss.item()))
                entropies.append(float(entropy.mean().item()))
                ratios.append(float(ratio.mean().item()))
            if early_stop:
                break
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "ratio": float(np.mean(ratios)) if ratios else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "optimizer_steps": optimizer_steps,
            "early_stop": int(early_stop),
            "early_stop_kl": float(early_stop_kl),
        }

    def _build_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return RUNS_DIR / f"standing_turbulence_stage1_grpo_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "initial_distance": INITIAL_DISTANCE,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
            "symmetric_self_play_rollout": True,
            "algorithm": "grpo",
            "stage": 1,
            "objective": "survival_duration",
            "group_size": GROUP_SIZE,
            "episodes_per_update": EPISODES_PER_UPDATE,
            "update_epochs": UPDATE_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "max_updates": MAX_UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "eval_episodes": EVAL_EPISODES,
            "learning_rate": LEARNING_RATE,
            "clip_eps": CLIP_EPS,
            "entropy_coef": ENTROPY_COEF,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "target_kl": TARGET_KL,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "fall_height_threshold": FALL_HEIGHT_THRESHOLD,
            "fall_upright_threshold": FALL_UPRIGHT_THRESHOLD,
            "fall_grace_steps": FALL_GRACE_STEPS,
            "advantage_mode": "episode_return_group_normalized",
            "rollout_workers": ROLLOUT_WORKERS,
            "eval_workers": EVAL_WORKERS,
            "seed": SEED,
            "resume_from": str(self.resume_from) if self.resume_from is not None else None,
        }
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def _save_checkpoint(self, path: Path) -> None:
        payload = {
            "algorithm": "grpo",
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": ACTOR_HIDDEN_DIM,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_eval_score": self.best_eval_score,
        }
        torch.save(payload, path)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        if int(payload.get("obs_dim", OBS_DIM)) != OBS_DIM:
            raise ValueError(f"Checkpoint obs_dim mismatch: expected {OBS_DIM}, got {payload.get('obs_dim')}")
        if int(payload.get("action_dim", ACTION_DIM)) != ACTION_DIM:
            raise ValueError(f"Checkpoint action_dim mismatch: expected {ACTION_DIM}, got {payload.get('action_dim')}")
        self.actor.load_state_dict(payload["state_dict"])
        optimizer_state_dict = payload.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            with suppress(ValueError):
                self.optimizer.load_state_dict(optimizer_state_dict)
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(self.device)
        best_eval_score = payload.get("best_eval_score")
        if isinstance(best_eval_score, (list, tuple)) and len(best_eval_score) == 2:
            self.best_eval_score = (float(best_eval_score[0]), float(best_eval_score[1]))

    def _write_history(self) -> None:
        with (self.run_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    def _print_record(self, record: Dict[str, Any]) -> None:
        keys = [
            "update",
            "train_mean_survival_steps",
            "train_mean_survival_seconds",
            "train_success_rate",
            "policy_loss",
            "entropy",
            "ratio",
            "approx_kl",
        ]
        if "optimizer_steps" in record:
            keys.append("optimizer_steps")
        if record.get("early_stop", 0):
            keys.extend([
                "early_stop",
                "early_stop_kl",
            ])
        if "eval_mean_survival_steps" in record:
            keys.extend([
                "eval_mean_survival_steps",
                "eval_mean_survival_seconds",
                "eval_success_rate",
            ])
        message = " | ".join(f"{key}={record[key]:.4f}" if isinstance(record[key], float) else f"{key}={record[key]}" for key in keys)
        print(message, flush=True)


def normalize_group_returns(returns: np.ndarray, group_size: int) -> np.ndarray:
    advantages = np.zeros_like(returns, dtype=np.float32)
    for start in range(0, len(returns), group_size):
        group = returns[start:start + group_size]
        group_mean = float(group.mean())
        group_std = float(group.std())
        advantages[start:start + group_size] = (group - group_mean) / (group_std + 1e-6)
    return advantages


def build_runtime() -> EnvRuntime:
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))
    robot_a_initial_perturbation = InitialStatePerturbationPlugin(
        target_robot="robot_a",
        joint_pos_delta_max=0.05,
        joint_vel_delta_max=0.05,
        root_xy_offset_max=0.05,
        root_tilt_deg_max=10.0,
        root_linear_velocity_delta_max=[0.5, 0.5, 0.0],
        root_angular_velocity_delta_max=[0.5, 0.5, 0.2],
        random_seed=None,
    )
    robot_b_initial_perturbation = InitialStatePerturbationPlugin(
        target_robot="robot_b",
        joint_pos_delta_max=0.05,
        joint_vel_delta_max=0.05,
        root_xy_offset_max=0.05,
        root_tilt_deg_max=10.0,
        root_linear_velocity_delta_max=[0.5, 0.5, 0.0],
        root_angular_velocity_delta_max=[0.5, 0.5, 0.2],
        random_seed=None,
    )
    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins={
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
        },
        plugins=[
            StandingTerminationPlugin(
                agent_id="robot_a",
                fall_height_threshold=FALL_HEIGHT_THRESHOLD,
                fall_upright_threshold=FALL_UPRIGHT_THRESHOLD,
                fall_grace_steps=FALL_GRACE_STEPS,
            ),
            StandingTerminationPlugin(
                agent_id="robot_b",
                fall_height_threshold=FALL_HEIGHT_THRESHOLD,
                fall_upright_threshold=FALL_UPRIGHT_THRESHOLD,
                fall_grace_steps=FALL_GRACE_STEPS,
            ),
            robot_a_initial_perturbation,
            robot_b_initial_perturbation,
        ],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=MAX_STEPS,
    )
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.action_space = Humanoid21Observer.get_action_space()
    runtime.initial_state_perturbation_plugins = {
        "robot_a": robot_a_initial_perturbation,
        "robot_b": robot_b_initial_perturbation,
    }
    return runtime


def collect_episode(
    runtime: EnvRuntime,
    actor: Actor,
    device: torch.device,
    deterministic: bool,
    group_seed: int,
) -> Dict[str, Any]:
    rollout_setup = _sample_rollout_setup(group_seed)
    controlled_agent = str(rollout_setup["controlled_agent"])
    opponent_agent = str(rollout_setup["opponent_agent"])
    initial_distance = float(rollout_setup["initial_distance"])
    _set_group_seed(runtime, group_seed)
    runtime.reset(seed=group_seed, options={"initial_distance": initial_distance})
    obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[float] = []
    for _ in range(MAX_STEPS):
        opponent_obs = np.asarray(runtime.get_observer_output(f"{opponent_agent}_obs"), dtype=np.float32)
        controlled_action, log_prob = _act(actor, obs, device, deterministic=deterministic)
        opponent_action, _ = _act(actor, opponent_obs, device, deterministic=deterministic)
        if controlled_agent == "robot_a":
            runtime.step(controlled_action, opponent_action)
        else:
            runtime.step(opponent_action, controlled_action)
        observations.append(obs.copy())
        actions.append(controlled_action.copy())
        if log_prob is not None:
            log_probs.append(log_prob)
        obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
        terminated, truncated = runtime.get_termination_flags()
        if terminated or truncated:
            break
    survival_steps = int(len(observations))
    survival_seconds = float(survival_steps / CONTROL_FREQUENCY)
    success = int(survival_steps >= MAX_STEPS)
    observations_array = np.asarray(observations, dtype=np.float32).reshape(survival_steps, OBS_DIM)
    actions_array = np.asarray(actions, dtype=np.float32).reshape(survival_steps, ACTION_DIM)
    log_probs_array = np.asarray(log_probs, dtype=np.float32).reshape(len(log_probs),)
    return {
        "observations": observations_array,
        "actions": actions_array,
        "log_probs": log_probs_array,
        "steps": survival_steps,
        "episode_return": float(survival_steps),
        "survival_steps": survival_steps,
        "survival_seconds": survival_seconds,
        "success": success,
        "controlled_agent": controlled_agent,
        "initial_distance": initial_distance,
        "group_seed": int(group_seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = GRPOTrainer(device=device, resume_from=args.resume_from)
    trainer.train()
    print(f"run_dir={trainer.run_dir}", flush=True)


if __name__ == "__main__":
    main()
