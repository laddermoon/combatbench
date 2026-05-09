from __future__ import annotations

import argparse
import json
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
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver

CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 3.0
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)
INITIAL_DISTANCE = 3.0
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5
ACTION_DIM = Humanoid21Observer.ACTION_DIM
OBS_DIM = Humanoid21Observer.OBS_DIM
EPISODES_PER_UPDATE = 256 * 32
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 4096 * 32
MAX_UPDATES = 10000
EVAL_INTERVAL = 5
EVAL_EPISODES = 16
LEARNING_RATE = 3e-4
GAMMA = float(os.environ.get("STANDING_GAMMA", "0.99"))
GAE_LAMBDA = float(os.environ.get("STANDING_GAE_LAMBDA", "0.95"))
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 1e-3
GRAD_CLIP_NORM = 1.0
TARGET_KL = float(os.environ.get("STANDING_TARGET_KL", "0.05"))
ACTOR_HIDDEN_DIM = 256
CRITIC_HIDDEN_DIM = 256
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0
FALL_HEIGHT_THRESHOLD = 1.10
FALL_UPRIGHT_THRESHOLD = 0.8
FALL_GRACE_STEPS = 3
SEED = 42
RUNS_DIR = Path(__file__).resolve().parent / "runs"
ROLLOUT_WORKERS = max(1, int(os.environ.get("STANDING_ROLLOUT_WORKERS", str(min(64, max(1, (os.cpu_count() or 1) // 2))))))
EVAL_WORKERS = max(1, int(os.environ.get("STANDING_EVAL_WORKERS", str(min(ROLLOUT_WORKERS, EVAL_EPISODES)))))
BALANCE_REWARD_MODE = os.environ.get("STANDING_BALANCE_REWARD_MODE", "delta").strip().lower()
BALANCE_INVALID_SCORE = float(os.environ.get("STANDING_BALANCE_INVALID_SCORE", "-1.0"))
BALANCE_FRONT_PENALTY_COEF = float(os.environ.get("STANDING_BALANCE_FRONT_PENALTY_COEF", "2.0"))
BALANCE_BACK_PENALTY_COEF = float(os.environ.get("STANDING_BALANCE_BACK_PENALTY_COEF", "2.0"))
BALANCE_CENTER_OFFSET_PENALTY_COEF = float(os.environ.get("STANDING_BALANCE_CENTER_OFFSET_PENALTY_COEF", "1.0"))
BALANCE_SUPPORT_AXIS_VELOCITY_COEF = float(os.environ.get("STANDING_BALANCE_SUPPORT_AXIS_VELOCITY_COEF", "0.25"))
BALANCE_SUPPORT_LATERAL_VELOCITY_COEF = float(os.environ.get("STANDING_BALANCE_SUPPORT_LATERAL_VELOCITY_COEF", "0.5"))
BALANCE_VELOCITY_CLIP = float(os.environ.get("STANDING_BALANCE_VELOCITY_CLIP", "1.5"))
BALANCE_SCORE_CLIP_MIN = float(os.environ.get("STANDING_BALANCE_SCORE_CLIP_MIN", "-4.0"))
BALANCE_SCORE_CLIP_MAX = float(os.environ.get("STANDING_BALANCE_SCORE_CLIP_MAX", "1.0"))

if BALANCE_REWARD_MODE not in {"absolute", "delta"}:
    raise ValueError(f"Unsupported STANDING_BALANCE_REWARD_MODE: {BALANCE_REWARD_MODE}")


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


def _split_sequence(values: Sequence[int], parts: int) -> List[List[int]]:
    values_list = list(values)
    if not values_list:
        return []
    bounded_parts = max(1, min(parts, len(values_list)))
    return [list(chunk) for chunk in np.array_split(np.asarray(values_list, dtype=np.int64), bounded_parts) if len(chunk) > 0]


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


def _compute_balance_reward_terms(balance_output: Dict[str, Any]) -> Dict[str, float]:
    if not bool(balance_output.get("ground_support_frame_defined", False)):
        return {
            "absolute_score": float(BALANCE_INVALID_SCORE),
            "front_distance": 0.0,
            "back_distance": 0.0,
            "center_offset_distance": 0.0,
            "support_axis_velocity_toward_center": 0.0,
            "support_lateral_velocity_toward_center": 0.0,
        }
    support_span = float(balance_output["support_span"])
    support_axis_projection_coordinate = float(balance_output["support_axis_projection_coordinate"])
    support_lateral_signed_distance = float(balance_output["support_lateral_signed_distance"])
    support_axis_velocity = float(balance_output["center_of_mass_velocity_along_support_axis"])
    support_lateral_velocity = float(balance_output["center_of_mass_velocity_along_support_lateral_axis"])
    required_values = np.asarray(
        [
            support_span,
            support_axis_projection_coordinate,
            support_lateral_signed_distance,
            support_axis_velocity,
            support_lateral_velocity,
        ],
        dtype=np.float64,
    )
    if support_span <= 0.0 or not np.all(np.isfinite(required_values)):
        return {
            "absolute_score": float(BALANCE_INVALID_SCORE),
            "front_distance": 0.0,
            "back_distance": 0.0,
            "center_offset_distance": 0.0,
            "support_axis_velocity_toward_center": 0.0,
            "support_lateral_velocity_toward_center": 0.0,
        }
    support_axis_center_offset = support_axis_projection_coordinate - 0.5 * support_span
    front_distance = max(support_lateral_signed_distance, 0.0)
    back_distance = max(-support_lateral_signed_distance, 0.0)
    center_offset_distance = abs(support_axis_center_offset)
    if center_offset_distance > 1e-6:
        support_axis_velocity_toward_center = -np.sign(support_axis_center_offset) * support_axis_velocity
    else:
        support_axis_velocity_toward_center = -abs(support_axis_velocity)
    if abs(support_lateral_signed_distance) > 1e-6:
        support_lateral_velocity_toward_center = -np.sign(support_lateral_signed_distance) * support_lateral_velocity
    else:
        support_lateral_velocity_toward_center = -abs(support_lateral_velocity)
    support_axis_velocity_toward_center = float(np.clip(support_axis_velocity_toward_center, -BALANCE_VELOCITY_CLIP, BALANCE_VELOCITY_CLIP))
    support_lateral_velocity_toward_center = float(np.clip(support_lateral_velocity_toward_center, -BALANCE_VELOCITY_CLIP, BALANCE_VELOCITY_CLIP))
    absolute_score = 1.0
    absolute_score -= BALANCE_FRONT_PENALTY_COEF * front_distance
    absolute_score -= BALANCE_BACK_PENALTY_COEF * back_distance
    absolute_score -= BALANCE_CENTER_OFFSET_PENALTY_COEF * center_offset_distance
    absolute_score += BALANCE_SUPPORT_AXIS_VELOCITY_COEF * support_axis_velocity_toward_center
    absolute_score += BALANCE_SUPPORT_LATERAL_VELOCITY_COEF * support_lateral_velocity_toward_center
    absolute_score = float(np.clip(absolute_score, BALANCE_SCORE_CLIP_MIN, BALANCE_SCORE_CLIP_MAX))
    return {
        "absolute_score": absolute_score,
        "front_distance": float(front_distance),
        "back_distance": float(back_distance),
        "center_offset_distance": float(center_offset_distance),
        "support_axis_velocity_toward_center": support_axis_velocity_toward_center,
        "support_lateral_velocity_toward_center": support_lateral_velocity_toward_center,
    }


def _compute_balance_step_reward(
    current_balance_terms: Dict[str, float],
    previous_balance_terms: Optional[Dict[str, float]],
) -> float:
    current_score = float(current_balance_terms["absolute_score"])
    if BALANCE_REWARD_MODE == "absolute":
        return current_score
    if previous_balance_terms is None:
        return 0.0
    previous_score = float(previous_balance_terms["absolute_score"])
    return current_score - previous_score


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


def _act_with_value(
    actor: Actor,
    critic: Critic,
    obs: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> tuple[np.ndarray, Optional[float], float]:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        value = critic(obs_tensor).item()
        if deterministic:
            action = actor.deterministic_action(obs_tensor)
            log_prob = None
        else:
            action, log_prob = actor.sample_action(obs_tensor)
    action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
    if log_prob is None:
        return action_np, None, float(value)
    return action_np, float(log_prob.item()), float(value)


_ROLLOUT_RUNTIME: Optional[EnvRuntime] = None
_ROLLOUT_ACTOR: Optional[Actor] = None
_ROLLOUT_CRITIC: Optional[Critic] = None
_CPU_DEVICE = torch.device("cpu")


def _set_episode_seed(runtime: EnvRuntime, episode_seed: int) -> None:
    plugins = getattr(runtime, "initial_state_perturbation_plugins", None)
    if not isinstance(plugins, dict):
        return
    robot_a_plugin = plugins.get("robot_a")
    robot_b_plugin = plugins.get("robot_b")
    if robot_a_plugin is not None:
        robot_a_plugin.set_episode_seed(episode_seed * 2)
    if robot_b_plugin is not None:
        robot_b_plugin.set_episode_seed(episode_seed * 2 + 1)


def _limit_worker_threads() -> None:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _init_rollout_worker() -> None:
    global _ROLLOUT_RUNTIME, _ROLLOUT_ACTOR, _ROLLOUT_CRITIC
    _limit_worker_threads()
    _ROLLOUT_RUNTIME = build_runtime()
    _ROLLOUT_ACTOR = Actor(OBS_DIM, ACTION_DIM, ACTOR_HIDDEN_DIM).to(_CPU_DEVICE)
    _ROLLOUT_ACTOR.eval()
    _ROLLOUT_CRITIC = Critic(OBS_DIM, CRITIC_HIDDEN_DIM).to(_CPU_DEVICE)
    _ROLLOUT_CRITIC.eval()


def _collect_episode_chunk(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    global _ROLLOUT_RUNTIME, _ROLLOUT_ACTOR, _ROLLOUT_CRITIC
    if _ROLLOUT_RUNTIME is None or _ROLLOUT_ACTOR is None or _ROLLOUT_CRITIC is None:
        _init_rollout_worker()
    _ROLLOUT_ACTOR.load_state_dict(task["actor_state_dict"])
    _ROLLOUT_ACTOR.eval()
    _ROLLOUT_CRITIC.load_state_dict(task["critic_state_dict"])
    _ROLLOUT_CRITIC.eval()
    return [
        collect_episode(
            _ROLLOUT_RUNTIME,
            _ROLLOUT_ACTOR,
            _ROLLOUT_CRITIC,
            _CPU_DEVICE,
            deterministic=bool(task["deterministic"]),
            seed=int(seed),
        )
        for seed in task["seeds"]
    ]


class PPOTrainer:
    def __init__(self, device: torch.device, resume_from: Optional[Path] = None):
        self.device = device
        self.actor = Actor(OBS_DIM, ACTION_DIM, ACTOR_HIDDEN_DIM).to(device)
        self.critic = Critic(OBS_DIM, CRITIC_HIDDEN_DIM).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LEARNING_RATE,
        )
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
                seeds = [
                    SEED + update_index * EPISODES_PER_UPDATE + episode_index
                    for episode_index in range(EPISODES_PER_UPDATE)
                ]
                episodes = self._collect_episodes(
                    seeds=seeds,
                    deterministic=False,
                    worker_limit=ROLLOUT_WORKERS,
                )
                update_stats = self._update_policy(episodes)
                mean_episode_reward = float(np.mean([episode["episode_reward"] for episode in episodes]))
                mean_survival_steps = float(np.mean([episode["survival_steps"] for episode in episodes]))
                mean_survival_seconds = float(np.mean([episode["survival_seconds"] for episode in episodes]))
                success_rate = float(np.mean([episode["success"] for episode in episodes]))
                mean_balance_score = float(np.mean([episode["mean_balance_score"] for episode in episodes]))
                record = {
                    "update": update_index,
                    "train_mean_reward": mean_episode_reward,
                    "train_mean_survival_steps": mean_survival_steps,
                    "train_mean_survival_seconds": mean_survival_seconds,
                    "train_success_rate": success_rate,
                    "train_mean_balance_score": mean_balance_score,
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
        seeds: Sequence[int],
        deterministic: bool,
        worker_limit: int,
    ) -> List[Dict[str, Any]]:
        if self.rollout_executor is None:
            if self.train_runtime is None:
                self.train_runtime = build_runtime()
            return [
                collect_episode(self.train_runtime, self.actor, self.critic, self.device, deterministic=deterministic, seed=int(seed))
                for seed in seeds
            ]
        actor_state_dict = _snapshot_module_state_dict(self.actor)
        critic_state_dict = _snapshot_module_state_dict(self.critic)
        seed_chunks = _split_sequence(list(seeds), max(1, min(worker_limit, ROLLOUT_WORKERS)))
        tasks = [
            {
                "actor_state_dict": actor_state_dict,
                "critic_state_dict": critic_state_dict,
                "deterministic": deterministic,
                "seeds": seed_chunk,
            }
            for seed_chunk in seed_chunks
            if seed_chunk
        ]
        episodes: List[Dict[str, Any]] = []
        for chunk_episodes in self.rollout_executor.map(_collect_episode_chunk, tasks):
            episodes.extend(chunk_episodes)
        return episodes

    def evaluate_actor(self) -> Dict[str, float]:
        seeds = [SEED + 100000 + episode_index for episode_index in range(EVAL_EPISODES)]
        episodes = self._collect_episodes(
            seeds=seeds,
            deterministic=True,
            worker_limit=EVAL_WORKERS,
        )
        return {
            "mean_reward": float(np.mean([episode["episode_reward"] for episode in episodes])),
            "mean_survival_steps": float(np.mean([episode["survival_steps"] for episode in episodes])),
            "mean_survival_seconds": float(np.mean([episode["survival_seconds"] for episode in episodes])),
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
            "mean_balance_score": float(np.mean([episode["mean_balance_score"] for episode in episodes])),
        }

    def _update_policy(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        advantage_batches: List[np.ndarray] = []
        return_batches: List[np.ndarray] = []
        for episode in episodes:
            advantages, returns = _compute_gae(
                rewards=np.asarray(episode["rewards"], dtype=np.float32),
                values=np.asarray(episode["values"], dtype=np.float32),
                bootstrap_value=float(episode["bootstrap_value"]),
            )
            advantage_batches.append(advantages)
            return_batches.append(returns)
        obs_batch = np.concatenate([episode["observations"] for episode in episodes], axis=0)
        action_batch = np.concatenate([episode["actions"] for episode in episodes], axis=0)
        old_log_prob_batch = np.concatenate([episode["log_probs"] for episode in episodes], axis=0)
        advantage_batch = np.concatenate(advantage_batches, axis=0)
        return_batch = np.concatenate(return_batches, axis=0)
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-6)
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        old_log_prob_tensor = torch.as_tensor(old_log_prob_batch, dtype=torch.float32, device=self.device)
        advantage_tensor = torch.as_tensor(advantage_batch, dtype=torch.float32, device=self.device)
        return_tensor = torch.as_tensor(return_batch, dtype=torch.float32, device=self.device)
        total_steps = obs_tensor.shape[0]
        policy_losses: List[float] = []
        value_losses: List[float] = []
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
                batch_returns = return_tensor[batch_indices]
                new_log_prob, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                value_pred = self.critic(batch_obs)
                ratio = torch.exp(new_log_prob - batch_old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                objective = torch.min(ratio * batch_advantage, clipped_ratio * batch_advantage)
                policy_loss = -objective.mean()
                value_loss = torch.nn.functional.mse_loss(value_pred, batch_returns)
                approx_kl = float((batch_old_log_prob - new_log_prob).mean().item())
                approx_kls.append(approx_kl)
                if TARGET_KL > 0.0 and approx_kl > TARGET_KL:
                    early_stop = True
                    early_stop_kl = approx_kl
                    break
                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), GRAD_CLIP_NORM)
                self.optimizer.step()
                optimizer_steps += 1
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.mean().item()))
                ratios.append(float(ratio.mean().item()))
            if early_stop:
                break
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "ratio": float(np.mean(ratios)) if ratios else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "optimizer_steps": optimizer_steps,
            "early_stop": int(early_stop),
            "early_stop_kl": float(early_stop_kl),
        }

    def _build_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return RUNS_DIR / f"standing_turbulence_stage1_ppo_balance_dense_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "initial_distance": INITIAL_DISTANCE,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
            "symmetric_self_play_rollout": True,
            "algorithm": "ppo",
            "stage": 1,
            "objective": "balance_dense_reward_only",
            "episodes_per_update": EPISODES_PER_UPDATE,
            "update_epochs": UPDATE_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "max_updates": MAX_UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "eval_episodes": EVAL_EPISODES,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_eps": CLIP_EPS,
            "value_loss_coef": VALUE_LOSS_COEF,
            "entropy_coef": ENTROPY_COEF,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "target_kl": TARGET_KL,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "critic_hidden_dim": CRITIC_HIDDEN_DIM,
            "fall_height_threshold": FALL_HEIGHT_THRESHOLD,
            "fall_upright_threshold": FALL_UPRIGHT_THRESHOLD,
            "fall_grace_steps": FALL_GRACE_STEPS,
            "reward_mode": BALANCE_REWARD_MODE,
            "balance_invalid_score": BALANCE_INVALID_SCORE,
            "balance_front_penalty_coef": BALANCE_FRONT_PENALTY_COEF,
            "balance_back_penalty_coef": BALANCE_BACK_PENALTY_COEF,
            "balance_center_offset_penalty_coef": BALANCE_CENTER_OFFSET_PENALTY_COEF,
            "balance_support_axis_velocity_coef": BALANCE_SUPPORT_AXIS_VELOCITY_COEF,
            "balance_support_lateral_velocity_coef": BALANCE_SUPPORT_LATERAL_VELOCITY_COEF,
            "balance_velocity_clip": BALANCE_VELOCITY_CLIP,
            "balance_score_clip_min": BALANCE_SCORE_CLIP_MIN,
            "balance_score_clip_max": BALANCE_SCORE_CLIP_MAX,
            "advantage_mode": "gae",
            "rollout_workers": ROLLOUT_WORKERS,
            "eval_workers": EVAL_WORKERS,
            "seed": SEED,
            "resume_from": str(self.resume_from) if self.resume_from is not None else None,
        }
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def _save_checkpoint(self, path: Path) -> None:
        payload = {
            "algorithm": "ppo",
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": ACTOR_HIDDEN_DIM,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "critic_hidden_dim": CRITIC_HIDDEN_DIM,
            "state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
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
        critic_state_dict = payload.get("critic_state_dict")
        if critic_state_dict is None:
            raise ValueError(f"Checkpoint missing critic_state_dict: {path}")
        self.critic.load_state_dict(critic_state_dict)
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
            "train_mean_reward",
            "train_mean_survival_steps",
            "train_mean_survival_seconds",
            "train_success_rate",
            "train_mean_balance_score",
            "policy_loss",
            "value_loss",
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
                "eval_mean_reward",
                "eval_mean_survival_steps",
                "eval_mean_survival_seconds",
                "eval_success_rate",
                "eval_mean_balance_score",
            ])
        message = " | ".join(f"{key}={record[key]:.4f}" if isinstance(record[key], float) else f"{key}={record[key]}" for key in keys)
        print(message, flush=True)


def _compute_gae(rewards: np.ndarray, values: np.ndarray, bootstrap_value: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = float(bootstrap_value)
    for step_index in range(len(rewards) - 1, -1, -1):
        delta = rewards[step_index] + GAMMA * next_value - values[step_index]
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages[step_index] = gae
        next_value = float(values[step_index])
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


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
            "robot_a_balance": Humanoid21BalanceAnalysisObserver("robot_a"),
            "robot_b_balance": Humanoid21BalanceAnalysisObserver("robot_b"),
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
    critic: Critic,
    device: torch.device,
    deterministic: bool,
    seed: int,
) -> Dict[str, Any]:
    rollout_setup = _sample_rollout_setup(seed)
    controlled_agent = str(rollout_setup["controlled_agent"])
    opponent_agent = str(rollout_setup["opponent_agent"])
    initial_distance = float(rollout_setup["initial_distance"])
    _set_episode_seed(runtime, seed)
    runtime.reset(seed=seed, options={"initial_distance": initial_distance})
    obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[float] = []
    values: List[float] = []
    rewards: List[float] = []
    balance_scores: List[float] = []
    bootstrap_value = 0.0
    previous_balance_terms = _compute_balance_reward_terms(
        runtime.get_observer_output(f"{controlled_agent}_balance")
    )
    for _ in range(MAX_STEPS):
        opponent_obs = np.asarray(runtime.get_observer_output(f"{opponent_agent}_obs"), dtype=np.float32)
        controlled_action, log_prob, value = _act_with_value(actor, critic, obs, device, deterministic=deterministic)
        opponent_action, _, _ = _act_with_value(actor, critic, opponent_obs, device, deterministic=deterministic)
        if controlled_agent == "robot_a":
            runtime.step(controlled_action, opponent_action)
        else:
            runtime.step(opponent_action, controlled_action)
        current_balance_terms = _compute_balance_reward_terms(
            runtime.get_observer_output(f"{controlled_agent}_balance")
        )
        observations.append(obs.copy())
        actions.append(controlled_action.copy())
        values.append(value)
        rewards.append(_compute_balance_step_reward(current_balance_terms, previous_balance_terms))
        balance_scores.append(float(current_balance_terms["absolute_score"]))
        previous_balance_terms = current_balance_terms
        if log_prob is not None:
            log_probs.append(log_prob)
        obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
        terminated, truncated = runtime.get_termination_flags()
        if terminated or truncated:
            bootstrap_value = 0.0
            break
        _, _, bootstrap_value = _act_with_value(actor, critic, obs, device, deterministic=True)
    survival_steps = int(len(observations))
    survival_seconds = float(survival_steps / CONTROL_FREQUENCY)
    success = int(survival_steps >= MAX_STEPS)
    observations_array = np.asarray(observations, dtype=np.float32).reshape(survival_steps, OBS_DIM)
    actions_array = np.asarray(actions, dtype=np.float32).reshape(survival_steps, ACTION_DIM)
    log_probs_array = np.asarray(log_probs, dtype=np.float32).reshape(len(log_probs),)
    values_array = np.asarray(values, dtype=np.float32).reshape(len(values),)
    rewards_array = np.asarray(rewards, dtype=np.float32).reshape(len(rewards),)
    return {
        "observations": observations_array,
        "actions": actions_array,
        "log_probs": log_probs_array,
        "values": values_array,
        "rewards": rewards_array,
        "bootstrap_value": float(bootstrap_value),
        "steps": survival_steps,
        "episode_reward": float(np.sum(rewards_array, dtype=np.float32)),
        "survival_steps": survival_steps,
        "survival_seconds": survival_seconds,
        "success": success,
        "mean_balance_score": float(np.mean(balance_scores, dtype=np.float32)) if balance_scores else float(previous_balance_terms["absolute_score"]),
        "final_balance_score": float(previous_balance_terms["absolute_score"]),
        "controlled_agent": controlled_agent,
        "initial_distance": initial_distance,
        "seed": int(seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PPOTrainer(device=device, resume_from=args.resume_from)
    trainer.train()
    print(f"run_dir={trainer.run_dir}", flush=True)


if __name__ == "__main__":
    main()
