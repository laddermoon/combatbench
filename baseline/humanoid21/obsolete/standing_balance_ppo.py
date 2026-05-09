"""PPO training for standing with balance-value reward.

All rollout infrastructure is reused from baseline.humanoid21.base:
  * Actor / Critic
  * RolloutCollector (owns the worker pool, actor state sync, and episode loop)
  * snapshot_module_state_dict helper

Episode termination uses BalanceScoreTerminationPlugin: the episode ends when
the absolute balance score stays below a threshold for a number of consecutive
steps. Height/uprightness-based termination is deliberately avoided because it
can conflict with the balance objective (a slightly lower stance can still be
perfectly balanced).

Reward comes from BalanceValueRewarder (rewards.py) attached as an observer
plugin for each agent. Episodes returned from RolloutCollector already contain
the per-step absolute balance scores under key "rewards"; the trainer
post-processes them on the main process to compute:
  * per-step critic values (one batched forward pass)
  * bootstrap value from final_obs for truncated episodes
  * delta reward if requested via STANDING_BALANCE_REWARD_MODE

Initial-state perturbation settings are copied verbatim from
standing_turbulence_dense_reward_ppo.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from baseline.humanoid21.base import (
    Actor,
    Critic,
    RolloutCollector,
)
from baseline.humanoid21.rewards import BalanceScoreTerminationPlugin, BalanceValueRewarder
from envs.framework import EnvRuntime
from envs.humanoid21 import Humanoid21Observer, MujocoCombatSimulator
from envs.humanoid21.disturbance_plugins import InitialStatePerturbationPlugin


# ---------------------------------------------------------------------------
# Environment / episode config
# ---------------------------------------------------------------------------
CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = float(os.environ.get("STANDING_MATCH_DURATION_SECONDS", "3.0"))
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5

ACTION_DIM = Humanoid21Observer.ACTION_DIM
OBS_DIM = Humanoid21Observer.OBS_DIM

# Balance-score-based termination: end the episode when the absolute balance
# score reported by BalanceValueRewarder stays below BALANCE_TERMINATION_SCORE_THRESHOLD
# for BALANCE_TERMINATION_GRACE_STEPS consecutive steps.
BALANCE_TERMINATION_SCORE_THRESHOLD = float(os.environ.get("STANDING_BALANCE_TERMINATION_SCORE_THRESHOLD", "0.3"))
BALANCE_TERMINATION_GRACE_STEPS = int(os.environ.get("STANDING_BALANCE_TERMINATION_GRACE_STEPS", "3"))

# Disturbance settings (verbatim from standing_turbulence_dense_reward_ppo.py)
PERTURBATION_JOINT_POS_DELTA_MAX = 0.05
PERTURBATION_JOINT_VEL_DELTA_MAX = 0.05
PERTURBATION_ROOT_XY_OFFSET_MAX = 0.05
PERTURBATION_ROOT_TILT_DEG_MAX = 10.0
PERTURBATION_ROOT_LINEAR_VELOCITY_DELTA_MAX = [0.5, 0.5, 0.0]
PERTURBATION_ROOT_ANGULAR_VELOCITY_DELTA_MAX = [0.5, 0.5, 0.2]

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------
EPISODES_PER_UPDATE = int(os.environ.get("STANDING_EPISODES_PER_UPDATE", str(256 * 32)))
UPDATE_EPOCHS = int(os.environ.get("STANDING_UPDATE_EPOCHS", "4"))
MINIBATCH_SIZE = int(os.environ.get("STANDING_MINIBATCH_SIZE", str(4096 * 32)))
MAX_UPDATES = int(os.environ.get("STANDING_MAX_UPDATES", "10000"))
EVAL_INTERVAL = int(os.environ.get("STANDING_EVAL_INTERVAL", "5"))
EVAL_EPISODES = int(os.environ.get("STANDING_EVAL_EPISODES", "16"))

LEARNING_RATE = float(os.environ.get("STANDING_LEARNING_RATE", "3e-4"))
GAMMA = float(os.environ.get("STANDING_GAMMA", "0.99"))
GAE_LAMBDA = float(os.environ.get("STANDING_GAE_LAMBDA", "0.95"))
CLIP_EPS = float(os.environ.get("STANDING_CLIP_EPS", "0.2"))
VALUE_LOSS_COEF = float(os.environ.get("STANDING_VALUE_LOSS_COEF", "0.5"))
ENTROPY_COEF = float(os.environ.get("STANDING_ENTROPY_COEF", "1e-3"))
GRAD_CLIP_NORM = float(os.environ.get("STANDING_GRAD_CLIP_NORM", "1.0"))
TARGET_KL = float(os.environ.get("STANDING_TARGET_KL", "0.05"))

ACTOR_HIDDEN_DIM = int(os.environ.get("STANDING_ACTOR_HIDDEN_DIM", "256"))
CRITIC_HIDDEN_DIM = int(os.environ.get("STANDING_CRITIC_HIDDEN_DIM", "256"))

SEED = int(os.environ.get("STANDING_SEED", "42"))
RUNS_DIR = Path(__file__).resolve().parent / "runs"
ROLLOUT_WORKERS = max(1, int(os.environ.get(
    "STANDING_ROLLOUT_WORKERS",
    str(min(64, max(1, (os.cpu_count() or 1) // 2))),
)))
EVAL_WORKERS = max(1, int(os.environ.get(
    "STANDING_EVAL_WORKERS",
    str(min(ROLLOUT_WORKERS, EVAL_EPISODES)),
)))

BALANCE_REWARD_MODE = os.environ.get("STANDING_BALANCE_REWARD_MODE", "absolute").strip().lower()
if BALANCE_REWARD_MODE not in {"absolute", "delta"}:
    raise ValueError(f"Unsupported STANDING_BALANCE_REWARD_MODE: {BALANCE_REWARD_MODE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_runtime(seed: int) -> EnvRuntime:
    """Build a fresh runtime per seed.

    The RNG used here is decorrelated from the RNG inside
    ``baseline.humanoid21.base._collect_actor_episode`` (which picks the
    controlled agent), so changing the controlled-agent coin flip does not
    perturb the initial-distance sampling.
    """
    distance_rng = np.random.default_rng(int(seed) ^ 0xA1B2C3D4)
    initial_distance = float(distance_rng.uniform(ROLLOUT_INITIAL_DISTANCE_MIN, ROLLOUT_INITIAL_DISTANCE_MAX))
    simulator = MujocoCombatSimulator(initial_distance=initial_distance)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))

    perturbations: Dict[str, InitialStatePerturbationPlugin] = {}
    for robot in ("robot_a", "robot_b"):
        perturbations[robot] = InitialStatePerturbationPlugin(
            target_robot=robot,
            joint_pos_delta_max=PERTURBATION_JOINT_POS_DELTA_MAX,
            joint_vel_delta_max=PERTURBATION_JOINT_VEL_DELTA_MAX,
            root_xy_offset_max=PERTURBATION_ROOT_XY_OFFSET_MAX,
            root_tilt_deg_max=PERTURBATION_ROOT_TILT_DEG_MAX,
            root_linear_velocity_delta_max=PERTURBATION_ROOT_LINEAR_VELOCITY_DELTA_MAX,
            root_angular_velocity_delta_max=PERTURBATION_ROOT_ANGULAR_VELOCITY_DELTA_MAX,
            random_seed=None,
        )

    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins={
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
            "robot_a_reward": BalanceValueRewarder(agent_id="robot_a"),
            "robot_b_reward": BalanceValueRewarder(agent_id="robot_b"),
        },
        plugins=[
            BalanceScoreTerminationPlugin(
                agent_id="robot_a",
                score_threshold=BALANCE_TERMINATION_SCORE_THRESHOLD,
                grace_steps=BALANCE_TERMINATION_GRACE_STEPS,
            ),
            BalanceScoreTerminationPlugin(
                agent_id="robot_b",
                score_threshold=BALANCE_TERMINATION_SCORE_THRESHOLD,
                grace_steps=BALANCE_TERMINATION_GRACE_STEPS,
            ),
            perturbations["robot_a"],
            perturbations["robot_b"],
        ],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=MAX_STEPS,
    )
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.action_space = Humanoid21Observer.get_action_space()
    runtime.initial_state_perturbation_plugins = perturbations
    return runtime


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
) -> tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
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
        self.collector = RolloutCollector(
            runtime_builder=build_runtime,
            actor=self.actor,
            max_workers=ROLLOUT_WORKERS,
            worker_device="cpu",
        )
        self.run_dir = self._build_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from.resolve() if resume_from is not None else None
        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)
        self._save_config()

    def close(self) -> None:
        self.collector.close()

    # ----- rollout / post-processing -----
    def _collect_episodes(
        self,
        seeds: Sequence[int],
        deterministic: bool,
        worker_limit: int,
    ) -> List[Dict[str, Any]]:
        episodes = self.collector.collect_episodes(
            seeds=seeds,
            worker_limit=worker_limit,
            deterministic=deterministic,
        )
        return self._postprocess_episodes(episodes)

    def _postprocess_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach critic values, bootstrap values, delta rewards, and summary stats."""
        valid = [ep for ep in episodes if int(ep["steps"]) > 0]
        if not valid:
            return []

        # Batched critic forward pass over all observations.
        steps_per_ep = [int(ep["observations"].shape[0]) for ep in valid]
        offsets = np.cumsum([0] + steps_per_ep).tolist()
        all_obs = np.concatenate([ep["observations"] for ep in valid], axis=0)
        obs_tensor = torch.as_tensor(all_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            all_values = self.critic(obs_tensor).detach().cpu().numpy().astype(np.float32)

        # Bootstrap values: only for truncated episodes where we captured final_obs.
        bootstrap_indices: List[int] = []
        bootstrap_obs_list: List[np.ndarray] = []
        for index, ep in enumerate(valid):
            if bool(ep.get("truncated", False)) and ep.get("final_obs") is not None:
                bootstrap_indices.append(index)
                bootstrap_obs_list.append(np.asarray(ep["final_obs"], dtype=np.float32))
        bootstrap_values: Dict[int, float] = {}
        if bootstrap_obs_list:
            boot_tensor = torch.as_tensor(np.stack(bootstrap_obs_list), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                boot_vals = self.critic(boot_tensor).detach().cpu().numpy().astype(np.float32)
            for index, value in zip(bootstrap_indices, boot_vals):
                bootstrap_values[index] = float(value)

        for index, ep in enumerate(valid):
            start, end = offsets[index], offsets[index + 1]
            ep["values"] = all_values[start:end]
            ep["bootstrap_value"] = bootstrap_values.get(index, 0.0)

            absolute_rewards = np.asarray(ep["rewards"], dtype=np.float32)
            ep["balance_scores"] = absolute_rewards.copy()
            if BALANCE_REWARD_MODE == "delta":
                delta_rewards = np.zeros_like(absolute_rewards)
                if absolute_rewards.size > 1:
                    delta_rewards[1:] = absolute_rewards[1:] - absolute_rewards[:-1]
                ep["rewards"] = delta_rewards
            else:
                ep["rewards"] = absolute_rewards.astype(np.float32)

            ep["survival_steps"] = int(ep["steps"])
            ep["survival_seconds"] = float(ep["steps"] / CONTROL_FREQUENCY)
            ep["success"] = int(ep["steps"] >= MAX_STEPS)
            ep["episode_reward"] = float(np.sum(ep["rewards"], dtype=np.float32))
            ep["mean_balance_score"] = float(np.mean(absolute_rewards)) if absolute_rewards.size else 0.0
            ep["final_balance_score"] = float(absolute_rewards[-1]) if absolute_rewards.size else 0.0
        return valid

    # ----- PPO update -----
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
        obs_batch = np.concatenate([ep["observations"] for ep in episodes], axis=0)
        action_batch = np.concatenate([ep["actions"] for ep in episodes], axis=0)
        old_log_prob_batch = np.concatenate([ep["log_probs"] for ep in episodes], axis=0)
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
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    GRAD_CLIP_NORM,
                )
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

    # ----- driver -----
    def train(self) -> None:
        try:
            for update_index in range(1, MAX_UPDATES + 1):
                seeds = [
                    SEED + update_index * EPISODES_PER_UPDATE + episode_index
                    for episode_index in range(EPISODES_PER_UPDATE)
                ]
                episodes = self._collect_episodes(seeds=seeds, deterministic=False, worker_limit=ROLLOUT_WORKERS)
                if not episodes:
                    print(f"update={update_index} | no valid episodes collected", flush=True)
                    continue
                update_stats = self._update_policy(episodes)
                record = {
                    "update": update_index,
                    "train_mean_reward": float(np.mean([ep["episode_reward"] for ep in episodes])),
                    "train_mean_survival_steps": float(np.mean([ep["survival_steps"] for ep in episodes])),
                    "train_mean_survival_seconds": float(np.mean([ep["survival_seconds"] for ep in episodes])),
                    "train_success_rate": float(np.mean([ep["success"] for ep in episodes])),
                    "train_mean_balance_score": float(np.mean([ep["mean_balance_score"] for ep in episodes])),
                    **update_stats,
                }
                if update_index % EVAL_INTERVAL == 0:
                    eval_stats = self.evaluate_actor()
                    record.update({f"eval_{k}": v for k, v in eval_stats.items()})
                    eval_score = (float(eval_stats["success_rate"]), float(eval_stats["mean_survival_steps"]))
                    if eval_score > self.best_eval_score:
                        self.best_eval_score = eval_score
                        self._save_checkpoint(self.run_dir / "best_model.pt")
                self.history.append(record)
                self._print_record(record)
                if update_index % EVAL_INTERVAL == 0:
                    self._write_history()
                if update_index % 25 == 0:
                    self._save_checkpoint(self.checkpoint_dir / f"update_{update_index}.pt")
            self._save_checkpoint(self.run_dir / "final_model.pt")
            self._write_history()
        finally:
            self.close()

    def evaluate_actor(self) -> Dict[str, float]:
        seeds = [SEED + 100000 + episode_index for episode_index in range(EVAL_EPISODES)]
        episodes = self._collect_episodes(seeds=seeds, deterministic=True, worker_limit=EVAL_WORKERS)
        if not episodes:
            return {
                "mean_reward": 0.0,
                "mean_survival_steps": 0.0,
                "mean_survival_seconds": 0.0,
                "success_rate": 0.0,
                "mean_balance_score": 0.0,
            }
        return {
            "mean_reward": float(np.mean([ep["episode_reward"] for ep in episodes])),
            "mean_survival_steps": float(np.mean([ep["survival_steps"] for ep in episodes])),
            "mean_survival_seconds": float(np.mean([ep["survival_seconds"] for ep in episodes])),
            "success_rate": float(np.mean([ep["success"] for ep in episodes])),
            "mean_balance_score": float(np.mean([ep["mean_balance_score"] for ep in episodes])),
        }

    # ----- checkpoint / config / logging -----
    def _build_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return RUNS_DIR / f"standing_balance_ppo_{BALANCE_REWARD_MODE}_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "algorithm": "ppo",
            "reward": "BalanceValueRewarder",
            "reward_mode": BALANCE_REWARD_MODE,
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
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
            "balance_termination_score_threshold": BALANCE_TERMINATION_SCORE_THRESHOLD,
            "balance_termination_grace_steps": BALANCE_TERMINATION_GRACE_STEPS,
            "perturbation": {
                "joint_pos_delta_max": PERTURBATION_JOINT_POS_DELTA_MAX,
                "joint_vel_delta_max": PERTURBATION_JOINT_VEL_DELTA_MAX,
                "root_xy_offset_max": PERTURBATION_ROOT_XY_OFFSET_MAX,
                "root_tilt_deg_max": PERTURBATION_ROOT_TILT_DEG_MAX,
                "root_linear_velocity_delta_max": PERTURBATION_ROOT_LINEAR_VELOCITY_DELTA_MAX,
                "root_angular_velocity_delta_max": PERTURBATION_ROOT_ANGULAR_VELOCITY_DELTA_MAX,
            },
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
            keys.extend(["early_stop", "early_stop_kl"])
        if "eval_mean_survival_steps" in record:
            keys.extend([
                "eval_mean_reward",
                "eval_mean_survival_steps",
                "eval_mean_survival_seconds",
                "eval_success_rate",
                "eval_mean_balance_score",
            ])
        message = " | ".join(
            f"{key}={record[key]:.4f}" if isinstance(record[key], float) else f"{key}={record[key]}"
            for key in keys
        )
        print(message, flush=True)


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
