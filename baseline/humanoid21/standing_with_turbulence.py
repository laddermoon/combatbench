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

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from envs.framework import EnvRuntime
from envs.humanoid21 import Humanoid21Observer, MujocoCombatSimulator
from envs.humanoid21.disturbance_plugins import ContinuousWindPlugin, RandomPushPlugin

from baseline.humanoid21.standing import (
    ACTION_DIM,
    CLIP_EPS,
    CONTROL_FREQUENCY,
    ENTROPY_COEF,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    FALL_GRACE_STEPS,
    FALL_HEIGHT_THRESHOLD,
    FALL_UPRIGHT_THRESHOLD,
    GRAD_CLIP_NORM,
    GROUP_SIZE,
    HIDDEN_DIM,
    LEARNING_RATE,
    LOG_STD_MAX,
    LOG_STD_MIN,
    MATCH_DURATION_SECONDS,
    MAX_STEPS,
    MAX_UPDATES,
    MINIBATCH_SIZE,
    OBS_DIM,
    UPDATE_EPOCHS,
    Actor,
    StandingRewardObserver,
    export_policy_artifacts,
    normalize_group_returns,
    set_seed,
)

INITIAL_DISTANCE = 3.0
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5
SEED = 42
RUNS_DIR = Path(__file__).resolve().parent / "runs"
ROLLOUT_WORKERS = max(1, int(os.environ.get("STANDING_TURBULENCE_ROLLOUT_WORKERS", str(min(64, max(1, (os.cpu_count() or 1) // 2))))))
EVAL_WORKERS = max(1, int(os.environ.get("STANDING_TURBULENCE_EVAL_WORKERS", str(min(ROLLOUT_WORKERS, EVAL_EPISODES)))))

WIND_DIRECTION = np.array([1.0, 0.35, 0.0], dtype=np.float32)
WIND_STRENGTH = 45.0
WIND_GUST_PROBABILITY = 0.03
WIND_GUST_MULTIPLIER = 2.0
PUSH_FORCE_MAGNITUDE = 120.0
PUSH_MIN_INTERVAL = 120
PUSH_MAX_INTERVAL = 280
PUSH_BODY_NAME = "torso"
ENABLE_RANDOM_PUSH = True
ENABLE_CONTINUOUS_WIND = True

_ROLLOUT_RUNTIME: Optional[EnvRuntime] = None
_ROLLOUT_ACTOR: Optional[Actor] = None
_CPU_DEVICE = torch.device("cpu")


def _limit_worker_threads() -> None:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _snapshot_actor_state_dict(actor: Actor) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in actor.state_dict().items()}


def _split_sequence(values: Sequence[int], parts: int) -> List[List[int]]:
    if not values:
        return []
    bounded_parts = max(1, min(parts, len(values)))
    chunk_size = (len(values) + bounded_parts - 1) // bounded_parts
    return [list(values[start:start + chunk_size]) for start in range(0, len(values), chunk_size)]


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


def _build_disturbance_plugins() -> List[Any]:
    plugins: List[Any] = []
    if ENABLE_CONTINUOUS_WIND:
        plugins.append(
            ContinuousWindPlugin(
                target_robot="robot_a",
                wind_direction=WIND_DIRECTION.copy(),
                wind_strength=WIND_STRENGTH,
                gust_probability=WIND_GUST_PROBABILITY,
                gust_multiplier=WIND_GUST_MULTIPLIER,
            )
        )
        plugins.append(
            ContinuousWindPlugin(
                target_robot="robot_b",
                wind_direction=WIND_DIRECTION.copy(),
                wind_strength=WIND_STRENGTH,
                gust_probability=WIND_GUST_PROBABILITY,
                gust_multiplier=WIND_GUST_MULTIPLIER,
            )
        )
    if ENABLE_RANDOM_PUSH:
        plugins.append(
            RandomPushPlugin(
                target_robot="robot_a",
                target_body=PUSH_BODY_NAME,
                force_magnitude=PUSH_FORCE_MAGNITUDE,
                min_interval=PUSH_MIN_INTERVAL,
                max_interval=PUSH_MAX_INTERVAL,
                random_seed=SEED,
            )
        )
        plugins.append(
            RandomPushPlugin(
                target_robot="robot_b",
                target_body=PUSH_BODY_NAME,
                force_magnitude=PUSH_FORCE_MAGNITUDE,
                min_interval=PUSH_MIN_INTERVAL,
                max_interval=PUSH_MAX_INTERVAL,
                random_seed=SEED + 1,
            )
        )
    return plugins


def build_runtime() -> EnvRuntime:
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))
    observer_plugins = {
        "robot_a_obs": Humanoid21Observer("robot_a"),
        "robot_b_obs": Humanoid21Observer("robot_b"),
        "robot_a_reward": StandingRewardObserver(
            agent_id="robot_a",
            fall_height_threshold=FALL_HEIGHT_THRESHOLD,
            fall_upright_threshold=FALL_UPRIGHT_THRESHOLD,
            fall_grace_steps=FALL_GRACE_STEPS,
        ),
        "robot_b_reward": StandingRewardObserver(
            agent_id="robot_b",
            fall_height_threshold=FALL_HEIGHT_THRESHOLD,
            fall_upright_threshold=FALL_UPRIGHT_THRESHOLD,
            fall_grace_steps=FALL_GRACE_STEPS,
        ),
    }
    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins=observer_plugins,
        plugins=_build_disturbance_plugins(),
        phy_steps_per_action=phy_steps_per_action,
        max_steps=MAX_STEPS,
    )
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.action_space = Humanoid21Observer.get_action_space()
    return runtime


def collect_episode(
    runtime: EnvRuntime,
    actor: Actor,
    device: torch.device,
    deterministic: bool,
    seed: int,
) -> Dict[str, Any]:
    rollout_setup = _sample_rollout_setup(seed)
    controlled_agent = str(rollout_setup["controlled_agent"])
    opponent_agent = str(rollout_setup["opponent_agent"])
    initial_distance = float(rollout_setup["initial_distance"])
    runtime.reset(seed=seed, options={"initial_distance": initial_distance})
    obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[float] = []
    reward_info = dict(runtime.get_observer_output(f"{controlled_agent}_reward"))
    for _ in range(MAX_STEPS):
        opponent_obs = np.asarray(runtime.get_observer_output(f"{opponent_agent}_obs"), dtype=np.float32)
        controlled_action, log_prob = actor.act_numpy(obs, device, deterministic=deterministic)
        opponent_action, _ = actor.act_numpy(opponent_obs, device, deterministic=deterministic)
        if controlled_agent == "robot_a":
            runtime.step(controlled_action, opponent_action)
        else:
            runtime.step(opponent_action, controlled_action)
        observations.append(obs.copy())
        actions.append(controlled_action.copy())
        if log_prob is not None:
            log_probs.append(log_prob)
        reward_info = dict(runtime.get_observer_output(f"{controlled_agent}_reward"))
        obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
        terminated, truncated = runtime.get_termination_flags()
        if terminated or truncated or reward_info["is_fallen"]:
            break
    episode_reward = float(len(observations))
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "steps": len(observations),
        "episode_reward": episode_reward,
        "reward_info": reward_info,
        "controlled_agent": controlled_agent,
        "initial_distance": initial_distance,
    }


def _init_rollout_worker() -> None:
    global _ROLLOUT_RUNTIME, _ROLLOUT_ACTOR
    _limit_worker_threads()
    _ROLLOUT_RUNTIME = build_runtime()
    _ROLLOUT_ACTOR = Actor(OBS_DIM, ACTION_DIM, HIDDEN_DIM).to(_CPU_DEVICE)
    _ROLLOUT_ACTOR.eval()


def _collect_episode_chunk(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    global _ROLLOUT_RUNTIME, _ROLLOUT_ACTOR
    if _ROLLOUT_RUNTIME is None or _ROLLOUT_ACTOR is None:
        _init_rollout_worker()
    _ROLLOUT_ACTOR.load_state_dict(task["state_dict"])
    _ROLLOUT_ACTOR.eval()
    return [
        collect_episode(
            _ROLLOUT_RUNTIME,
            _ROLLOUT_ACTOR,
            _CPU_DEVICE,
            deterministic=bool(task["deterministic"]),
            seed=int(seed),
        )
        for seed in task["seeds"]
    ]


class GRPOTrainer:
    def __init__(self, device: torch.device):
        self.device = device
        self.actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.best_eval_reward = -float("inf")
        self.history: List[Dict[str, Any]] = []
        self.train_runtime = build_runtime() if ROLLOUT_WORKERS == 1 else None
        self.rollout_executor = ProcessPoolExecutor(
            max_workers=ROLLOUT_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_init_rollout_worker,
        ) if ROLLOUT_WORKERS > 1 else None
        self.run_dir = self._build_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.policy_dir = self.run_dir / "policy"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    def train(self) -> None:
        try:
            for update_index in range(1, MAX_UPDATES + 1):
                seeds = [SEED + update_index * EPISODES_PER_UPDATE + episode_index for episode_index in range(EPISODES_PER_UPDATE)]
                episodes = self._collect_episodes(seeds=seeds, deterministic=False, worker_limit=ROLLOUT_WORKERS)
                update_stats = self._update_actor(episodes)
                mean_episode_reward = float(np.mean([episode["episode_reward"] for episode in episodes]))
                mean_episode_length = float(np.mean([episode["steps"] for episode in episodes]))
                fall_rate = float(np.mean([1.0 if episode["reward_info"]["is_fallen"] else 0.0 for episode in episodes]))
                record = {
                    "update": update_index,
                    "train_mean_reward": mean_episode_reward,
                    "train_mean_length": mean_episode_length,
                    "train_fall_rate": fall_rate,
                    **update_stats,
                }
                if update_index % EVAL_INTERVAL == 0:
                    eval_stats = self.evaluate_actor()
                    record.update({f"eval_{key}": value for key, value in eval_stats.items()})
                    if eval_stats["mean_reward"] > self.best_eval_reward:
                        self.best_eval_reward = float(eval_stats["mean_reward"])
                        self._save_checkpoint(self.run_dir / "best_model.pt")
                        self._export_policy(self.policy_dir, self.run_dir / "best_model.pt")
                self.history.append(record)
                self._print_record(record)
                if update_index % EVAL_INTERVAL == 0:
                    self._write_history()
                if update_index % 25 == 0:
                    self._save_checkpoint(self.checkpoint_dir / f"update_{update_index}.pt")
            final_model_path = self.run_dir / "final_model.pt"
            self._save_checkpoint(final_model_path)
            if not self.policy_dir.exists():
                self._export_policy(self.policy_dir, final_model_path)
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
                collect_episode(self.train_runtime, self.actor, self.device, deterministic=deterministic, seed=int(seed))
                for seed in seeds
            ]
        actor_state_dict = _snapshot_actor_state_dict(self.actor)
        seed_chunks = _split_sequence(list(seeds), max(1, min(worker_limit, ROLLOUT_WORKERS)))
        tasks = [
            {
                "state_dict": actor_state_dict,
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
        episodes = self._collect_episodes(seeds=seeds, deterministic=True, worker_limit=EVAL_WORKERS)
        return {
            "mean_reward": float(np.mean([episode["episode_reward"] for episode in episodes])),
            "mean_length": float(np.mean([episode["steps"] for episode in episodes])),
            "fall_rate": float(np.mean([1.0 if episode["reward_info"]["is_fallen"] else 0.0 for episode in episodes])),
        }

    def _update_actor(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        episode_rewards = np.asarray([episode["episode_reward"] for episode in episodes], dtype=np.float32)
        advantages_per_episode = normalize_group_returns(episode_rewards, GROUP_SIZE)
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
        losses: List[float] = []
        entropies: List[float] = []
        ratios: List[float] = []
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
                loss = -objective.mean() - ENTROPY_COEF * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()
                losses.append(float(loss.item()))
                entropies.append(float(entropy.mean().item()))
                ratios.append(float(ratio.mean().item()))
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "ratio": float(np.mean(ratios)) if ratios else 0.0,
        }

    def _build_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return RUNS_DIR / f"standing_turbulence_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "initial_distance": INITIAL_DISTANCE,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
            "symmetric_self_play_rollout": True,
            "wind_direction": WIND_DIRECTION.tolist(),
            "wind_strength": WIND_STRENGTH,
            "wind_gust_probability": WIND_GUST_PROBABILITY,
            "wind_gust_multiplier": WIND_GUST_MULTIPLIER,
            "push_force_magnitude": PUSH_FORCE_MAGNITUDE,
            "push_min_interval": PUSH_MIN_INTERVAL,
            "push_max_interval": PUSH_MAX_INTERVAL,
            "push_body_name": PUSH_BODY_NAME,
            "enable_random_push": ENABLE_RANDOM_PUSH,
            "enable_continuous_wind": ENABLE_CONTINUOUS_WIND,
            "group_size": GROUP_SIZE,
            "update_epochs": UPDATE_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "max_updates": MAX_UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "eval_episodes": EVAL_EPISODES,
            "learning_rate": LEARNING_RATE,
            "clip_eps": CLIP_EPS,
            "entropy_coef": ENTROPY_COEF,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "hidden_dim": HIDDEN_DIM,
            "log_std_min": LOG_STD_MIN,
            "log_std_max": LOG_STD_MAX,
            "rollout_workers": ROLLOUT_WORKERS,
            "eval_workers": EVAL_WORKERS,
            "seed": SEED,
        }
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def _save_checkpoint(self, path: Path) -> None:
        payload = {
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": HIDDEN_DIM,
            "state_dict": self.actor.state_dict(),
        }
        torch.save(payload, path)

    def _export_policy(self, policy_dir: Path, model_path: Path) -> None:
        export_policy_artifacts(model_path, policy_dir)

    def _write_history(self) -> None:
        with (self.run_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    def _print_record(self, record: Dict[str, Any]) -> None:
        keys = [
            "update",
            "train_mean_reward",
            "train_mean_length",
            "train_fall_rate",
            "loss",
            "entropy",
            "ratio",
        ]
        if "eval_mean_reward" in record:
            keys.extend(["eval_mean_reward", "eval_mean_length", "eval_fall_rate"])
        message = " | ".join(
            f"{key}={record[key]:.4f}" if isinstance(record[key], float) else f"{key}={record[key]}"
            for key in keys
        )
        print(message, flush=True)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = GRPOTrainer(device=device)
    trainer.train()
    print(f"run_dir={trainer.run_dir}", flush=True)
    print(f"policy_dir={trainer.policy_dir}", flush=True)


if __name__ == "__main__":
    main()
