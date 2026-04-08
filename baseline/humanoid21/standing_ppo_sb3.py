from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from torch import nn

COMBATBENCH_DIR = Path(__file__).resolve().parents[2]
if str(COMBATBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_DIR))

from envs.framework import BaseObserverPlugin, EnvRuntime, ReadOnlySimContext
from envs.humanoid21 import Humanoid21Observer, MujocoCombatSimulator

CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 5.0
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)
INITIAL_DISTANCE = 3.0
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5
ACTION_DIM = Humanoid21Observer.ACTION_DIM
OBS_DIM = Humanoid21Observer.OBS_DIM
EPISODES_PER_UPDATE = 256
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 4096
MAX_UPDATES = 10000
EVAL_INTERVAL = 5
EVAL_EPISODES = 16
NUM_ENVS = max(1, int(os.environ.get("STANDING_SB3_ENVS", str(min(8, max(1, (os.cpu_count() or 1) // 2))))))
VEC_ENV_START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 1e-3
GRAD_CLIP_NORM = 1.0
ACTOR_HIDDEN_DIM = 256
CRITIC_HIDDEN_DIM = 256
TARGET_HEIGHT = 1.28
HEIGHT_REWARD_WEIGHT = 8.0
UPRIGHTNESS_REWARD_WEIGHT = 2.0
FOOT_STABILITY_WEIGHT = 1.5
JOINT_VEL_ENERGY_WEIGHT = 0.02
FALL_HEIGHT_THRESHOLD = 1.10
FALL_UPRIGHT_THRESHOLD = 0.8
FALL_GRACE_STEPS = 3
SEED = 42
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandingRewardObserver(BaseObserverPlugin):
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
        self._output: Dict[str, Any] = {}
        self._step_count = 0
        self._fall_streak = 0
        self._fallen = False
        self._reference_feet_positions: Optional[np.ndarray] = None
        self._last_reward_terms: Dict[str, float] = self._zero_reward_terms()

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._step_count = 0
        self._fall_streak = 0
        self._fallen = False
        self._reference_feet_positions = self._get_feet_world_positions(ctx)
        self._last_reward_terms = self._zero_reward_terms()
        self._output = self._build_output(ctx, is_standing=True, reward_terms=self._last_reward_terms)

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])
        is_standing = bool(
            height >= self.fall_height_threshold and uprightness >= self.fall_upright_threshold
        )
        self._step_count += 1
        if is_standing:
            self._fall_streak = 0
        else:
            self._fall_streak += 1
        if self._fall_streak >= self.fall_grace_steps:
            self._fallen = True
        reward_terms = self._compute_reward_terms(ctx, height=height, uprightness=uprightness)
        self._last_reward_terms = reward_terms
        self._output = self._build_output(ctx, is_standing=is_standing, reward_terms=reward_terms)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_output(ctx, reward_terms=self._last_reward_terms)

    def get_output(self) -> Any:
        return self._output

    def _zero_reward_terms(self) -> Dict[str, float]:
        return {
            "height_reward": 0.0,
            "uprightness_reward": 0.0,
            "foot_stability_reward": 0.0,
            "energy_reward": 0.0,
            "total_reward": 0.0,
            "foot_position_penalty": 0.0,
            "joint_velocity_energy": 0.0,
        }

    def _get_feet_world_positions(self, ctx: ReadOnlySimContext) -> np.ndarray:
        accessor = ctx.accessor
        cache = accessor._robot_cache[self.agent_id]
        foot_right_pos = accessor.data.xpos[cache["foot_right_body_id"]].copy()
        foot_left_pos = accessor.data.xpos[cache["foot_left_body_id"]].copy()
        return np.stack([foot_right_pos, foot_left_pos], axis=0).astype(np.float32)

    def _compute_reward_terms(
        self,
        ctx: ReadOnlySimContext,
        height: float,
        uprightness: float,
    ) -> Dict[str, float]:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        action = np.asarray(ctx.accessor.get_action()[self.agent_id], dtype=np.float32)
        current_feet_positions = self._get_feet_world_positions(ctx)
        reference_feet_positions = current_feet_positions if self._reference_feet_positions is None else self._reference_feet_positions
        feet_displacement = current_feet_positions - reference_feet_positions
        foot_position_penalty = float(np.mean(np.sum(feet_displacement ** 2, axis=-1)))
        joint_velocity_energy = float(np.mean(np.square(core_state["joint_vel_norm"])))
        reward_terms = {
            "height_reward": -HEIGHT_REWARD_WEIGHT * float((height - TARGET_HEIGHT) ** 2),
            "uprightness_reward": UPRIGHTNESS_REWARD_WEIGHT * uprightness,
            "foot_stability_reward": -FOOT_STABILITY_WEIGHT * foot_position_penalty,
            "energy_reward": - JOINT_VEL_ENERGY_WEIGHT * joint_velocity_energy,
        }
        total_reward = float(sum(reward_terms.values()))
        reward_terms["total_reward"] = total_reward
        reward_terms["foot_position_penalty"] = foot_position_penalty
        reward_terms["joint_velocity_energy"] = joint_velocity_energy
        return reward_terms

    def _build_output(
        self,
        ctx: ReadOnlySimContext,
        is_standing: Optional[bool] = None,
        reward_terms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])
        if is_standing is None:
            is_standing = bool(
                height >= self.fall_height_threshold and uprightness >= self.fall_upright_threshold
            )
        if reward_terms is None:
            reward_terms = self._zero_reward_terms()
        return {
            "steps": int(self._step_count),
            "height": height,
            "uprightness": uprightness,
            "is_standing": bool(is_standing),
            "is_fallen": bool(self._fallen),
            "is_terminated": bool(ctx.is_terminated),
            **reward_terms,
        }


def export_policy_artifacts(model_path: Path, policy_dir: Path) -> None:
    policy_dir.mkdir(parents=True, exist_ok=True)
    copy2(model_path, policy_dir / "model.zip")
    policy_code = """import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3 import PPO

for parent in Path(__file__).resolve().parents:
    if (parent / "policy" / "base.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
    if (parent / "combatbench" / "policy" / "base.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

try:
    from policy.base import BaseCombatPolicy
except ImportError:
    from combatbench.policy.base import BaseCombatPolicy


class StandingCombatPolicy(BaseCombatPolicy):
    def __init__(self, model_path: Optional[str] = None, observation_space: Any = None, action_space: Any = None, **kwargs: Any):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.zip"
        self.model = PPO.load(str(payload_path), device="cpu")

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        action, _ = self.model.predict(obs_array, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def reset(self) -> None:
        return None
"""
    with (policy_dir / "policy.py").open("w", encoding="utf-8") as handle:
        handle.write(policy_code)


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
        plugins=[],
        phy_steps_per_action=phy_steps_per_action,
        max_steps=MAX_STEPS,
    )
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.action_space = Humanoid21Observer.get_action_space()
    return runtime


_REWARD_TERM_KEYS = [
    "height_reward",
    "uprightness_reward",
    "foot_stability_reward",
    "energy_reward",
    "total_reward",
    "foot_position_penalty",
    "joint_velocity_energy",
]


def _mean_reward_terms(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    all_keys = sorted({key for episode in episodes for key in episode.get("mean_reward_terms", {}).keys()})
    return {
        key: float(np.mean([float(episode.get("mean_reward_terms", {}).get(key, 0.0)) for episode in episodes]))
        for key in all_keys
    }


def _extract_reward_terms(reward_info: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(reward_info[key]) for key in _REWARD_TERM_KEYS}


def _summarize_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    if not episodes:
        return {
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "fall_rate": 0.0,
            "mean_height": 0.0,
            "mean_uprightness": 0.0,
            **{key: 0.0 for key in _REWARD_TERM_KEYS},
        }
    reward_terms = _mean_reward_terms(episodes)
    return {
        "mean_reward": float(np.mean([episode["episode_reward"] for episode in episodes])),
        "mean_length": float(np.mean([episode["steps"] for episode in episodes])),
        "fall_rate": float(np.mean([1.0 if episode["reward_info"]["is_fallen"] else 0.0 for episode in episodes])),
        "mean_height": float(np.mean([episode["mean_height"] for episode in episodes])),
        "mean_uprightness": float(np.mean([episode["mean_uprightness"] for episode in episodes])),
        **reward_terms,
    }


class StandingSelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_rank: int = 0, deterministic_opponent: bool = False):
        super().__init__()
        self.runtime = build_runtime()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.deterministic_opponent = deterministic_opponent
        self.policy_model: Optional[PPO] = None
        self.opponent_model: Optional[PPO] = None
        self._seed_offset = env_rank * 1000000
        self._rng = np.random.default_rng(SEED + self._seed_offset)
        self._controlled_agent = "robot_a"
        self._opponent_agent = "robot_b"
        self._initial_distance = INITIAL_DISTANCE
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._heights: List[float] = []
        self._uprightnesses: List[float] = []
        self._reward_terms_history: List[Dict[str, float]] = []

    def set_policy_model(self, model: PPO) -> None:
        self.policy_model = model

    def load_opponent_model(self, model_path: str) -> None:
        self.opponent_model = PPO.load(model_path, device="cpu")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed + self._seed_offset)
        self._controlled_agent = "robot_a" if int(self._rng.integers(0, 2)) == 0 else "robot_b"
        self._opponent_agent = "robot_b" if self._controlled_agent == "robot_a" else "robot_a"
        self._initial_distance = float(self._rng.uniform(ROLLOUT_INITIAL_DISTANCE_MIN, ROLLOUT_INITIAL_DISTANCE_MAX))
        self.runtime.reset(seed=seed, options={"initial_distance": self._initial_distance})
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._heights = []
        self._uprightnesses = []
        self._reward_terms_history = []
        obs = np.asarray(self.runtime.get_observer_output(f"{self._controlled_agent}_obs"), dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        controlled_action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        opponent_obs = np.asarray(self.runtime.get_observer_output(f"{self._opponent_agent}_obs"), dtype=np.float32)
        opponent_action = self._predict_action(opponent_obs)
        if self._controlled_agent == "robot_a":
            self.runtime.step(controlled_action, opponent_action)
        else:
            self.runtime.step(opponent_action, controlled_action)
        reward_info = dict(self.runtime.get_observer_output(f"{self._controlled_agent}_reward"))
        reward = float(reward_info["total_reward"])
        terminated, truncated = self.runtime.get_termination_flags()
        terminated = bool(terminated or reward_info["is_fallen"])
        next_obs = np.asarray(self.runtime.get_observer_output(f"{self._controlled_agent}_obs"), dtype=np.float32)
        self._episode_reward += reward
        self._episode_steps += 1
        self._heights.append(float(reward_info["height"]))
        self._uprightnesses.append(float(reward_info["uprightness"]))
        self._reward_terms_history.append(_extract_reward_terms(reward_info))
        info: Dict[str, Any] = {}
        if terminated or truncated:
            info["episode_metrics"] = self._build_episode(reward_info)
        return next_obs, reward, terminated, bool(truncated), info

    def close(self) -> None:
        self.runtime.close()

    def _predict_action(self, obs: np.ndarray) -> np.ndarray:
        model = self.policy_model if self.policy_model is not None else self.opponent_model
        if model is None:
            raise RuntimeError("Opponent policy is not initialized")
        action, _ = model.predict(obs, deterministic=self.deterministic_opponent)
        return np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)

    def _build_episode(self, reward_info: Dict[str, Any]) -> Dict[str, Any]:
        mean_reward_terms = {
            key: float(np.mean([float(reward_terms.get(key, 0.0)) for reward_terms in self._reward_terms_history]))
            for key in sorted({key for reward_terms in self._reward_terms_history for key in reward_terms.keys()})
        } if self._reward_terms_history else {}
        return {
            "steps": self._episode_steps,
            "episode_reward": float(self._episode_reward),
            "mean_height": float(np.mean(self._heights)) if self._heights else float(reward_info["height"]),
            "mean_uprightness": float(np.mean(self._uprightnesses)) if self._uprightnesses else float(reward_info["uprightness"]),
            "mean_reward_terms": mean_reward_terms,
            "reward_info": dict(reward_info),
        }


def make_env(env_rank: int) -> Any:
    def _factory() -> StandingSelfPlayEnv:
        return StandingSelfPlayEnv(env_rank=env_rank, deterministic_opponent=False)
    return _factory


class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.episodes: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode_metrics = info.get("episode_metrics")
            if episode_metrics is not None:
                self.episodes.append(dict(episode_metrics))
        return True


def evaluate_actor(env: StandingSelfPlayEnv, model: PPO) -> Dict[str, float]:
    episodes: List[Dict[str, Any]] = []
    for episode_index in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=SEED + 100000 + episode_index)
        done = False
        info: Dict[str, Any] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
        episodes.append(dict(info["episode_metrics"]))
    return _summarize_episodes(episodes)


class StandingSB3Trainer:
    def __init__(self, device: torch.device):
        self.device = device
        self.run_dir = self._build_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.policy_dir = self.run_dir / "policy"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.best_eval_reward = -float("inf")
        self.num_envs = NUM_ENVS
        self.rollout_steps = EPISODES_PER_UPDATE * MAX_STEPS
        self.rollout_steps_per_env = max(1, self.rollout_steps // self.num_envs)
        self.rollout_steps = self.rollout_steps_per_env * self.num_envs
        self.train_env = SubprocVecEnv([make_env(env_rank) for env_rank in range(self.num_envs)], start_method=VEC_ENV_START_METHOD)
        self.eval_env = StandingSelfPlayEnv(deterministic_opponent=True)
        self.model = PPO(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=LEARNING_RATE,
            n_steps=self.rollout_steps_per_env,
            batch_size=min(MINIBATCH_SIZE, self.rollout_steps),
            n_epochs=UPDATE_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_EPS,
            ent_coef=ENTROPY_COEF,
            vf_coef=VALUE_LOSS_COEF,
            max_grad_norm=GRAD_CLIP_NORM,
            policy_kwargs={
                "activation_fn": nn.Tanh,
                "net_arch": {"pi": [ACTOR_HIDDEN_DIM, ACTOR_HIDDEN_DIM], "vf": [CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM]},
            },
            verbose=0,
            seed=SEED,
            device=str(device),
        )
        self.opponent_model_path = self.run_dir / "latest_opponent.zip"
        self.eval_env.set_policy_model(self.model)
        self._sync_opponent_policy()
        self._save_config()

    def train(self) -> None:
        try:
            for update_index in range(1, MAX_UPDATES + 1):
                callback = EpisodeStatsCallback()
                self.model.learn(total_timesteps=self.rollout_steps, reset_num_timesteps=False, progress_bar=False, callback=callback)
                self._sync_opponent_policy()
                train_stats = _summarize_episodes(callback.episodes)
                logger_values = dict(getattr(self.model.logger, "name_to_value", {}))
                record = {
                    "update": update_index,
                    "train_mean_reward": float(train_stats["mean_reward"]),
                    "train_mean_length": float(train_stats["mean_length"]),
                    "train_fall_rate": float(train_stats["fall_rate"]),
                    "train_mean_height": float(train_stats["mean_height"]),
                    "train_mean_uprightness": float(train_stats["mean_uprightness"]),
                    **{f"train_{key}": float(train_stats.get(key, 0.0)) for key in _REWARD_TERM_KEYS},
                    "policy_loss": float(logger_values.get("train/policy_gradient_loss", 0.0)),
                    "value_loss": float(logger_values.get("train/value_loss", 0.0)),
                    "entropy": float(-logger_values.get("train/entropy_loss", 0.0)),
                    "clip_fraction": float(logger_values.get("train/clip_fraction", 0.0)),
                    "approx_kl": float(logger_values.get("train/approx_kl", 0.0)),
                }
                if update_index % EVAL_INTERVAL == 0:
                    eval_stats = evaluate_actor(self.eval_env, self.model)
                    record.update({f"eval_{key}": float(value) for key, value in eval_stats.items()})
                    if eval_stats["mean_reward"] > self.best_eval_reward:
                        self.best_eval_reward = float(eval_stats["mean_reward"])
                        best_model_path = self.run_dir / "best_model.zip"
                        self.model.save(best_model_path)
                        self._export_policy(self.policy_dir, best_model_path)
                self.history.append(record)
                self._print_record(record)
                if update_index % EVAL_INTERVAL == 0:
                    self._write_history()
                if update_index % 25 == 0:
                    self.model.save(self.checkpoint_dir / f"update_{update_index}.zip")
            final_model_path = self.run_dir / "final_model.zip"
            self.model.save(final_model_path)
            if not self.policy_dir.exists():
                self._export_policy(self.policy_dir, final_model_path)
            self._write_history()
        finally:
            self.close()

    def close(self) -> None:
        self.train_env.close()
        self.eval_env.close()

    def _build_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return RUNS_DIR / f"standing_ppo_sb3_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "algorithm": "stable_baselines3.PPO",
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "num_envs": self.num_envs,
            "vec_env_start_method": VEC_ENV_START_METHOD,
            "max_steps": MAX_STEPS,
            "episodes_per_update": EPISODES_PER_UPDATE,
            "rollout_steps_per_env": self.rollout_steps_per_env,
            "rollout_steps": self.rollout_steps,
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
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "critic_hidden_dim": CRITIC_HIDDEN_DIM,
            "target_height": TARGET_HEIGHT,
            "height_reward_weight": HEIGHT_REWARD_WEIGHT,
            "uprightness_reward_weight": UPRIGHTNESS_REWARD_WEIGHT,
            "foot_stability_weight": FOOT_STABILITY_WEIGHT,
            "joint_vel_energy_weight": JOINT_VEL_ENERGY_WEIGHT,
            "seed": SEED,
        }
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def _export_policy(self, policy_dir: Path, model_path: Path) -> None:
        export_policy_artifacts(model_path, policy_dir)

    def _write_history(self) -> None:
        with (self.run_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    def _sync_opponent_policy(self) -> None:
        self.model.save(self.opponent_model_path)
        self.train_env.env_method("load_opponent_model", str(self.opponent_model_path))

    def _print_record(self, record: Dict[str, Any]) -> None:
        keys = [
            "update",
            "train_mean_reward",
            "train_mean_length",
            "train_fall_rate",
            "train_mean_height",
            "train_mean_uprightness",
            "train_height_reward",
            "train_uprightness_reward",
            "train_foot_stability_reward",
            "train_energy_reward",
            "policy_loss",
            "value_loss",
            "entropy",
            "clip_fraction",
            "approx_kl",
        ]
        if "eval_mean_reward" in record:
            keys.extend([
                "eval_mean_reward",
                "eval_mean_length",
                "eval_fall_rate",
                "eval_mean_height",
                "eval_mean_uprightness",
                "eval_height_reward",
                "eval_uprightness_reward",
                "eval_foot_stability_reward",
                "eval_energy_reward",
            ])
        print(" | ".join(f"{key}={record[key]:.4f}" if isinstance(record[key], float) else f"{key}={record[key]}" for key in keys), flush=True)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = StandingSB3Trainer(device=device)
    trainer.train()
    print(f"run_dir={trainer.run_dir}", flush=True)
    print(f"policy_dir={trainer.policy_dir}", flush=True)


if __name__ == "__main__":
    main()
