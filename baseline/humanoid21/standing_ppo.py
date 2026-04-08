from __future__ import annotations

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
EPISODES_PER_UPDATE = 256 # 1024
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 4096 # 32768
MAX_UPDATES = 10000
EVAL_INTERVAL = 5
EVAL_EPISODES = 16
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 1e-3
GRAD_CLIP_NORM = 1.0
ACTOR_HIDDEN_DIM = 256
CRITIC_HIDDEN_DIM = 256
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0
TARGET_HEIGHT = 1.28
HEIGHT_REWARD_WEIGHT = 8.0
UPRIGHTNESS_REWARD_WEIGHT = 2.0
FOOT_STABILITY_WEIGHT = 1.5
ROOT_XY_STABILITY_WEIGHT = 0.15
JOINT_POSE_STABILITY_WEIGHT = 0.015
ACTION_ENERGY_WEIGHT = 0.03
JOINT_VEL_ENERGY_WEIGHT = 0.0015
FALL_PENALTY = 5.0
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
        self._reference_root_xy: Optional[np.ndarray] = None
        self._reference_joint_pos: Optional[np.ndarray] = None
        self._previous_posture_terms: Dict[str, float] = {}
        self._last_reward_terms: Dict[str, float] = self._zero_reward_terms()

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._step_count = 0
        self._fall_streak = 0
        self._fallen = False
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])
        self._reference_root_xy = np.asarray(core_state["root_pos"][:2], dtype=np.float32).copy()
        self._reference_joint_pos = np.asarray(core_state["joint_pos_norm"], dtype=np.float32).copy()
        self._previous_posture_terms = self._compute_posture_terms(ctx, height=height, uprightness=uprightness)
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
            "pose_reward": 0.0,
            "energy_reward": 0.0,
            "fall_penalty": 0.0,
            "total_reward": 0.0,
            "foot_position_penalty": 0.0,
            "action_energy": 0.0,
            "joint_velocity_energy": 0.0,
            "posture_score": 0.0,
            "root_xy_penalty": 0.0,
            "joint_pose_penalty": 0.0,
        }

    def _compute_posture_terms(
        self,
        ctx: ReadOnlySimContext,
        height: float,
        uprightness: float,
    ) -> Dict[str, float]:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        root_xy = np.asarray(core_state["root_pos"][:2], dtype=np.float32)
        joint_pos = np.asarray(core_state["joint_pos_norm"], dtype=np.float32)
        reference_root_xy = root_xy if self._reference_root_xy is None else self._reference_root_xy
        reference_joint_pos = joint_pos if self._reference_joint_pos is None else self._reference_joint_pos
        root_xy_penalty = float(np.mean(np.square(root_xy - reference_root_xy)))
        joint_pose_penalty = float(np.mean(np.square(joint_pos - reference_joint_pos)))
        joint_velocity_penalty = float(np.mean(np.square(core_state["joint_vel_norm"])))
        height_score = -HEIGHT_REWARD_WEIGHT * float((height - TARGET_HEIGHT) ** 2)
        uprightness_score = UPRIGHTNESS_REWARD_WEIGHT * uprightness
        root_xy_score = -ROOT_XY_STABILITY_WEIGHT * root_xy_penalty
        joint_pose_score = -JOINT_POSE_STABILITY_WEIGHT * joint_pose_penalty
        joint_velocity_score = -JOINT_VEL_ENERGY_WEIGHT * joint_velocity_penalty
        total_score = float(height_score + uprightness_score + root_xy_score + joint_pose_score + joint_velocity_score)
        return {
            "height_score": float(height_score),
            "uprightness_score": float(uprightness_score),
            "root_xy_score": float(root_xy_score),
            "joint_pose_score": float(joint_pose_score),
            "joint_velocity_score": float(joint_velocity_score),
            "total_score": total_score,
            "root_xy_penalty": root_xy_penalty,
            "joint_pose_penalty": joint_pose_penalty,
            "joint_velocity_penalty": joint_velocity_penalty,
        }

    def _compute_reward_terms(
        self,
        ctx: ReadOnlySimContext,
        height: float,
        uprightness: float,
    ) -> Dict[str, float]:
        posture_terms = self._compute_posture_terms(ctx, height=height, uprightness=uprightness)
        previous_posture_terms = posture_terms if not self._previous_posture_terms else self._previous_posture_terms
        reward_terms = {
            "height_reward": float(posture_terms["height_score"] - previous_posture_terms["height_score"]),
            "uprightness_reward": float(posture_terms["uprightness_score"] - previous_posture_terms["uprightness_score"]),
            "foot_stability_reward": float(posture_terms["root_xy_score"] - previous_posture_terms["root_xy_score"]),
            "pose_reward": float(posture_terms["joint_pose_score"] - previous_posture_terms["joint_pose_score"]),
            "energy_reward": float(posture_terms["joint_velocity_score"] - previous_posture_terms["joint_velocity_score"]),
            "fall_penalty": 0.0,
        }
        total_reward = float(sum(reward_terms.values()))
        reward_terms["total_reward"] = total_reward
        reward_terms["foot_position_penalty"] = float(posture_terms["root_xy_penalty"])
        reward_terms["action_energy"] = 0.0
        reward_terms["joint_velocity_energy"] = float(posture_terms["joint_velocity_penalty"])
        reward_terms["posture_score"] = float(posture_terms["total_score"])
        reward_terms["root_xy_penalty"] = float(posture_terms["root_xy_penalty"])
        reward_terms["joint_pose_penalty"] = float(posture_terms["joint_pose_penalty"])
        self._previous_posture_terms = posture_terms
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


def _snapshot_module_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _get_actor_hidden_dim() -> int:
    return ACTOR_HIDDEN_DIM


def _split_sequence(values: Sequence[int], parts: int) -> List[List[int]]:
    if not values:
        return []
    bounded_parts = max(1, min(parts, len(values)))
    return [list(chunk) for chunk in np.array_split(np.asarray(values, dtype=np.int64), bounded_parts) if len(chunk) > 0]


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


def export_policy_artifacts(model_path: Path, policy_dir: Path) -> None:
    policy_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(model_path, map_location="cpu")
    export_payload = dict(payload)
    export_payload["state_dict"] = {
        key: value.detach().cpu()
        for key, value in payload["state_dict"].items()
        if key != "log_std"
    }
    export_payload["hidden_dim"] = int(payload.get("actor_hidden_dim", payload.get("hidden_dim", ACTOR_HIDDEN_DIM)))
    torch.save(export_payload, policy_dir / "model.pt")
    policy_code = """import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

for parent in Path(__file__).resolve().parents:
    if (parent / \"policy\" / \"base.py\").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
    if (parent / \"combatbench\" / \"policy\" / \"base.py\").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

try:
    from policy.base import BaseCombatPolicy
except ImportError:
    from combatbench.policy.base import BaseCombatPolicy


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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class StandingCombatPolicy(BaseCombatPolicy):
    def __init__(self, model_path: Optional[str] = None, observation_space: Any = None, action_space: Any = None, **kwargs: Any):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / \"model.pt\"
        payload = torch.load(payload_path, map_location=\"cpu\")
        hidden_dim = int(payload.get(\"hidden_dim\", payload.get(\"actor_hidden_dim\", 256)))
        self.actor = Actor(payload[\"obs_dim\"], payload[\"action_dim\"], hidden_dim)
        model_state_dict = self.actor.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in payload[\"state_dict\"].items()
            if key in model_state_dict
        }
        incompatible = self.actor.load_state_dict(filtered_state_dict, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(f\"Missing keys in exported standing policy: {incompatible.missing_keys}\")
        if incompatible.unexpected_keys:
            raise RuntimeError(f\"Unexpected keys in exported standing policy: {incompatible.unexpected_keys}\")
        self.actor.eval()

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        return None
"""
    with (policy_dir / "policy.py").open("w", encoding="utf-8") as handle:
        handle.write(policy_code)

def _mean_reward_terms(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    all_keys = sorted({key for episode in episodes for key in episode.get("mean_reward_terms", {}).keys()})
    return {
        key: float(np.mean([float(episode.get("mean_reward_terms", {}).get(key, 0.0)) for episode in episodes]))
        for key in all_keys
    }


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


_ROLLOUT_RUNTIME: Optional[EnvRuntime] = None
_ROLLOUT_ACTOR: Optional[Actor] = None
_ROLLOUT_CRITIC: Optional[Critic] = None
_CPU_DEVICE = torch.device("cpu")


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
    def __init__(self, device: torch.device):
        self.device = device
        self.actor = Actor(OBS_DIM, ACTION_DIM, ACTOR_HIDDEN_DIM).to(device)
        self.critic = Critic(OBS_DIM, CRITIC_HIDDEN_DIM).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LEARNING_RATE,
        )
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
                update_stats = self._update_policy(episodes)
                mean_reward_terms = _mean_reward_terms(episodes)
                mean_episode_reward = float(np.mean([episode["episode_reward"] for episode in episodes]))
                mean_episode_length = float(np.mean([episode["steps"] for episode in episodes]))
                fall_rate = float(np.mean([1.0 if episode["reward_info"]["is_fallen"] else 0.0 for episode in episodes]))
                mean_height = float(np.mean([episode["mean_height"] for episode in episodes]))
                mean_uprightness = float(np.mean([episode["mean_uprightness"] for episode in episodes]))
                record = {
                    "update": update_index,
                    "train_mean_reward": mean_episode_reward,
                    "train_mean_length": mean_episode_length,
                    "train_fall_rate": fall_rate,
                    "train_mean_height": mean_height,
                    "train_mean_uprightness": mean_uprightness,
                    **{f"train_{key}": value for key, value in mean_reward_terms.items()},
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
        episodes = self._collect_episodes(seeds=seeds, deterministic=True, worker_limit=EVAL_WORKERS)
        reward_terms = _mean_reward_terms(episodes)
        return {
            "mean_reward": float(np.mean([episode["episode_reward"] for episode in episodes])),
            "mean_length": float(np.mean([episode["steps"] for episode in episodes])),
            "fall_rate": float(np.mean([1.0 if episode["reward_info"]["is_fallen"] else 0.0 for episode in episodes])),
            "mean_height": float(np.mean([episode["mean_height"] for episode in episodes])),
            "mean_uprightness": float(np.mean([episode["mean_uprightness"] for episode in episodes])),
            **reward_terms,
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
                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), GRAD_CLIP_NORM)
                self.optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.mean().item()))
                ratios.append(float(ratio.mean().item()))
                approx_kls.append(float((batch_old_log_prob - new_log_prob).mean().item()))
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "ratio": float(np.mean(ratios)) if ratios else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def _build_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return RUNS_DIR / f"standing_ppo_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "initial_distance": INITIAL_DISTANCE,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
            "symmetric_self_play_rollout": True,
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
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "critic_hidden_dim": CRITIC_HIDDEN_DIM,
            "target_height": TARGET_HEIGHT,
            "height_reward_weight": HEIGHT_REWARD_WEIGHT,
            "uprightness_reward_weight": UPRIGHTNESS_REWARD_WEIGHT,
            "foot_stability_weight": FOOT_STABILITY_WEIGHT,
            "action_energy_weight": ACTION_ENERGY_WEIGHT,
            "joint_vel_energy_weight": JOINT_VEL_ENERGY_WEIGHT,
            "fall_penalty": FALL_PENALTY,
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
            "hidden_dim": ACTOR_HIDDEN_DIM,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "critic_hidden_dim": CRITIC_HIDDEN_DIM,
            "state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
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
            "train_mean_height",
            "train_mean_uprightness",
            "train_height_reward",
            "train_uprightness_reward",
            "train_foot_stability_reward",
            "train_energy_reward",
            "policy_loss",
            "value_loss",
            "entropy",
            "ratio",
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
    runtime.reset(seed=seed, options={"initial_distance": initial_distance})
    obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[float] = []
    values: List[float] = []
    rewards: List[float] = []
    heights: List[float] = []
    uprightnesses: List[float] = []
    reward_terms_history: List[Dict[str, float]] = []
    reward_info = dict(runtime.get_observer_output(f"{controlled_agent}_reward"))
    bootstrap_value = 0.0
    for _ in range(MAX_STEPS):
        opponent_obs = np.asarray(runtime.get_observer_output(f"{opponent_agent}_obs"), dtype=np.float32)
        controlled_action, log_prob, value = _act_with_value(actor, critic, obs, device, deterministic=deterministic)
        opponent_action, _, _ = _act_with_value(actor, critic, opponent_obs, device, deterministic=deterministic)
        if controlled_agent == "robot_a":
            runtime.step(controlled_action, opponent_action)
        else:
            runtime.step(opponent_action, controlled_action)
        reward_info = dict(runtime.get_observer_output(f"{controlled_agent}_reward"))
        step_reward = float(reward_info["total_reward"])
        reward_terms = {
            key: float(reward_info[key])
            for key in [
                "height_reward",
                "uprightness_reward",
                "foot_stability_reward",
                "energy_reward",
                "fall_penalty",
                "total_reward",
                "foot_position_penalty",
                "action_energy",
                "joint_velocity_energy",
            ]
        }
        observations.append(obs.copy())
        actions.append(controlled_action.copy())
        values.append(value)
        rewards.append(step_reward)
        heights.append(float(reward_info["height"]))
        uprightnesses.append(float(reward_info["uprightness"]))
        reward_terms_history.append(reward_terms)
        if log_prob is not None:
            log_probs.append(log_prob)
        obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
        terminated, truncated = runtime.get_termination_flags()
        if terminated or truncated or reward_info["is_fallen"]:
            bootstrap_value = 0.0
            break
        _, _, bootstrap_value = _act_with_value(actor, critic, obs, device, deterministic=True)
    if not observations:
        _, _, bootstrap_value = _act_with_value(actor, critic, obs, device, deterministic=True)
    episode_reward = float(np.sum(rewards, dtype=np.float32))
    mean_reward_terms: Dict[str, float] = {}
    if reward_terms_history:
        reward_term_keys = sorted({key for reward_terms in reward_terms_history for key in reward_terms.keys()})
        mean_reward_terms = {
            key: float(np.mean([float(reward_terms.get(key, 0.0)) for reward_terms in reward_terms_history]))
            for key in reward_term_keys
        }
    observations_array = np.asarray(observations, dtype=np.float32).reshape(len(observations), OBS_DIM)
    actions_array = np.asarray(actions, dtype=np.float32).reshape(len(actions), ACTION_DIM)
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
        "steps": len(observations),
        "episode_reward": episode_reward,
        "mean_height": float(np.mean(heights)) if heights else float(reward_info["height"]),
        "mean_uprightness": float(np.mean(uprightnesses)) if uprightnesses else float(reward_info["uprightness"]),
        "mean_reward_terms": mean_reward_terms,
        "reward_terms_last": reward_terms_history[-1] if reward_terms_history else {},
        "reward_info": reward_info,
        "controlled_agent": controlled_agent,
        "initial_distance": initial_distance,
    }


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PPOTrainer(device=device)
    trainer.train()
    print(f"run_dir={trainer.run_dir}", flush=True)
    print(f"policy_dir={trainer.policy_dir}", flush=True)


if __name__ == "__main__":
    main()
