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

from envs.framework import BaseObserverPlugin, BasePlugin, EnvRuntime, ReadOnlySimContext, SimContext, TerminationReason
from envs.humanoid21 import Humanoid21Observer, MujocoCombatSimulator
from envs.humanoid21.disturbance_plugins import RandomPushPlugin, InitialStatePerturbationPlugin

CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 10.0
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)
INITIAL_DISTANCE = 3.0
ROLLOUT_INITIAL_DISTANCE_MIN = 1.5
ROLLOUT_INITIAL_DISTANCE_MAX = 3.5
ACTION_DIM = Humanoid21Observer.ACTION_DIM
OBS_DIM = Humanoid21Observer.OBS_DIM
GROUP_SIZE = max(1, int(os.environ.get("STANDING_GROUP_SIZE", "8")))
EPISODES_PER_UPDATE = 256
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 4096
MAX_UPDATES = 10000
EVAL_INTERVAL = 5
EVAL_EPISODES = 16
LEARNING_RATE = 1e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 1e-3
GRAD_CLIP_NORM = 1.0
TARGET_KL = float(os.environ.get("STANDING_TARGET_KL", "0.05"))
ACTOR_HIDDEN_DIM = 256
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0
STANDING_SCORE_MAX = 1.0
TARGET_HEIGHT = 1.28
HEIGHT_FULL_PENALTY_DELTA = 0.20
UPRIGHT_TILT_FULL_PENALTY_DEGREES = 30.0
UPRIGHT_FULL_PENALTY_COSINE = float(np.cos(np.deg2rad(UPRIGHT_TILT_FULL_PENALTY_DEGREES)))
ROOT_XY_FULL_PENALTY_DISTANCE = 1.5
JOINT_POSE_FULL_PENALTY_MEAN_ABS = 0.2
JOINT_VEL_FULL_PENALTY_MEAN_ABS = 1
FALL_HEIGHT_THRESHOLD = 1.10
FALL_UPRIGHT_THRESHOLD = 0.8
FALL_GRACE_STEPS = 3
POSTURE_SCORE_VERBOSE = os.environ.get("STANDING_SCORE_VERBOSE", "0") == "1"
POSTURE_SCORE_VERBOSE_STRIDE = max(1, int(os.environ.get("STANDING_SCORE_VERBOSE_STRIDE", "10")))
POSTURE_SCORE_VERBOSE_AGENT = os.environ.get("STANDING_SCORE_VERBOSE_AGENT", "robot_a")
RTG_GAMMA = float(os.environ.get("STANDING_RTG_GAMMA", "0.9"))
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
        verbose: bool = False,
        verbose_stride: int = 1,
    ):
        self.agent_id = agent_id
        self.verbose = bool(verbose)
        self.verbose_stride = max(1, int(verbose_stride))
        self._output = 0.0
        self._reference_root_xy: Optional[np.ndarray] = None
        self._reference_joint_pos: Optional[np.ndarray] = None
        self._previous_total_score: float = 0.0
        self._last_total_score: float = 0.0
        self._step_index: int = 0

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])
        self._reference_root_xy = np.asarray(core_state["root_pos"][:2], dtype=np.float32).copy()
        self._reference_joint_pos = np.asarray(core_state["joint_pos_norm"], dtype=np.float32).copy()
        initial_total_score = self._compute_reward_terms(ctx, height=height, uprightness=uprightness)
        self._previous_total_score = initial_total_score
        self._last_total_score = initial_total_score
        self._step_index = 0
        self._output = 0.0

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])
        total_score = self._compute_reward_terms(ctx, height=height, uprightness=uprightness)
        reward = total_score - self._previous_total_score
        self._previous_total_score = total_score
        self._last_total_score = total_score
        self._step_index += 1
        self._output = self._build_output(reward)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_output(0.0)

    def get_output(self) -> float:
        return self._output

    def _compute_posture_terms(
        self,
        ctx: ReadOnlySimContext,
        height: float,
        uprightness: float,
        verbose: bool = False,
    ) -> Dict[str, float]:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        root_xy = np.asarray(core_state["root_pos"][:2], dtype=np.float32)
        joint_pos = np.asarray(core_state["joint_pos_norm"], dtype=np.float32)
        reference_root_xy = root_xy if self._reference_root_xy is None else self._reference_root_xy
        reference_joint_pos = joint_pos if self._reference_joint_pos is None else self._reference_joint_pos
        root_xy_distance = float(np.linalg.norm(root_xy - reference_root_xy))
        joint_pose_mean_abs = float(np.mean(np.abs(joint_pos - reference_joint_pos)))
        joint_velocity_mean_abs = float(np.mean(np.abs(np.asarray(core_state["joint_vel_norm"], dtype=np.float32))))
        height_deficit = max(0.0, TARGET_HEIGHT - height)
        tilt_angle_radians = float(np.arccos(np.clip(uprightness, -1.0, 1.0)))
        tilt_angle_degrees = float(np.degrees(tilt_angle_radians))
        height_penalty = float((height_deficit / HEIGHT_FULL_PENALTY_DELTA) ** 2)
        uprightness_penalty = float((tilt_angle_degrees / UPRIGHT_TILT_FULL_PENALTY_DEGREES) ** 2)
        root_xy_penalty = float((root_xy_distance / ROOT_XY_FULL_PENALTY_DISTANCE) ** 2)
        joint_pose_penalty = float((joint_pose_mean_abs / JOINT_POSE_FULL_PENALTY_MEAN_ABS) ** 2)
        joint_velocity_penalty = float((joint_velocity_mean_abs / JOINT_VEL_FULL_PENALTY_MEAN_ABS) ** 2)
        total_penalty = float(
            height_penalty
            + uprightness_penalty
            + root_xy_penalty
            + joint_pose_penalty
            + joint_velocity_penalty
        )
        total_score = float(STANDING_SCORE_MAX - total_penalty)
        if verbose:
            contribution_values = {
                "height": abs(float(height_penalty)),
                "uprightness": abs(float(uprightness_penalty)),
                "root_xy": abs(float(root_xy_penalty)),
                "joint_pose": abs(float(joint_pose_penalty)),
                "joint_velocity": abs(float(joint_velocity_penalty)),
            }
            contribution_total = max(float(sum(contribution_values.values())), 1e-8)
            contribution_ratios = {
                key: float(value / contribution_total)
                for key, value in contribution_values.items()
            }
            print(
                f"posture_contrib[{self.agent_id}] step={ctx.episode_step} "
                f"height={contribution_ratios['height']:.4f} "
                f"uprightness={contribution_ratios['uprightness']:.4f} "
                f"root_xy={contribution_ratios['root_xy']:.4f} "
                f"joint_pose={contribution_ratios['joint_pose']:.4f} "
                f"joint_velocity={contribution_ratios['joint_velocity']:.4f} "
                f"penalty=({float(height_penalty):.4f}, {float(uprightness_penalty):.4f}, {float(root_xy_penalty):.4f}, {float(joint_pose_penalty):.4f}, {float(joint_velocity_penalty):.4f}) "
                f"standing_score={total_score:.4f}",
                flush=True,
            )
        return {
            "height_penalty": float(height_penalty),
            "uprightness_penalty": float(uprightness_penalty),
            "root_xy_penalty": float(root_xy_penalty),
            "joint_pose_penalty": float(joint_pose_penalty),
            "joint_velocity_penalty": float(joint_velocity_penalty),
            "total_score": total_score,
            "height_deficit": float(height_deficit),
            "tilt_angle_degrees": float(tilt_angle_degrees),
            "root_xy_distance": float(root_xy_distance),
            "joint_pose_mean_abs": float(joint_pose_mean_abs),
            "joint_velocity_mean_abs": float(joint_velocity_mean_abs),
        }

    def _compute_reward_terms(
        self,
        ctx: ReadOnlySimContext,
        height: float,
        uprightness: float,
    ) -> float:
        posture_terms = self._compute_posture_terms(
            ctx,
            height=height,
            uprightness=uprightness,
            verbose=self.verbose and (ctx.episode_step % self.verbose_stride == 0),
        )
        return float(posture_terms["total_score"])

    def _build_output(self, reward: float) -> float:
        return float(reward)


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
    def __init__(self, device: torch.device, resume_from: Optional[Path] = None):
        self.device = device
        self.actor = Actor(OBS_DIM, ACTION_DIM, ACTOR_HIDDEN_DIM).to(device)
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
        self.resume_from = resume_from.resolve() if resume_from is not None else None
        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)
        self._save_config()

    def train(self) -> None:
        try:
            for update_index in range(1, MAX_UPDATES + 1):
                seeds = [SEED + update_index * EPISODES_PER_UPDATE + episode_index for episode_index in range(EPISODES_PER_UPDATE)]
                episodes = self._collect_episodes(seeds=seeds, deterministic=False, worker_limit=ROLLOUT_WORKERS)
                update_stats = self._update_actor(episodes)
                mean_episode_reward = float(np.mean([episode["episode_reward"] for episode in episodes]))
                mean_episode_length = float(np.mean([episode["steps"] for episode in episodes]))
                record = {
                    "update": update_index,
                    "train_mean_reward": mean_episode_reward,
                    "train_mean_length": mean_episode_length,
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
        actor_state_dict = _snapshot_module_state_dict(self.actor)
        seed_chunks = _split_sequence(list(seeds), max(1, min(worker_limit, ROLLOUT_WORKERS)))
        tasks = [
            {
                "actor_state_dict": actor_state_dict,
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
        }

    def _update_actor(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        advantage_sequences = build_group_normalized_reward_to_go(episodes, GROUP_SIZE, RTG_GAMMA)
        obs_batch = np.concatenate([episode["observations"] for episode in episodes], axis=0)
        action_batch = np.concatenate([episode["actions"] for episode in episodes], axis=0)
        old_log_prob_batch = np.concatenate([episode["log_probs"] for episode in episodes], axis=0)
        advantage_batch = np.concatenate(advantage_sequences, axis=0)
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
        return RUNS_DIR / f"standing_grpo_rtg_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "initial_distance": INITIAL_DISTANCE,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
            "symmetric_self_play_rollout": True,
            "algorithm": "grpo_rtg",
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
            "standing_score_max": STANDING_SCORE_MAX,
            "target_height": TARGET_HEIGHT,
            "height_full_penalty_delta": HEIGHT_FULL_PENALTY_DELTA,
            "upright_tilt_full_penalty_degrees": UPRIGHT_TILT_FULL_PENALTY_DEGREES,
            "upright_full_penalty_cosine": UPRIGHT_FULL_PENALTY_COSINE,
            "root_xy_full_penalty_distance": ROOT_XY_FULL_PENALTY_DISTANCE,
            "joint_pose_full_penalty_mean_abs": JOINT_POSE_FULL_PENALTY_MEAN_ABS,
            "joint_vel_full_penalty_mean_abs": JOINT_VEL_FULL_PENALTY_MEAN_ABS,
            "rtg_gamma": RTG_GAMMA,
            "advantage_mode": "reward_to_go_group_normalized",
            "rollout_workers": ROLLOUT_WORKERS,
            "eval_workers": EVAL_WORKERS,
            "seed": SEED,
            "resume_from": str(self.resume_from) if self.resume_from is not None else None,
        }
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def _save_checkpoint(self, path: Path) -> None:
        payload = {
            "algorithm": "grpo_rtg",
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": ACTOR_HIDDEN_DIM,
            "actor_hidden_dim": ACTOR_HIDDEN_DIM,
            "state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
        if "eval_mean_reward" in record:
            keys.extend([
                "eval_mean_reward",
                "eval_mean_length",
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


def reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        rtg[index] = running
    return rtg


def build_group_normalized_reward_to_go(
    episodes: Sequence[Dict[str, Any]],
    group_size: int,
    gamma: float,
) -> List[np.ndarray]:
    advantage_sequences: List[np.ndarray] = []
    for start in range(0, len(episodes), group_size):
        group_episodes = episodes[start:start + group_size]
        group_returns = [
            reward_to_go(np.asarray(episode["rewards"], dtype=np.float32), gamma)
            for episode in group_episodes
        ]
        flattened_group_returns = np.concatenate(group_returns, axis=0)
        group_mean = float(flattened_group_returns.mean())
        group_std = float(flattened_group_returns.std())
        for group_return in group_returns:
            advantage_sequences.append((group_return - group_mean) / (group_std + 1e-6))
    return advantage_sequences


def build_runtime() -> EnvRuntime:
    simulator = MujocoCombatSimulator(initial_distance=INITIAL_DISTANCE)
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / CONTROL_FREQUENCY)))
    observer_plugins = {
        "robot_a_obs": Humanoid21Observer("robot_a"),
        "robot_b_obs": Humanoid21Observer("robot_b"),
        "robot_a_reward": StandingRewardObserver(
            agent_id="robot_a",
            verbose=POSTURE_SCORE_VERBOSE and POSTURE_SCORE_VERBOSE_AGENT in {"robot_a", "all"},
            verbose_stride=POSTURE_SCORE_VERBOSE_STRIDE,
        ),
        "robot_b_reward": StandingRewardObserver(
            agent_id="robot_b",
            verbose=POSTURE_SCORE_VERBOSE and POSTURE_SCORE_VERBOSE_AGENT in {"robot_b", "all"},
            verbose_stride=POSTURE_SCORE_VERBOSE_STRIDE,
        ),
    }
    runtime = EnvRuntime(
        simulator=simulator,
        observer_plugins=observer_plugins,
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
            InitialStatePerturbationPlugin(
                target_robot="robot_a",
                joint_pos_delta_max=0.05,
                joint_vel_delta_max=0.05,
                root_xy_offset_max=0.05,
                root_tilt_deg_max=10.0,
                root_linear_velocity_delta_max=[0.5, 0.5, 0.0],
                root_angular_velocity_delta_max=[0.5, 0.5, 0.2],
                random_seed=None,
            ),
            InitialStatePerturbationPlugin(
                target_robot="robot_b",
                joint_pos_delta_max=0.05,
                joint_vel_delta_max=0.05,
                root_xy_offset_max=0.05,
                root_tilt_deg_max=10.0,
                root_linear_velocity_delta_max=[0.5, 0.5, 0.0],
                root_angular_velocity_delta_max=[0.5, 0.5, 0.2],
                random_seed=None,
            ),
        ],
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
    rewards: List[float] = []
    for _ in range(MAX_STEPS):
        opponent_obs = np.asarray(runtime.get_observer_output(f"{opponent_agent}_obs"), dtype=np.float32)
        controlled_action, log_prob = _act(actor, obs, device, deterministic=deterministic)
        opponent_action, _ = _act(actor, opponent_obs, device, deterministic=deterministic)
        if controlled_agent == "robot_a":
            runtime.step(controlled_action, opponent_action)
        else:
            runtime.step(opponent_action, controlled_action)
        step_reward = float(runtime.get_observer_output(f"{controlled_agent}_reward"))
        observations.append(obs.copy())
        actions.append(controlled_action.copy())
        rewards.append(step_reward)
        if log_prob is not None:
            log_probs.append(log_prob)
        obs = np.asarray(runtime.get_observer_output(f"{controlled_agent}_obs"), dtype=np.float32)
        terminated, truncated = runtime.get_termination_flags()
        if terminated or truncated:
            break
    episode_reward = float(np.sum(rewards, dtype=np.float32))
    observations_array = np.asarray(observations, dtype=np.float32).reshape(len(observations), OBS_DIM)
    actions_array = np.asarray(actions, dtype=np.float32).reshape(len(actions), ACTION_DIM)
    log_probs_array = np.asarray(log_probs, dtype=np.float32).reshape(len(log_probs),)
    rewards_array = np.asarray(rewards, dtype=np.float32).reshape(len(rewards),)
    return {
        "observations": observations_array,
        "actions": actions_array,
        "log_probs": log_probs_array,
        "rewards": rewards_array,
        "steps": len(observations),
        "episode_reward": episode_reward,
        "controlled_agent": controlled_agent,
        "initial_distance": initial_distance,
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
    print(f"policy_dir={trainer.policy_dir}", flush=True)


if __name__ == "__main__":
    main()
