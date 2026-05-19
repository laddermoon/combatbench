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
GROUP_SIZE = 8 # 64
EPISODES_PER_UPDATE = 256 # 1024
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 4096 # 32768
MAX_UPDATES = 10000
EVAL_INTERVAL = 5
EVAL_EPISODES = 16
LEARNING_RATE = 3e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 1e-3
GRAD_CLIP_NORM = 1.0
HIDDEN_DIM = 256
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

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._step_count = 0
        self._fall_streak = 0
        self._fallen = False
        self._output = self._build_output(ctx, is_standing=True)

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
        self._output = self._build_output(ctx, is_standing=is_standing)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_output(ctx)

    def get_output(self) -> Any:
        return self._output

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "fall_height_threshold": self.fall_height_threshold,
            "fall_upright_threshold": self.fall_upright_threshold,
            "fall_grace_steps": self.fall_grace_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandingRewardObserver":
        return cls(**config)

    def _build_output(
        self,
        ctx: ReadOnlySimContext,
        is_standing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if is_standing is None:
            core_state = ctx.accessor.get_core_state()[self.agent_id]
            derived_state = ctx.accessor.get_derived_state()[self.agent_id]
            height = float(core_state["root_pos"][2])
            uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])
            is_standing = bool(
                height >= self.fall_height_threshold and uprightness >= self.fall_upright_threshold
            )
        return {
            "steps": int(self._step_count),
            "is_standing": bool(is_standing),
            "is_fallen": bool(self._fallen),
            "is_terminated": bool(ctx.is_terminated),
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


_ROLLOUT_RUNTIME: Optional[EnvRuntime] = None
_ROLLOUT_ACTOR: Optional[Actor] = None
_CPU_DEVICE = torch.device("cpu")


def _limit_worker_threads() -> None:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _snapshot_actor_state_dict(actor: Actor) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in actor.state_dict().items()}


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
    torch.save(export_payload, policy_dir / "model.pt")
    policy_code = """import sys\nfrom pathlib import Path\nfrom typing import Any, Dict, Optional\n\nimport numpy as np\nimport torch\nfrom torch import nn\n\nfor parent in Path(__file__).resolve().parents:\n    if (parent / \"policy\" / \"base.py\").exists():\n        if str(parent) not in sys.path:\n            sys.path.insert(0, str(parent))\n        break\n    if (parent / \"combatbench\" / \"policy\" / \"base.py\").exists():\n        if str(parent) not in sys.path:\n            sys.path.insert(0, str(parent))\n        break\n\ntry:\n    from policy.base import BaseCombatPolicy\nexcept ImportError:\n    from combatbench.policy.base import BaseCombatPolicy\n\n\nclass Actor(nn.Module):\n    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(obs_dim, hidden_dim),\n            nn.Tanh(),\n            nn.Linear(hidden_dim, hidden_dim),\n            nn.Tanh(),\n            nn.Linear(hidden_dim, action_dim),\n        )\n\n    def forward(self, obs: torch.Tensor) -> torch.Tensor:\n        return torch.tanh(self.net(obs))\n\n\nclass StandingCombatPolicy(BaseCombatPolicy):\n    def __init__(self, model_path: Optional[str] = None, observation_space: Any = None, action_space: Any = None, **kwargs: Any):\n        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / \"model.pt\"\n        payload = torch.load(payload_path, map_location=\"cpu\")\n        self.actor = Actor(payload[\"obs_dim\"], payload[\"action_dim\"], payload[\"hidden_dim\"])\n        model_state_dict = self.actor.state_dict()\n        filtered_state_dict = {\n            key: value\n            for key, value in payload[\"state_dict\"].items()\n            if key in model_state_dict\n        }\n        incompatible = self.actor.load_state_dict(filtered_state_dict, strict=False)\n        if incompatible.missing_keys:\n            raise RuntimeError(f\"Missing keys in standing policy checkpoint: {incompatible.missing_keys}\")\n        unexpected_keys = sorted(set(payload[\"state_dict\"].keys()) - set(filtered_state_dict.keys()))\n        allowed_unexpected_keys = {\"log_std\"}\n        disallowed_unexpected_keys = [key for key in unexpected_keys if key not in allowed_unexpected_keys]\n        if disallowed_unexpected_keys:\n            raise RuntimeError(f\"Unexpected keys in standing policy checkpoint: {disallowed_unexpected_keys}\")\n        self.actor.eval()\n\n    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:\n        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)\n        with torch.no_grad():\n            action = self.actor(obs_tensor)\n        return action.squeeze(0).cpu().numpy().astype(np.float32)\n"""
    with (policy_dir / "policy.py").open("w", encoding="utf-8") as handle:
        handle.write(policy_code)


def _split_sequence(values: Sequence[int], parts: int) -> List[List[int]]:
    if not values:
        return []
    bounded_parts = max(1, min(parts, len(values)))
    chunk_size = (len(values) + bounded_parts - 1) // bounded_parts
    return [list(values[start:start + chunk_size]) for start in range(0, len(values), chunk_size)]


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
        return RUNS_DIR / f"standing_{timestamp}"

    def _save_config(self) -> None:
        config = {
            "control_frequency": CONTROL_FREQUENCY,
            "match_duration_seconds": MATCH_DURATION_SECONDS,
            "max_steps": MAX_STEPS,
            "initial_distance": INITIAL_DISTANCE,
            "rollout_initial_distance_min": ROLLOUT_INITIAL_DISTANCE_MIN,
            "rollout_initial_distance_max": ROLLOUT_INITIAL_DISTANCE_MAX,
            "symmetric_self_play_rollout": True,
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
            "hidden_dim": HIDDEN_DIM,
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



def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = GRPOTrainer(device=device)
    trainer.train()
    print(f"run_dir={trainer.run_dir}", flush=True)
    print(f"policy_dir={trainer.policy_dir}", flush=True)


if __name__ == "__main__":
    main()
