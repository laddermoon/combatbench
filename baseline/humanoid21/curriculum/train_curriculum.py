"""Curriculum PPO trainer using ParallelRollouter + blueprints.

Follows the design in ``TRAINDESIGN.md``:

  * Env: ``blueprints/curriculum_env.yaml`` (parameterized).
  * Policy init: ``blueprints/init_policy.yaml``.
  * Rollout: ``ParallelRollouter`` with per-episode ``(policy_bp, policy_bp, env_bp, seed, options)``.
  * Reward: stage-dependent combination of cross_support / opponent_relation / damage.
  * Eval-driven curriculum stage gate.

Data contract: all data lives in :class:`Episode` objects (numpy arrays).
There is no RolloutBatch; PPO buffers are built directly from Episode fields.
"""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from baseline.common.algos import compute_gae, ppo_loss
from baseline.common.policies import (
    CriticMLP,
    TanhGaussianMLPPolicy,
    export_actor_policy_artifacts,
)
from baseline.common.rollout import Episode, ParallelRollouter
from baseline.humanoid21.curriculum.common import (
    CONTROL_FREQUENCY,
    CurriculumConfig,
    CurriculumStageGate,
    ROLLOUT_INITIAL_DISTANCE_MAX,
    ROLLOUT_INITIAL_DISTANCE_MIN,
    set_seed,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

# ---------------------------------------------------------------------------
# Data helpers – work directly on Episode numpy arrays
# ---------------------------------------------------------------------------

def _extract_per_step_scalar(
    observer_outputs: Any,
    observer_name: str,
    expected_len: int,
) -> np.ndarray:
    """Pull a (T,) float32 reward signal from stacked observer outputs."""
    node = observer_outputs.get(observer_name)
    if node is None:
        return np.zeros(expected_len, dtype=np.float32)
    values = next(iter(node.values())) if isinstance(node, dict) else node
    if values is None:
        return np.zeros(expected_len, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_len:
        if expected_len == 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0, len(arr) - 1, expected_len)
        arr = np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)
    return arr


def _agent_from_rollout_seed(seed: int) -> str:
    rng = np.random.default_rng(int(seed) + 937)
    return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"


# ---------------------------------------------------------------------------
# PPO buffer – flat numpy arrays assembled from a list of Episodes
# ---------------------------------------------------------------------------

class _PPOBuffer:
    """PPO buffer built from a list of :class:`Episode` objects.

    Processes ALL episodes regardless of agent_id — each episode's data
    is extracted using its own target agent (from ``episode_options``).
    """

    __slots__ = (
        "obs", "actions", "log_probs",
        "r_fall_rewards", "r_cross_rewards", "r_relation_rewards", "r_damage_rewards",
        "final_obs", "is_terminated", "ep_lengths",
        "r_fall_sums", "r_cross_sums", "r_relation_sums", "r_damage_sums",
    )

    def __init__(
        self,
        episodes: Sequence[Episode],
        stage_weights: Tuple[float, float, float, float],
        actor: TanhGaussianMLPPolicy,
        device: torch.device,
        terminal_fall_penalty: float,
    ):
        # Buffer only stores raw per-step rewards for each component.
        # No scaling applied here - scaling happens during optimization.
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        lp_list: List[np.ndarray] = []
        r_cross_per_step: List[np.ndarray] = []
        r_relation_per_step: List[np.ndarray] = []
        r_damage_per_step: List[np.ndarray] = []
        fin_list: List[np.ndarray] = []
        terms: List[bool] = []
        ep_lens: List[int] = []

        for ep in episodes:
            # Use each episode's own target agent (supports mixed robot_a/robot_b rollout).
            ep_target = str(ep.episode_options.get("agent_id", "robot_a"))
            obs = ep.observations.get(ep_target)
            acts = ep.actions.get(ep_target)
            fin = ep.final_observation.get(ep_target)
            if obs is None or acts is None or fin is None:
                print(f"[DEBUG] Skipping episode {len(obs_list)+1}: obs={obs is not None} acts={acts is not None} fin={fin is not None}", flush=True)
                continue
            T = int(acts.shape[0])
            if T == 0:
                continue

            oo = ep.observer_outputs
            # Per-step dense signals from environment observers.
            # Naming follows the 4-component reward scheme:
            #   r_cross     -> cross_support (balance)
            #   r_relation  -> opponent_relation (distance + heading)
            #   r_damage    -> damage reward
            r_cross = _extract_per_step_scalar(oo, "cross_support", T)
            r_relation = _extract_per_step_scalar(oo, "opponent_relation", T)
            r_damage = _extract_per_step_scalar(oo, "damage", T)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(acts, dtype=torch.float32, device=device)
            with torch.no_grad():
                lp, _ = actor.evaluate_actions(obs_t, act_t)
            lp_np = lp.cpu().numpy().astype(np.float32)

            obs_list.append(obs)
            act_list.append(acts)
            lp_list.append(lp_np)
            r_cross_per_step.append(r_cross)
            r_relation_per_step.append(r_relation)
            r_damage_per_step.append(r_damage)
            fin_list.append(np.asarray(fin, dtype=np.float32))
            terms.append(bool(ep.is_terminated))
            ep_lens.append(T)

        if not ep_lens:
            print(f"[DEBUG] _PPOBuffer: no valid episodes from {len(episodes)} input episodes", flush=True)
            self.obs = np.zeros((0,), np.float32)
            self.actions = np.zeros((0,), np.float32)
            self.log_probs = np.zeros((0,), np.float32)
            self.r_fall_rewards = []
            self.r_cross_rewards = []
            self.r_relation_rewards = []
            self.r_damage_rewards = []
            self.final_obs = []
            self.is_terminated = []
            self.ep_lengths = []
            self.r_fall_sums = []
            self.r_cross_sums = []
            self.r_relation_sums = []
            self.r_damage_sums = []
            return

        # Store raw reward components separately for multi-critic training.
        # ``r_fall`` is its own sparse component (zeros except at the
        # terminal step of fallen episodes); it is NOT folded into r_cross.
        r_fall_rew_list: List[np.ndarray] = []
        r_cross_rew_list: List[np.ndarray] = []
        r_relation_rew_list: List[np.ndarray] = []
        r_damage_rew_list: List[np.ndarray] = []
        r_fall_sums: List[float] = []
        r_cross_sums: List[float] = []
        r_relation_sums: List[float] = []
        r_damage_sums: List[float] = []

        for i, T in enumerate(ep_lens):
            # Store raw rewards without any scaling.
            # Scaling will be applied during optimization.
            r_cross_seg = r_cross_per_step[i].astype(np.float32)
            r_relation_seg = r_relation_per_step[i].astype(np.float32)
            r_damage_seg = r_damage_per_step[i].astype(np.float32)

            # r_fall: dedicated sparse signal — single negative spike on
            # the terminal step iff the episode ended in a fall.
            r_fall_seg = np.zeros(T, dtype=np.float32)
            if terms[i] and terminal_fall_penalty > 0.0:
                r_fall_seg[-1] = -float(terminal_fall_penalty)

            r_fall_rew_list.append(r_fall_seg)
            r_cross_rew_list.append(r_cross_seg)
            r_relation_rew_list.append(r_relation_seg)
            r_damage_rew_list.append(r_damage_seg)
            r_fall_sums.append(float(r_fall_seg.sum()))
            r_cross_sums.append(float(r_cross_seg.sum()))
            r_relation_sums.append(float(r_relation_seg.sum()))
            r_damage_sums.append(float(r_damage_seg.sum()))

        self.obs = np.concatenate(obs_list, axis=0)
        self.actions = np.concatenate(act_list, axis=0)
        self.log_probs = np.concatenate(lp_list, axis=0)
        self.r_fall_rewards = r_fall_rew_list
        self.r_cross_rewards = r_cross_rew_list
        self.r_relation_rewards = r_relation_rew_list
        self.r_damage_rewards = r_damage_rew_list
        self.final_obs = fin_list
        self.is_terminated = terms
        self.ep_lengths = ep_lens
        self.r_fall_sums = r_fall_sums
        self.r_cross_sums = r_cross_sums
        self.r_relation_sums = r_relation_sums
        self.r_damage_sums = r_damage_sums

    def __len__(self) -> int:
        return sum(self.ep_lengths)

    def is_empty(self) -> bool:
        return len(self.ep_lengths) == 0


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

REWARD_KEYS: Tuple[str, ...] = ("r_fall", "r_cross", "r_relation", "r_damage")


def _ppo_update(
    actor: TanhGaussianMLPPolicy,
    critics: Dict[str, CriticMLP],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    buf: _PPOBuffer,
    cfg: CurriculumConfig,
    device: torch.device,
    stage_weights: Tuple[float, float, float, float],
) -> Dict[str, float]:
    # Multi-critic GAE: compute separate advantages for each reward component.
    # Each critic is optimized independently with its own optimizer.
    # Actor is optimized separately with its own optimizer.
    obs_all_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=device)

    # Compute values for each critic
    values_all: Dict[str, np.ndarray] = {}
    for key, critic in critics.items():
        with torch.no_grad():
            values_all[key] = critic(obs_all_t).squeeze(-1).cpu().numpy().astype(np.float32)

    # Compute GAE for each reward component
    advs_all: Dict[str, np.ndarray] = {}
    rets_all: Dict[str, np.ndarray] = {}

    for key in REWARD_KEYS:
        advs_list = []
        rets_list = []
        offset = 0

        for i, T in enumerate(buf.ep_lengths):
            values = values_all[key][offset : offset + T]
            offset += T
            last_value = 0.0
            if not buf.is_terminated[i] and buf.final_obs[i] is not None:
                fin_t = torch.as_tensor(
                    buf.final_obs[i][None], dtype=torch.float32, device=device,
                )
                with torch.no_grad():
                    last_value = float(critics[key](fin_t).squeeze(-1).item())

            rewards = getattr(buf, f"{key}_rewards")[i]
            adv, ret = compute_gae(
                rewards=rewards,
                values=values,
                last_value=last_value,
                gamma=cfg.gamma,
                lam=cfg.gae_lambda,
            )
            advs_list.append(adv)
            rets_list.append(ret)

        advs_all[key] = np.concatenate(advs_list)
        rets_all[key] = np.concatenate(rets_list)

    # Prepare tensors
    obs_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(buf.actions, dtype=torch.float32, device=device)
    old_lp_t = torch.as_tensor(buf.log_probs, dtype=torch.float32, device=device)

    # Normalize advantages per component and combine with stage weights
    def _normalize_adv(adv: np.ndarray) -> np.ndarray:
        mean = float(adv.mean())
        std = float(adv.std())
        if std < 1e-8:
            return np.zeros_like(adv, dtype=np.float32)
        return ((adv - mean) / std).astype(np.float32)

    if len(stage_weights) != len(REWARD_KEYS):
        raise ValueError(
            f"stage_weights must have {len(REWARD_KEYS)} entries (one per "
            f"reward in {REWARD_KEYS}); got {stage_weights!r}"
        )
    combined_adv = np.zeros_like(advs_all[REWARD_KEYS[0]], dtype=np.float32)
    for w, key in zip(stage_weights, REWARD_KEYS):
        if w == 0.0:
            continue
        combined_adv = combined_adv + float(w) * _normalize_adv(advs_all[key])
    adv_t = torch.as_tensor(combined_adv, dtype=torch.float32, device=device)
    
    n = obs_t.shape[0]
    pol_losses: List[float] = []
    val_losses: Dict[str, List[float]] = {key: [] for key in REWARD_KEYS}
    kls: List[float] = []
    early_stop_kl = 0.0
    
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=device)
        early_stop = False
        
        for s in range(0, n, cfg.minibatch_size):
            idx = perm[s : s + cfg.minibatch_size]
            idx_cpu = idx.cpu().numpy()
            
            # Step 1: Update each critic independently
            for key in REWARD_KEYS:
                critic_optimizers[key].zero_grad()
                new_val = critics[key](obs_t[idx]).squeeze(-1)
                ret_val = torch.as_tensor(
                    rets_all[key][idx_cpu], dtype=torch.float32, device=device,
                )
                val_loss = ((new_val - ret_val) ** 2).mean()
                val_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critics[key].parameters(), cfg.grad_clip_norm,
                )
                critic_optimizers[key].step()
                val_losses[key].append(float(val_loss))
            
            # Step 2: Update actor (after all critics are updated)
            new_lp, entropy = actor.evaluate_actions(obs_t[idx], act_t[idx])
            
            with torch.no_grad():
                approx_kl = float((old_lp_t[idx] - new_lp).mean().item())
            kls.append(approx_kl)
            if cfg.target_kl > 0.0 and approx_kl > cfg.target_kl:
                early_stop_kl = approx_kl
                early_stop = True
                break
            
            # Policy loss with combined normalized advantages
            ratio = torch.exp(new_lp - old_lp_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Actor loss (no value loss here - critics are updated separately)
            loss = policy_loss - cfg.entropy_coef * entropy.mean()
            
            actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.grad_clip_norm,
            )
            actor_optimizer.step()
            pol_losses.append(float(policy_loss))
        
        if early_stop:
            break
    
    # Aggregate value losses per critic
    total_val_losses = [np.mean(val_losses[key]) if val_losses[key] else 0.0 for key in REWARD_KEYS]
    
    # Per-critic detailed losses for structured logging
    per_critic_losses: Dict[str, float] = {
        f"vloss_{key}": float(np.mean(val_losses[key])) if val_losses[key] else 0.0
        for key in REWARD_KEYS
    }
    
    return {
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
        "value_loss": float(np.mean(total_val_losses)),
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "early_stop_kl": early_stop_kl,
        **per_critic_losses,
    }


# ---------------------------------------------------------------------------
# Rollout jobs
# ---------------------------------------------------------------------------

def _build_rollout_jobs(
    env_pb: ParameterizedEnvBlueprint,
    policy_bp: PolicyBlueprint,
    base_seed: int,
    n_episodes: int,
    max_steps: int,
) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
    """Prepare ``n`` jobs – each with a randomly chosen target agent and initial distance.

    ``initial_distance`` is passed via ``options`` (not baked into the env
    blueprint) so that all episodes sharing the same ``agent_id`` reuse the
    same cached ``EnvRuntime`` inside each worker.  The simulator's
    ``reset(options={"initial_distance": ...})`` already supports this.
    """
    rng = np.random.default_rng(base_seed)

    # Pre-materialize 2 env blueprints (one per agent_id) — cache-friendly.
    env_bps: Dict[str, EnvBlueprint] = {
        aid: env_pb.materialize(max_steps=max_steps, agent_id=aid)
        for aid in ("robot_a", "robot_b")
    }

    jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
    for i in range(n_episodes):
        seed = int(base_seed + i)
        agent_id = _agent_from_rollout_seed(seed)
        initial_distance = float(
            rng.uniform(ROLLOUT_INITIAL_DISTANCE_MIN, ROLLOUT_INITIAL_DISTANCE_MAX)
        )
        jobs.append((
            policy_bp, policy_bp,
            env_bps[agent_id], seed,
            {"agent_id": agent_id, "initial_distance": initial_distance},
        ))
    return jobs


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _batch_summary(buf: _PPOBuffer, max_steps: int) -> Dict[str, float]:
    n = len(buf.ep_lengths)
    if n == 0:
        return {"term_rate": 0.0, "mean_length": 0.0, "len_ratio": 0.0,
                "final_in_zone_ratio": 0.0}
    mean_len = float(np.mean(buf.ep_lengths))
    return {
        "term_rate": float(sum(buf.is_terminated) / n),
        "mean_length": mean_len,
        "len_ratio": mean_len / float(max_steps),
        "final_in_zone_ratio": 0.0,
    }


def _reward_summary(buf: _PPOBuffer) -> Dict[str, Any]:
    """Return per-reward statistics including mean and std for diagnostics."""
    empty_return: Dict[str, Any] = {
        "r_fall_mean": 0.0, "r_fall_std": 0.0,
        "r_cross_mean": 0.0, "r_cross_std": 0.0,
        "r_relation_mean": 0.0, "r_relation_std": 0.0,
        "r_damage_mean": 0.0, "r_damage_std": 0.0,
    }
    if not buf.ep_lengths:
        return empty_return

    def _mean_std(arr: List[float]) -> Tuple[float, float]:
        if not arr:
            return 0.0, 0.0
        mean = float(np.mean(arr))
        std = float(np.std(arr))  # population std
        return mean, std

    r_fall_mean, r_fall_std = _mean_std(buf.r_fall_sums)
    r_cross_mean, r_cross_std = _mean_std(buf.r_cross_sums)
    r_relation_mean, r_relation_std = _mean_std(buf.r_relation_sums)
    r_damage_mean, r_damage_std = _mean_std(buf.r_damage_sums)

    return {
        "r_fall_mean": r_fall_mean, "r_fall_std": r_fall_std,
        "r_cross_mean": r_cross_mean, "r_cross_std": r_cross_std,
        "r_relation_mean": r_relation_mean, "r_relation_std": r_relation_std,
        "r_damage_mean": r_damage_mean, "r_damage_std": r_damage_std,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    gate: CurriculumStageGate,
    update: int,
    best_eval: tuple,
    cfg: CurriculumConfig,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critics_state_dict": {k: v.state_dict() for k, v in critics.items()},
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizers_state_dict": {k: v.state_dict() for k, v in critic_optimizers.items()},
            "gate_stage": gate.stage,
            "update": update,
            "best_eval": best_eval,
            "cfg": cfg.__dict__,
        },
        ckpt_path,
    )


def _load_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    gate: CurriculumStageGate,
    cfg: CurriculumConfig,
) -> int:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["actor_state_dict"])

    # Critic loading. Three formats are supported for backwards compat:
    #   * new (4-critic): payload["critics_state_dict"] keyed by REWARD_KEYS
    #   * old (3-critic): payload["critics_state_dict"] without ``r_fall``
    #   * very old (single-critic): payload["critic_state_dict"]
    if "critics_state_dict" in payload:
        saved = payload["critics_state_dict"]
        loaded_keys = []
        for k, v in critics.items():
            if k in saved:
                v.load_state_dict(saved[k])
                loaded_keys.append(k)
            else:
                print(
                    f"[INFO] Critic {k!r} not in checkpoint -> fresh init",
                    flush=True,
                )
        print(
            f"[INFO] Loaded multi-critic weights for {loaded_keys}",
            flush=True,
        )
    elif "critic_state_dict" in payload:
        critics["r_cross"].load_state_dict(payload["critic_state_dict"])
        print(
            "[INFO] Loaded legacy single-critic weights into r_cross critic",
            flush=True,
        )
    else:
        print(
            "[WARNING] No critic weights found in checkpoint, using random init",
            flush=True,
        )

    # Load optimizer states if available (new format with separate optimizers)
    if "actor_optimizer_state_dict" in payload:
        try:
            actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        except RuntimeError as e:
            print(f"[WARNING] Actor optimizer state mismatch: {e}", flush=True)
    elif "optimizer_state_dict" in payload:
        # Legacy format - ignore, start fresh optimizers
        print("[INFO] Legacy combined optimizer found, using fresh optimizer states", flush=True)
    
    if "critic_optimizers_state_dict" in payload:
        saved_crit_opt = payload["critic_optimizers_state_dict"]
        for k, opt in critic_optimizers.items():
            if k in saved_crit_opt:
                try:
                    opt.load_state_dict(saved_crit_opt[k])
                except RuntimeError as e:
                    print(f"[WARNING] Critic {k} optimizer state mismatch: {e}", flush=True)

    gate.stage = int(payload.get("gate_stage", 1))
    return int(payload.get("update", 0))


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    cfg: CurriculumConfig,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
) -> None:
    # Kill entire process group on SIGTERM/SIGINT so spawn workers don't
    # become orphans when the main process is killed.
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load blueprints
    blueprint_dir = Path(__file__).resolve().parent.parent / "blueprints"
    env_pb = ParameterizedEnvBlueprint.load(blueprint_dir / "curriculum_env.yaml")
    init_policy_bp = PolicyBlueprint.load(blueprint_dir / "init_policy.yaml")

    # 2. Build models
    actor: TanhGaussianMLPPolicy = init_policy_bp.build()
    actor = actor.to(device)
    
    # Multi-critic architecture: one critic per reward component.
    # ``r_fall`` is its own component (terminal-only sparse penalty) so
    # the cross_support critic doesn't have to memorize the fall spike.
    critics = {
        key: CriticMLP(obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim).to(device)
        for key in REWARD_KEYS
    }
    
    # Separate optimizers for actor and each critic (can have different learning rates)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    critic_optimizers = {
        key: torch.optim.Adam(critics[key].parameters(), lr=cfg.critic_learning_rate)
        for key in REWARD_KEYS
    }

    gate = CurriculumStageGate(
        max_steps=cfg.max_steps,
        pass_len_ratio=cfg.stage1_pass_len_ratio,
        pass_final_in_zone=cfg.stage2_pass_final_in_zone,
    )

    start_update = 1
    best_eval: tuple = (-1, -float("inf"))

    # 3. Resume
    if resume_from is not None:
        start_update = _load_checkpoint(
            Path(resume_from), actor=actor, critics=critics,
            actor_optimizer=actor_optimizer, critic_optimizers=critic_optimizers,
            gate=gate, cfg=cfg,
        )
        print(f"[resume] loaded from {resume_from}, starting at update={start_update}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    ckpt_dir = run_dir / "checkpoints"
    print(f"run_dir={run_dir}", flush=True)

    # 4. Training loop
    print(f"[DEBUG] rollout_workers={cfg.rollout_workers}  episodes_per_update={cfg.episodes_per_update}  max_steps={cfg.max_steps}  update_epochs={cfg.update_epochs}  minibatch_size={cfg.minibatch_size}", flush=True)
    with ParallelRollouter(num_workers=cfg.rollout_workers) as rollouter:
        for u in range(start_update, cfg.max_updates + 1):
            t_update_start = time.perf_counter()

            # 4.1 Export policy blueprint (stochastic for training rollouts)
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(dest_path=str(export_dir))
            policy_bp.config["stochastic"] = True
            t_export = time.perf_counter() - t0

            # 4.2 Prepare rollout jobs
            t0 = time.perf_counter()
            rollout_seed = cfg.seed + u * cfg.episodes_per_update
            jobs = _build_rollout_jobs(
                env_pb, policy_bp, rollout_seed,
                cfg.episodes_per_update, max_steps=cfg.max_steps,
            )
            t_jobs = time.perf_counter() - t0

            # 4.3 Rollout
            t0 = time.perf_counter()
            episodes: List[Episode] = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            # 4.4 Build per-agent PPO buffers directly from Episodes
            t0 = time.perf_counter()
            gate_info = gate.current_state()
            stage_weights: Tuple[float, float, float, float] = gate_info["weights"]

            buf = _PPOBuffer(
                episodes=episodes,
                stage_weights=stage_weights,
                actor=actor,
                device=device,
                terminal_fall_penalty=cfg.terminal_fall_penalty,
            )
            t_buffer = time.perf_counter() - t0
            

            # 4.5 PPO update
            t0 = time.perf_counter()
            stats = _ppo_update(
                actor, critics, actor_optimizer, critic_optimizers,
                buf, cfg, device, stage_weights,
            )
            t_ppo = time.perf_counter() - t0

            # 4.6 Logging (structured with reward std and per-critic losses)
            bsum = _batch_summary(buf, cfg.max_steps)
            rsum = _reward_summary(buf)
            line = (
                f"update={u:4d} stage={gate_info['stage']} "
                f"weights={tuple(round(w, 2) for w in gate_info['weights'])} "
                f"len={bsum['mean_length']:6.2f} term={bsum['term_rate']:.3f} "
                f"r_fall={rsum['r_fall_mean']:+.3f}±{rsum['r_fall_std']:.3f} "
                f"r_cross={rsum['r_cross_mean']:+.3f}±{rsum['r_cross_std']:.3f} "
                f"r_relation={rsum['r_relation_mean']:+.3f}±{rsum['r_relation_std']:.3f} "
                f"r_damage={rsum['r_damage_mean']:+.3f}±{rsum['r_damage_std']:.3f} "
                f"policy_loss={stats['policy_loss']:+.5f} "
                f"vloss_r_fall={stats.get('vloss_r_fall', 0.0):.4f} "
                f"vloss_r_cross={stats.get('vloss_r_cross', 0.0):.4f} "
                f"vloss_r_relation={stats.get('vloss_r_relation', 0.0):.4f} "
                f"vloss_r_damage={stats.get('vloss_r_damage', 0.0):.4f} "
                f"kl={stats['approx_kl']:.4f}"
            )

            # 4.7 Eval (deterministic)
            t_eval = 0.0
            if u % cfg.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = cfg.seed + 100_000 + u * 97
                det_bp = actor.to_blueprint(dest_path=str(export_dir))
                eval_jobs = _build_rollout_jobs(
                    env_pb, det_bp, eval_seed,
                    cfg.eval_episodes, max_steps=cfg.max_steps,
                )
                eval_episodes: List[Episode] = rollouter.collect(eval_jobs)
                eval_buf = _PPOBuffer(
                    episodes=eval_episodes,
                    stage_weights=stage_weights,
                    actor=actor,
                    device=device,
                    terminal_fall_penalty=0.0,
                )
                if not eval_buf.is_empty():
                    esum = _batch_summary(eval_buf, cfg.max_steps)
                    line += (
                        f"| len={esum['mean_length']:6.2f}"
                        f" term={esum['term_rate']:.3f}"
                    )
                    # Stage gate decision
                    prev_stage = gate.stage
                    gate_info = gate.assign_from_eval(esum)
                    if gate_info["stage"] != prev_stage:
                        line += f"  [stage {prev_stage}->{gate_info['stage']} {gate_info['reason']}]"
                    # Best-of-run snapshot
                    score = (gate_info["stage"], esum["mean_length"])
                    if score > best_eval:
                        best_eval = score
                        export_actor_policy_artifacts(
                            actor=actor,
                            policy_dir=policy_dir,
                            extra_payload={
                                "algorithm": "ppo_curriculum",
                                "update": u,
                                "stage": gate_info["stage"],
                                "weights": list(gate_info["weights"]),
                                "best_eval_length": esum["mean_length"],
                            },
                        )
                        line += "  [new_best]"
                t_eval = time.perf_counter() - t0

            t_total = time.perf_counter() - t_update_start
            line += (
                f" | time: total={t_total:.1f}s"
                f" export={t_export:.2f}s"
                f" jobs={t_jobs:.2f}s"
                f" rollout={t_rollout:.1f}s"
                f" buffer={t_buffer:.2f}s"
                f" ppo={t_ppo:.2f}s"
                f" eval={t_eval:.1f}s"
            )
            print(line, flush=True)

            # 4.8 Periodic checkpoint
            if u % cfg.eval_interval == 0 or u == 1:
                _save_checkpoint(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actor=actor, critics=critics,
                    actor_optimizer=actor_optimizer, critic_optimizers=critic_optimizers,
                    gate=gate, update=u, best_eval=best_eval, cfg=cfg,
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curriculum PPO trainer with ParallelRollouter.")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--terminal-fall-penalty", type=float, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Short smoke run (max_updates=2, episodes_per_update=8, eval_episodes=4).",
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = CurriculumConfig()
    if args.smoke:
        cfg.max_updates = 2
        cfg.episodes_per_update = 8
        cfg.eval_episodes = 4
        cfg.eval_interval = 1
        cfg.rollout_workers = 2
        cfg.minibatch_size = 64
    if args.max_updates is not None:
        cfg.max_updates = int(args.max_updates)
    if args.episodes_per_update is not None:
        cfg.episodes_per_update = int(args.episodes_per_update)
    if args.rollout_workers is not None:
        cfg.rollout_workers = int(args.rollout_workers)
    if args.terminal_fall_penalty is not None:
        cfg.terminal_fall_penalty = float(args.terminal_fall_penalty)

    run_name = args.run_name or f"curriculum_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    resume_from = Path(args.resume_from) if args.resume_from else None
    train(cfg, run_dir=run_dir, resume_from=resume_from)


if __name__ == "__main__":
    main()
