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
from baseline.humanoid21.common import (
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
        "r1_rewards", "r2_rewards", "r3_rewards",
        "final_obs", "is_terminated", "ep_lengths",
        "r1_sums", "r2_sums", "r3_sums",
    )

    def __init__(
        self,
        episodes: Sequence[Episode],
        stage_weights: Tuple[float, float, float],
        actor: TanhGaussianMLPPolicy,
        device: torch.device,
        terminal_fall_penalty: float,
    ):
        w1, w2, w3 = stage_weights
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        lp_list: List[np.ndarray] = []
        r1_per_step: List[np.ndarray] = []
        r2_per_step: List[np.ndarray] = []
        r3_per_step: List[np.ndarray] = []
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
            r1 = _extract_per_step_scalar(oo, "cross_support", T)
            r2 = _extract_per_step_scalar(oo, "opponent_relation", T)
            r3 = _extract_per_step_scalar(oo, "damage", T)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(acts, dtype=torch.float32, device=device)
            with torch.no_grad():
                lp, _ = actor.evaluate_actions(obs_t, act_t)
            lp_np = lp.cpu().numpy().astype(np.float32)

            obs_list.append(obs)
            act_list.append(acts)
            lp_list.append(lp_np)
            r1_per_step.append(r1)
            r2_per_step.append(r2)
            r3_per_step.append(r3)
            fin_list.append(np.asarray(fin, dtype=np.float32))
            terms.append(bool(ep.is_terminated))
            ep_lens.append(T)

        if not ep_lens:
            print(f"[DEBUG] _PPOBuffer: no valid episodes from {len(episodes)} input episodes", flush=True)
            self.obs = np.zeros((0,), np.float32)
            self.actions = np.zeros((0,), np.float32)
            self.log_probs = np.zeros((0,), np.float32)
            self.r1_rewards = []
            self.r2_rewards = []
            self.r3_rewards = []
            self.final_obs = []
            self.is_terminated = []
            self.ep_lengths = []
            self.r1_sums = []
            self.r2_sums = []
            self.r3_sums = []
            return

        # Store raw reward components separately for multi-critic training
        r1_rew_list: List[np.ndarray] = []
        r2_rew_list: List[np.ndarray] = []
        r3_rew_list: List[np.ndarray] = []
        r1_sums: List[float] = []
        r2_sums: List[float] = []
        r3_sums: List[float] = []

        for i, T in enumerate(ep_lens):
            r1_seg = r1_per_step[i].copy()
            r2_seg = r2_per_step[i].copy()
            r3_seg = r3_per_step[i].copy()
            
            # Apply terminal fall penalty only to r1 (cross_support) component
            if terms[i] and terminal_fall_penalty > 0.0:
                r1_seg[-1] -= terminal_fall_penalty
            
            r1_rew_list.append(r1_seg)
            r2_rew_list.append(r2_seg)
            r3_rew_list.append(r3_seg)
            r1_sums.append(float(r1_seg.sum()))
            r2_sums.append(float(r2_seg.sum()))
            r3_sums.append(float(r3_seg.sum()))

        self.obs = np.concatenate(obs_list, axis=0)
        self.actions = np.concatenate(act_list, axis=0)
        self.log_probs = np.concatenate(lp_list, axis=0)
        self.r1_rewards = r1_rew_list
        self.r2_rewards = r2_rew_list
        self.r3_rewards = r3_rew_list
        self.final_obs = fin_list
        self.is_terminated = terms
        self.ep_lengths = ep_lens
        self.r1_sums = r1_sums
        self.r2_sums = r2_sums
        self.r3_sums = r3_sums

    def __len__(self) -> int:
        return sum(self.ep_lengths)

    def is_empty(self) -> bool:
        return len(self.ep_lengths) == 0


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _ppo_update(
    actor: TanhGaussianMLPPolicy,
    critics: Dict[str, CriticMLP],
    optimizer: torch.optim.Optimizer,
    buf: _PPOBuffer,
    cfg: CurriculumConfig,
    device: torch.device,
    stage_weights: Tuple[float, float, float],
) -> Dict[str, float]:
    # Multi-critic GAE: compute separate advantages for each reward component
    obs_all_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=device)
    
    # Compute values for each critic
    values_all = {}
    for key, critic in critics.items():
        with torch.no_grad():
            values_all[key] = critic(obs_all_t).squeeze(-1).cpu().numpy().astype(np.float32)
    
    # Compute GAE for each reward component
    advs_all = {}
    rets_all = {}
    val_all = {}
    
    for key in ['r1', 'r2', 'r3']:
        advs_list = []
        rets_list = []
        val_list = []
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
            val_list.append(values)
        
        advs_all[key] = np.concatenate(advs_list)
        rets_all[key] = np.concatenate(rets_list)
        val_all[key] = np.concatenate(val_list)
    
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
    
    w1, w2, w3 = stage_weights
    combined_adv = (
        w1 * _normalize_adv(advs_all['r1']) +
        w2 * _normalize_adv(advs_all['r2']) +
        w3 * _normalize_adv(advs_all['r3'])
    )
    adv_t = torch.as_tensor(combined_adv, dtype=torch.float32, device=device)
    
    n = obs_t.shape[0]
    pol_losses: List[float] = []
    val_losses: List[float] = []
    kls: List[float] = []
    early_stop_kl = 0.0
    
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=device)
        early_stop = False
        
        for s in range(0, n, cfg.minibatch_size):
            idx = perm[s : s + cfg.minibatch_size]
            new_lp, entropy = actor.evaluate_actions(obs_t[idx], act_t[idx])
            
            # Compute value losses for all critics
            total_val_loss = 0.0
            for key in ['r1', 'r2', 'r3']:
                new_val = critics[key](obs_t[idx]).squeeze(-1)
                idx_cpu = idx.cpu().numpy()
                ret_val = torch.as_tensor(rets_all[key][idx_cpu], dtype=torch.float32, device=device)
                
                # MSE loss for value function
                val_loss = ((new_val - ret_val) ** 2).mean()
                total_val_loss += val_loss
            
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
            
            # Total loss
            loss = policy_loss + cfg.value_loss_coef * total_val_loss - cfg.entropy_coef * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + [p for c in critics.values() for p in c.parameters()],
                cfg.grad_clip_norm,
            )
            optimizer.step()
            pol_losses.append(float(policy_loss))
            val_losses.append(float(total_val_loss))
        
        if early_stop:
            break
    
    return {
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
        "value_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "early_stop_kl": early_stop_kl,
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


def _reward_summary(buf: _PPOBuffer) -> Dict[str, float]:
    if not buf.ep_lengths:
        return {"r1_mean": 0.0, "r2_mean": 0.0, "r3_mean": 0.0, "ep_reward_mean": 0.0}
    
    # Calculate total episode rewards from components
    ep_rewards = []
    for i in range(len(buf.r1_rewards)):
        total_reward = buf.r1_rewards[i].sum() + buf.r2_rewards[i].sum() + buf.r3_rewards[i].sum()
        ep_rewards.append(total_reward)
    
    return {
        "r1_mean": float(np.mean(buf.r1_sums)),
        "r2_mean": float(np.mean(buf.r2_sums)),
        "r3_mean": float(np.mean(buf.r3_sums)),
        "ep_reward_mean": float(np.mean(ep_rewards)),
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
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
            "optimizer_state_dict": optimizer.state_dict(),
            "gate_stage": gate.stage,
            "gate_stage_training_counts": gate.stage_training_counts,
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
    optimizer: torch.optim.Optimizer,
    gate: CurriculumStageGate,
    cfg: CurriculumConfig,
) -> int:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["actor_state_dict"])
    
    # Handle both old single-critic and new multi-critic checkpoints
    is_legacy_checkpoint = False
    if "critics_state_dict" in payload:
        # New multi-critic format
        for k, v in critics.items():
            if k in payload["critics_state_dict"]:
                v.load_state_dict(payload["critics_state_dict"][k])
            else:
                print(f"[WARNING] Critic {k} not found in checkpoint, using random initialization", flush=True)
    elif "critic_state_dict" in payload:
        # Old single-critic format: load into r1 critic (stage 1)
        critics["r1"].load_state_dict(payload["critic_state_dict"])
        print(f"[INFO] Loaded legacy single-critic weights into r1 critic for stage 1", flush=True)
        is_legacy_checkpoint = True
    else:
        print(f"[WARNING] No critic weights found in checkpoint, using random initialization", flush=True)
    
    # For legacy checkpoints, recreate optimizer due to parameter count mismatch
    if is_legacy_checkpoint:
        print(f"[INFO] Recreating optimizer for multi-critic architecture", flush=True)
        critic_params = []
        for c in critics.values():
            critic_params.extend(list(c.parameters()))
        new_optimizer = torch.optim.Adam(
            list(actor.parameters()) + critic_params,
            lr=cfg.learning_rate,
        )
        # Copy learning rate and other states from old optimizer if possible
        if "optimizer_state_dict" in payload:
            old_opt_state = payload["optimizer_state_dict"]
            if "param_groups" in old_opt_state and len(old_opt_state["param_groups"]) > 0:
                new_optimizer.param_groups[0]["lr"] = old_opt_state["param_groups"][0].get("lr", cfg.learning_rate)
        # Update the reference in the calling function
        optimizer.load_state_dict(new_optimizer.state_dict())
    else:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    
    gate.stage = int(payload.get("gate_stage", 1))
    # Restore stage training counts if available (new format)
    if "gate_stage_training_counts" in payload:
        gate.stage_training_counts = payload["gate_stage_training_counts"]
        print(f"[INFO] Restored stage training counts: {gate.stage_training_counts}", flush=True)
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
    
    # Multi-critic architecture: one critic per reward component
    critics = {
        'r1': CriticMLP(obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim).to(device),
        'r2': CriticMLP(obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim).to(device),
        'r3': CriticMLP(obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim).to(device),
    }
    
    # Optimizer includes all parameters
    critic_params = []
    for c in critics.values():
        critic_params.extend(list(c.parameters()))
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + critic_params,
        lr=cfg.learning_rate,
    )

    gate = CurriculumStageGate(
        max_steps=cfg.max_steps,
        pass_len_ratio=cfg.stage1_pass_len_ratio,
        pass_final_in_zone=cfg.stage2_pass_final_in_zone,
    )

    start_update = 1
    best_eval: tuple = (-1, -float("inf"), -float("inf"))

    # 3. Resume
    if resume_from is not None:
        start_update = _load_checkpoint(
            Path(resume_from), actor=actor, critics=critics,
            optimizer=optimizer, gate=gate, cfg=cfg,
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
            stage_weights: Tuple[float, float, float] = gate_info["weights"]
            target_agent = _agent_from_rollout_seed(rollout_seed)

            buf = _PPOBuffer(
                episodes=episodes,
                stage_weights=stage_weights,
                actor=actor,
                device=device,
                terminal_fall_penalty=cfg.terminal_fall_penalty,
            )
            t_buffer = time.perf_counter() - t0
            if buf.is_empty():
                print(f"update={u} | no episodes for target={target_agent}", flush=True)
                continue

            # 4.5 PPO update
            t0 = time.perf_counter()
            stats = _ppo_update(actor, critics, optimizer, buf, cfg, device, stage_weights)
            # Increment training count for progressive weight scaling
            gate.increment_training_count()
            t_ppo = time.perf_counter() - t0

            # 4.6 Logging
            bsum = _batch_summary(buf, cfg.max_steps)
            rsum = _reward_summary(buf)
            line = (
                f"update={u:4d} target={target_agent} stage={gate_info['stage']} "
                f"weights={tuple(round(w, 2) for w in gate_info['weights'])} "
                f"ep_reward={rsum['ep_reward_mean']:+.4f} "
                f"len={bsum['mean_length']:6.2f} term={bsum['term_rate']:.3f} "
                f"r1={rsum['r1_mean']:+.3f} r2={rsum['r2_mean']:+.3f} r3={rsum['r3_mean']:+.3f} "
                f"policy_loss={stats['policy_loss']:+.5f} "
                f"value_loss={stats['value_loss']:+.5f} "
                f"kl={stats['approx_kl']:.4f}"
            )

            # 4.7 Eval (deterministic)
            t_eval = 0.0
            if u % cfg.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = cfg.seed + 100_000 + u * 97
                eval_target = _agent_from_rollout_seed(eval_seed)
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
                    ersum = _reward_summary(eval_buf)
                    line += (
                        f" | eval target={eval_target}"
                        f" ep_reward={ersum['ep_reward_mean']:+.4f}"
                        f" len={esum['mean_length']:6.2f}"
                        f" term={esum['term_rate']:.3f}"
                    )
                    # Stage gate decision
                    prev_stage = gate.stage
                    gate_info = gate.assign_from_eval(esum)
                    if gate_info["stage"] != prev_stage:
                        line += f"  [stage {prev_stage}->{gate_info['stage']} {gate_info['reason']}]"
                    # Best-of-run snapshot
                    score = (gate_info["stage"], esum["mean_length"], ersum["ep_reward_mean"])
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
                                "best_eval_reward": ersum["ep_reward_mean"],
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
                    actor=actor, critics=critics, optimizer=optimizer,
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
