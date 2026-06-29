"""SAC training loop (v2).

Off-policy training loop using Soft Actor-Critic.  Collects episodes via
ParallelRollouter, adds transitions to a ReplayBuffer, and performs
SAC gradient updates.  Shares evaluation, curriculum, checkpointing,
and logging patterns with ppo_loop.py.
"""
from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline.common.policies import export_actor_policy_artifacts
from baseline.common.rollout import Episode, ParallelRollouter

from .experiment import CommonParams, Experiment, SACParams
from .ppo_trainer import set_seed
from .sac_trainer import (
    QCriticMLP,
    ReplayBuffer,
    sac_update,
    soft_copy,
)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    q1: torch.nn.Module,
    q2: torch.nn.Module,
    q1_target: torch.nn.Module,
    q2_target: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    q1_optimizer: torch.optim.Optimizer,
    q2_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
    experiment: Experiment,
    cp: CommonParams,
    update: int,
    total_transitions: int,
    best_eval: dict,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "sac",
        "actor_state_dict": actor.state_dict(),
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "q1_target_state_dict": q1_target.state_dict(),
        "q2_target_state_dict": q2_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q1_optimizer_state_dict": q1_optimizer.state_dict(),
        "q2_optimizer_state_dict": q2_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "experiment_name": cp.name,
        "reward_keys": cp.reward_keys,
        "scheduler_state": experiment.scheduler_state(),
        "training_state": experiment.training_state(),
        "update": update,
        "total_transitions": total_transitions,
        "best_eval": best_eval,
    }
    if alpha_optimizer is not None:
        payload["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    q1: torch.nn.Module,
    q2: torch.nn.Module,
    q1_target: torch.nn.Module,
    q2_target: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    q1_optimizer: torch.optim.Optimizer,
    q2_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
    experiment: Experiment,
    cp: CommonParams,
    load_experiment_state: bool = False,
) -> Tuple[int, int]:
    """Load checkpoint and return (start_update, total_transitions)."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    actor.load_state_dict(payload["actor_state_dict"])
    q1.load_state_dict(payload["q1_state_dict"])
    q2.load_state_dict(payload["q2_state_dict"])
    q1_target.load_state_dict(payload["q1_target_state_dict"])
    q2_target.load_state_dict(payload["q2_target_state_dict"])

    try:
        actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    except RuntimeError as e:
        print(f"[checkpoint] Actor optimizer mismatch: {e}", flush=True)
    try:
        q1_optimizer.load_state_dict(payload["q1_optimizer_state_dict"])
    except RuntimeError as e:
        print(f"[checkpoint] Q1 optimizer mismatch: {e}", flush=True)
    try:
        q2_optimizer.load_state_dict(payload["q2_optimizer_state_dict"])
    except RuntimeError as e:
        print(f"[checkpoint] Q2 optimizer mismatch: {e}", flush=True)

    saved_log_alpha = payload.get("log_alpha")
    if saved_log_alpha is not None:
        log_alpha.data.copy_(saved_log_alpha)

    if alpha_optimizer is not None and "alpha_optimizer_state_dict" in payload:
        try:
            alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])
        except RuntimeError as e:
            print(f"[checkpoint] Alpha optimizer mismatch: {e}", flush=True)

    # Force align optimizer LR
    for pg in actor_optimizer.param_groups:
        pg["lr"] = cp.learning_rate
    print(f"[checkpoint] Force aligned actor optimizer LR to {cp.learning_rate:.2e}", flush=True)

    if load_experiment_state:
        saved_exp = payload.get("experiment_name", "")
        if saved_exp == cp.name:
            experiment.load_scheduler_state(payload.get("scheduler_state", {}))
            experiment.load_training_state(payload.get("training_state", {}))

    start_update = int(payload.get("update", 0))
    total_transitions = int(payload.get("total_transitions", 0))
    return start_update, total_transitions


# ---------------------------------------------------------------------------
# Video recording helper (shared with PPO)
# ---------------------------------------------------------------------------

def spawn_video_render(
    *,
    env_blueprint: str,
    policy_blueprint: Path,
    video_path: Path,
    seed: int,
    log_path: Path,
) -> Optional[subprocess.Popen]:
    """Spawn a non-blocking ``round_runner`` subprocess to render one match."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "envs.framework.round_runner",
        "--env-blueprint", str(env_blueprint),
        "--policy-a-blueprint", str(policy_blueprint),
        "--policy-b-blueprint", str(policy_blueprint),
        "--video", str(video_path),
        "--seed", str(seed),
    ]
    try:
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc
    except Exception as e:
        print(f"[WARN] Failed to spawn video render: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Eval helper — compute episode metrics from collected eval episodes
# ---------------------------------------------------------------------------

def _eval_summary(
    episodes: List[Episode],
    experiment: Experiment,
) -> Optional[Dict[str, float]]:
    """Compute batch-level eval summary from episodes."""
    metrics_list: List[Dict[str, float]] = []
    lengths: List[int] = []
    for ep in episodes:
        ep_target = str(ep.episode_options.get("agent_id", "robot_a"))
        acts = ep.actions.get(ep_target)
        if acts is None:
            continue
        T = int(acts.shape[0])
        if T == 0:
            continue
        metrics_list.append(experiment.compute_episode_metrics(ep))
        lengths.append(T)

    if not metrics_list:
        return None

    result: Dict[str, float] = {"mean_length": float(np.mean(lengths))}
    for k in metrics_list[0].keys():
        result[k] = float(np.mean([m.get(k, 0.0) for m in metrics_list]))
    return result


# ---------------------------------------------------------------------------
# Train (SAC)
# ---------------------------------------------------------------------------

def train_sac(
    experiment: Experiment,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
) -> None:
    """SAC training loop using the v2 unified experiment interface."""
    experiment.run_dir = run_dir
    cp = experiment.common_params()
    sp = experiment.sac_params()

    # Kill entire process group on SIGTERM/SIGINT
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(cp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    actor = experiment.build_actor(device)

    # SAC uses a single combined Q for all reward components
    # (rewards are combined via stage weights in the replay buffer)
    q1 = experiment.build_q_critic("combined", device)
    q2 = experiment.build_q_critic("combined", device)
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)

    # Freeze target networks
    for p in q1_target.parameters():
        p.requires_grad = False
    for p in q2_target.parameters():
        p.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cp.learning_rate)
    q1_optimizer = torch.optim.Adam(q1.parameters(), lr=cp.critic_learning_rate)
    q2_optimizer = torch.optim.Adam(q2.parameters(), lr=cp.critic_learning_rate)

    # Entropy temperature
    log_alpha = torch.tensor(
        np.log(sp.init_alpha), dtype=torch.float32, device=device, requires_grad=True,
    )
    alpha_optimizer: Optional[torch.optim.Optimizer] = None
    if sp.auto_alpha:
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=cp.learning_rate)

    # Replay buffer
    replay_buffer = ReplayBuffer(sp.replay_buffer_size, cp.obs_dim, cp.action_dim)

    # Combined gamma (use mean of per-component gammas for the single Q)
    gamma = float(np.mean(list(cp.gammas.values())))

    # Curriculum weights
    weights = experiment.initial_weights()

    start_update = 1
    total_transitions = 0
    best_esum: Dict[str, float] = {}

    # Resume
    if resume_from is not None:
        start_update, total_transitions = load_checkpoint(
            Path(resume_from),
            actor=actor,
            q1=q1, q2=q2,
            q1_target=q1_target, q2_target=q2_target,
            actor_optimizer=actor_optimizer,
            q1_optimizer=q1_optimizer,
            q2_optimizer=q2_optimizer,
            log_alpha=log_alpha,
            alpha_optimizer=alpha_optimizer,
            experiment=experiment,
            cp=cp,
        )
        print(
            f"[resume] loaded from {resume_from}, starting at update={start_update}, "
            f"transitions={total_transitions}",
            flush=True,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    ckpt_dir = run_dir / "checkpoints"
    video_dir = run_dir / "videos"
    print(f"run_dir={run_dir} experiment={cp.name} algo=sac", flush=True)

    # Video recording state
    n_evals_done = 0
    last_video_proc: Optional[subprocess.Popen] = None
    video_dir.mkdir(parents=True, exist_ok=True)
    video_env_bp_path = video_dir / "video_env_blueprint.yaml"

    # Normalize weights
    def _norm_weights(w: Tuple[float, ...]) -> Tuple[float, ...]:
        total = sum(w)
        if total <= 0:
            return tuple(1.0 if i == 0 else 0.0 for i in range(len(w)))
        return tuple(v / total for v in w)

    print(
        f"[DEBUG] rollout_workers={cp.rollout_workers} "
        f"episodes_per_update={cp.episodes_per_update} "
        f"replay_buffer_size={sp.replay_buffer_size} "
        f"batch_size={sp.batch_size} "
        f"warmup_steps={sp.warmup_steps} "
        f"updates_per_step={sp.updates_per_step} "
        f"tau={sp.tau} gamma={gamma:.4f} "
        f"reward_keys={cp.reward_keys}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    with ParallelRollouter(num_workers=cp.rollout_workers) as rollouter:
        for u in range(start_update, cp.max_updates + 1):
            t_update_start = time.perf_counter()
            norm_weights = _norm_weights(weights)

            # 1. Export policy blueprint (stochastic for training rollouts)
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(dest_path=str(export_dir))
            policy_bp.config["stochastic"] = True
            t_export = time.perf_counter() - t0

            # 2. Collect episodes
            t0 = time.perf_counter()
            rollout_seed = cp.seed + u * cp.episodes_per_update
            jobs = experiment.build_rollout_jobs(policy_bp, rollout_seed)
            episodes: List[Episode] = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            # 3. Add episodes to replay buffer & compute batch metrics
            t0 = time.perf_counter()
            episode_metrics: List[Dict[str, float]] = []
            episode_lengths: List[int] = []
            transitions_added = 0
            reward_accum: Dict[str, List[float]] = {k: [] for k in cp.reward_keys}

            for ep in episodes:
                ep_target = str(ep.episode_options.get("agent_id", "robot_a"))
                acts = ep.actions.get(ep_target)
                if acts is None:
                    continue
                T = int(acts.shape[0])
                if T == 0:
                    continue

                episode_metrics.append(experiment.compute_episode_metrics(ep))
                episode_lengths.append(T)

                # Compute per-component reward stats for logging
                reward_dict = experiment.extract_rewards(ep)
                for key in cp.reward_keys:
                    r = reward_dict.get(key, np.zeros(T, dtype=np.float32))
                    reward_accum[key].append(float(r.mean()))

                n = replay_buffer.add_episode(ep, experiment, norm_weights, cp)
                transitions_added += n

            total_transitions += transitions_added
            t_buffer = time.perf_counter() - t0

            # Batch summary
            bsum: Dict[str, float] = {"mean_length": 0.0}
            if episode_lengths:
                bsum["mean_length"] = float(np.mean(episode_lengths))
                if episode_metrics:
                    for k in episode_metrics[0].keys():
                        bsum[k] = float(np.mean([m.get(k, 0.0) for m in episode_metrics]))

            # Reward summary
            rsum: Dict[str, float] = {}
            for key in cp.reward_keys:
                vals = reward_accum[key]
                rsum[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
                rsum[f"{key}_std"] = float(np.std(vals)) if vals else 0.0

            # 4. SAC gradient updates
            t0 = time.perf_counter()
            sac_stats_accum: Dict[str, List[float]] = {}
            n_updates = 0

            if replay_buffer.size >= sp.warmup_steps:
                # Do updates_per_step * transitions_added gradient steps
                n_gradient_steps = max(1, sp.updates_per_step * transitions_added)
                for _ in range(n_gradient_steps):
                    batch = replay_buffer.sample(sp.batch_size, device)
                    step_stats = sac_update(
                        actor=actor,
                        q1=q1, q2=q2,
                        q1_target=q1_target, q2_target=q2_target,
                        log_alpha=log_alpha,
                        target_entropy=sp.target_entropy,
                        actor_optimizer=actor_optimizer,
                        q1_optimizer=q1_optimizer,
                        q2_optimizer=q2_optimizer,
                        alpha_optimizer=alpha_optimizer,
                        batch=batch,
                        gamma=gamma,
                        tau=sp.tau,
                        grad_clip_norm=cp.grad_clip_norm,
                    )
                    for k, v in step_stats.items():
                        sac_stats_accum.setdefault(k, []).append(v)
                    n_updates += 1
            else:
                print(
                    f"  [warmup] buffer_size={replay_buffer.size}/{sp.warmup_steps}, "
                    f"skipping updates",
                    flush=True,
                )

            t_sac = time.perf_counter() - t0

            # Aggregate SAC stats
            stats: Dict[str, Any] = {}
            for k, vals in sac_stats_accum.items():
                stats[k] = float(np.mean(vals))
            stats["n_updates"] = n_updates
            stats["buffer_size"] = replay_buffer.size
            stats["total_transitions"] = total_transitions
            stats["transitions_added"] = transitions_added
            stats["ep_len_mean"] = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            stats["ep_len_min"] = float(np.min(episode_lengths)) if episode_lengths else 0.0
            stats["ep_len_max"] = float(np.max(episode_lengths)) if episode_lengths else 0.0
            stats["n_episodes"] = len(episode_lengths)

            # --- Eval ---
            esum = None
            t_eval = 0.0
            if u % cp.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = cp.seed + 100_000 + u * 97
                det_bp = actor.to_blueprint(dest_path=str(export_dir))
                eval_jobs = experiment.build_eval_jobs(det_bp, eval_seed)
                eval_episodes: List[Episode] = rollouter.collect(eval_jobs)
                esum = _eval_summary(eval_episodes, experiment)

                if esum is not None:
                    ep_parts = [f"{k}={v:.3f}" for k, v in esum.items()]
                    eval_line = f"[eval {u:4d}] [ep " + " ".join(ep_parts) + "]"

                    # Update curriculum weights
                    prev_weights = weights
                    weights = experiment.next_weights(esum, weights)
                    norm_weights = _norm_weights(weights)
                    if weights != prev_weights:
                        eval_line += (
                            f"  [weights {tuple(round(w, 2) for w in _norm_weights(prev_weights))}"
                            f" -> {tuple(round(w, 2) for w in norm_weights)}]"
                        )

                    # Best-of-run snapshot
                    if experiment.compare_eval(esum, best_esum):
                        best_esum = esum
                        if hasattr(actor, "export_policy_artifacts"):
                            actor.export_policy_artifacts(
                                policy_dir=policy_dir,
                                extra_payload={
                                    "algorithm": "sac_curriculum_v2",
                                    "experiment": cp.name,
                                    "update": u,
                                    "weights": list(norm_weights),
                                    "best_eval_esum": best_esum,
                                },
                            )
                        else:
                            export_actor_policy_artifacts(
                                actor=actor,
                                policy_dir=policy_dir,
                                extra_payload={
                                    "algorithm": "sac_curriculum_v2",
                                    "experiment": cp.name,
                                    "update": u,
                                    "weights": list(norm_weights),
                                    "best_eval_esum": best_esum,
                                },
                            )
                        eval_line += "  [new_best]"

                    print(eval_line, flush=True)
                t_eval = time.perf_counter() - t0

                # Video render
                n_evals_done += 1
                if (
                    cp.video_eval_interval > 0
                    and n_evals_done % cp.video_eval_interval == 0
                ):
                    if last_video_proc is not None and last_video_proc.poll() is None:
                        print(f"  [video_skip:prev_running]", flush=True)
                    else:
                        policy_bp_path = export_dir / "policy_blueprint.yaml"
                        video_path = video_dir / f"u{u:05d}.mp4"
                        log_path = video_dir / f"u{u:05d}.log"
                        experiment.video_env_blueprint().save(video_env_bp_path)
                        last_video_proc = spawn_video_render(
                            env_blueprint=video_env_bp_path,
                            policy_blueprint=policy_bp_path,
                            video_path=video_path,
                            seed=eval_seed,
                            log_path=log_path,
                        )
                        if last_video_proc is not None:
                            print(f"  [video:{video_path.name}]", flush=True)

            # --- Logging ---
            sinfo = experiment.scheduler_info()

            # 1. Header
            sched_str = " ".join(
                [f"weights={tuple(round(w, 2) for w in norm_weights)}"]
                + ([f"{k}={v}" for k, v in sinfo.items()] if sinfo else [])
            )
            print(f"[update {u:4d}] [{sched_str}]", flush=True)

            # 2. Rollout
            ep_len_str = (
                f"len={stats.get('ep_len_mean', 0.0):.1f} "
                f"(min={stats.get('ep_len_min', 0.0):.1f}, "
                f"max={stats.get('ep_len_max', 0.0):.1f})"
            )
            ep_metrics_str = " ".join([f"{k}={v:.3f}" for k, v in bsum.items()])
            print(f"  [Rollout] {ep_len_str} | {ep_metrics_str}", flush=True)

            # 3. SAC stats
            alpha_val = stats.get("alpha", sp.init_alpha)
            q1_loss = stats.get("q1_loss", 0.0)
            q2_loss = stats.get("q2_loss", 0.0)
            actor_loss = stats.get("actor_loss", 0.0)
            q_target_mean = stats.get("q_target_mean", 0.0)
            log_prob_mean = stats.get("log_prob_mean", 0.0)
            print(
                f"  [SAC    ] alpha={alpha_val:.4f} actor_loss={actor_loss:.4f} "
                f"q1_loss={q1_loss:.4f} q2_loss={q2_loss:.4f} "
                f"q_target={q_target_mean:.3f} log_prob={log_prob_mean:.2f}",
                flush=True,
            )
            print(
                f"  [Buffer ] size={replay_buffer.size} added={transitions_added} "
                f"updates={n_updates}",
                flush=True,
            )

            # 4. Per-reward summary
            for key in cp.reward_keys:
                mk, sk = f"{key}_mean", f"{key}_std"
                rew_flow = f"{rsum.get(mk, 0.0):+.3f}±{rsum.get(sk, 0.0):.3f}"
                print(f"    - {key:<12} | reward={rew_flow}", flush=True)

            # 5. Machine-readable raw log
            t_total = time.perf_counter() - t_update_start
            raw_log_dict = {
                "update": u,
                "algo": "sac",
                "weights": list(norm_weights),
                "sinfo": sinfo,
                "bsum": bsum,
                "rsum": rsum,
                "stats": stats,
                "timing": {
                    "total": round(t_total, 2),
                    "export": round(t_export, 2),
                    "rollout": round(t_rollout, 2),
                    "buffer": round(t_buffer, 2),
                    "sac": round(t_sac, 2),
                    "eval": round(t_eval, 2),
                },
            }
            if esum is not None:
                raw_log_dict["esum"] = esum
            print(f"__RAW_STATS__ {json.dumps(raw_log_dict)}", flush=True)

            # Timing
            print(
                f"  | time: total={t_total:.1f}s"
                f" export={t_export:.2f}s"
                f" rollout={t_rollout:.1f}s"
                f" buffer={t_buffer:.2f}s"
                f" sac={t_sac:.2f}s"
                f" eval={t_eval:.1f}s",
                flush=True,
            )

            # Checkpoint
            if u % cp.eval_interval == 0 or u == 1:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actor=actor,
                    q1=q1, q2=q2,
                    q1_target=q1_target, q2_target=q2_target,
                    actor_optimizer=actor_optimizer,
                    q1_optimizer=q1_optimizer,
                    q2_optimizer=q2_optimizer,
                    log_alpha=log_alpha,
                    alpha_optimizer=alpha_optimizer,
                    experiment=experiment,
                    cp=cp,
                    update=u,
                    total_transitions=total_transitions,
                    best_eval=best_esum,
                )
