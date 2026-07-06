"""PPO training loop (v2).

Adapted from framework/training_loop.py for the unified v2 experiment
interface.  Uses CommonParams + PPOParams instead of FrameworkParams.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline.common.policies import export_actor_policy_artifacts
from baseline.common.rollout import Episode, ParallelRollouter

from .experiment import CommonParams, Experiment, PPOParams
from .ppo_trainer import (
    PPOBuffer,
    ppo_update,
    set_seed,
)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    experiment: Experiment,
    cp: CommonParams,
    update: int,
    best_eval: tuple,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "ppo",
            "actor_state_dict": actor.state_dict(),
            "critics_state_dict": {k: v.state_dict() for k, v in critics.items()},
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizers_state_dict": {
                k: v.state_dict() for k, v in critic_optimizers.items()
            },
            "experiment_name": cp.name,
            "reward_keys": cp.reward_keys,
            "scheduler_state": experiment.scheduler_state(),
            "training_state": experiment.training_state(),
            "update": update,
            "best_eval": best_eval,
        },
        ckpt_path,
    )


def load_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    experiment: Experiment,
    cp: CommonParams,
    pp: PPOParams,
    load_experiment_state: bool = False,
) -> int:
    """Load model weights and optimizer states from checkpoint.

    Returns
    -------
    start_update : int
        The update number to resume from.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    actor.load_state_dict(payload["actor_state_dict"])

    saved = payload["critics_state_dict"]
    for k, v in critics.items():
        if k in saved:
            v.load_state_dict(saved[k])
        else:
            print(f"[checkpoint] critic '{k}' not in checkpoint -> fresh init", flush=True)

    try:
        actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    except RuntimeError as e:
        print(f"[checkpoint] Actor optimizer state mismatch: {e}", flush=True)

    saved_crit_opt = payload["critic_optimizers_state_dict"]
    for k, opt in critic_optimizers.items():
        if k in saved_crit_opt:
            try:
                opt.load_state_dict(saved_crit_opt[k])
            except RuntimeError as e:
                print(f"[checkpoint] Critic {k} optimizer state mismatch: {e}", flush=True)

    # Force align optimizer learning rate and policy standard deviation boundaries to currently configured experiment config
    for pg in actor_optimizer.param_groups:
        pg["lr"] = cp.learning_rate
    actor.log_std_min = float(pp.log_std_min)
    print(f"[checkpoint] Force aligned actor optimizer LR to {cp.learning_rate:.2e} and log_std_min to {pp.log_std_min}", flush=True)

    if load_experiment_state:
        saved_exp = payload.get("experiment_name", "")
        if saved_exp == cp.name:
            experiment.load_scheduler_state(payload.get("scheduler_state", {}))
            experiment.load_training_state(payload.get("training_state", {}))
            # Restore optimizer LR from experiment
            for pg in actor_optimizer.param_groups:
                pg["lr"] = cp.learning_rate
            print(
                f"[checkpoint] restored LR={cp.learning_rate:.2e}",
                flush=True,
            )
        else:
            print(
                f"[checkpoint] experiment changed ({saved_exp} -> {cp.name}), "
                f"resetting scheduler and training state",
                flush=True,
            )

    return int(payload.get("update", 0))


# ---------------------------------------------------------------------------
# Video recording helper
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
# Train (PPO)
# ---------------------------------------------------------------------------

def train_ppo(
    experiment: Experiment,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
    use_confidence: bool = True,
) -> None:
    """PPO training loop using the v2 unified experiment interface."""
    # Attach run_dir to the experiment so it can discover saved models for opponent pool.
    experiment.run_dir = run_dir
    cp = experiment.common_params()
    pp = experiment.ppo_params()

    # Kill entire process group on SIGTERM/SIGINT
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(cp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Build models (delegated to experiment)
    actor = experiment.build_actor(device)
    critics = {
        key: experiment.build_v_critic(key, device)
        for key in cp.reward_keys
    }

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cp.learning_rate)
    critic_optimizers = {
        key: torch.optim.Adam(critics[key].parameters(), lr=cp.critic_learning_rate)
        for key in cp.reward_keys
    }

    # 3. Initialize weights from experiment
    weights = experiment.initial_weights()

    start_update = 1
    best_esum: Dict[str, float] = {}

    # 4. Resume
    if resume_from is not None:
        start_update = load_checkpoint(
            Path(resume_from),
            actor=actor,
            critics=critics,
            actor_optimizer=actor_optimizer,
            critic_optimizers=critic_optimizers,
            experiment=experiment,
            cp=cp,
            pp=pp,
        )
        print(
            f"[resume] loaded from {resume_from}, starting at update={start_update}",
            flush=True,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    ckpt_dir = run_dir / "checkpoints"
    video_dir = run_dir / "videos"
    print(f"run_dir={run_dir} experiment={cp.name} algo=ppo", flush=True)

    # Plateau detection state — sliding window of reward stats per key
    PLATEAU_WINDOW = 30
    PLATEAU_THRESHOLD = 0.05  # 5% relative波动才算 plateau
    reward_stat_history: Dict[str, deque] = {
        key: deque(maxlen=PLATEAU_WINDOW) for key in cp.reward_keys
    }

    # Video recording state
    n_evals_done = 0
    last_video_proc: Optional[subprocess.Popen] = None
    video_dir.mkdir(parents=True, exist_ok=True)
    video_env_bp_path = video_dir / "video_env_blueprint.yaml"

    # Normalize weights for display
    def _norm_weights(w: Tuple[float, ...]) -> Tuple[float, ...]:
        total = sum(w)
        if total <= 0:
            return tuple(1.0 if i == 0 else 0.0 for i in range(len(w)))
        return tuple(v / total for v in w)

    # 5. Training loop
    print(
        f"[DEBUG] rollout_workers={cp.rollout_workers} "
        f"episodes_per_update={cp.episodes_per_update} "
        f"update_epochs={pp.update_epochs} "
        f"minibatch_size={pp.minibatch_size} "
        f"reward_keys={cp.reward_keys}",
        flush=True,
    )
    with ParallelRollouter(num_workers=cp.rollout_workers) as rollouter:
        for u in range(start_update, cp.max_updates + 1):
            t_update_start = time.perf_counter()

            # 5.1 Export policy blueprint (stochastic for training rollouts)
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(dest_path=str(export_dir))
            policy_bp.config["stochastic"] = True
            t_export = time.perf_counter() - t0

            # 5.2 Prepare rollout jobs
            t0 = time.perf_counter()
            rollout_seed = cp.seed + u * cp.episodes_per_update
            jobs = experiment.build_rollout_jobs(policy_bp, rollout_seed)
            t_jobs = time.perf_counter() - t0

            # 5.3 Rollout
            t0 = time.perf_counter()
            episodes: List[Episode] = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            # 5.4 Build PPO buffer
            t0 = time.perf_counter()
            norm_weights = _norm_weights(weights)
            buf = PPOBuffer(
                episodes=episodes,
                stage_weights=norm_weights,
                actor=actor,
                device=device,
                experiment=experiment,
                common_params=cp,
            )
            t_buffer = time.perf_counter() - t0

            # 5.5 PPO update
            t0 = time.perf_counter()
            stats = ppo_update(
                actor=actor,
                critics=critics,
                actor_optimizer=actor_optimizer,
                critic_optimizers=critic_optimizers,
                buf=buf,
                reward_keys=cp.reward_keys,
                gammas=cp.gammas,
                gae_lambda=pp.gae_lambda,
                clip_eps=pp.clip_eps,
                entropy_coef=pp.entropy_coef,
                grad_clip_norm=cp.grad_clip_norm,
                target_kl=pp.target_kl,
                update_epochs=pp.update_epochs,
                device=device,
                stage_weights=norm_weights,
                minibatch_size=pp.minibatch_size,
                experiment=experiment,
                use_confidence=use_confidence,
            )
            t_ppo = time.perf_counter() - t0

            # --- Eval (Run BEFORE logging to synchronize weights & raw stats) ---
            esum = None
            t_eval = 0.0
            if u % cp.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = cp.seed + 100_000 + u * 97
                det_bp = actor.to_blueprint(dest_path=str(export_dir))
                eval_jobs = experiment.build_eval_jobs(det_bp, eval_seed)
                eval_episodes: List[Episode] = rollouter.collect(eval_jobs)
                
                eval_buf = PPOBuffer(
                    episodes=eval_episodes,
                    stage_weights=norm_weights,
                    actor=actor,
                    device=device,
                    experiment=experiment,
                    common_params=cp,
                )
                if not eval_buf.is_empty():
                    esum = eval_buf.batch_summary()

                    ep_parts = [f"{k}={v:.3f}" for k, v in esum.items()]
                    eval_line = f"[eval {u:4d}] [ep " + " ".join(ep_parts) + "]"

                    # Update weights from experiment scheduler
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
                        # Use actor.to_blueprint() if it handles custom export
                        # (e.g. HybridActor), otherwise use standard export.
                        if hasattr(actor, "export_policy_artifacts"):
                            actor.export_policy_artifacts(
                                policy_dir=policy_dir,
                                extra_payload={
                                    "algorithm": "ppo_curriculum_v2",
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
                                    "algorithm": "ppo_curriculum_v2",
                                    "experiment": cp.name,
                                    "update": u,
                                    "weights": list(norm_weights),
                                    "best_eval_esum": best_esum,
                                },
                            )
                        eval_line += "  [new_best]"

                    print(eval_line, flush=True)
                t_eval = time.perf_counter() - t0

                # 5.7.1 Video render
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

            # 5.6 Logging
            bsum = buf.batch_summary()
            rsum = buf.reward_summary()
            sinfo = experiment.scheduler_info()

            # --- High Observability Logging ---
            # 1. Header & Scheduler Info
            sched_str = " ".join([f"weights={tuple(round(w, 2) for w in norm_weights)}"] + ([f"{k}={v}" for k, v in sinfo.items()] if sinfo else []))
            print(f"[update {u:4d}] [{sched_str}]", flush=True)

            # 2. Rollout / Episode metrics
            ep_len_str = f"len={stats.get('ep_len_mean', 0.0):.1f} (min={stats.get('ep_len_min', 0.0):.1f}, max={stats.get('ep_len_max', 0.0):.1f})"
            ep_metrics_str = " ".join([f"{k}={v:.3f}" for k, v in bsum.items()]) if bsum else ""
            print(f"  [Rollout] {ep_len_str} | {ep_metrics_str}", flush=True)

            # 3. Policy & Optimization stats
            policy_loss = stats.get('policy_loss', 0.0)
            entropy = stats.get('entropy', 0.0)
            std_mean = stats.get('std_mean', 0.0)
            std_min = stats.get('std_min', 0.0)
            std_max = stats.get('std_max', 0.0)
            epochs_done = stats.get('epochs_done', 0)
            approx_kl = stats.get('approx_kl', 0.0)
            max_kl = stats.get('max_kl', 0.0)
            early_stop_kl = stats.get('early_stop_kl', 0.0)
            
            print(f"  [Policy ] loss={policy_loss:.4f} entropy={entropy:.2f} std={std_mean:.3f} (min={std_min:.3f}, max={std_max:.3f})", flush=True)
            print(f"  [PPO Opt] epochs={epochs_done}/{pp.update_epochs} kl_mean={approx_kl:.4f} kl_max={max_kl:.4f} (stop_kl={early_stop_kl:.4f})", flush=True)

            # 4. Critics details
            value_loss = stats.get('value_loss', 0.0)
            print(f"  [Critics] total_vloss={value_loss:.4f}", flush=True)
            for key in cp.reward_keys:
                mk, sk = f"{key}_mean", f"{key}_std"
                rew_flow = f"{rsum.get(mk, 0.0):+.3f}±{rsum.get(sk, 0.0):.3f}"
                vloss_key = f"vloss_{key}"
                ev_key = f"ev_{key}"
                adv_std_key = f"adv_std_{key}"
                conf_key = f"confidence_{key}"
                # Plateau diagnostics
                mu_k = rsum.get(mk, 0.0)
                sigma_k = rsum.get(sk, 0.0)
                reward_stat_history[key].append((mu_k, sigma_k))
                plateau_factor = 1.0
                cv_mu = 0.0
                cv_sigma = 0.0
                if len(reward_stat_history[key]) >= 5:
                    hist = reward_stat_history[key]
                    mus = np.array([h[0] for h in hist])
                    sigmas = np.array([h[1] for h in hist])
                    cv_mu = float(np.std(mus) / (np.abs(np.mean(mus)) + 1e-8))
                    cv_sigma = float(np.std(sigmas) / (np.abs(np.mean(sigmas)) + 1e-8))
                    stability = max(cv_mu, cv_sigma)
                    if stability < PLATEAU_THRESHOLD:
                        plateau_factor = stability / PLATEAU_THRESHOLD
                print(f"    - {key:<12} | reward={rew_flow} | val_loss={stats.get(vloss_key, 0.0):.4f} | explained_var={stats.get(ev_key, 0.0):+.3f} | confidence={stats.get(conf_key, 1.0):.3f} | plateau={plateau_factor:.3f} (cv_mu={cv_mu:.3f} cv_sig={cv_sigma:.3f}) | adv_std={stats.get(adv_std_key, 0.0):.2f}", flush=True)
                stats[f"plateau_factor_{key}"] = plateau_factor
                stats[f"cv_mu_{key}"] = cv_mu
                stats[f"cv_sigma_{key}"] = cv_sigma

            # 5.6b Machine-readable raw logging for monitoring script
            t_total = time.perf_counter() - t_update_start
            raw_log_dict = {
                "update": u,
                "algo": "ppo",
                "weights": list(norm_weights) if norm_weights else [],
                "sinfo": sinfo,
                "bsum": bsum,
                "rsum": rsum,
                "stats": stats,
                "timing": {
                    "total": round(t_total, 2),
                    "export": round(t_export, 2),
                    "jobs": round(t_jobs, 2),
                    "rollout": round(t_rollout, 2),
                    "buffer": round(t_buffer, 2),
                    "ppo": round(t_ppo, 2),
                    "eval": round(t_eval, 2),
                },
            }
            if esum is not None:
                raw_log_dict["esum"] = esum
            print(f"__RAW_STATS__ {json.dumps(raw_log_dict)}", flush=True)

            # --- Timing (human-readable) ---
            print(
                f"  | time: total={t_total:.1f}s"
                f" export={t_export:.2f}s"
                f" jobs={t_jobs:.2f}s"
                f" rollout={t_rollout:.1f}s"
                f" buffer={t_buffer:.2f}s"
                f" ppo={t_ppo:.2f}s"
                f" eval={t_eval:.1f}s",
                flush=True,
            )

            # 5.8 Periodic checkpoint
            if u % cp.eval_interval == 0 or u == 1:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actor=actor,
                    critics=critics,
                    actor_optimizer=actor_optimizer,
                    critic_optimizers=critic_optimizers,
                    experiment=experiment,
                    cp=cp,
                    update=u,
                    best_eval=best_esum,
                )
