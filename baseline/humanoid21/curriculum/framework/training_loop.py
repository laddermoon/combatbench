"""Unified curriculum training loop.

Contains ``TrainConfig``, ``train()``, checkpoint save/load, and
video rendering — generic over ``ExperimentConfig``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline.common.policies import (
    CriticMLP,
    TanhGaussianMLPPolicy,
    export_actor_policy_artifacts,
)
from baseline.common.rollout import Episode, ParallelRollouter
from envs.framework.policy import PolicyBlueprint

from .config import ExperimentConfig
from .ppo_trainer import (
    PPOBuffer,
    ppo_update,
    set_seed,
)

# ---------------------------------------------------------------------------
# Env-side constants
# ---------------------------------------------------------------------------
CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 10.0

CURRICULUM_TERMINAL_FALL_PENALTY = float(
    os.environ.get("CURRICULUM_TERMINAL_FALL_PENALTY", "1.0")
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
    experiment: ExperimentConfig,
    update: int,
    best_eval: tuple,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critics_state_dict": {k: v.state_dict() for k, v in critics.items()},
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizers_state_dict": {
                k: v.state_dict() for k, v in critic_optimizers.items()
            },
            "experiment_name": experiment.name,
            "reward_keys": experiment.reward_keys,
            "scheduler_state": experiment.scheduler_state(),
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
    experiment: ExperimentConfig,
    load_experiment_state: bool = False,
) -> int:
    """Load model weights and optimizer states from checkpoint.

    Parameters
    ----------
    load_experiment_state : bool
        If True, restore the experiment scheduler state (stage, phase, etc.)
        so the curriculum resumes from where it left off.  If False, the
        experiment keeps its default initial state.

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

    if load_experiment_state:
        saved_exp = payload.get("experiment_name", "")
        if saved_exp == experiment.name:
            experiment.load_scheduler_state(payload.get("scheduler_state", {}))
        else:
            print(
                f"[checkpoint] experiment changed ({saved_exp} -> {experiment.name}), "
                f"resetting scheduler",
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
# Train
# ---------------------------------------------------------------------------

def train(
    experiment: ExperimentConfig,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
) -> None:
    # Kill entire process group on SIGTERM/SIGINT
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load init policy blueprint.
    # The env blueprint lifecycle is owned by the experiment (see
    # ExperimentConfig.build_rollout_jobs / current_env_blueprint), so the
    # train loop never loads or holds an env blueprint itself.
    blueprint_dir = Path(__file__).resolve().parent.parent.parent / "blueprints"
    init_policy_bp = PolicyBlueprint.load(blueprint_dir / "init_policy.yaml")

    # 2. Build models
    actor: TanhGaussianMLPPolicy = init_policy_bp.build()
    actor = actor.to(device)

    critics = {
        key: CriticMLP(obs_dim=experiment.obs_dim, hidden_dim=experiment.critic_hidden_dim).to(device)
        for key in experiment.reward_keys
    }

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=experiment.learning_rate)
    critic_optimizers = {
        key: torch.optim.Adam(critics[key].parameters(), lr=experiment.critic_learning_rate)
        for key in experiment.reward_keys
    }

    # 3. Initialize weights from experiment
    weights = experiment.initial_weights()

    start_update = 1
    best_eval: tuple = (-1, -float("inf"))

    # 4. Resume
    if resume_from is not None:
        start_update = load_checkpoint(
            Path(resume_from),
            actor=actor,
            critics=critics,
            actor_optimizer=actor_optimizer,
            critic_optimizers=critic_optimizers,
            experiment=experiment,
        )
        print(
            f"[resume] loaded from {resume_from}, starting at update={start_update}",
            flush=True,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    ckpt_dir = run_dir / "checkpoints"
    video_dir = run_dir / "videos"
    print(f"run_dir={run_dir} experiment={experiment.name}", flush=True)

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
        f"[DEBUG] rollout_workers={experiment.rollout_workers} "
        f"episodes_per_update={experiment.episodes_per_update} "
        f"max_steps={experiment.custom_config['max_steps']} "
        f"update_epochs={experiment.update_epochs} "
        f"minibatch_size={experiment.minibatch_size} "
        f"reward_keys={experiment.reward_keys}",
        flush=True,
    )
    with ParallelRollouter(num_workers=experiment.rollout_workers) as rollouter:
        for u in range(start_update, experiment.max_updates + 1):
            t_update_start = time.perf_counter()

            # 5.1 Export policy blueprint (stochastic for training rollouts)
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(dest_path=str(export_dir))
            policy_bp.config["stochastic"] = True
            t_export = time.perf_counter() - t0

            # 5.2 Prepare rollout jobs
            t0 = time.perf_counter()
            rollout_seed = experiment.seed + u * experiment.episodes_per_update
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
                reward_keys=experiment.reward_keys,
                gammas=experiment.gammas,
                gae_lambda=experiment.gae_lambda,
                clip_eps=experiment.clip_eps,
                entropy_coef=experiment.entropy_coef,
                grad_clip_norm=experiment.grad_clip_norm,
                target_kl=experiment.target_kl,
                update_epochs=experiment.update_epochs,
                minibatch_size=experiment.minibatch_size,
                device=device,
                stage_weights=norm_weights,
            )
            t_ppo = time.perf_counter() - t0

            # 5.6 Logging
            bsum = buf.batch_summary()
            rsum = buf.reward_summary()
            sinfo = experiment.scheduler_info()

            # --- Train log ---
            parts = [f"[update {u:4d}]"]

            # Schedule info
            sched_parts = [f"weights={tuple(round(w, 2) for w in norm_weights)}"]
            if sinfo:
                sched_parts += [f"{k}={v}" for k, v in sinfo.items()]
            parts.append("[" + " ".join(sched_parts) + "]")

            # Episode metrics (from batch_summary)
            if bsum:
                ep_parts = [f"{k}={v:.3f}" for k, v in bsum.items()]
                parts.append("[ep " + " ".join(ep_parts) + "]")

            # Reward stats
            r_parts = []
            for key in experiment.reward_keys:
                mk, sk = f"{key}_mean", f"{key}_std"
                r_parts.append(f"{key}={rsum.get(mk, 0.0):+.3f}±{rsum.get(sk, 0.0):.3f}")
            if r_parts:
                parts.append("[reward " + " ".join(r_parts) + "]")

            # Training stats
            train_parts = [f"policy_loss={stats['policy_loss']:+.5f}"]
            for key in experiment.reward_keys:
                vk = f"vloss_{key}"
                train_parts.append(f"{vk}={stats.get(vk, 0.0):.4f}")
            train_parts.append(f"kl={stats['approx_kl']:.4f}")
            parts.append("[train " + " ".join(train_parts) + "]")

            print(" ".join(parts), flush=True)

            # --- Eval ---
            t_eval = 0.0
            if u % experiment.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = experiment.seed + 100_000 + u * 97
                det_bp = actor.to_blueprint(dest_path=str(export_dir))
                eval_jobs = experiment.build_eval_jobs(det_bp, eval_seed)
                eval_episodes: List[Episode] = rollouter.collect(eval_jobs)
                
                eval_buf = PPOBuffer(
                    episodes=eval_episodes,
                    stage_weights=norm_weights,
                    actor=actor,
                    device=device,
                    experiment=experiment,
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
                    score = (esum["mean_length"], esum.get("in_zone", 0.0))
                    if score > best_eval:
                        best_eval = score
                        export_actor_policy_artifacts(
                            actor=actor,
                            policy_dir=policy_dir,
                            extra_payload={
                                "algorithm": "ppo_curriculum",
                                "experiment": experiment.name,
                                "update": u,
                                "weights": list(norm_weights),
                                "best_eval_length": esum["mean_length"],
                            },
                        )
                        eval_line += "  [new_best]"

                    print(eval_line, flush=True)
                t_eval = time.perf_counter() - t0

                # 5.7.1 Video render
                n_evals_done += 1
                if (
                    experiment.video_eval_interval > 0
                    and n_evals_done % experiment.video_eval_interval == 0
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

            # --- Timing ---
            t_total = time.perf_counter() - t_update_start
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
            if u % experiment.eval_interval == 0 or u == 1:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actor=actor,
                    critics=critics,
                    actor_optimizer=actor_optimizer,
                    critic_optimizers=critic_optimizers,
                    experiment=experiment,
                    update=u,
                    best_eval=best_eval,
                )
