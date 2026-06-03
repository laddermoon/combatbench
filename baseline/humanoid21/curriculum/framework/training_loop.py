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
from dataclasses import dataclass, field
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
    batch_summary,
    ppo_update,
    reward_summary,
    set_seed,
)

# ---------------------------------------------------------------------------
# Env-side constants
# ---------------------------------------------------------------------------
CONTROL_FREQUENCY = 20
MATCH_DURATION_SECONDS = 10.0
MAX_STEPS = int(CONTROL_FREQUENCY * MATCH_DURATION_SECONDS)

CURRICULUM_TERMINAL_FALL_PENALTY = float(
    os.environ.get("CURRICULUM_TERMINAL_FALL_PENALTY", "1.0")
)
CURRICULUM_STAGE1_PASS_LEN_RATIO = 0.98
CURRICULUM_STAGE2_PASS_FINAL_IN_ZONE = 0.5


@dataclass
class TrainConfig:
    """Training hyperparameters (PPO, rollout schedule, runtime).

    Reward-specific fields (gammas, reward_keys, weight scheduling) come from
    ``ExperimentConfig`` rather than here.  This config covers everything else
    that controls *how* training runs: network shape, optimizer settings,
    rollout parallelism, eval frequency, checkpointing, etc.
    """

    # Network shape.
    obs_dim: int = 96
    action_dim: int = 21
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    log_std_min: float = -4.0
    log_std_max: float = 0.0

    # PPO knobs.
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 8

    # GAE.
    gae_lambda: float = 0.95

    # Rollout schedule.
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # Video recording.
    video_eval_interval: int = 5
    video_env_blueprint: Optional[str] = None

    # Runtime horizon.
    max_steps: int = MAX_STEPS

    # Terminal fall penalty.
    terminal_fall_penalty: float = CURRICULUM_TERMINAL_FALL_PENALTY

    # Stage-gate pass thresholds.
    stage1_pass_len_ratio: float = CURRICULUM_STAGE1_PASS_LEN_RATIO
    stage2_pass_final_in_zone: float = CURRICULUM_STAGE2_PASS_FINAL_IN_ZONE

    # Initial-state perturbation toggle.
    enable_perturbation: bool = True

    # Parallelism.
    rollout_workers: int = field(default_factory=lambda: max(
        1, (os.cpu_count() or 1) // 2
    ))
    eval_workers: int = field(default_factory=lambda: max(
        1, (os.cpu_count() or 1) // 4
    ))

    seed: int = 42


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
    current_weights: Tuple[float, ...],
    update: int,
    best_eval: tuple,
    cfg: TrainConfig,
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
            "current_weights": current_weights,
            "experiment_name": experiment.name,
            "reward_keys": experiment.reward_keys,
            "scheduler_state": experiment.scheduler_state(),
            "update": update,
            "best_eval": best_eval,
            "cfg": cfg.__dict__,
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
    cfg: TrainConfig,
) -> Tuple[int, Tuple[float, ...]]:
    """Load checkpoint with cross-experiment compatibility.

    Returns (start_update, current_weights).
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Actor — always loads (same architecture)
    actor.load_state_dict(payload["actor_state_dict"])

    # Critics — match by key name
    if "critics_state_dict" in payload:
        saved = payload["critics_state_dict"]
        loaded_keys = []
        for k, v in critics.items():
            if k in saved:
                v.load_state_dict(saved[k])
                loaded_keys.append(k)
            else:
                print(
                    f"[checkpoint] critic '{k}' not in checkpoint -> fresh init",
                    flush=True,
                )
        print(
            f"[checkpoint] Loaded critic weights for {loaded_keys}",
            flush=True,
        )
    elif "critic_state_dict" in payload:
        # Legacy single-critic format
        if "r_cross" in critics:
            critics["r_cross"].load_state_dict(payload["critic_state_dict"])
            print(
                "[checkpoint] Loaded legacy single-critic weights into r_cross",
                flush=True,
            )
    else:
        print(
            "[checkpoint] No critic weights found, using random init",
            flush=True,
        )

    # Optimizer states
    if "actor_optimizer_state_dict" in payload:
        try:
            actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        except RuntimeError as e:
            print(f"[checkpoint] Actor optimizer state mismatch: {e}", flush=True)
    elif "optimizer_state_dict" in payload:
        print("[checkpoint] Legacy combined optimizer found, using fresh optimizers", flush=True)

    if "critic_optimizers_state_dict" in payload:
        saved_crit_opt = payload["critic_optimizers_state_dict"]
        for k, opt in critic_optimizers.items():
            if k in saved_crit_opt:
                try:
                    opt.load_state_dict(saved_crit_opt[k])
                except RuntimeError as e:
                    print(f"[checkpoint] Critic {k} optimizer state mismatch: {e}", flush=True)

    # Weights — only restore if compatible
    saved_weights = payload.get("current_weights")
    saved_keys = payload.get("reward_keys")
    if (
        saved_weights is not None
        and saved_keys is not None
        and tuple(saved_keys) == experiment.reward_keys
    ):
        weights = tuple(saved_weights)
    else:
        weights = experiment.initial_weights()
        print(
            f"[checkpoint] reward keys differ (saved={saved_keys}, "
            f"current={experiment.reward_keys}), using initial weights",
            flush=True,
        )

    # Scheduler state — only restore if experiment matches
    saved_exp = payload.get("experiment_name", "")
    if saved_exp == experiment.name:
        experiment.load_scheduler_state(payload.get("scheduler_state", {}))
    else:
        print(
            f"[checkpoint] experiment changed ({saved_exp} -> {experiment.name}), "
            f"resetting scheduler",
            flush=True,
        )

    # Handle legacy gate_stage format (no current_weights)
    if saved_weights is None and "gate_stage" in payload:
        gate_stage = int(payload.get("gate_stage", 1))
        print(f"[checkpoint] Legacy gate_stage={gate_stage} (no weights saved)", flush=True)

    start_update = int(payload.get("update", 0))
    return start_update, weights


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
    cfg: TrainConfig,
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

    set_seed(cfg.seed)
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
        key: CriticMLP(obs_dim=cfg.obs_dim, hidden_dim=cfg.critic_hidden_dim).to(device)
        for key in experiment.reward_keys
    }

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    critic_optimizers = {
        key: torch.optim.Adam(critics[key].parameters(), lr=cfg.critic_learning_rate)
        for key in experiment.reward_keys
    }

    # 3. Initialize weights from experiment
    weights = experiment.initial_weights()

    start_update = 1
    best_eval: tuple = (-1, -float("inf"))

    # 4. Resume
    if resume_from is not None:
        start_update, weights = load_checkpoint(
            Path(resume_from),
            actor=actor,
            critics=critics,
            actor_optimizer=actor_optimizer,
            critic_optimizers=critic_optimizers,
            experiment=experiment,
            cfg=cfg,
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
    video_env_bp = (
        cfg.video_env_blueprint
        if cfg.video_env_blueprint is not None
        else "envs/humanoid21/blueprint.yaml"
    )

    # Normalize weights for display
    def _norm_weights(w: Tuple[float, ...]) -> Tuple[float, ...]:
        total = sum(w)
        if total <= 0:
            return tuple(1.0 if i == 0 else 0.0 for i in range(len(w)))
        return tuple(v / total for v in w)

    # 5. Training loop
    print(
        f"[DEBUG] rollout_workers={cfg.rollout_workers} "
        f"episodes_per_update={cfg.episodes_per_update} "
        f"max_steps={cfg.max_steps} "
        f"update_epochs={cfg.update_epochs} "
        f"minibatch_size={cfg.minibatch_size} "
        f"reward_keys={experiment.reward_keys}",
        flush=True,
    )
    with ParallelRollouter(num_workers=cfg.rollout_workers) as rollouter:
        for u in range(start_update, cfg.max_updates + 1):
            t_update_start = time.perf_counter()

            # 5.1 Export policy blueprint (stochastic for training rollouts)
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(dest_path=str(export_dir))
            policy_bp.config["stochastic"] = True
            t_export = time.perf_counter() - t0

            # 5.2 Prepare rollout jobs
            t0 = time.perf_counter()
            rollout_seed = cfg.seed + u * cfg.episodes_per_update
            jobs = experiment.build_rollout_jobs(
                policy_bp, rollout_seed,
                cfg.episodes_per_update, max_steps=cfg.max_steps,
            )
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
                terminal_fall_penalty=cfg.terminal_fall_penalty,
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
                gae_lambda=cfg.gae_lambda,
                clip_eps=cfg.clip_eps,
                entropy_coef=cfg.entropy_coef,
                grad_clip_norm=cfg.grad_clip_norm,
                target_kl=cfg.target_kl,
                update_epochs=cfg.update_epochs,
                minibatch_size=cfg.minibatch_size,
                device=device,
                stage_weights=norm_weights,
            )
            t_ppo = time.perf_counter() - t0

            # 5.6 Logging
            bsum = batch_summary(buf, cfg.max_steps)
            rsum = reward_summary(buf)
            sinfo = experiment.scheduler_info()

            line = (
                f"update={u:4d} "
                f"weights={tuple(round(w, 2) for w in norm_weights)} "
            )
            if sinfo:
                info_parts = [f"{k}={v}" for k, v in sinfo.items()]
                line += " ".join(info_parts) + " "
            line = (
                f"update={u:4d} "
                f"weights={tuple(round(w, 2) for w in norm_weights)} "
            )
            if sinfo:
                info_parts = [f"{k}={v}" for k, v in sinfo.items()]
                line += " ".join(info_parts) + " "

            # Generic training stats
            n_eps = len(buf.ep_lengths)
            term_rate = float(sum(buf.is_terminated) / n_eps) if n_eps > 0 else 0.0
            line += (
                f"\n  len={bsum['mean_length']:6.2f} "
                f"term={term_rate:.3f}"
            )
            # Episode metrics from experiment (aggregated by batch_summary)
            for mk, mv in bsum.items():
                if mk not in ("mean_length", "len_ratio"):
                    line += f" {mk}={mv:.3f}"
            # Reward summary
            for key in experiment.reward_keys:
                mk, sk = f"{key}_mean", f"{key}_std"
                line += f" {key}={rsum.get(mk, 0.0):+.3f}±{rsum.get(sk, 0.0):.3f}"

            line += (
                f"\n  policy_loss={stats['policy_loss']:+.5f}"
            )
            for key in experiment.reward_keys:
                vk = f"vloss_{key}"
                line += f" {vk}={stats.get(vk, 0.0):.4f}"
            line += f" kl={stats['approx_kl']:.4f}"

            # 5.7 Eval (deterministic)
            t_eval = 0.0
            if u % cfg.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = cfg.seed + 100_000 + u * 97
                det_bp = actor.to_blueprint(dest_path=str(export_dir))
                eval_jobs = experiment.build_rollout_jobs(
                    det_bp, eval_seed,
                    cfg.eval_episodes, max_steps=cfg.max_steps,
                )
                eval_episodes: List[Episode] = rollouter.collect(eval_jobs)
                eval_buf = PPOBuffer(
                    episodes=eval_episodes,
                    stage_weights=norm_weights,
                    actor=actor,
                    device=device,
                    terminal_fall_penalty=0.0,
                    experiment=experiment,
                )
                if not eval_buf.is_empty():
                    esum = batch_summary(eval_buf, cfg.max_steps)
                    n_eval_eps = len(eval_buf.ep_lengths)
                    eval_term_rate = (
                        float(sum(eval_buf.is_terminated) / n_eval_eps)
                        if n_eval_eps > 0 else 0.0
                    )
                    line += (
                        f"\n  [eval] len={esum['mean_length']:6.2f}"
                        f" term={eval_term_rate:.3f}"
                    )
                    for mk, mv in esum.items():
                        if mk not in ("mean_length", "len_ratio"):
                            line += f" {mk}={mv:.3f}"

                    # Update weights from experiment scheduler
                    prev_weights = weights
                    weights = experiment.next_weights(esum, weights)
                    norm_weights = _norm_weights(weights)
                    if weights != prev_weights:
                        line += (
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
                        line += "  [new_best]"
                t_eval = time.perf_counter() - t0

                # 5.7.1 Video render
                n_evals_done += 1
                if (
                    cfg.video_eval_interval > 0
                    and n_evals_done % cfg.video_eval_interval == 0
                ):
                    if last_video_proc is not None and last_video_proc.poll() is None:
                        line += "  [video_skip:prev_running]"
                    else:
                        policy_bp_path = export_dir / "policy_blueprint.yaml"
                        video_path = video_dir / f"u{u:05d}.mp4"
                        log_path = video_dir / f"u{u:05d}.log"
                        last_video_proc = spawn_video_render(
                            env_blueprint=video_env_bp,
                            policy_blueprint=policy_bp_path,
                            video_path=video_path,
                            seed=eval_seed,
                            log_path=log_path,
                        )
                        if last_video_proc is not None:
                            line += f"  [video:{video_path.name}]"

            t_total = time.perf_counter() - t_update_start
            line += (
                f"\n  | time: total={t_total:.1f}s"
                f" export={t_export:.2f}s"
                f" jobs={t_jobs:.2f}s"
                f" rollout={t_rollout:.1f}s"
                f" buffer={t_buffer:.2f}s"
                f" ppo={t_ppo:.2f}s"
                f" eval={t_eval:.1f}s"
            )
            print(line, flush=True)

            # 5.8 Periodic checkpoint
            if u % cfg.eval_interval == 0 or u == 1:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actor=actor,
                    critics=critics,
                    actor_optimizer=actor_optimizer,
                    critic_optimizers=critic_optimizers,
                    experiment=experiment,
                    current_weights=weights,
                    update=u,
                    best_eval=best_eval,
                    cfg=cfg,
                )
