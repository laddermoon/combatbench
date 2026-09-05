"""SAC V2 training loop — synchronous, env_step-based clock.

Orchestrates: rollout → slice insertion → gradient steps → eval →
log → checkpoint. Uses env_step as the primary clock (not update
count) so that UTD ratio changes don't break schedule comparability.

Key features:
- Warmup period before first gradient step.
- UTD-driven gradient step count.
- Divergence guardrails (Q magnitude, TD error, alpha collapse).
- Per-channel diagnostics (Q values, gradient shares, buffer stats).
- Checkpoint/resume (model only, buffer re-warmups).
- Video rendering (reuses PPO V2's subprocess approach).
- Machine-readable __RAW_STATS__ logging.
"""
from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from baseline.framework.rollout import Episode, ParallelRollouter

from .experiment import (
    CommonParamsSAC,
    ExperimentSAC,
    SACParams,
    SACRewardChannel,
)
from .networks import MultiHeadQCritic
from .replay import TaggedReplay
from .trainer import GradNormStats, sac_update_v2


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------

def save_run_config_sac(
    experiment: ExperimentSAC,
    run_dir: Path,
    *,
    smoke: bool = False,
) -> None:
    cp = experiment.common_params()
    sp = experiment.sac_params()
    channels = experiment.reward_channels()

    payload = {
        "experiment": {
            "name": cp.name,
            "reward_channels": [
                {
                    "name": ch.name,
                    "gamma": ch.gamma,
                    "n_step": ch.n_step,
                    "n_critics": ch.n_critics,
                    "in_target_min": ch.in_target_min,
                    "trunk_group": ch.trunk_group,
                    "actor_weight_share": ch.actor_weight_share,
                }
                for ch in channels
            ],
            "common_params": dataclasses.asdict(cp),
            "sac_params": dataclasses.asdict(sp),
            "state": experiment.state(),
        },
        "algorithm": "sac",
        "smoke": smoke,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint_sac(
    ckpt_path: Path,
    *,
    actor: nn.Module,
    critic: MultiHeadQCritic,
    actor_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
    experiment: ExperimentSAC,
    cp: CommonParamsSAC,
    env_step: int,
    grad_step: int,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algorithm": "sac",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "alpha_optimizer_state_dict": (
            alpha_optimizer.state_dict() if alpha_optimizer is not None else None
        ),
        "experiment_name": cp.name,
        "state": experiment.state(),
        "env_step": env_step,
        "grad_step": grad_step,
    }, ckpt_path)


def load_checkpoint_sac(
    ckpt_path: Path,
    *,
    actor: nn.Module,
    critic: MultiHeadQCritic,
    actor_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: Optional[torch.optim.Optimizer],
    experiment: ExperimentSAC,
    cp: CommonParamsSAC,
) -> tuple[int, int]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    actor.load_state_dict(payload["actor_state_dict"])
    critic.load_state_dict(payload["critic_state_dict"])

    try:
        actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    except (RuntimeError, ValueError) as e:
        print(f"[checkpoint] Actor optimizer mismatch: {e}", flush=True)

    saved_log_alpha = payload.get("log_alpha")
    if saved_log_alpha is not None:
        log_alpha.data.copy_(saved_log_alpha)

    if alpha_optimizer is not None and payload.get("alpha_optimizer_state_dict"):
        try:
            alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])
        except (RuntimeError, ValueError) as e:
            print(f"[checkpoint] Alpha optimizer mismatch: {e}", flush=True)

    for pg in actor_optimizer.param_groups:
        pg["lr"] = cp.learning_rate

    saved_exp = payload.get("experiment_name", "")
    if saved_exp == cp.name:
        experiment.load_state(payload.get("state", {}))

    env_step = int(payload.get("env_step", 0))
    grad_step = int(payload.get("grad_step", 0))
    return env_step, grad_step


# ---------------------------------------------------------------------------
# Video rendering (shared pattern with PPO V2)
# ---------------------------------------------------------------------------

def _spawn_video_render(
    *,
    env_blueprint: str,
    policy_a_blueprint: Path,
    policy_b_blueprint: Path,
    video_path: Path,
    seed: int,
    log_path: Path,
    options_json: Optional[Path] = None,
) -> Optional[subprocess.Popen]:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "envs.framework.round_runner",
        "--env-blueprint", str(env_blueprint),
        "--policy-a-blueprint", str(policy_a_blueprint),
        "--policy-b-blueprint", str(policy_b_blueprint),
        "--video", str(video_path),
        "--seed", str(seed),
    ]
    if options_json is not None:
        cmd.extend(["--options-json", str(options_json)])
    try:
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc
    except Exception as e:
        print(f"[WARN] Failed to spawn video render: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Episode stats
# ---------------------------------------------------------------------------

def _episode_stats(episodes: List[Episode]) -> Dict[str, Any]:
    if not episodes:
        return {
            "n_episodes": 0, "ep_len_mean": 0.0,
            "ep_len_min": 0, "ep_len_max": 0,
            "termination_reasons": {},
        }
    lengths = [ep.num_frames for ep in episodes]
    term_counts: Dict[str, int] = {}
    for ep in episodes:
        for agent_id, reason in ep.agent_termination_reason.items():
            if reason:
                term_counts[reason] = term_counts.get(reason, 0) + 1
    return {
        "n_episodes": len(episodes),
        "ep_len_mean": float(np.mean(lengths)),
        "ep_len_min": int(np.min(lengths)),
        "ep_len_max": int(np.max(lengths)),
        "termination_reasons": term_counts,
    }


# ---------------------------------------------------------------------------
# Divergence guardrails
# ---------------------------------------------------------------------------

class DivergenceGuard:
    """Monitors SAC training for divergence signatures.

    Checks:
    - Q value magnitude explosion.
    - TD error explosion.
    - Alpha collapse to zero.
    - Target-online Q divergence.
    """

    def __init__(
        self,
        q_magnitude_limit: float = 1e4,
        td_error_limit: float = 1e3,
        alpha_min: float = 1e-6,
        target_div_ratio: float = 10.0,
    ):
        self.q_magnitude_limit = q_magnitude_limit
        self.td_error_limit = td_error_limit
        self.alpha_min = alpha_min
        self.target_div_ratio = target_div_ratio
        self.warnings: List[str] = []

    def check(self, stats: Dict[str, float]) -> Optional[str]:
        """Return a warning message if divergence is detected, else None."""
        self.warnings = []

        q1_mean = abs(stats.get("q1_mean", 0.0))
        if q1_mean > self.q_magnitude_limit:
            msg = f"Q magnitude explosion: |q1_mean|={q1_mean:.1f} > {self.q_magnitude_limit}"
            self.warnings.append(msg)

        alpha = stats.get("alpha", 0.0)
        if alpha < self.alpha_min:
            msg = f"Alpha collapse: alpha={alpha:.2e} < {self.alpha_min}"
            self.warnings.append(msg)

        q1_loss = stats.get("q1_loss", 0.0)
        if q1_loss > self.td_error_limit:
            msg = f"TD error explosion: q1_loss={q1_loss:.1f} > {self.td_error_limit}"
            self.warnings.append(msg)

        if self.warnings:
            return " | ".join(self.warnings)
        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sac(
    experiment: ExperimentSAC,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
    reset_update: bool = False,
) -> None:
    """SAC training loop using the ExperimentSAC interface."""
    cp = experiment.common_params()
    sp = experiment.sac_params()
    channels = experiment.reward_channels()
    channel_names = tuple(ch.name for ch in channels)
    n_steps = {ch.name: ch.n_step for ch in channels}

    # Signal handling
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(cp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build models ---
    actor = experiment.build_actor(device)
    critic = MultiHeadQCritic(
        obs_dim=actor.obs_dim,
        action_dim=actor.action_dim,
        channels=channels,
        hidden_dim=sp.q_hidden_dim,
        layer_norm=sp.q_layer_norm,
        critic_lr=cp.critic_learning_rate,
        device=device,
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cp.learning_rate)

    # Entropy temperature
    log_alpha = torch.tensor(
        np.log(sp.init_alpha), dtype=torch.float32, device=device,
        requires_grad=True,
    )
    alpha_optimizer: Optional[torch.optim.Optimizer] = None
    if sp.auto_alpha:
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=sp.alpha_lr)

    # Gradient norm stats
    grad_norm_stats = GradNormStats(
        channel_names=channel_names,
        ema_decay=sp.grad_norm_ema_decay,
    ) if sp.use_grad_norm else None

    # Replay buffer
    replay = TaggedReplay(
        capacity=sp.replay_buffer_size,
        obs_dim=actor.obs_dim,
        action_dim=actor.action_dim,
        channel_names=channel_names,
    )

    # Divergence guard
    guard = DivergenceGuard()

    # --- Resume ---
    start_env_step = 0
    grad_step = 0
    if resume_from is not None:
        start_env_step, grad_step = load_checkpoint_sac(
            Path(resume_from),
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            log_alpha=log_alpha,
            alpha_optimizer=alpha_optimizer,
            experiment=experiment,
            cp=cp,
        )
        if reset_update:
            start_env_step = 0
            grad_step = 0
        print(
            f"[resume] loaded from {resume_from}, "
            f"env_step={start_env_step}, grad_step={grad_step}",
            flush=True,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = run_dir / "policy"
    ckpt_dir = run_dir / "checkpoints"
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir} experiment={cp.name} algo=sac", flush=True)

    n_evals_done = 0
    last_video_proc: Optional[subprocess.Popen] = None

    print(
        f"[DEBUG] rollout_workers={cp.rollout_workers} "
        f"episodes_per_update={cp.episodes_per_update} "
        f"replay_buffer_size={sp.replay_buffer_size} "
        f"batch_size={sp.batch_size} "
        f"warmup_steps={sp.warmup_steps} "
        f"utd_ratio={sp.utd_ratio} "
        f"tau={sp.tau} "
        f"init_alpha={sp.init_alpha} "
        f"auto_alpha={sp.auto_alpha} "
        f"use_grad_norm={sp.use_grad_norm} "
        f"channels={channel_names} "
        f"n_networks={critic.n_networks}",
        flush=True,
    )

    env_step = start_env_step
    rollout_round = 0

    # --- Main training loop ---
    with ParallelRollouter(num_workers=cp.rollout_workers) as rollouter:
        while env_step < cp.max_env_steps:
            t_round_start = time.perf_counter()
            rollout_round += 1

            # 1. Export stochastic policy for rollout
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"r{rollout_round:05d}"
            policy_bp = actor.to_blueprint(
                dest_path=str(export_dir), stochastic=True,
            )
            t_export = time.perf_counter() - t0

            # 2. Build rollout jobs and collect
            t0 = time.perf_counter()
            rollout_seed = cp.seed + rollout_round * cp.episodes_per_update
            jobs = experiment.build_jobs(
                policy_bp, rollout_seed, cp.episodes_per_update,
            )
            t_jobs = time.perf_counter() - t0

            t0 = time.perf_counter()
            episodes: List[Episode] = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            # 3. Build slices and insert into replay
            t0 = time.perf_counter()
            slices = experiment.build_slices(episodes)
            transitions_added = replay.add_slices(slices)

            # Count env steps
            ep_stats = _episode_stats(episodes)
            env_step += ep_stats["n_episodes"] * 0  # actual steps counted below
            actual_steps = sum(ep.num_frames for ep in episodes)
            env_step += actual_steps
            t_buffer = time.perf_counter() - t0

            # 4. SAC gradient updates
            t0 = time.perf_counter()
            sac_stats_accum: Dict[str, List[float]] = {}
            n_grad_steps = 0

            if replay.size >= sp.warmup_steps:
                n_grad_steps = max(1, int(sp.utd_ratio * transitions_added))
                # Cap to prevent runaway in large batches
                n_grad_steps = min(n_grad_steps, sp.max_grad_steps_per_round)

                for _ in range(n_grad_steps):
                    batch = replay.sample_nstep(
                        sp.batch_size, device, n_steps,
                    )
                    step_stats = sac_update_v2(
                        actor=actor,
                        critic=critic,
                        actor_optimizer=actor_optimizer,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
                        batch=batch,
                        channels=channels,
                        sp=sp,
                        grad_clip_norm=cp.grad_clip_norm,
                        device=device,
                        grad_norm_stats=grad_norm_stats,
                        grad_norm_step=grad_step,
                    )
                    for k, v in step_stats.items():
                        sac_stats_accum.setdefault(k, []).append(v)
                    grad_step += 1

                    # Divergence check
                    if grad_step % 100 == 0:
                        warning = guard.check(step_stats)
                        if warning:
                            print(f"  [DIVERGENCE WARNING] {warning}", flush=True)
                            if "explosion" in warning:
                                print(
                                    f"  [DIVERGENCE] Stopping training at "
                                    f"env_step={env_step}, grad_step={grad_step}",
                                    flush=True,
                                )
                                save_checkpoint_sac(
                                    ckpt_dir / f"checkpoint_s{env_step:08d}.pt",
                                    actor=actor, critic=critic,
                                    actor_optimizer=actor_optimizer,
                                    log_alpha=log_alpha,
                                    alpha_optimizer=alpha_optimizer,
                                    experiment=experiment, cp=cp,
                                    env_step=env_step, grad_step=grad_step,
                                )
                                return
            else:
                print(
                    f"  [warmup] buffer={replay.size}/{sp.warmup_steps}, "
                    f"skipping updates",
                    flush=True,
                )

            t_sac = time.perf_counter() - t0

            # Aggregate SAC stats
            stats: Dict[str, Any] = {}
            for k, vals in sac_stats_accum.items():
                stats[k] = float(np.mean(vals))
            stats["n_grad_steps"] = n_grad_steps
            stats["grad_step"] = grad_step
            stats["buffer_size"] = replay.size
            stats["env_step"] = env_step
            stats["transitions_added"] = transitions_added

            # --- Eval ---
            eval_info: Optional[Dict[str, Any]] = None
            t_eval = 0.0
            do_eval = env_step >= cp.eval_interval and (
                env_step // cp.eval_interval > (env_step - actual_steps) // cp.eval_interval
            )

            if do_eval:
                t0 = time.perf_counter()
                eval_seed = cp.seed + 100_000 + rollout_round * 97
                eval_export_dir = run_dir / "policy_exports" / f"r{rollout_round:05d}_eval"
                det_bp = actor.to_blueprint(
                    dest_path=str(eval_export_dir), stochastic=False,
                )
                eval_jobs = experiment.build_jobs(
                    det_bp, eval_seed, cp.eval_episodes,
                )
                eval_episodes: List[Episode] = rollouter.collect(eval_jobs)

                result = experiment.on_eval(eval_episodes, env_step)
                eval_info = result.get("info", {})
                is_new_best = result.get("is_new_best", False)

                # Relabel request
                if result.get("request_relabel", False):
                    n_relabeled = replay.relabel(experiment.relabel, {
                        "env_step": env_step,
                        "eval_info": eval_info,
                    })
                    if n_relabeled > 0:
                        print(
                            f"  [relabel] {n_relabeled} transitions relabeled",
                            flush=True,
                        )

                if result.get("stop_training", False):
                    print(
                        f"[early_stop] requested by experiment at "
                        f"env_step={env_step}",
                        flush=True,
                    )
                    save_checkpoint_sac(
                        ckpt_dir / f"checkpoint_s{env_step:08d}.pt",
                        actor=actor, critic=critic,
                        actor_optimizer=actor_optimizer,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
                        experiment=experiment, cp=cp,
                        env_step=env_step, grad_step=grad_step,
                    )
                    break

                # Best-of-run snapshot
                info_parts = [
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in eval_info.items()
                ]
                eval_line = f"[eval s{env_step:7d}] " + " ".join(info_parts)

                if is_new_best:
                    if hasattr(actor, "export_policy_artifacts"):
                        actor.export_policy_artifacts(
                            policy_dir=policy_dir,
                            extra_payload={
                                "algorithm": "sac_v2",
                                "experiment": cp.name,
                                "env_step": env_step,
                                "best_eval_info": eval_info,
                            },
                        )
                    else:
                        actor.to_blueprint(dest_path=str(policy_dir), stochastic=False)
                    eval_line += "  [new_best]"

                print(eval_line, flush=True)
                t_eval = time.perf_counter() - t0

                # Video
                n_evals_done += 1
                if (
                    cp.video_eval_interval > 0
                    and n_evals_done % cp.video_eval_interval == 0
                ):
                    if last_video_proc is not None and last_video_proc.poll() is None:
                        print(f"  [video_skip:prev_running]", flush=True)
                    elif eval_jobs:
                        v_p_a, v_p_b, v_env, v_seed, v_options = eval_jobs[0]
                        video_path = video_dir / f"s{env_step:08d}.mp4"
                        log_path = video_dir / f"s{env_step:08d}.log"
                        v_env_path = video_dir / "video_env_blueprint.yaml"
                        v_p_a_path = video_dir / "video_policy_a.yaml"
                        v_p_b_path = video_dir / "video_policy_b.yaml"
                        v_env.save(v_env_path)
                        v_p_a.save(v_p_a_path)
                        v_p_b.save(v_p_b_path)
                        v_options_path: Optional[Path] = None
                        if v_options:
                            v_options_path = video_dir / "video_options.json"
                            with open(v_options_path, "w") as f:
                                json.dump(v_options, f)
                        last_video_proc = _spawn_video_render(
                            env_blueprint=v_env_path,
                            policy_a_blueprint=v_p_a_path,
                            policy_b_blueprint=v_p_b_path,
                            video_path=video_path,
                            seed=v_seed,
                            log_path=log_path,
                            options_json=v_options_path,
                        )
                        if last_video_proc is not None:
                            print(f"  [video:{video_path.name}]", flush=True)

            # --- Logging ---
            print(
                f"[round {rollout_round:4d}] "
                f"[env_step={env_step:7d}/{cp.max_env_steps}] "
                f"[episodes={ep_stats['n_episodes']} "
                f"len={ep_stats['ep_len_mean']:.1f} "
                f"(min={ep_stats['ep_len_min']}, max={ep_stats['ep_len_max']})] "
                f"[buffer={replay.size} added={transitions_added} "
                f"grad_steps={n_grad_steps}]",
                flush=True,
            )

            # SAC stats
            alpha_val = stats.get("alpha", sp.init_alpha)
            actor_loss = stats.get("actor_loss", 0.0)
            q1_loss = stats.get("q1_loss", 0.0)
            q1_mean = stats.get("q1_mean", 0.0)
            log_prob = stats.get("log_prob_mean", 0.0)
            print(
                f"  [SAC    ] alpha={alpha_val:.4f} actor_loss={actor_loss:.4f} "
                f"q1_loss={q1_loss:.4f} q1_mean={q1_mean:.3f} "
                f"log_prob={log_prob:.2f}",
                flush=True,
            )

            # Per-channel Q stats
            for ch in channel_names:
                q1lk = stats.get(f"q1_loss_{ch}", 0.0)
                q2lk = stats.get(f"q2_loss_{ch}", 0.0)
                q1mk = stats.get(f"q1_mean_{ch}", 0.0)
                aw_mean = float(np.mean([
                    v for v in [stats.get(f"actor_weights_{ch}_mean")]
                    if v is not None
                ])) if False else 0.0  # aw comes from buffer stats
                grad_share = stats.get(f"grad_share_{ch}", None)
                grad_scale = stats.get(f"grad_scale_{ch}", None)
                line = (
                    f"    - {ch:<12} q1_loss={q1lk:.4f} q2_loss={q2lk:.4f} "
                    f"q1_mean={q1mk:.3f}"
                )
                if grad_share is not None:
                    line += f" grad_share={grad_share:.1%}"
                if grad_scale is not None:
                    line += f" grad_scale={grad_scale:.4f}"
                print(line, flush=True)

            # Buffer stats
            buf_stats = replay.buffer_stats()
            print(
                f"  [Buffer ] size={replay.size}/{sp.replay_buffer_size} "
                f"util={buf_stats['utilization']:.1%} "
                f"n_trajs={buf_stats['n_trajectories']}",
                flush=True,
            )

            # Per-channel buffer stats
            for ch in channel_names:
                cs = buf_stats["per_channel"].get(ch, {})
                r_mean = cs.get("reward_mean", 0.0)
                r_std = cs.get("reward_std", 0.0)
                aw_mean = cs.get("aw_mean", 0.0)
                done_rate = cs.get("done_rate", 0.0)
                active_rate = cs.get("active_rate", 0.0)
                print(
                    f"    - {ch:<12} reward={r_mean:+.4f}±{r_std:.4f} "
                    f"aw={aw_mean:.3f} done={done_rate:.1%} "
                    f"active={active_rate:.1%}",
                    flush=True,
                )

            # Raw stats
            t_total = time.perf_counter() - t_round_start
            raw_log = {
                "round": rollout_round,
                "algo": "sac",
                "env_step": env_step,
                "grad_step": grad_step,
                "episode_stats": ep_stats,
                "buffer_stats": buf_stats,
                "stats": stats,
                "timing": {
                    "total": round(t_total, 2),
                    "export": round(t_export, 2),
                    "jobs": round(t_jobs, 2),
                    "rollout": round(t_rollout, 2),
                    "buffer": round(t_buffer, 2),
                    "sac": round(t_sac, 2),
                    "eval": round(t_eval, 2),
                },
            }
            if eval_info is not None:
                raw_log["eval_info"] = eval_info
            print(f"__RAW_STATS__ {json.dumps(raw_log, default=str)}", flush=True)

            print(
                f"  | time: total={t_total:.1f}s"
                f" export={t_export:.2f}s"
                f" jobs={t_jobs:.2f}s"
                f" rollout={t_rollout:.1f}s"
                f" buffer={t_buffer:.2f}s"
                f" sac={t_sac:.1f}s"
                f" eval={t_eval:.1f}s",
                flush=True,
            )

            # Checkpoint
            if do_eval or rollout_round == 1:
                save_checkpoint_sac(
                    ckpt_dir / f"checkpoint_s{env_step:08d}.pt",
                    actor=actor, critic=critic,
                    actor_optimizer=actor_optimizer,
                    log_alpha=log_alpha,
                    alpha_optimizer=alpha_optimizer,
                    experiment=experiment, cp=cp,
                    env_step=env_step, grad_step=grad_step,
                )

    print(f"[done] env_step={env_step}, grad_step={grad_step}", flush=True)
