"""PPO training loop for ExperimentV2.

Clean rewrite of ppo_loop.py for the V2 experiment interface.
Key differences from v1:

- Uses ``ExperimentV2`` (PPO-only, no SAC).
- ``build_jobs()`` replaces separate ``build_rollout_jobs`` / ``build_eval_jobs``.
- ``build_trajectories()`` called directly — no ``resolve_trajectories`` funnel.
- ``on_eval()`` replaces ``compute_episode_metrics`` + ``compare_eval`` +
  ``next_weights`` + ``scheduler_info``.
- ``state()`` / ``load_state()`` replaces split scheduler/training state.
- ``to_blueprint(stochastic=...)`` replaces manual ``config["stochastic"]`` hack.
- Framework builds ``config.json`` from experiment's public interface.
- No ``_current_actor_weights`` hack.
- No plateau detection (experiment can do this in ``on_eval`` if needed).
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

from baseline.common.policies import export_actor_policy_artifacts
from baseline.common.rollout import Episode, ParallelRollouter

from .experiment_v2 import CommonParams, ExperimentV2, PPOParams, TrainablePolicy
from .ppo_trainer_v2 import PPOBufferV2, ppo_update_v2, set_seed


# ---------------------------------------------------------------------------
# Episode-level stats (framework-computed, no experiment involvement)
# ---------------------------------------------------------------------------

def _episode_stats(episodes: List[Episode]) -> Dict[str, Any]:
    """Compute episode-level stats from raw rollout episodes for logging."""
    if not episodes:
        return {
            "n_episodes": 0,
            "ep_len_mean": 0.0,
            "ep_len_min": 0,
            "ep_len_max": 0,
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
# Config serialization (framework's job, not experiment's)
# ---------------------------------------------------------------------------

def save_run_config_v2(
    experiment: ExperimentV2,
    run_dir: Path,
    *,
    smoke: bool = False,
    algo: str = "ppo",
) -> None:
    """Build and save ``run_dir/config.json`` from experiment's public interface."""
    cp = experiment.common_params()
    pp = experiment.ppo_params()
    channels = experiment.reward_channels()

    payload = {
        "experiment": {
            "name": cp.name,
            "reward_channels": [
                {"name": ch.name, "gamma": ch.gamma, "gae_lambda": ch.gae_lambda}
                for ch in channels
            ],
            "common_params": dataclasses.asdict(cp),
            "ppo_params": dataclasses.asdict(pp),
            "state": experiment.state(),
        },
        "algorithm": algo,
        "smoke": smoke,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint_v2(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    experiment: ExperimentV2,
    cp: CommonParams,
    update: int,
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
            "state": experiment.state(),
            "update": update,
        },
        ckpt_path,
    )


def load_checkpoint_v2(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    experiment: ExperimentV2,
    cp: CommonParams,
    pp: PPOParams,
) -> int:
    """Load model weights and optimizer states from checkpoint.

    Returns the update number to resume from.
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

    # Force align LR and log_std bounds to current config
    for pg in actor_optimizer.param_groups:
        pg["lr"] = cp.learning_rate
    actor.log_std_min = float(pp.log_std_min)
    print(
        f"[checkpoint] Force aligned actor optimizer LR to {cp.learning_rate:.2e} "
        f"and log_std_min to {pp.log_std_min}",
        flush=True,
    )

    # Restore experiment state
    saved_exp = payload.get("experiment_name", "")
    if saved_exp == cp.name:
        experiment.load_state(payload.get("state", {}))
        for pg in actor_optimizer.param_groups:
            pg["lr"] = cp.learning_rate
        print(f"[checkpoint] restored LR={cp.learning_rate:.2e}", flush=True)
    else:
        print(
            f"[checkpoint] experiment changed ({saved_exp} -> {cp.name}), "
            f"resetting state",
            flush=True,
        )

    return int(payload.get("update", 0))


# ---------------------------------------------------------------------------
# Video recording (reused from v1)
# ---------------------------------------------------------------------------

def _spawn_video_render(
    *,
    env_blueprint: str,
    policy_a_blueprint: Path,
    policy_b_blueprint: Path,
    video_path: Path,
    seed: int,
    log_path: Path,
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
# Train (PPO V2)
# ---------------------------------------------------------------------------

def train_ppo_v2(
    experiment: ExperimentV2,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
    use_confidence: bool = True,
) -> None:
    """PPO training loop using the ExperimentV2 interface."""
    cp = experiment.common_params()
    pp = experiment.ppo_params()
    channels = experiment.reward_channels()
    reward_keys = tuple(ch.name for ch in channels)

    # Kill entire process group on SIGTERM/SIGINT
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(cp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build models ---
    actor = experiment.build_actor(device)
    critics = {
        ch.name: experiment.build_critic(ch.name, device)
        for ch in channels
    }

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cp.learning_rate)
    critic_optimizers = {
        ch.name: torch.optim.Adam(critics[ch.name].parameters(), lr=cp.critic_learning_rate)
        for ch in channels
    }

    start_update = 1

    # --- Resume ---
    if resume_from is not None:
        start_update = load_checkpoint_v2(
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
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir} experiment={cp.name} algo=ppo", flush=True)

    # Video recording state
    n_evals_done = 0
    last_video_proc: Optional[subprocess.Popen] = None

    print(
        f"[DEBUG] rollout_workers={cp.rollout_workers} "
        f"episodes_per_update={cp.episodes_per_update} "
        f"update_epochs={pp.update_epochs} "
        f"minibatch_size={pp.minibatch_size} "
        f"reward_keys={reward_keys}",
        flush=True,
    )

    with ParallelRollouter(num_workers=cp.rollout_workers) as rollouter:
        for u in range(start_update, cp.max_updates + 1):
            t_update_start = time.perf_counter()

            # 1. Export stochastic policy blueprint for training rollouts
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(
                dest_path=str(export_dir), stochastic=True,
            )
            t_export = time.perf_counter() - t0

            # 2. Build rollout jobs
            t0 = time.perf_counter()
            rollout_seed = cp.seed + u * cp.episodes_per_update
            jobs = experiment.build_jobs(
                policy_bp, rollout_seed, cp.episodes_per_update,
            )
            t_jobs = time.perf_counter() - t0

            # 3. Rollout
            t0 = time.perf_counter()
            episodes: List[Episode] = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            # 4. Build trajectories (batch call — experiment sees all episodes)
            t0 = time.perf_counter()
            all_trajs = experiment.build_trajectories(episodes)
            buf = PPOBufferV2(
                trajectories=all_trajs,
                actor=actor,
                device=device,
                reward_keys=reward_keys,
            )
            t_buffer = time.perf_counter() - t0

            # 5. PPO update
            t0 = time.perf_counter()
            stats = ppo_update_v2(
                actor=actor,
                critics=critics,
                actor_optimizer=actor_optimizer,
                critic_optimizers=critic_optimizers,
                buf=buf,
                reward_channels=channels,
                pp=pp,
                grad_clip_norm=cp.grad_clip_norm,
                device=device,
                use_confidence=use_confidence,
            )
            t_ppo = time.perf_counter() - t0

            # 6. Eval
            eval_info: Optional[Dict[str, Any]] = None
            t_eval = 0.0
            if u % cp.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = cp.seed + 100_000 + u * 97
                eval_export_dir = run_dir / "policy_exports" / f"u{u:05d}_eval"
                det_bp = actor.to_blueprint(
                    dest_path=str(eval_export_dir), stochastic=False,
                )
                eval_jobs = experiment.build_jobs(
                    det_bp, eval_seed, cp.eval_episodes,
                )
                eval_episodes: List[Episode] = rollouter.collect(eval_jobs)

                # on_eval handles metrics, best-of-run, and state updates
                result = experiment.on_eval(eval_episodes, u)
                eval_info = result.get("info", {})
                is_new_best = result.get("is_new_best", False)

                # Build eval line from info dict
                info_parts = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in eval_info.items()]
                eval_line = f"[eval {u:4d}] " + " ".join(info_parts)

                # Best-of-run snapshot
                if is_new_best:
                    if hasattr(actor, "export_policy_artifacts"):
                        actor.export_policy_artifacts(
                            policy_dir=policy_dir,
                            extra_payload={
                                "algorithm": "ppo_v2",
                                "experiment": cp.name,
                                "update": u,
                                "best_eval_info": eval_info,
                            },
                        )
                    else:
                        export_actor_policy_artifacts(
                            actor=actor,
                            policy_dir=policy_dir,
                            extra_payload={
                                "algorithm": "ppo_v2",
                                "experiment": cp.name,
                                "update": u,
                                "best_eval_info": eval_info,
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
                    elif eval_jobs:
                        v_p_a, v_p_b, v_env, v_seed, _ = eval_jobs[0]
                        video_path = video_dir / f"u{u:05d}.mp4"
                        log_path = video_dir / f"u{u:05d}.log"
                        v_env_path = video_dir / "video_env_blueprint.yaml"
                        v_p_a_path = video_dir / "video_policy_a.yaml"
                        v_p_b_path = video_dir / "video_policy_b.yaml"
                        v_env.save(v_env_path)
                        v_p_a.save(v_p_a_path)
                        v_p_b.save(v_p_b_path)
                        last_video_proc = _spawn_video_render(
                            env_blueprint=v_env_path,
                            policy_a_blueprint=v_p_a_path,
                            policy_b_blueprint=v_p_b_path,
                            video_path=video_path,
                            seed=v_seed,
                            log_path=log_path,
                        )
                        if last_video_proc is not None:
                            print(f"  [video:{video_path.name}]", flush=True)

            # 7. Logging — framework-computed stats from Trajectory + Episode
            ep_stats = _episode_stats(episodes)
            traj_stats = buf.trajectory_stats()
            reward_stats = buf.reward_stats()

            # [update] header
            print(
                f"[update {u:4d}] "
                f"[episodes={ep_stats['n_episodes']} "
                f"len={ep_stats['ep_len_mean']:.1f} "
                f"(min={ep_stats['ep_len_min']}, max={ep_stats['ep_len_max']})] "
                f"[trajs={traj_stats['n_trajectories']} "
                f"steps={traj_stats['total_steps']}]",
                flush=True,
            )

            # [Rollout] — episode + trajectory + termination stats
            term_strs = " ".join(
                f"{k}:{v}" for k, v in ep_stats["termination_reasons"].items()
            )
            print(
                f"  [Rollout] "
                f"len={ep_stats['ep_len_mean']:.1f} "
                f"(min={ep_stats['ep_len_min']}, max={ep_stats['ep_len_max']}) | "
                f"n_episodes={ep_stats['n_episodes']} "
                f"n_trajs={traj_stats['n_trajectories']} | "
                f"terms={{{term_strs}}}",
                flush=True,
            )

            # [Policy] & [PPO Opt]
            policy_loss = stats.get("policy_loss", 0.0)
            entropy = stats.get("entropy", 0.0)
            std_mean = stats.get("std_mean", 0.0)
            std_min = stats.get("std_min", 0.0)
            std_max = stats.get("std_max", 0.0)
            epochs_done = stats.get("epochs_done", 0)
            approx_kl = stats.get("approx_kl", 0.0)
            max_kl = stats.get("max_kl", 0.0)
            early_stop_kl = stats.get("early_stop_kl", 0.0)

            print(
                f"  [Policy ] loss={policy_loss:.4f} entropy={entropy:.2f} "
                f"std={std_mean:.3f} (min={std_min:.3f}, max={std_max:.3f})",
                flush=True,
            )
            print(
                f"  [PPO Opt] epochs={epochs_done}/{pp.update_epochs} "
                f"kl_mean={approx_kl:.4f} kl_max={max_kl:.4f} "
                f"(stop_kl={early_stop_kl:.4f})",
                flush=True,
            )

            # [Critics] — per-channel with actor_weight and active_ratio
            value_loss = stats.get("value_loss", 0.0)
            print(f"  [Critics] total_vloss={value_loss:.4f}", flush=True)
            chan_stats = traj_stats["per_channel"]
            for key in reward_keys:
                r_mean, r_std = reward_stats.get(key, (0.0, 0.0))
                rew_flow = f"{r_mean:+.3f}±{r_std:.3f}"
                vloss_key = f"vloss_{key}"
                ev_key = f"ev_{key}"
                adv_std_key = f"adv_std_{key}"
                conf_key = f"confidence_{key}"
                cs = chan_stats.get(key, {})
                aw_mean = cs.get("actor_weight_mean", 0.0)
                active_ratio = cs.get("active_ratio", 0.0)
                print(
                    f"    - {key:<12} | reward={rew_flow} | "
                    f"val_loss={stats.get(vloss_key, 0.0):.4f} | "
                    f"ev={stats.get(ev_key, 0.0):+.3f} | "
                    f"conf={stats.get(conf_key, 1.0):.3f} | "
                    f"aw={aw_mean:.2f} | "
                    f"active={active_ratio*100:.0f}% | "
                    f"adv_std={stats.get(adv_std_key, 0.0):.2f}",
                    flush=True,
                )

            # Machine-readable raw logging
            t_total = time.perf_counter() - t_update_start
            raw_log_dict = {
                "update": u,
                "algo": "ppo",
                "episode_stats": ep_stats,
                "trajectory_stats": traj_stats,
                "reward_stats": {
                    k: {"mean": v[0], "std": v[1]}
                    for k, v in reward_stats.items()
                },
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
            if eval_info is not None:
                raw_log_dict["eval_info"] = eval_info
            print(f"__RAW_STATS__ {json.dumps(raw_log_dict, default=str)}", flush=True)

            # Timing
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

            # 8. Periodic checkpoint
            if u % cp.eval_interval == 0 or u == 1:
                save_checkpoint_v2(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actor=actor,
                    critics=critics,
                    actor_optimizer=actor_optimizer,
                    critic_optimizers=critic_optimizers,
                    experiment=experiment,
                    cp=cp,
                    update=u,
                )
