"""PPO training loop for ExperimentPPO.

Clean rewrite of ppo_loop.py for the V2 experiment interface.

Design intent
-------------
The V2 loop is a thin orchestrator: it owns the training *process*
(rollout → trajectory → buffer → PPO update → eval → log → checkpoint)
but delegates all *semantics* to the experiment — reward shaping,
trajectory segmentation, actor_weight scheduling, and eval metrics.

The framework never interprets rewards or decides how to slice episodes.
It simply calls ``experiment.build_trajectories(all_episodes)`` with the
full batch, letting the experiment compute global statistics (e.g. phase
frame-count ratios) and adjust per-trajectory weights before returning.

Key differences from v1
-----------------------
- Uses ``ExperimentPPO`` (PPO-only, no SAC).
- ``build_jobs()`` replaces separate ``build_rollout_jobs`` / ``build_eval_jobs``.
- ``build_trajectories(episodes)`` receives *all* episodes at once — no
  per-episode funnel — so experiments can do cross-episode balancing.
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

from .experiment import (
    CommonParams,
    ExperimentPPO,
    ExplorationSpec,
    PPOParams,
    TrainablePolicy,
)
from .trainer import PPOBuffer, ppo_update, set_seed


# ---------------------------------------------------------------------------
# Episode-level stats (framework-computed, no experiment involvement)
#
# These are pure diagnostics for logging. The experiment never sees them
# and they do not influence training. Keeping them framework-owned avoids
# boilerplate in every experiment subclass.
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
#
# The framework serializes the experiment's public interface (common params,
# ppo params, reward channels, state) into a reproducible config.json.
# Experiments don't need to implement any serialization themselves.
# ---------------------------------------------------------------------------

def save_run_config(
    experiment: ExperimentPPO,
    run_dir: Path,
    *,
    smoke: bool = False,
    algo: str = "ppo",
) -> None:
    """Build and save ``run_dir/config.json`` from experiment's public interface."""
    cp = experiment.common_params()
    pp = experiment.ppo_params()
    channels = experiment.reward_channels()

    # log_std bounds and entropy_coef left PPOParams, so record the initial
    # ExplorationSpec too — otherwise config.json would silently lose the
    # exploration configuration and stop being reproducible.
    initial_spec = experiment.exploration(1)

    payload = {
        "experiment": {
            "name": cp.name,
            "reward_channels": [
                {"name": ch.name, "gamma": ch.gamma, "gae_lambda": ch.gae_lambda}
                for ch in channels
            ],
            "common_params": dataclasses.asdict(cp),
            "ppo_params": dataclasses.asdict(pp),
            "initial_exploration": (
                dataclasses.asdict(initial_spec) if initial_spec is not None else None
            ),
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
#
# Checkpoints bundle actor + all critics + optimizers + experiment state
# so training can resume from any point. On resume, the framework force-
# aligns LR and log_std bounds to the *current* config, allowing config
# changes (e.g. LR decay) between resume runs.
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    experiment: ExperimentPPO,
    cp: CommonParams,
    update: int,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # A4: Atomic checkpoint write.  Write to a temporary file then rename,
    # so a SIGKILL mid-write cannot leave a truncated .pt that silently
    # corrupts resume.  os.replace is atomic on POSIX.
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
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
        tmp_path,
    )
    os.replace(tmp_path, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    *,
    actor: torch.nn.Module,
    critics: Dict[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: Dict[str, torch.optim.Optimizer],
    experiment: ExperimentPPO,
    cp: CommonParams,
    reset_update: bool = False,
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
    except (RuntimeError, ValueError) as e:
        print(f"[checkpoint] Actor optimizer state mismatch: {e}", flush=True)

    saved_crit_opt = payload["critic_optimizers_state_dict"]
    for k, opt in critic_optimizers.items():
        if k in saved_crit_opt:
            try:
                opt.load_state_dict(saved_crit_opt[k])
            except (RuntimeError, ValueError) as e:
                print(f"[checkpoint] Critic {k} optimizer state mismatch: {e}", flush=True)

    # Force align LR to current config so a config change between resume
    # runs takes effect immediately.  Both actor AND critic optimizers
    # are aligned — previously only the actor was, so changing
    # ``critic_learning_rate`` in config had no effect on resume.
    #
    # log_std bounds used to be force-aligned here too. That is now both
    # unnecessary and wrong to do from the framework: ``build_actor()``
    # runs *before* this function and already sets the bounds from the
    # experiment, and ``load_state_dict`` cannot clobber them because they
    # are plain Python floats rather than parameters or buffers. Reaching
    # into a policy-specific attribute from the loop was exactly the
    # coupling that broke non-Gaussian actors.
    for pg in actor_optimizer.param_groups:
        pg["lr"] = cp.learning_rate
    print(
        f"[checkpoint] Force aligned actor optimizer LR to {cp.learning_rate:.2e}",
        flush=True,
    )
    for key, opt in critic_optimizers.items():
        for pg in opt.param_groups:
            pg["lr"] = cp.critic_learning_rate
    print(
        f"[checkpoint] Force aligned {len(critic_optimizers)} critic optimizer(s) "
        f"LR to {cp.critic_learning_rate:.2e}",
        flush=True,
    )

    if reset_update:
        saved_update = payload.get("update", 0)
        payload["update"] = 0
        state = payload.get("state", {})
        state["update"] = 0
        payload["state"] = state
        print(f"[checkpoint] update counter reset to 0 (was {saved_update})", flush=True)

    # Restore experiment state
    saved_exp = payload.get("experiment_name", "")
    if saved_exp == cp.name:
        experiment.load_state(payload.get("state", {}))
        print(f"[checkpoint] restored experiment state", flush=True)
    else:
        print(
            f"[checkpoint] experiment changed ({saved_exp} -> {cp.name}), "
            f"resetting state",
            flush=True,
        )

    # Return the next update to run.  The checkpoint stores the update
    # that was *completed* and saved; resuming should start from the next
    # one, not rerun the completed update.  Previously this returned the
    # raw stored value, causing the resumed run to rerun the last update
    # with the same rollout seed (``seed + u * episodes_per_update``),
    # wasting a cycle and producing duplicate data.
    if reset_update:
        return 1
    saved_update = int(payload.get("update", 0))
    next_update = saved_update + 1
    print(
        f"[checkpoint] resuming from update {next_update} "
        f"(checkpoint was at update {saved_update})",
        flush=True,
    )
    return next_update


# ---------------------------------------------------------------------------
# Video recording (reused from v1)
#
# Video is rendered in a subprocess via round_runner to avoid blocking
# the training loop. If a previous render is still running, the new one
# is skipped rather than queued.
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
#
# Core training loop. Each iteration:
#   1. Export stochastic policy blueprint for rollout sampling
#   2. Build rollout jobs (experiment decides agent/distance/seed)
#   3. Collect episodes via parallel workers
#   4. Build trajectories — experiment receives ALL episodes at once,
#      enabling cross-episode statistics (e.g. phase frame balancing)
#   5. PPO update — per-channel GAE, confidence-weighted advantage
#      combination, clipped surrogate + value loss
#   6. Eval — deterministic rollout, experiment computes metrics and
#      decides best-of-run; framework handles checkpoint/video
#   7. Logging — framework-computed episode/trajectory/reward stats
#      + machine-readable __RAW_STATS__ line for external parsing
#   8. Periodic checkpoint (aligned with eval_interval)
# ---------------------------------------------------------------------------

def train_ppo(
    experiment: ExperimentPPO,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
    use_confidence: bool = True,
    reset_update: bool = False,
) -> None:
    """PPO training loop using the ExperimentPPO interface."""
    cp = experiment.common_params()
    pp = experiment.ppo_params()
    channels = experiment.reward_channels()
    reward_keys = tuple(ch.name for ch in channels)

    # --- Signal handling: kill entire process group (including rollout
    #     workers) on SIGTERM/SIGINT so --background runs can be cleanly
    #     stopped without orphaned subprocesses. ---
    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    set_seed(cp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build models ---
    # One critic per reward channel. Each critic learns V(s) for its
    # channel's reward stream, enabling per-channel GAE and confidence-
    # weighted advantage combination in ppo_update.
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

    # --- Resume from checkpoint ---
    # Restores model weights, optimizer states, and experiment state.
    # LR and log_std bounds are force-aligned to current config so
    # hyperparameter changes between runs take effect immediately.
    if resume_from is not None:
        start_update = load_checkpoint(
            Path(resume_from),
            actor=actor,
            critics=critics,
            actor_optimizer=actor_optimizer,
            critic_optimizers=critic_optimizers,
            experiment=experiment,
            cp=cp,
            reset_update=reset_update,
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

    # --- Main training loop ---
    # ParallelRollouter maintains long-lived EnvRuntime instances across
    # workers, amortizing environment construction cost over many updates.
    # Exploration state carried across updates.
    #   exploration — the spec currently in force; kept so ppo_update can
    #                 read its trust-region fields.
    exploration: Optional[ExplorationSpec] = None
    last_effective_exploration: Dict[str, float] = {}

    with ParallelRollouter(num_workers=cp.rollout_workers) as rollouter:
        for u in range(start_update, cp.max_updates + 1):
            t_update_start = time.perf_counter()

            # 0. Exploration scheduling — must precede the blueprint export,
            #    since a temperature change has to be baked into the artifact
            #    the rollout workers sample from. If it happened after, the
            #    workers would sample from one distribution while
            #    evaluate_actions scored those actions under another, silently
            #    breaking the on-policy assumption.
            #
            #    The experiment expresses intent (ExplorationSpec); the policy
            #    decides what that means for its distribution family and
            #    reports back what it actually honoured. We log the *effective*
            #    config, not the request, because a policy may clamp or ignore
            #    fields it does not support.
            #
            #    exploration() reads internal state that on_update() has
            #    accumulated from previous updates' stats. On the first
            #    update of a process, on_update() has not been called yet,
            #    so the experiment's initial state is used.
            spec = experiment.exploration(u)
            if spec is not None:
                exploration = spec
                effective = actor.set_exploration(spec)
                if effective != last_effective_exploration:
                    print(f"  [explore] {effective}", flush=True)
                    last_effective_exploration = dict(effective)

            # 1. Export stochastic policy blueprint for training rollouts.
            #    Stochastic (log_std included) so rollout samples explore.
            #    A fresh export each update ensures workers use the latest weights.
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(
                dest_path=str(export_dir), stochastic=True,
            )
            t_export = time.perf_counter() - t0

            # 2. Build rollout jobs.
            #    Experiment decides agent assignment, initial distance, seeds.
            #    Rollout seed is offset by update * batch size for reproducibility.
            t0 = time.perf_counter()
            rollout_seed = cp.seed + u * cp.episodes_per_update
            jobs = experiment.build_jobs(
                policy_bp, rollout_seed, cp.episodes_per_update,
            )
            t_jobs = time.perf_counter() - t0

            # 3. Rollout — parallel episode collection across workers.
            t0 = time.perf_counter()
            episodes: List[Episode] = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            # 4. Build trajectories — experiment receives ALL episodes at once.
            #    This is the key V2 design point: the experiment can compute
            #    global statistics (e.g. struggle/stability frame ratios) and
            #    adjust per-trajectory actor_weight before returning. The
            #    framework then wraps trajectories into a flat PPOBuffer.
            t0 = time.perf_counter()
            all_trajs = experiment.build_trajectories(episodes)
            buf = PPOBuffer(
                trajectories=all_trajs,
                actor=actor,
                device=device,
                reward_keys=reward_keys,
            )
            t_buffer = time.perf_counter() - t0

            # 5. PPO update — per-channel GAE, z-score normalized advantages,
            #    confidence-weighted combination, clipped surrogate loss.
            #    See ppo.trainer.py for the full algorithm.
            t0 = time.perf_counter()
            stats = ppo_update(
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
                exploration=exploration,
            )
            t_ppo = time.perf_counter() - t0
            # 5b. Update feedback — let the experiment absorb this update's
            #     training stats into internal state (e.g. KL history for
            #     closed-loop exploration scheduling).  exploration() on the
            #     next update will read whatever on_update() writes here.
            experiment.on_update(stats, u)

            # 6. Eval — deterministic policy rollout + experiment-defined metrics.
            #    Experiment's on_eval returns {is_new_best, info}. Framework
            #    saves best-of-run policy and spawns video on schedule.
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

                # on_eval handles metrics, best-of-run selection, and any
                # internal state updates (e.g. curriculum advancement).
                result = experiment.on_eval(eval_episodes, u)
                eval_info = result.get("info", {})
                is_new_best = result.get("is_new_best", False)

                if result.get("stop_training", False):
                    print(f"[early_stop] no improvement for {getattr(experiment, '_no_improvement_limit', '?')} evals, stopping at update {u}", flush=True)
                    save_checkpoint(
                        ckpt_dir / f"checkpoint_u{u:05d}.pt",
                        actor=actor,
                        critics=critics,
                        actor_optimizer=actor_optimizer,
                        critic_optimizers=critic_optimizers,
                        experiment=experiment,
                        cp=cp,
                        update=u,
                    )
                    break

                # Build eval line from info dict
                info_parts = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in eval_info.items()]
                eval_line = f"[eval {u:4d}] " + " ".join(info_parts)

                # Best-of-run snapshot — exported as clean inference policy
                # (no log_std) for deployment and video rendering.
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
                        v_p_a, v_p_b, v_env, v_seed, v_options = eval_jobs[0]
                        video_path = video_dir / f"u{u:05d}.mp4"
                        log_path = video_dir / f"u{u:05d}.log"
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

            # 7. Logging — framework-computed stats from Trajectory + Episode.
            #    Two layers: human-readable summary lines + machine-readable
            #    __RAW_STATS__ JSON for external log parsing / plotting.
            ep_stats = _episode_stats(episodes)
            buf_stats = buf.buffer_stats()

            # [update] header
            print(
                f"[update {u:4d}] "
                f"[episodes={ep_stats['n_episodes']} "
                f"len={ep_stats['ep_len_mean']:.1f} "
                f"(min={ep_stats['ep_len_min']}, max={ep_stats['ep_len_max']})] "
                f"[trajs={buf_stats['n_trajectories']} "
                f"steps={buf_stats['total_steps']}]",
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
                f"n_trajs={buf_stats['n_trajectories']} | "
                f"terms={{{term_strs}}}",
                flush=True,
            )

            # [Policy] & [PPO Opt]
            policy_loss = stats.policy_loss
            epochs_done = stats.epochs_done
            approx_kl = stats.approx_kl
            max_kl = stats.max_kl
            early_stop_kl = stats.early_stop_kl

            # Exploration diagnostics are rendered generically from whatever
            # the policy reported. Hard-coding entropy/std here would reassert
            # the Gaussian assumption this refactor removed; a mixture or
            # diffusion policy contributes different keys and they still show
            # up in the log without a framework change.
            explore_str = " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(buf.actor_stats.items())
            )
            print(
                f"  [Policy ] loss={policy_loss:.4f}"
                + (f" | {explore_str}" if explore_str else ""),
                flush=True,
            )
            print(
                f"  [PPO Opt] epochs={epochs_done}/{pp.update_epochs} "
                f"kl_mean={approx_kl:.4f} kl_max={max_kl:.4f} "
                f"(stop_kl={early_stop_kl:.4f})",
                flush=True,
            )

            # [Critics] — per-channel with reward, actor_weight, traj stats
            value_loss = stats.value_loss
            print(f"  [Critics] total_vloss={value_loss:.4f}", flush=True)
            chan_stats = buf_stats["per_channel"]
            for key in reward_keys:
                cs = chan_stats.get(key, {})
                r_mean = cs.get("reward_mean", 0.0)
                r_std = cs.get("reward_std", 0.0)
                r_min = cs.get("reward_min", 0.0)
                r_max = cs.get("reward_max", 0.0)
                rew_flow = f"{r_mean:+.3f}±{r_std:.3f}"
                aw_mean = cs.get("actor_weight_mean", 0.0)
                aw_min = cs.get("actor_weight_min", 0.0)
                aw_max = cs.get("actor_weight_max", 0.0)
                active_ratio = cs.get("active_ratio", 0.0)
                n_active = cs.get("n_active_trajs", 0)
                tl_mean = cs.get("traj_len_mean", 0.0)
                tl_min = cs.get("traj_len_min", 0)
                tl_max = cs.get("traj_len_max", 0)
                print(
                    f"    - {key:<12} | reward={rew_flow} "
                    f"[{r_min:+.2f},{r_max:+.2f}] | "
                    f"val_loss={stats.critic_losses.get(key, 0.0):.4f} | "
                    f"ev={stats.explained_variance.get(key, 0.0):+.3f} | "
                    f"conf={stats.confidence.get(key, 1.0):.3f} | "
                    f"aw={aw_mean:.2f} [{aw_min:.2f},{aw_max:.2f}] | "
                    f"trajs={n_active} len={tl_mean:.0f}({tl_min}-{tl_max}) | "
                    f"active={active_ratio*100:.0f}% | "
                    f"adv_std={stats.adv_std.get(key, 0.0):.2f}",
                    flush=True,
                )

            # Machine-readable raw logging — one JSON line per update.
            # Contains all stats needed for offline analysis / plotting.
            t_total = time.perf_counter() - t_update_start
            raw_log_dict = {
                "update": u,
                "algo": "ppo",
                "episode_stats": ep_stats,
                "buffer_stats": buf_stats,
                "stats": stats.to_log_dict(),
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

            # 8. Periodic checkpoint — saved at eval intervals and at u=1
            #    so the first update is always recoverable.
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
                )
