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
- ``to_blueprint()`` exports a policy that implements both Policy and StochasticPolicy.
- Framework builds ``config.json`` from experiment's public interface.
- No ``_current_actor_weights`` hack.
- No plateau detection (experiment can do this in ``on_eval`` if needed).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from baseline.framework.rollout import Episode, ParallelRollouter

from .experiment import (
    CommonParams,
    DUMP_GRADSIG_SAMPLE_SIZE,
    ExperimentPPO,
    ExplorationSpec,
    GradDiagSpec,
    LRSpec,
    PPOParams,
    TrainablePolicy,
    resolve_update_params,
)
from .trainer import PPOBuffer, ppo_update, set_seed
from .dumpkit.dump_request import DumpRequest, poll_dump_request
from .dumpkit.dump_capture import capture_dump


# ---------------------------------------------------------------------------
# Episode-level stats (framework-computed, no experiment involvement)
#
# These are pure diagnostics for logging. The experiment never sees them
# and they do not influence training. Keeping them framework-owned avoids
# boilerplate in every experiment subclass.
# ---------------------------------------------------------------------------

_EXP_METRIC_KEY = re.compile(r"^[a-z0-9_]+$")


def _sanitize_exp_metrics(m: Any) -> Dict[str, float]:
    """Keep only finite scalar metrics with safe key names.

    ``on_update()`` may return arbitrary objects; anything that is not a
    finite ``int``/``float``/``bool`` scalar, or whose key is not a
    lowercase ``[a-z0-9_]`` identifier (dots would corrupt the viewer's
    ``exp.*`` flattening), is silently dropped.  ``None`` and
    non-mapping returns yield ``{}`` — no experiment metrics.
    """
    if not isinstance(m, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in m.items():
        if not isinstance(k, str) or not _EXP_METRIC_KEY.match(k):
            continue
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = float(v)
    return out


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
    dump_at: Optional[List[int]] = None,
) -> None:
    """Build and save ``run_dir/config.json`` from experiment's public interface."""
    cp = experiment.common_params()
    pp = experiment.ppo_params()
    channels = experiment.reward_channels()

    # log_std bounds and uncertainty_coef left PPOParams, so record the initial
    # ExplorationSpec too — otherwise config.json would silently lose the
    # exploration configuration and stop being reproducible.
    initial_spec = experiment.exploration(1)
    initial_lr = experiment.lr_schedule(1)

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
            "initial_lr_schedule": (
                dataclasses.asdict(initial_lr) if initial_lr is not None else None
            ),
            "state": experiment.state(),
        },
        "algorithm": algo,
        "smoke": smoke,
        "dump_at": sorted(dump_at) if dump_at else [],
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
    prev_gvec: Optional[np.ndarray] = None,
    n_evals_done: int = 0,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # A4: Atomic checkpoint write.  Write to a temporary file then rename,
    # so a SIGKILL mid-write cannot leave a truncated .pt that silently
    # corrupts resume.  os.replace is atomic on POSIX.
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "algorithm": "ppo",
            # Checkpoint format version.  v1 = weights/optimizer/state
            # only; v2 adds rng_state + loop_state so a resumed run can
            # continue bit-identically to an uninterrupted one.
            "checkpoint_format": 2,
            "actor_state_dict": actor.state_dict(),
            "critics_state_dict": {k: v.state_dict() for k, v in critics.items()},
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizers_state_dict": {
                k: v.state_dict() for k, v in critic_optimizers.items()
            },
            "experiment_name": cp.name,
            "state": experiment.state(),
            "update": update,
            # Global RNG states captured at the end of this update —
            # exactly the stream position an uninterrupted run would
            # have entering the next update.  torch.randperm (minibatch
            # order) consumes the CUDA RNG every epoch; without this the
            # resumed run silently diverges.
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else []
                ),
            },
            # Loop-local state that is not derivable from the model:
            # prev_gvec feeds the next update's grad_sig_dir_cos;
            # n_evals_done keeps video cadence aligned.
            "loop_state": {
                "prev_gvec": prev_gvec,
                "n_evals_done": int(n_evals_done),
                # Forensic snapshot of the per-update parameter
                # overrides in force when this checkpoint was written.
                "param_overrides": dict(param_overrides or {}),
            },
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
    resume_ctx: Optional[Dict[str, Any]] = None,
) -> int:
    """Load model weights and optimizer states from checkpoint.

    Returns the update number to resume from.  If ``resume_ctx`` is
    given, it is filled with restored loop-local state (``prev_gvec``,
    ``n_evals_done``) for the caller to pick up.
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

    # Restore global RNG + loop-local state (v2 checkpoints only).
    # This is what makes a resumed continuation bit-identical to an
    # uninterrupted run: torch.randperm (minibatch order) consumes the
    # CUDA RNG every epoch, so without restoring it the resumed run
    # silently follows a different update trajectory.  With
    # reset_update=True the semantics are "fresh run with warm weights"
    # — the RNG deliberately stays at its post-init position, matching
    # a brand-new run, and loop counters start at zero.
    if not reset_update:
        rng_state = payload.get("rng_state")
        if rng_state is None:
            print(
                "[checkpoint] v1 format: no RNG state saved — minibatch "
                "order will diverge; continuation is NOT bit-identical",
                flush=True,
            )
        else:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch_cpu"])
            cuda_states = rng_state.get("torch_cuda") or []
            if cuda_states and torch.cuda.is_available():
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)
                else:
                    print(
                        f"[checkpoint] CUDA device count changed "
                        f"({len(cuda_states)} -> {torch.cuda.device_count()}); "
                        f"skipping CUDA RNG restore — not bit-identical",
                        flush=True,
                    )
            elif cuda_states:
                print(
                    "[checkpoint] checkpoint has CUDA RNG state but CUDA "
                    "is unavailable; skipping — not bit-identical",
                    flush=True,
                )
        loop_state = payload.get("loop_state") or {}
        if resume_ctx is not None:
            resume_ctx["prev_gvec"] = loop_state.get("prev_gvec")
            resume_ctx["n_evals_done"] = int(loop_state.get("n_evals_done", 0))

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
    param_patches: Optional[List[Tuple[int, str, Any]]] = None,
    dump_updates: Optional[Set[int]] = None,
    dump_hypothesis: str = "",
    dump_full_grad: bool = False,
) -> None:
    """PPO training loop using the ExperimentPPO interface.

    ``param_patches``: ``[(from_update, field, value), ...]`` applied on
    top of ``experiment.param_overrides(u)`` each update (CLI wins on
    conflicts).  See ``resolve_update_params`` for routing/validation.

    ``dump_updates``: absolute update indices that carry a scheduled
    dump request (``--dump-at``) — the update runs the full capture +
    gradsig diagnostic as if a sentinel had been written in advance.
    A sentinel found at the same update wins (manual request beats the
    schedule).  ``dump_hypothesis``/``dump_full_grad`` shape the
    synthesized request.
    """
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
    # resume_ctx carries restored loop-local state (RNG is restored
    # inside load_checkpoint itself).
    resume_ctx: Dict[str, Any] = {}
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
            resume_ctx=resume_ctx,
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

    # Video recording state — n_evals_done is restored on resume so
    # the video cadence (n_evals_done % video_eval_interval) continues
    # where the interrupted run left off.
    n_evals_done = resume_ctx.get("n_evals_done", 0)
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
    # Previous diagnostic update's aggregate gradient G, held in memory
    # and passed back via GradDiagSpec.prev_g so the trainer can emit
    # grad_sig_dir_cos (direction persistence across updates).  Persisted
    # in v2 checkpoints so a resumed run keeps dir_cos continuity.
    prev_gvec: Optional[np.ndarray] = resume_ctx.get("prev_gvec")

    patches = param_patches or []
    prev_applied_ovr: Optional[Dict[str, Any]] = None

    # Scheduled dumps (--dump-at): absolute update indices.  Warn early
    # about requests that can never fire — a typo'd index should be
    # visible at launch, not discovered hours later.
    scheduled_dumps: Set[int] = set(dump_updates or ())
    for u_req in sorted(scheduled_dumps):
        if u_req < start_update:
            print(
                f"[dump] --dump-at {u_req} < start_update {start_update} "
                f"— will never fire",
                flush=True,
            )
        elif u_req > cp.max_updates:
            print(
                f"[dump] --dump-at {u_req} > max_updates {cp.max_updates} "
                f"— will never fire",
                flush=True,
            )

    with ParallelRollouter(num_workers=cp.rollout_workers) as rollouter:
        u = start_update
        while True:
            t_update_start = time.perf_counter()

            # 0. Dump request poll — sentinel file first (a manual
            #    request beats the schedule), then the launch-time
            #    --dump-at set.  Either way the update gets the full
            #    capture + gradsig diagnostic after ppo_update.
            dump_req = poll_dump_request(run_dir, u)
            if dump_req is None and u in scheduled_dumps:
                dump_req = DumpRequest(
                    hypothesis=(
                        dump_hypothesis
                        or f"scheduled dump (--dump-at u{u})"
                    ),
                    include_full_grad=dump_full_grad,
                    source="cli",
                )
            dump_collector: Dict[str, Dict[str, Any]] = {}

            # 0a. Per-update parameter resolution — experiment hook
            #     first, CLI patches last (CLI wins on conflicts).  The
            #     resolved cp_u/pp_u are used everywhere below so that
            #     rollout, update, diagnostics and logging all see the
            #     same effective parameters.
            exp_ovr = experiment.param_overrides(u)
            cli_ovr = {k: v for frm, k, v in patches if frm <= u}
            cp_u, pp_u, applied_ovr = resolve_update_params(
                cp, pp, {**(exp_ovr or {}), **cli_ovr},
            )
            if u > cp_u.max_updates:
                break
            if applied_ovr != prev_applied_ovr:
                if applied_ovr:
                    print(
                        f"[params] u{u} effective overrides: {applied_ovr}",
                        flush=True,
                    )
                elif prev_applied_ovr:
                    print(
                        f"[params] u{u} overrides cleared — back to base",
                        flush=True,
                    )
                prev_applied_ovr = applied_ovr

            # Apply LR overrides to the optimizer param_groups when they
            # differ from the currently-applied values; lr_schedule may
            # still override them below.
            if cp_u.learning_rate != actor_optimizer.param_groups[0]["lr"]:
                for pg in actor_optimizer.param_groups:
                    pg["lr"] = cp_u.learning_rate
            if critic_optimizers:
                _copt0 = next(iter(critic_optimizers.values()))
                if cp_u.critic_learning_rate != _copt0.param_groups[0]["lr"]:
                    for copt in critic_optimizers.values():
                        for pg in copt.param_groups:
                            pg["lr"] = cp_u.critic_learning_rate

            # 0b. Exploration scheduling — resolve PPO update parameters
            #    (uncertainty_floor, uncertainty_coef) for this update.
            #    explore_factor is NOT here — it is decided inside
            #    build_jobs and placed into each Job's fields.
            spec = experiment.exploration(u)
            if spec is not None:
                exploration = spec
            else:
                exploration = None

            # 0c. LR scheduling — per-update absolute LR overrides.
            #     Symmetric with exploration(): the experiment returns an
            #     LRSpec (absolute values, consistent with the resume
            #     force-align path) or None to keep the current LR.
            lr_spec = experiment.lr_schedule(u)
            if lr_spec is not None:
                if lr_spec.actor_lr is not None:
                    for pg in actor_optimizer.param_groups:
                        pg["lr"] = lr_spec.actor_lr
                if lr_spec.critic_lr is not None:
                    for copt in critic_optimizers.values():
                        for pg in copt.param_groups:
                            pg["lr"] = lr_spec.critic_lr

            # 1. Export stochastic policy blueprint for training rollouts.
            #    Stochastic (log_std included) so rollout samples explore.
            #    A fresh export each update ensures workers use the latest weights.
            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"u{u:05d}"
            policy_bp = actor.to_blueprint(
                dest_path=str(export_dir),
            )
            t_export = time.perf_counter() - t0

            # 2. Build rollout jobs.
            #    Experiment decides agent assignment, initial distance, seeds,
            #    and explore_factor (internally, placed into Job fields).
            t0 = time.perf_counter()
            rollout_seed = cp.seed + u * cp_u.episodes_per_update
            jobs = experiment.build_jobs(
                policy_bp, rollout_seed, cp_u.episodes_per_update,
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

            # 4b. ADV gradient-signal diagnostic spec — DUMP-ONLY.
            #     When this update carries a dump request, the trainer
            #     computes the full-buffer aggregate gradient G via
            #     chunked backwards, then samples DUMP_GRADSIG_SAMPLE_SIZE
            #     frames at theta_old and computes per-frame training-
            #     loss gradients (surrogate + floor) projected onto G.
            #     Norm-bin edges are frozen in gradsig/meta.json after
            #     the first diagnostic update so all dumps in the run
            #     share a comparable axis; explicit grad_sig_norm_lo/hi
            #     config overrides the frozen range.  The seed is derived
            #     from (run seed, update) and consumed by a dedicated
            #     RNG — the training RNG stream is untouched, preserving
            #     bit-identical reproduction.
            grad_diag: Optional[GradDiagSpec] = None
            if dump_req is not None:
                gradsig_dir = run_dir / "gradsig"
                meta_path = gradsig_dir / "meta.json"
                norm_edges: Optional[np.ndarray] = None
                if pp_u.grad_sig_norm_lo > 0.0:
                    # Explicit fixed axis (log-spaced, absolute scale).
                    norm_edges = np.geomspace(
                        pp_u.grad_sig_norm_lo, pp_u.grad_sig_norm_hi,
                        pp_u.grad_sig_norm_bins + 1,
                    )
                elif meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        # norm_axis guards against reusing edges frozen
                        # under a different axis semantics (e.g. the old
                        # pairwise geomean format).
                        if (
                            meta.get("norm_bins") == pp_u.grad_sig_norm_bins
                            and meta.get("norm_axis") == "per_frame_norm"
                        ):
                            norm_edges = np.asarray(
                                meta["norm_edges"], dtype=np.float64,
                            )
                    except (json.JSONDecodeError, OSError, KeyError,
                            TypeError, ValueError):
                        norm_edges = None  # re-derive below
                grad_diag = GradDiagSpec(
                    sample_size=DUMP_GRADSIG_SAMPLE_SIZE,
                    cos_bins=pp_u.grad_sig_cos_bins,
                    norm_bins=pp_u.grad_sig_norm_bins,
                    norm_edges=norm_edges,
                    seed=cp.seed * 1000003 + u,
                    prev_g=prev_gvec,
                )

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
                pp=pp_u,
                grad_clip_norm=cp_u.grad_clip_norm,
                device=device,
                use_confidence=use_confidence,
                exploration=exploration,
                dump_callback=(
                    lambda stage, data: dump_collector.__setitem__(stage, data)
                ) if dump_req is not None else None,
                include_full_grad=(
                    dump_req.include_full_grad
                    if dump_req is not None else False
                ),
                grad_diag=grad_diag,
                update_index=u,
            )
            t_ppo = time.perf_counter() - t0

            # 5a-i. Gradient-signal payload → the dump's gradsig.npz.
            #     The scalars already went into train.log via
            #     to_log_dict() (grad_sig_ran marks the update); the 2D
            #     histogram + raw per-frame arrays are merged into
            #     dump_collector["gradsig"] so the dump artifact is
            #     self-contained.  meta.json records the frozen bin
            #     edges (written on first derived update, or when the
            #     file is missing).  The aggregate gradient vector
            #     rides back in memory via stats.grad_sig_gvec for the
            #     next diagnostic update's dir_cos.
            if stats.grad_sig_gvec is not None:
                prev_gvec = stats.grad_sig_gvec
            if stats.grad_sig_payload is not None and dump_req is not None:
                try:
                    payload = stats.grad_sig_payload
                    dump_collector.setdefault("gradsig", {}).update(
                        payload
                    )
                    gradsig_dir = run_dir / "gradsig"
                    gradsig_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = gradsig_dir / "meta.json"
                    if (
                        bool(payload["norm_edges_derived"])
                        or not meta_path.exists()
                    ):
                        meta_path.write_text(json.dumps({
                            "version": 2,
                            "norm_axis": "per_frame_norm",
                            "sample_size": DUMP_GRADSIG_SAMPLE_SIZE,
                            "cos_bins": pp_u.grad_sig_cos_bins,
                            "norm_bins": pp_u.grad_sig_norm_bins,
                            "cos_edges": payload["cos_edges"].tolist(),
                            "norm_edges": payload["norm_edges"].tolist(),
                            "norm_edges_derived": bool(
                                payload["norm_edges_derived"]
                            ),
                            "derived_update": u,
                        }, indent=2))
                except Exception as e:
                    print(f"[gradsig] write failed: {e}", flush=True)

            # 5a. Dump capture — when a request was polled, write the
            #     complete update data (episodes, trajectories, buffer,
            #     GAE, combine, gradients) + RECORD_GUIDE.md to
            #     <run_dir>/dumps/u{u:05d}/.  All data from actual training
            #     objects — no reconstruction.
            if dump_req is not None and not stats.is_empty:
                try:
                    capture_dump(
                        run_dir=run_dir,
                        update=u,
                        request=dump_req,
                        episodes=episodes,
                        trajectories=all_trajs,
                        buf=buf,
                        stats=stats,
                        jobs=jobs,
                        dump_collector=dump_collector,
                        experiment_name=cp_u.name,
                    )
                except Exception as e:
                    print(f"[dump] capture failed: {e}", flush=True)
            # B8: Print diagnostics collected by ppo_update (kept pure
            # by deferring all printing to the loop).
            for line in stats.diagnostics:
                print(line, flush=True)
            # 5b. Update feedback — let the experiment absorb this update's
            #     training stats into internal state (e.g. KL history for
            #     closed-loop exploration scheduling).  exploration() on the
            #     next update will read whatever on_update() writes here.
            #
            # P0-3: Skip on_update for empty-buffer updates so the
            # experiment's KL history doesn't get polluted with zeros
            # (which would be misread as "KL too flat, push exploration").
            exp_metrics: Optional[Dict[str, float]] = None
            if not stats.is_empty:
                exp_metrics = _sanitize_exp_metrics(
                    experiment.on_update(stats, u)
                )
            else:
                print(f"  [skip] build_trajectories returned no usable frames "
                      f"(episodes={len(episodes)}); PPO update skipped",
                      flush=True)

            # 6. Eval — deterministic policy rollout + experiment-defined metrics.
            #    Experiment's on_eval returns {is_new_best, info}. Framework
            #    saves best-of-run policy and spawns video on schedule.
            eval_info: Optional[Dict[str, Any]] = None
            t_eval = 0.0
            if u % cp_u.eval_interval == 0:
                t0 = time.perf_counter()
                eval_seed = cp.seed + 100_000 + u * 97
                eval_export_dir = run_dir / "policy_exports" / f"u{u:05d}_eval"
                det_bp = actor.to_blueprint(
                    dest_path=str(eval_export_dir),
                )
                eval_jobs = experiment.build_jobs(
                    det_bp, eval_seed, cp_u.eval_episodes,
                    stochastic=False,
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
                        prev_gvec=prev_gvec,
                        n_evals_done=n_evals_done,
                        param_overrides=applied_ovr,
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
                                "experiment": cp_u.name,
                                "update": u,
                                "best_eval_info": eval_info,
                            },
                        )
                    else:
                        # Generic path: use to_blueprint to export the
                        # best policy for deployment and video rendering.
                        actor.to_blueprint(dest_path=str(policy_dir))
                    eval_line += "  [new_best]"

                print(eval_line, flush=True)
                t_eval = time.perf_counter() - t0

                # Video render
                n_evals_done += 1
                if (
                    cp_u.video_eval_interval > 0
                    and n_evals_done % cp_u.video_eval_interval == 0
                ):
                    if last_video_proc is not None and last_video_proc.poll() is None:
                        print(f"  [video_skip:prev_running]", flush=True)
                    elif eval_jobs:
                        v_job = eval_jobs[0]
                        v_p_a, v_p_b = v_job.policy_a_bp, v_job.policy_b_bp
                        v_env, v_seed = v_job.env_bp, v_job.seed
                        v_options = v_job.episode_options
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
                f"frames={buf_stats['total_frames']}]",
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
            policy_loss_mean = stats.policy_loss_mean
            epochs_done = stats.epochs_done
            actor_epochs_done = stats.actor_epochs_done
            kl_mean = stats.kl_mean
            kl_max = stats.kl_max
            early_stop_kl_mean = stats.early_stop_kl_mean

            # Exploration diagnostics are rendered generically from whatever
            # the policy reported. Hard-coding uncertainty/std here would reassert
            # the Gaussian assumption this refactor removed; a mixture or
            # diffusion policy contributes different keys and they still show
            # up in the log without a framework change.
            explore_str = " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(buf.actor_stats.items())
            )
            print(
                f"  [Policy ] loss={policy_loss_mean:.4f}"
                + (f" | {explore_str}" if explore_str else ""),
                flush=True,
            )
            print(
                f"  [PPO Opt] epochs={epochs_done}/{pp_u.update_epochs} "
                f"actor_epochs={actor_epochs_done}/{pp_u.update_epochs} "
                f"kl_mean={kl_mean:.4f} kl_max={kl_max:.4f} "
                f"(stop_kl={early_stop_kl_mean:.4f})",
                flush=True,
            )

            # [Critics] — per-channel with reward, actor_weight, traj stats
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
                    f"val_loss={stats.critic_loss_mean.get(key, 0.0):.4f} | "
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
            # Applied (not requested) LRs — read from param_groups so the
            # realized schedule is what gets logged, including resume
            # force-aligns and any external adjustment.
            _first_copt = next(iter(critic_optimizers.values()), None)
            raw_log_dict = {
                "update": u,
                "algo": "ppo",
                "episode_stats": ep_stats,
                "buffer_stats": buf_stats,
                "stats": {
                    "actor_lr": actor_optimizer.param_groups[0]["lr"],
                    "critic_lr": (
                        _first_copt.param_groups[0]["lr"]
                        if _first_copt is not None else 0.0
                    ),
                    **stats.to_log_dict(),
                },
                # Policy-contributed stats as a separate sub-mapping so
                # consumers can tell them apart from framework-guaranteed
                # keys (stats.to_log_dict() spreads them into `stats` for
                # legacy flat-format consumers).
                "policy_stats": dict(stats.policy_stats),
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
            if exp_metrics:
                raw_log_dict["experiment"] = exp_metrics
            if applied_ovr:
                # Effective per-update parameter overrides in force for
                # this update — the dashboard draws a config-change
                # marker; no dedicated panel needed.
                raw_log_dict["param_overrides"] = applied_ovr
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
            if u % cp_u.eval_interval == 0 or u == 1:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_u{u:05d}.pt",
                    actor=actor,
                    critics=critics,
                    actor_optimizer=actor_optimizer,
                    critic_optimizers=critic_optimizers,
                    experiment=experiment,
                    cp=cp,
                    update=u,
                    prev_gvec=prev_gvec,
                    n_evals_done=n_evals_done,
                    param_overrides=applied_ovr,
                )

            u += 1
