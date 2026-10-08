"""SAC training loop with canonical clocks, metrics, and checkpoint bundles.

The loop is synchronous and SAC-owned: collection produces
``CollectedEpisode`` objects, experiments produce validated
``sac_transition_v2`` slices, and replay admits agent transitions with
stable ``sample_id``/``source_key`` identity.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import (
    load_checkpoint_bundle,
    load_model_only,
    save_checkpoint_bundle,
)
from .clocks import SACClockState
from .collection import create_rollouter
from .diagnostics import SACTickRing
from .debugkit import (
    begin_critic_tick_dump,
    fail_critic_tick_dump,
    finish_critic_tick_dump,
)
from .experiment import CommonParamsSAC, ExperimentSAC, SACParams, SACRewardChannel
from .metrics import SACMetricsWriter
from .networks import MultiHeadQCritic
from .replay import SACReplayBuffer
from .trainer import (
    load_model_state,
    load_trainer_state,
    sac_update_v2,
    trainer_state_dict,
)


RESUME_ALLOWED_OVERRIDES = (
    "saved_at",
    "experiment.state",
    "experiment.common_params.learning_rate",
    "experiment.common_params.critic_learning_rate",
    "experiment.sac_params.utd_ratio",
    "experiment.sac_params.max_grad_steps_per_round",
    "experiment.sac_params.alpha_lr",
)
TICK_METRIC_INTERVAL = 64
TICK_RING_CAPACITY = 4096


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def save_run_config_sac(
    experiment: ExperimentSAC,
    run_dir: Path,
    *,
    smoke: bool = False,
) -> Dict[str, Any]:
    cp = experiment.common_params()
    sp = experiment.sac_params()
    channels = experiment.reward_channels()
    payload = {
        "experiment": {
            "name": cp.name,
            "reward_channels": [dataclasses.asdict(ch) for ch in channels],
            "common_params": dataclasses.asdict(cp),
            "sac_params": dataclasses.asdict(sp),
            "state": experiment.state(),
        },
        "algorithm": "sac",
        "smoke": bool(smoke),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload


def _spawn_video_render(
    *,
    env_blueprint: Path,
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
        sys.executable,
        "-m",
        "envs.framework.round_runner",
        "--env-blueprint",
        str(env_blueprint),
        "--policy-a-blueprint",
        str(policy_a_blueprint),
        "--policy-b-blueprint",
        str(policy_b_blueprint),
        "--video",
        str(video_path),
        "--seed",
        str(seed),
    ]
    if options_json is not None:
        cmd.extend(["--options-json", str(options_json)])
    try:
        log_f = open(log_path, "w")
        return subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as e:
        print(f"[WARN] Failed to spawn video render: {e}", flush=True)
        return None


def _episode_stats(episodes: List[Any]) -> Dict[str, Any]:
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
        for reason in ep.agent_termination_reason.values():
            if reason:
                term_counts[reason] = term_counts.get(reason, 0) + 1
    return {
        "n_episodes": len(episodes),
        "ep_len_mean": float(np.mean(lengths)),
        "ep_len_min": int(np.min(lengths)),
        "ep_len_max": int(np.max(lengths)),
        "termination_reasons": term_counts,
    }


def _rng_state() -> Dict[str, Any]:
    return {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _planned_updates(
    *,
    transitions_added: int,
    replay_size: int,
    warmup_steps: int,
    batch_size: int,
    utd_ratio: float,
    max_updates: int,
    utd_credit: float,
) -> tuple[int, int, float]:
    """Fractional UTD accounting; capped updates are dropped, not rolled."""
    minimum = max(int(warmup_steps), int(batch_size))
    if replay_size < minimum:
        return 0, 0, float(utd_credit)
    credit = float(utd_credit) + float(utd_ratio) * int(transitions_added)
    requested = int(credit)
    planned = min(requested, int(max_updates))
    dropped = requested - planned
    return planned, dropped, credit - requested


def _tick_metrics(step_stats: Mapping[str, Any], batch_size: int, replay_size: int, tau: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "batch.size": float(batch_size),
        "critic.loss": float(step_stats.get("critic_loss", 0.0)),
        "actor.loss": float(step_stats.get("actor_loss", 0.0)),
        "actor.log_prob_mean": float(step_stats.get("log_prob_mean", 0.0)),
        "temperature.alpha": float(step_stats.get("alpha", 0.0)),
        "temperature.loss": float(step_stats.get("alpha_loss", 0.0)),
        "target.tau": float(tau),
        "target.pair1_frac": float(step_stats.get("target_pair1_frac", 0.0)),
        "actor.pair1_frac": float(step_stats.get("actor_pair1_frac", 0.0)),
        "actor.valid_count": float(step_stats.get("actor_valid_count", 0.0)),
        "replay.size": float(replay_size),
    }
    q_values = [
        float(v) for k, v in step_stats.items()
        if k.startswith("q1_mean_") or k.startswith("q2_mean_")
    ]
    td_values = [
        float(v) for k, v in step_stats.items() if k.startswith("td_abs_mean_")
    ]
    if q_values:
        metrics["critic.q_mean"] = float(np.mean(q_values))
    if td_values:
        metrics["critic.td_mean"] = float(np.mean(td_values))
    for name, value in step_stats.items():
        if name.startswith("q1_loss_"):
            metrics[f"critic.q1_loss.{name[8:]}"] = float(value)
        elif name.startswith("q2_loss_"):
            metrics[f"critic.q2_loss.{name[8:]}"] = float(value)
        elif name.startswith("actor_weight_mean_"):
            metrics[f"actor.weight.{name[18:]}"] = float(value)
        elif name.startswith("actor_weight_next_mean_"):
            metrics[f"actor.weight_next.{name[23:]}"] = float(value)
        elif name.startswith("critic_updated_"):
            metrics[f"critic.updated.{name[15:]}"] = float(value)
        elif name.startswith("critic_valid_weight_"):
            metrics[f"critic.valid_weight.{name[20:]}"] = float(value)
    return metrics


class DivergenceGuard:
    """Simple divergence guardrails for the first SAC loop."""

    def __init__(self, q_limit: float = 1e4, loss_limit: float = 1e3, alpha_min: float = 1e-6):
        self.q_limit = q_limit
        self.loss_limit = loss_limit
        self.alpha_min = alpha_min

    def check(self, stats: Mapping[str, float]) -> Optional[str]:
        warnings: List[str] = []
        if abs(float(stats.get("q1_mean", 0.0))) > self.q_limit:
            warnings.append("Q magnitude explosion")
        if float(stats.get("critic_loss", 0.0)) > self.loss_limit:
            warnings.append("TD loss explosion")
        if float(stats.get("alpha", 1.0)) < self.alpha_min:
            warnings.append("alpha collapse")
        return " | ".join(warnings) if warnings else None


def train_sac(
    experiment: ExperimentSAC,
    *,
    run_dir: Path,
    resume_from: Optional[Path] = None,
    reset_update: bool = False,
    config_lock: bool = False,
    rollouter: Optional[Any] = None,
    dump_ticks: Optional[Iterable[int]] = None,
    dump_hypothesis: str = "",
    debug_strict: bool = False,
    dump_keep_last: int = 8,
) -> None:
    cp = experiment.common_params()
    sp = experiment.sac_params()
    channels = experiment.reward_channels()
    channel_names = tuple(ch.name for ch in channels)
    sources = experiment.data_sources()
    if not sources:
        raise ValueError("SAC experiment must declare at least one data source")
    unsupported_sources = [src.kind for src in sources if src.kind != "self"]
    if unsupported_sources:
        raise ValueError(
            "sac_replay_v2 supports only self data sources; got "
            f"{unsupported_sources}"
        )
    if any(src.sampling_share <= 0 for src in sources):
        raise ValueError("SAC data source sampling_share must be positive")

    def _shutdown_handler(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)
    set_seed(cp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir.mkdir(parents=True, exist_ok=True)
    config = save_run_config_sac(experiment, run_dir)
    metrics = SACMetricsWriter(run_dir)
    clocks = SACClockState()
    tick_ring = SACTickRing(TICK_RING_CAPACITY)
    metrics.emit_config(clocks, config, run_id=run_dir.name)

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
    log_alpha = torch.tensor(
        np.log(sp.init_alpha), dtype=torch.float32, device=device,
        requires_grad=True,
    )
    alpha_optimizer = (
        torch.optim.Adam([log_alpha], lr=sp.alpha_lr)
        if sp.auto_alpha else None
    )
    replay = SACReplayBuffer(
        capacity=sp.replay_buffer_size,
        obs_dim=actor.obs_dim,
        action_dim=actor.action_dim,
        channel_names=channel_names,
        rng_seed=cp.seed + 17,
        replay_plan=experiment.replay_plan(),
    )
    guard = DivergenceGuard()
    utd_credit = 0.0
    n_evals_done = 0

    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            bundle = load_checkpoint_bundle(
                resume_path,
                expected_config=config,
                allowed_overrides=RESUME_ALLOWED_OVERRIDES,
                config_lock=config_lock,
                warm_start=reset_update,
            )
            if bundle.resume_mode == "full":
                load_trainer_state(
                    bundle.trainer_state,
                    actor=actor,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    log_alpha=log_alpha,
                    alpha_optimizer=alpha_optimizer,
                )
                replay = bundle.replay
                runtime_state = dict(bundle.runtime_state or {})
                clocks = SACClockState.from_mapping(runtime_state.get("clocks", {}))
                utd_credit = float(runtime_state.get("utd_credit", 0.0))
                n_evals_done = int(runtime_state.get("n_evals_done", 0))
                experiment.load_state(bundle.experiment_state)
                _restore_rng_state(runtime_state["rng"])
            else:
                load_model_state(
                    bundle.trainer_state,
                    actor=actor,
                    critic=critic,
                )
            print(
                f"[resume:{bundle.resume_mode}] {resume_path} "
                f"clocks={clocks.snapshot()}",
                flush=True,
            )
        else:
            payload = load_model_only(resume_path)["model_payload"]
            if isinstance(payload, Mapping) and payload.get("schema") == "sac_trainer_v1":
                load_model_state(payload, actor=actor, critic=critic)
            elif isinstance(payload, Mapping) and "actor_state_dict" in payload:
                actor.load_state_dict(payload["actor_state_dict"])
            else:
                raise ValueError(f"unsupported model-only checkpoint: {resume_path}")
            print(f"[resume:warm_start] {resume_path}", flush=True)

    policy_dir = run_dir / "policy"
    ckpt_dir = run_dir / "checkpoints"
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    dump_root = run_dir / "debug_dumps"
    scheduled_dump_ticks = {int(t) for t in (dump_ticks or ())}
    last_video_proc: Optional[subprocess.Popen] = None

    def _checkpoint(path: Path) -> Path:
        clocks.tick_checkpoint()
        runtime_state = {
            "clocks": clocks.snapshot(),
            "utd_credit": utd_credit,
            "n_evals_done": n_evals_done,
            "rng": _rng_state(),
        }
        save_checkpoint_bundle(
            path,
            trainer_state=trainer_state_dict(
                actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
            ),
            replay=replay,
            runtime_state=runtime_state,
            experiment_state=experiment.state(),
            config=config,
            allowed_overrides=RESUME_ALLOWED_OVERRIDES,
        )
        metrics.emit_checkpoint(
            clocks,
            {"checkpoint.bytes": _directory_bytes(path)},
            path=str(path),
        )
        return path

    print(
        f"[DEBUG] rollout_workers={cp.rollout_workers} "
        f"episodes_per_update={cp.episodes_per_update} "
        f"replay_buffer_size={sp.replay_buffer_size} batch_size={sp.batch_size} "
        f"warmup_steps={sp.warmup_steps} utd_ratio={sp.utd_ratio} "
        f"channels={channel_names} n_networks={critic.n_networks}",
        flush=True,
    )

    active_rollouter = (
        rollouter if rollouter is not None
        else create_rollouter(num_workers=cp.rollout_workers)
    )
    with active_rollouter as rollouter:
        while clocks.env_step < cp.max_env_steps:
            t_round_start = time.perf_counter()
            round_index = clocks.collection_round + 1

            t0 = time.perf_counter()
            export_dir = run_dir / "policy_exports" / f"r{round_index:05d}"
            policy_bp = actor.to_blueprint(str(export_dir), stochastic=True)
            clocks.tick_export()
            t_export = time.perf_counter() - t0
            metrics.emit_export(
                clocks,
                {"export.bytes": _directory_bytes(export_dir), "timing.export_s": t_export},
                path=str(export_dir),
            )

            t0 = time.perf_counter()
            rollout_seed = cp.seed + round_index * cp.episodes_per_update
            jobs = experiment.build_jobs(
                policy_bp,
                rollout_seed,
                cp.episodes_per_update,
                collection_round=round_index,
                run_id=run_dir.name,
                deterministic=False,
            )
            t_jobs = time.perf_counter() - t0
            episodes = rollouter.collect(jobs)
            t_rollout = time.perf_counter() - t0

            t0 = time.perf_counter()
            slices = experiment.build_slices(episodes)
            transitions_added = replay.add_slices(slices)
            ep_stats = _episode_stats(episodes)
            actual_steps = sum(ep.num_frames for ep in episodes)
            clocks.advance_collection(
                env_steps=actual_steps,
                agent_transitions=transitions_added,
            )
            t_buffer = time.perf_counter() - t0

            t0 = time.perf_counter()
            accumulated: Dict[str, List[float]] = {}
            n_grad_steps = 0
            dropped_updates = 0
            min_replay = max(int(sp.warmup_steps), int(sp.batch_size))
            if replay.size >= min_replay:
                n_grad_steps, dropped_updates, utd_credit = _planned_updates(
                    transitions_added=transitions_added,
                    replay_size=replay.size,
                    warmup_steps=sp.warmup_steps,
                    batch_size=sp.batch_size,
                    utd_ratio=sp.utd_ratio,
                    max_updates=sp.max_grad_steps_per_round,
                    utd_credit=utd_credit,
                )

                for _ in range(n_grad_steps):
                    update_start = time.perf_counter()
                    batch = replay.sample(sp.batch_size, device)
                    next_tick = clocks.critic_tick + 1
                    dump_tmp: Optional[Path] = None
                    capture: Optional[Dict[str, Any]] = None
                    if next_tick in scheduled_dump_ticks:
                        try:
                            dump_tmp = begin_critic_tick_dump(
                                dump_root=dump_root,
                                critic_tick=next_tick,
                                clocks=clocks,
                                batch=batch,
                                trainer_pre_state=trainer_state_dict(
                                    actor, critic, actor_optimizer,
                                    log_alpha, alpha_optimizer,
                                ),
                                hypothesis=dump_hypothesis,
                            )
                            capture = {}
                        except Exception as exc:
                            if debug_strict:
                                raise
                            metrics.emit_debug(
                                clocks,
                                {"debug.status": 0.0},
                                stage="begin_dump",
                                critic_tick=next_tick,
                                error=str(exc),
                            )
                            print(
                                f"  [dump_failed:capture] tick={next_tick} {exc}",
                                flush=True,
                            )
                    try:
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
                            capture=capture,
                        )
                    except Exception as exc:
                        if dump_tmp is not None:
                            fail_critic_tick_dump(dump_tmp, error=exc)
                        raise
                    if dump_tmp is not None:
                        try:
                            dump_dir = finish_critic_tick_dump(
                                dump_tmp,
                                forward_capture=capture or {},
                                trainer_post_state=trainer_state_dict(
                                    actor, critic, actor_optimizer,
                                    log_alpha, alpha_optimizer,
                                ),
                                update_stats=step_stats,
                                actor=actor,
                                critic=critic,
                                channels=channels,
                                sp=sp,
                                grad_clip_norm=cp.grad_clip_norm,
                                critic_lr=cp.critic_learning_rate,
                                keep_last=dump_keep_last,
                            )
                            metrics.emit_debug(
                                clocks,
                                {
                                    "debug.status": 1.0,
                                    "debug.bytes": float(_directory_bytes(dump_dir)),
                                    "debug.capture_s": float(
                                        time.perf_counter() - update_start
                                    ),
                                },
                                stage="finish_dump",
                                critic_tick=next_tick,
                                path=str(dump_dir),
                            )
                        except Exception as exc:
                            fail_critic_tick_dump(dump_tmp, error=exc)
                            if debug_strict:
                                raise
                            metrics.emit_debug(
                                clocks,
                                {"debug.status": 0.0},
                                stage="finish_dump",
                                critic_tick=next_tick,
                                error=str(exc),
                            )
                            print(
                                f"  [dump_failed:finish] tick={next_tick} {exc}",
                                flush=True,
                            )
                    clocks.tick_critic()
                    clocks.tick_actor()
                    if alpha_optimizer is not None:
                        clocks.tick_temperature()
                    clocks.tick_target()
                    for k, v in step_stats.items():
                        accumulated.setdefault(k, []).append(float(v))
                    tick_metrics = _tick_metrics(
                        step_stats, sp.batch_size, replay.size, sp.tau,
                    )
                    tick_ring.append(
                        clocks=clocks,
                        metrics=tick_metrics,
                        sample_ids=batch["sample_ids"].detach().cpu().tolist(),
                    )
                    if clocks.critic_tick % TICK_METRIC_INTERVAL == 0:
                        tick_metrics["timing.update_s"] = time.perf_counter() - update_start
                        metrics.emit_tick(clocks, tick_metrics)
                    warning = guard.check(step_stats)
                    if warning and "explosion" in warning:
                        _checkpoint(ckpt_dir / f"checkpoint_s{clocks.env_step:08d}")
                        raise RuntimeError(
                            f"SAC divergence at env_step={clocks.env_step}: {warning}"
                        )
            else:
                print(
                    f"  [warmup] buffer={replay.size}/{min_replay}, skipping updates",
                    flush=True,
                )
            t_sac = time.perf_counter() - t0

            stats = {
                k: float(np.mean(v)) for k, v in accumulated.items()
            }
            stats["n_grad_steps"] = float(n_grad_steps)
            stats["utd_dropped"] = float(dropped_updates)

            eval_info: Optional[Dict[str, Any]] = None
            t_eval = 0.0
            prev_env = clocks.env_step - actual_steps
            do_eval = clocks.env_step >= cp.eval_interval and (
                clocks.env_step // cp.eval_interval > prev_env // cp.eval_interval
            )
            if do_eval:
                t0 = time.perf_counter()
                clocks.tick_eval()
                eval_seed = cp.seed + 100_000 + round_index * 97
                eval_export_dir = run_dir / "policy_exports" / f"r{round_index:05d}_eval"
                det_bp = actor.to_blueprint(str(eval_export_dir), stochastic=False)
                clocks.tick_export()
                eval_jobs = experiment.build_jobs(
                    det_bp,
                    eval_seed,
                    cp.eval_episodes,
                    collection_round=round_index,
                    run_id=run_dir.name,
                    deterministic=True,
                )
                eval_episodes = rollouter.collect(eval_jobs)
                result = experiment.on_eval(eval_episodes, clocks.env_step)
                eval_info = result.get("info", {})
                is_new_best = result.get("is_new_best", False)
                if result.get("request_relabel", False):
                    raise RuntimeError(
                        "SAC relabel requests are unsupported in sac_replay_v2"
                    )
                t_eval = time.perf_counter() - t0
                eval_metrics = {
                    "eval.episodes": float(len(eval_episodes)),
                    "timing.eval_s": t_eval,
                }
                for k, v in eval_info.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        eval_metrics[f"eval.{k}"] = float(v)
                metrics.emit_eval(clocks, eval_metrics)

                info_parts = [
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in eval_info.items()
                ]
                eval_line = f"[eval s{clocks.env_step:7d}] " + " ".join(info_parts)
                if is_new_best:
                    actor.to_blueprint(str(policy_dir), stochastic=False)
                    eval_line += "  [new_best]"
                print(eval_line, flush=True)

                n_evals_done += 1
                if (
                    cp.video_eval_interval > 0
                    and n_evals_done % cp.video_eval_interval == 0
                ):
                    if last_video_proc is not None and last_video_proc.poll() is None:
                        print("  [video_skip:prev_running]", flush=True)
                    elif eval_jobs:
                        job = eval_jobs[0]
                        video_path = video_dir / f"s{clocks.env_step:08d}.mp4"
                        log_path = video_dir / f"s{clocks.env_step:08d}.log"
                        v_env_path = video_dir / "video_env_blueprint.yaml"
                        v_p_a_path = video_dir / "video_policy_a.yaml"
                        v_p_b_path = video_dir / "video_policy_b.yaml"
                        job.env_bp.save(v_env_path)
                        job.policy_a_bp.save(v_p_a_path)
                        job.policy_b_bp.save(v_p_b_path)
                        v_options_path = None
                        if job.episode_options:
                            v_options_path = video_dir / "video_options.json"
                            v_options_path.write_text(json.dumps(job.episode_options))
                        last_video_proc = _spawn_video_render(
                            env_blueprint=v_env_path,
                            policy_a_blueprint=v_p_a_path,
                            policy_b_blueprint=v_p_b_path,
                            video_path=video_path,
                            seed=job.seed,
                            log_path=log_path,
                            options_json=v_options_path,
                        )

                if result.get("stop_training", False):
                    _checkpoint(ckpt_dir / f"checkpoint_s{clocks.env_step:08d}")
                    break

            buf_stats = replay.buffer_stats()
            t_total = time.perf_counter() - t_round_start
            round_metrics = {
                "collection.episodes": float(ep_stats["n_episodes"]),
                "collection.env_steps": float(actual_steps),
                "collection.agent_transitions": float(transitions_added),
                "collection.wall_time_s": float(t_rollout),
                "replay.transitions_added": float(transitions_added),
                "replay.size": float(replay.size),
                "timing.collection_s": float(t_rollout),
                "timing.slice_s": float(t_buffer),
            }
            for key, value in experiment.post_round_metrics(episodes).items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    round_metrics[f"task.{key}"] = float(value)
            metrics.emit_round(
                clocks,
                round_metrics,
                termination_reasons=ep_stats["termination_reasons"],
                utd_credit=utd_credit,
                dropped_updates=dropped_updates,
                replay_stats=buf_stats,
                tick_ring_size=len(tick_ring),
            )

            print(
                f"[round {clocks.collection_round:4d}] "
                f"env_step={clocks.env_step}/{cp.max_env_steps} "
                f"agent_transitions={clocks.agent_transition} "
                f"episodes={ep_stats['n_episodes']} "
                f"len={ep_stats['ep_len_mean']:.1f} "
                f"buffer={replay.size} added={transitions_added} "
                f"updates={n_grad_steps} dropped={dropped_updates}",
                flush=True,
            )
            print(
                f"  | time: total={t_total:.1f}s export={t_export:.2f}s "
                f"jobs={t_jobs:.2f}s rollout={t_rollout:.1f}s "
                f"slice={t_buffer:.2f}s train={t_sac:.1f}s eval={t_eval:.1f}s",
                flush=True,
            )

            if do_eval or clocks.collection_round == 1:
                _checkpoint(ckpt_dir / f"checkpoint_s{clocks.env_step:08d}")

    metrics.close()
    print(
        f"[done] env_step={clocks.env_step} critic_tick={clocks.critic_tick}",
        flush=True,
    )
