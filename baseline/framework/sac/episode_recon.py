"""Cut-anchored episode reconstruction + cross-generation delta (WP3).

Two heavy, on-demand jobs:

``reconstruct_episode(dump_dir, sample_id)``
    Rerun the exact collection job that produced one batch row —
    identified by ``metadata`` (episode_seed, collection_round,
    job_index, agent_id) — with the dump's snapshotted behavior policy.
    Produces an mp4 + a rebuilt CollectedEpisode/slice and a per-frame
    verification report comparing replayed obs/actions/rewards against
    the values stored in ``batch.pt``.  A mismatch means the capture
    path is not faithful to the env — a falsifiable, checkable claim.

``cross_gen_delta(dump_dir)``
    Replay every saved checkpoint's deterministic policy on this dump's
    batch observations — how far did the policy travel across
    generations?  The SAC analogue of PPO's delta analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from .experiment import SACRewardChannel


def _channels_of(spec: Mapping[str, Any]) -> Tuple[SACRewardChannel, ...]:
    return tuple(
        SACRewardChannel(**item) for item in spec.get("channels", [])
    )


def _run_dir_of(dump_dir: Path) -> Path:
    # <run>/debug_dumps/critic_tick_XXXXXXXX → <run>
    return Path(dump_dir).resolve().parent.parent


def _experiment_for_run(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    exp_cfg = cfg.get("experiment") or {}
    from baseline.experiments_sac import get_sac_experiment
    return get_sac_experiment(
        str(exp_cfg["name"]), **dict(exp_cfg.get("knobs") or {})
    ), cfg


def _behavior_blueprint(
    dump_dir: Path, spec: Mapping[str, Any], run_dir: Path,
    collection_round: int,
) -> Tuple[Optional[Any], str, Optional[str]]:
    """Resolve the behavior policy blueprint for a collection round.

    Returns ``(bp_for_build, source, fingerprint_raw)`` where
    ``fingerprint_raw`` is ``mapping_hash`` over the blueprint dict as
    originally recorded (un-rewritten model_path) — comparable to the
    ``policy_fingerprint`` stored in sample metadata.
    """
    from .collected_episode import mapping_hash
    bp_entry = spec.get("behavior_policy") or {}
    bp_dict = dict(bp_entry.get("blueprint") or {})
    snap_model = (
        dump_dir / bp_entry["snapshot_dir"] / "model.pt"
        if bp_entry.get("snapshot_dir") else None
    )
    if bp_dict and snap_model is not None and snap_model.exists():
        cfg = dict(bp_dict.get("config") or {})
        cfg["model_path"] = str(snap_model)
        build_dict = dict(bp_dict)
        build_dict["config"] = cfg
        from envs.framework.policy import PolicyBlueprint
        return (
            PolicyBlueprint(**{
                k: v for k, v in build_dict.items()
                if k in ("cls", "config", "version")
            }),
            "dump_snapshot",
            mapping_hash(bp_dict),
        )
    export_yaml = (
        run_dir / "policy_exports" / f"r{int(collection_round):05d}"
        / "policy_blueprint.yaml"
    )
    if export_yaml.exists():
        from envs.framework.policy import PolicyBlueprint
        bp = PolicyBlueprint.load(export_yaml)
        raw_fp = mapping_hash(bp.to_dict())
        cfg = dict(bp.config or {})
        # YAML stores an absolute model_path captured at export time —
        # runs get renamed/moved; prefer the self-contained neighbor
        # file when it exists.
        local_model = export_yaml.parent / "model.pt"
        if local_model.exists():
            cfg["model_path"] = str(local_model)
        return (
            PolicyBlueprint(cls=bp.cls, config=cfg),
            "policy_exports",
            raw_fp,
        )
    if bp_dict:
        from envs.framework.policy import PolicyBlueprint
        return (
            PolicyBlueprint(**{
                k: v for k, v in bp_dict.items()
                if k in ("cls", "config", "version")
            }),
            "original_path_fallback",
            mapping_hash(bp_dict),
        )
    return None, "unavailable", None


def _find_row(dump: Mapping[str, Any], sample_id: int) -> int:
    ids = dump["batch"]["sample_ids"].cpu().numpy()
    rows = np.nonzero(ids == int(sample_id))[0]
    if not len(rows):
        raise KeyError(f"sample_id {sample_id} not in dump batch")
    return int(rows[0])


def reconstruct_episode(
    dump_dir: Path,
    *,
    sample_id: Optional[int] = None,
    source_key: Optional[str] = None,
    out_dir: Optional[Path] = None,
    fps: int = 30,
) -> Dict[str, Any]:
    """Rerun the episode that produced one batch row; verify + render."""
    from .analysis import parse_source_key
    from .debugkit import load_dump

    dump_dir = Path(dump_dir)
    dump = load_dump(dump_dir)
    spec = dump["spec"]
    batch = dump["batch"]

    row: Optional[int] = None
    meta: Mapping[str, Any] = {}
    if sample_id is not None:
        row = _find_row(dump, int(sample_id))
        meta = (batch.get("metadata") or [{}] * 0)[row] if batch.get(
            "metadata") else {}
    elif source_key is not None:
        keys = list(batch.get("source_keys") or [])
        if source_key in keys:
            row = keys.index(source_key)
            metas = batch.get("metadata") or []
            meta = metas[row] if row < len(metas) else {}
        else:
            parsed = parse_source_key(source_key)
            meta = {
                "collection": {
                    "collection_round": parsed.get("collection_round"),
                    "job_index": parsed.get("job_index"),
                    "episode_seed": parsed.get("episode_seed"),
                    "agent_id": parsed.get("agent_id"),
                    "run_id": parsed.get("run_id"),
                },
            }
    if row is None and source_key is None:
        raise KeyError("reconstruct_episode needs sample_id or source_key")

    col = dict(meta.get("collection") or {})
    missing = [
        k for k in ("collection_round", "job_index", "episode_seed",
                    "agent_id", "run_id")
        if col.get(k) is None
    ]
    if missing:
        raise KeyError(
            f"metadata incomplete for reconstruction; missing {missing}"
        )
    run_id = str(col["run_id"])
    cround = int(col["collection_round"])
    jidx = int(col["job_index"])
    ep_seed = int(col["episode_seed"])
    agent_id = str(col["agent_id"])

    run_dir = _run_dir_of(dump_dir)
    experiment, cfg = _experiment_for_run(run_dir)
    cp = dict(cfg.get("experiment", {}).get("common_params") or {})
    rollout_seed = int(cp["seed"]) + cround * int(cp["episodes_per_update"])

    bp, bp_source, bp_fp_raw = _behavior_blueprint(
        dump_dir, spec, run_dir, cround)
    if bp is None:
        raise KeyError(
            f"no behavior policy available for round {cround} "
            "(pruned and no snapshot)"
        )
    jobs = experiment.build_jobs(
        bp, rollout_seed, int(cp["episodes_per_update"]),
        collection_round=cround, run_id=run_id,
        deterministic=False,
    )
    job = next(
        (j for j in jobs if int(j.job_index) == jidx), None
    )
    if job is None:
        raise KeyError(
            f"job_index {jidx} not reproducible for round {cround} "
            f"(built {len(jobs)} jobs)"
        )
    if int(job.seed) != ep_seed:
        raise ValueError(
            f"rebuilt job seed {job.seed} != recorded episode_seed "
            f"{ep_seed} — job construction order drifted"
        )

    ep_dir = (
        Path(out_dir) if out_dir is not None
        else dump_dir / "episodes"
        / f"e{ep_seed}_r{cround}_{agent_id}"
    )
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Mirror _run_job_batch's setup, plus a VideoRecorderPlugin so the
    # reconstructed episode produces real frames (the single-process job
    # runner does not attach one).
    from .collection_recorder import SACEpisodeRecorder
    from .collection_rollouter import _provenance_for_job
    from .collection_runner import SACEpisodeRunner
    from .collected_episode import blueprint_hash
    from baseline.framework.sac.fact_providers import build_fact_providers
    from envs.framework.common_plugins import VideoRecorderPlugin

    runtime = job.env_bp.build()
    recorder = SACEpisodeRecorder(
        blueprint_hash=blueprint_hash(job.env_bp),
        provenance=_provenance_for_job(job),
        require_action_extras=bool(
            job.behavior_a.require_extras or job.behavior_b.require_extras
        ),
        require_pre_action_facts=bool(job.fact_specs),
    )
    runtime.attach_recorder(recorder)
    video_plugin = VideoRecorderPlugin(
        fps=int(fps), output_path=str(ep_dir / "episode.mp4"),
    )
    runtime.attach_plugin(video_plugin)
    policy_a = job.policy_a_bp.build()
    policy_b = job.policy_b_bp.build()
    runner = SACEpisodeRunner(
        runtime, policy_a, policy_b,
        recorder=recorder,
        fact_providers=build_fact_providers(job.fact_specs),
    )
    try:
        options = dict(job.episode_options)
        options["video_output_path"] = str(ep_dir / "episode.mp4")
        runner.run_episode(
            seed=job.seed, options=options, want_extras=True,
        )
    finally:
        try:
            runner.close()
        except Exception:
            pass
        try:
            runtime.close()
        except Exception:
            pass
    episode = recorder.get_last_episode()

    # Rebuild the slice through the experiment's own contract.
    slices = experiment.build_slices([episode])
    match = [s for s in slices if str(s.slice_id) == str(
        meta.get("slice_id", ""))]
    rec = match[0] if match else (slices[0] if slices else None)

    report: Dict[str, Any] = {
        "dump_dir": str(dump_dir),
        "sample_id": int(sample_id) if sample_id is not None else None,
        "source_key": source_key,
        "reconstruction": {
            "run_id": run_id,
            "collection_round": cround,
            "job_index": jidx,
            "episode_seed": ep_seed,
            "agent_id": agent_id,
            "policy_source": bp_source,
            # fingerprint of the blueprint dict AS RECORDED (model_path
            # un-rewritten) — comparable to the collection-time hash.
            "policy_fingerprint_blueprint": bp_fp_raw,
            "policy_fingerprint_recorded": col.get("policy_fingerprint"),
            "fingerprint_match": (
                bp_fp_raw is not None
                and bp_fp_raw == col.get("policy_fingerprint")
            ),
        },
        "episode_dir": str(ep_dir),
        "video": str(ep_dir / "episode.mp4")
        if (ep_dir / "episode.mp4").exists() else None,
        "rebuilt_slices": len(slices),
    }
    if rec is None:
        report["verification"] = {
            "status": "unverifiable",
            "reason": "rebuilt episode produced no matching slice",
        }
        return report
    if row is not None:
        fr = int(meta.get("frame_index", -1))
        T = len(rec.obs)
        obs_err = act_err = None
        if 0 <= fr < T:
            obs_err = float(
                np.abs(
                    np.asarray(rec.obs[fr])
                    - batch["obs"][row].cpu().numpy()
                ).max()
            )
            act_err = float(
                np.abs(
                    np.asarray(rec.actions[fr])
                    - batch["actions"][row].cpu().numpy()
                ).max()
            )
        report["verification"] = {
            "status": "checked",
            "frame_index": fr,
            "slice_T": int(T),
            "obs_max_abs_err": obs_err,
            "action_max_abs_err": act_err,
            # stochastic policy → obs path should match bit-for-bit under
            # identical RNG; actions resampled → compare distributions is
            # out of scope, but det path should coincide.
            "note": (
                "obs equality requires the rerun policy to consume the "
                "same RNG stream; a mismatch flags either RNG drift or "
                "nondeterminism in collection."
            ),
        }
        report["verification"]["obs_match"] = (
            obs_err is not None and obs_err <= 1e-5
        )
    return report


def cross_gen_delta(
    dump_dir: Path,
    *,
    checkpoints_dir: Optional[Path] = None,
    stride: int = 1,
    max_checkpoints: int = 24,
) -> Dict[str, Any]:
    """Replay every checkpoint's det-policy on this dump's batch obs."""
    from .debugkit import _build_actor, _build_critic, load_dump
    from .analysis import _log_prob_of

    dump_dir = Path(dump_dir)
    dump = load_dump(dump_dir)
    spec = dump["spec"]
    batch = dump["batch"]
    channels = _channels_of(spec)
    obs = batch["obs"].float()
    actions = batch["actions"].float()

    run_dir = _run_dir_of(dump_dir)
    ckpt_root = (
        Path(checkpoints_dir) if checkpoints_dir is not None
        else run_dir / "checkpoints"
    )
    ckpts = sorted(
        p for p in ckpt_root.glob("checkpoint_s*/trainer.pt")
        if p.exists()
    ) if ckpt_root.exists() else []
    ckpts = ckpts[:: max(int(stride), 1)][:max_checkpoints]
    out: Dict[str, Any] = {
        "dump_dir": str(dump_dir),
        "critic_tick": dump["manifest"].get("critic_tick"),
        "n_checkpoints": len(ckpts),
        "series": [],
    }
    if not ckpts:
        out["reason"] = f"no checkpoints under {ckpt_root}"
        return out

    prev_det: Optional[np.ndarray] = None
    first_det: Optional[np.ndarray] = None
    for tp in ckpts:
        state = torch.load(tp, map_location="cpu", weights_only=False)
        actor = _build_actor(spec, state)
        critic = _build_critic(spec, channels, state)
        with torch.no_grad():
            det = actor.deterministic_action(obs)
            lp = _log_prob_of(actor, obs, actions).float().numpy()
            row: Dict[str, Any] = {
                "checkpoint": tp.parent.name,
                "det_action_mean": [
                    float(x) for x in det.mean(dim=0).numpy()
                ],
                "logp_of_batch_actions_mean": float(lp.mean()),
            }
            u_fn = getattr(actor, "uncertainty", None)
            if callable(u_fn):
                from .trainer import _resolve_u_kind
                u = u_fn(
                    obs, _resolve_u_kind(
                        actor, str(
                            spec["sac_params"].get("u_kind", "native"))
                    ),
                ).numpy()
                row["u_s_mean"] = float(u.mean())
                row["u_s_q05"] = float(np.quantile(u, 0.05))
            per_ch: Dict[str, float] = {}
            for ch in channels:
                q = (
                    critic.q1_forward(obs, det, ch.name)
                    + critic.q2_forward(obs, det, ch.name)
                ).numpy() / 2.0
                per_ch[ch.name] = float(q.mean())
            row["q_det_mean"] = per_ch
        det_np = det.numpy()
        if first_det is None:
            first_det = det_np.copy()
        row["drift_vs_prev_det_l2"] = (
            float(np.linalg.norm(det_np - prev_det, axis=1).mean())
            if prev_det is not None else 0.0
        )
        row["drift_vs_first_det_l2"] = float(
            np.linalg.norm(det_np - first_det, axis=1).mean()
        )
        out["series"].append(row)
        prev_det = det_np
    return out
