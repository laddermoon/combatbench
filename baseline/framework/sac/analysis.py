"""Read-only analysis layer for SAC runs, metrics, and dump bundles.

This module is the single computation layer for both the debugkit CLI and
the HTTP JSON server (D51).  All functions are pure/read-only: they take a
path or loaded artifact and return JSON-serializable dicts.  Missing v3-only
fields are reported as ``"unavailable"`` instead of being approximated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .metric_catalog import DEFAULT_METRIC_SPECS


_SOURCE_KEY_RE = re.compile(
    r"^(?P<run>.+)/r(?P<round>\d+)/j(?P<job>\d+)/(?P<agent>[^/]+)/"
    r"e(?P<seed>-?\d+)/f(?P<frame>\d+)/h(?P<hash>[^/]+)/v(?P<schema>.+)$"
)


def parse_source_key(source_key: str) -> Dict[str, Any]:
    """Decode a sac_transition source_key back to its provenance fields."""
    m = _SOURCE_KEY_RE.match(str(source_key))
    if not m:
        return {"raw": str(source_key), "parseable": False}
    return {
        "raw": str(source_key),
        "parseable": True,
        "run_id": m.group("run"),
        "collection_round": int(m.group("round")),
        "job_index": int(m.group("job")),
        "agent_id": m.group("agent"),
        "episode_seed": int(m.group("seed")),
        "frame_index": int(m.group("frame")),
        "env_hash": m.group("hash"),
        "schema": m.group("schema"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------- run level


def iter_events(run_dir: Path) -> Iterable[Dict[str, Any]]:
    path = Path(run_dir) / "metrics" / "events.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_summary(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists() else {}
    )
    event_counts: Dict[str, int] = {}
    last_clocks: Dict[str, Any] = {}
    last_eval: Optional[Dict[str, Any]] = None
    for ev in iter_events(run_dir):
        etype = ev.get("event_type", "?")
        event_counts[etype] = event_counts.get(etype, 0) + 1
        if ev.get("clocks"):
            last_clocks = ev["clocks"]
        if etype == "eval":
            last_eval = ev.get("metrics", {})
    dumps_root = run_dir / "debug_dumps"
    dumps = (
        sorted(p.name for p in dumps_root.iterdir() if p.is_dir())
        if dumps_root.exists() else []
    )
    ckpt_root = run_dir / "checkpoints"
    checkpoints = (
        sorted(p.name for p in ckpt_root.iterdir() if p.is_dir())
        if ckpt_root.exists() else []
    )
    vid_root = run_dir / "videos"
    videos = (
        sorted(p.name for p in vid_root.iterdir() if p.suffix == ".mp4")
        if vid_root.exists() else []
    )
    experiment = (config.get("experiment") or {})
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "experiment_name": experiment.get("name"),
        "actor_arch": (experiment.get("knobs") or {}).get("actor_arch"),
        "knobs": experiment.get("knobs") or {},
        "event_counts": event_counts,
        "last_clocks": last_clocks,
        "last_eval": last_eval,
        "dumps": dumps,
        "checkpoints": checkpoints,
        "videos": videos,
    }


def runs_index(runs_root: Path) -> List[Dict[str, Any]]:
    root = Path(runs_root)
    out = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if (child / "metrics" / "events.jsonl").exists() or (
            child / "config.json"
        ).exists():
            try:
                out.append(run_summary(child))
            except Exception as exc:  # fail-soft per run; one bad run ok
                out.append({"run_dir": str(child), "name": child.name,
                            "error": str(exc)})
    return out


def metric_series(
    run_dir: Path, key: str, event: Optional[str] = None,
    max_points: Optional[int] = None,
) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    for ev in iter_events(run_dir):
        if event is not None and ev.get("event_type") != event:
            continue
        metrics = ev.get("metrics") or {}
        if key in metrics:
            points.append({
                "value": metrics[key],
                "event_type": ev.get("event_type"),
                "clocks": ev.get("clocks"),
                "timestamp_unix": ev.get("timestamp_unix"),
            })
    total = len(points)
    if max_points and total > max_points:
        # Stride-downsample preserving first/last — dense tick series
        # (millions of points) would otherwise swamp the viewer.
        stride = (total + max_points - 1) // max_points
        down = points[::stride]
        if down[-1] is not points[-1]:
            down.append(points[-1])
        points = down
    return {
        "run_dir": str(run_dir),
        "metric": key,
        "event": event,
        "count": len(points),
        "total": total,
        "points": points,
    }


def round_tick_series(
    run_dir: Path, collection_round: int,
) -> Dict[str, Any]:
    """All tick events of one collection round — the in-round update
    timeline a critic-tick dump sits on (module ⑥).  Single pass over
    events.jsonl; each row carries its critic_tick plus the full tick
    metric dict so the viewer can chart any key."""
    ticks: List[Dict[str, Any]] = []
    for ev in iter_events(run_dir):
        if ev.get("event_type") != "tick":
            continue
        clocks = ev.get("clocks") or {}
        if clocks.get("collection_round") != int(collection_round):
            continue
        ticks.append({
            "critic_tick": clocks.get("critic_tick"),
            **(ev.get("metrics") or {}),
        })
    keys = sorted(
        {k for t in ticks for k in t if k != "critic_tick"})
    return {
        "run_dir": str(run_dir),
        "collection_round": int(collection_round),
        "n_ticks": len(ticks),
        "keys": keys,
        "ticks": ticks,
    }


def metric_catalog(prefix: Optional[str] = None) -> List[Dict[str, str]]:
    return [
        {"name": s.name, "event_type": s.event_type,
         "description": s.description}
        for s in DEFAULT_METRIC_SPECS
        if prefix is None or s.name.startswith(prefix)
    ]


# ---------------------------------------------------------------- dump level


def _channels_of(dump: Mapping[str, Any]) -> List[str]:
    return [ch["name"] for ch in dump["spec"].get("channels", [])]


def _field(dump: Mapping[str, Any], name: str) -> Any:
    """v3-only forward fields degrade to the string 'unavailable' on v2."""
    return dump["forward"].get(name, "unavailable")


def _accumulate_tensor_delta(pre_node: Any, post_node: Any, acc: List) -> None:
    """Recursively accumulate |post-pre| over matching tensor leaves.

    SAC state dicts nest by structure (critic group → network/optimizer
    → tensors); a shallow top-level loop silently counts zero (G6).
    ``acc`` is [sum_sq, max_abs, elements].
    """
    if isinstance(pre_node, torch.Tensor) and isinstance(
        post_node, torch.Tensor
    ):
        if pre_node.shape != post_node.shape:
            return
        delta = (post_node.float() - pre_node.float()).abs()
        acc[0] += float((delta ** 2).sum().item())
        if delta.numel():
            acc[1] = max(acc[1], float(delta.max().item()))
        acc[2] += int(delta.numel())
    elif isinstance(pre_node, Mapping) and isinstance(post_node, Mapping):
        for key, pre_v in pre_node.items():
            if key in post_node:
                _accumulate_tensor_delta(pre_v, post_node[key], acc)
    elif isinstance(pre_node, (list, tuple)) and isinstance(
        post_node, (list, tuple)
    ):
        for pre_v, post_v in zip(pre_node, post_node):
            _accumulate_tensor_delta(pre_v, post_v, acc)


def _delta_summary(acc: List) -> Dict[str, Any]:
    return {
        "l2": float(acc[0] ** 0.5),
        "max_abs_delta": float(acc[1]),
        "elements": int(acc[2]),
    }


def param_delta(dump: Mapping[str, Any]) -> Dict[str, Any]:
    """Pre→post update delta summary (actor + critic), nested-aware.

    Returns an aggregate per scope plus a ``by_key`` breakdown per
    top-level entry — e.g. each critic group separately, and within a
    group the q1/q2/target/optimizer sub-states are all recursed into,
    so network weights and optimizer momentum are both counted.
    """
    out: Dict[str, Any] = {}
    for scope in ("actor_state_dict", "critic_state_dict"):
        pre = dump["trainer_pre"].get(scope) or {}
        post = dump["trainer_post"].get(scope) or {}
        total = [0.0, 0.0, 0]
        by_key: Dict[str, Any] = {}
        for key, pre_v in pre.items():
            post_v = post.get(key)
            if post_v is None:
                by_key[key] = "missing_in_post"
                continue
            acc = [0.0, 0.0, 0]
            _accumulate_tensor_delta(pre_v, post_v, acc)
            by_key[key] = _delta_summary(acc)
            total[0] += acc[0]
            total[1] = max(total[1], acc[1])
            total[2] += acc[2]
        out[scope] = {
            **_delta_summary(total),
            "by_key": by_key,
        }
    return out


def dump_inspect(dump_dir: Path) -> Dict[str, Any]:
    from .debugkit import load_dump  # local import to avoid a cycle

    dump = load_dump(Path(dump_dir))
    batch, forward = dump["batch"], dump["forward"]
    channels = _channels_of(dump)
    sample_ids = batch["sample_ids"].cpu().numpy()
    round_counts: Dict[str, int] = {}
    policy_fp: Dict[str, int] = {}
    for meta in (batch.get("metadata") or [{}] * len(sample_ids)):
        collection = (meta or {}).get("collection") or {}
        versions = (meta or {}).get("versions") or {}
        r = str(collection.get("collection_round", "?"))
        round_counts[r] = round_counts.get(r, 0) + 1
        fp = str(collection.get("policy_fingerprint", "?"))[:12]
        policy_fp[fp] = policy_fp.get(fp, 0) + 1
    per_channel: Dict[str, Any] = {}
    for ch in channels:
        entry: Dict[str, Any] = {}
        for field in ("q1_pred", "q2_pred", "targets"):
            arr = forward.get(field, {})
            if isinstance(arr, Mapping) and ch in arr:
                v = arr[ch].float().numpy()
                entry[f"{field}_mean"] = float(v.mean())
                entry[f"{field}_absmax"] = float(np.abs(v).max())
        td1 = _field(dump, "td_q1")
        if isinstance(td1, Mapping) and ch in td1:
            td = td1[ch].float().numpy()
            entry["td_abs_mean"] = float(np.abs(td).mean())
            entry["td_abs_max"] = float(np.abs(td).max())
        per_channel[ch] = entry
    return {
        "dump_dir": str(Path(dump_dir)),
        "schema_version": dump["manifest"]["schema_version"],
        "critic_tick": dump["manifest"].get("critic_tick"),
        "clocks": dump["clocks"],
        "hypothesis": (dump["request"].get("hypothesis") or ""),
        "batch": {
            "size": int(len(sample_ids)),
            "sample_id_range": [
                int(sample_ids.min()), int(sample_ids.max()),
            ] if len(sample_ids) else None,
            "collection_round_counts": round_counts,
            "policy_fingerprint_counts": policy_fp,
            "terminated_rows": int(batch["terminated"].sum().item()),
            "truncated_rows": int(batch["truncated"].sum().item()),
        },
        "per_channel": per_channel,
        "pair_index": {
            "target_unique": (
                forward["target_pair_index"].unique().tolist()
                if "target_pair_index" in forward else "unavailable"
            ),
            "actor_unique": (
                forward["actor_pair_index"].unique().tolist()
                if "actor_pair_index" in forward else "unavailable"
            ),
        },
        "consistency": dump["analysis"].get("consistency", "unavailable"),
        "replay_stats": dump["analysis"].get("replay_stats", "unavailable"),
        "update_stats": dump["analysis"].get("update_stats", {}),
        "param_delta": param_delta(dump),
    }


def dump_samples(
    dump_dir: Path,
    *,
    sort: str = "td_abs",
    channel: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    from .debugkit import load_dump

    dump = load_dump(Path(dump_dir))
    batch, forward = dump["batch"], dump["forward"]
    channels = _channels_of(dump)
    if channel is not None and channel not in channels:
        raise ValueError(
            f"channel {channel!r} not in dump channels {channels}"
        )
    B = int(batch["sample_ids"].numel())
    sample_ids = batch["sample_ids"].cpu().numpy()
    sel_channels = [channel] if channel else channels

    td_abs = np.zeros(B)
    q_mean = np.zeros(B)
    for ch in sel_channels:
        td1 = _field(dump, "td_q1")
        td2 = _field(dump, "td_q2")
        if isinstance(td1, Mapping) and ch in td1:
            td_abs += (np.abs(td1[ch].float().numpy())
                       + np.abs(td2[ch].float().numpy())) / 2.0
        else:
            t = forward["targets"][ch].float().numpy()
            td_abs += (
                np.abs(forward["q1_pred"][ch].float().numpy() - t)
                + np.abs(forward["q2_pred"][ch].float().numpy() - t)
            ) / 2.0
        q_mean += (forward["q1_pred"][ch].float().numpy()
                   + forward["q2_pred"][ch].float().numpy()) / 2.0
    q_mean /= max(len(sel_channels), 1)

    logp_w = forward.get("next_integration_weights")
    logp = forward.get("next_log_probs")
    if isinstance(logp_w, torch.Tensor) and isinstance(logp, torch.Tensor):
        logp_mean = (logp_w * logp).sum(dim=(1, 2)).numpy()
    else:
        logp_mean = np.zeros(B)

    rows = []
    for i in range(B):
        meta = (batch.get("metadata") or [{}] * B)[i] or {}
        rows.append({
            "row": i,
            "sample_id": int(sample_ids[i]),
            "source_key": str(batch["source_keys"][i]),
            "td_abs": float(td_abs[i]),
            "q_mean": float(q_mean[i]),
            "logp_expectation": float(logp_mean[i]),
            "weighted_q": (
                float(forward["weighted_q"][i])
                if "weighted_q" in forward else None
            ),
            "reward_sum": float(batch["rewards"][i].sum().item()),
            "terminated": bool(batch["terminated"][i]),
            "truncated": bool(batch["truncated"][i]),
            "draw_count": (
                int(batch["draw_counts"][i])
                if "draw_counts" in batch else "unavailable"
            ),
            "collection_round": (
                meta.get("collection") or {}
            ).get("collection_round"),
        })
    sorters = {
        "td_abs": lambda r: r["td_abs"],
        "q_mean": lambda r: abs(r["q_mean"]),
        "logp": lambda r: r["logp_expectation"],
    }
    if sort not in sorters:
        raise ValueError(
            f"unknown sort {sort!r}; allowed={sorted(sorters)}"
        )
    rows.sort(key=sorters[sort], reverse=True)
    return {
        "dump_dir": str(Path(dump_dir)),
        "sort": sort,
        "channel": channel,
        "total": B,
        "rows": rows[: max(int(limit), 0)],
    }


def pipeline_summary(dump_dir: Path) -> Dict[str, Any]:
    """Dump-home pipeline payload — one call, per-module headline stats.

    SAC analog of the PPO dumpkit ``/api/pipeline``: the dump homepage's
    nine stage cards each render their scalar slice from ``stages``.
    Everything here is precomputed at capture (analysis.json/spec.json)
    or scalar-level (inspect's param_delta) — the heavy per-row views
    live on the tool pages.
    """
    ins = dump_inspect(Path(dump_dir))
    dump_dir = Path(dump_dir)
    spec: Dict[str, Any] = {}
    manifest: Dict[str, Any] = {}
    for name, dst in (("spec.json", "spec"), ("manifest.json", "manifest")):
        p = dump_dir / name
        if p.exists():
            try:
                if dst == "spec":
                    spec = json.loads(p.read_text(encoding="utf-8"))
                else:
                    manifest = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    clocks = ins.get("clocks") or {}
    batch = ins.get("batch") or {}
    us = ins.get("update_stats") or {}
    rs = ins.get("replay_stats") or {}
    if not isinstance(rs, dict):
        rs = {}
    pc = ins.get("per_channel") or {}
    sac_params = spec.get("sac_params") or {}
    spec_channels = spec.get("channels") or []
    channels = [c.get("name") for c in spec_channels
                if isinstance(c, dict)] or sorted(pc.keys())

    target_channels = []
    for ch in pc:
        e = pc[ch] or {}
        target_channels.append({
            "name": ch,
            "targets_mean": e.get("targets_mean"),
            "targets_absmax": e.get("targets_absmax"),
            "td_abs_mean": e.get("td_abs_mean"),
            "td_abs_max": e.get("td_abs_max"),
            "gamma": next(
                (c.get("gamma") for c in spec_channels
                 if isinstance(c, dict) and c.get("name") == ch), None),
        })
    critic_channels = []
    for ch in pc:
        critic_channels.append({
            "name": ch,
            "q1_loss": us.get(f"q1_loss_{ch}"),
            "q2_loss": us.get(f"q2_loss_{ch}"),
            "q1_mean": us.get(f"q1_mean_{ch}"),
            "q2_mean": us.get(f"q2_mean_{ch}"),
            "valid_weight": us.get(f"critic_valid_weight_{ch}"),
            "updated": us.get(f"critic_updated_{ch}"),
            "td_abs_mean": us.get(f"td_abs_mean_{ch}"),
        })
    actor_weights = {
        ch: {
            "weight": us.get(f"actor_weight_mean_{ch}"),
            "weight_next": us.get(f"actor_weight_next_mean_{ch}"),
        } for ch in pc
    }
    pd = ins.get("param_delta") or {}
    pd_out = {}
    for k, v in pd.items():
        if isinstance(v, dict):
            pd_out[k] = {
                "l2": v.get("l2"),
                "max_abs_delta": v.get("max_abs_delta"),
                "elements": v.get("elements"),
                "by_key": {
                    kk: {"l2": vv.get("l2"),
                         "max_abs_delta": vv.get("max_abs_delta")}
                    for kk, vv in (v.get("by_key") or {}).items()
                },
            }
    cons = ins.get("consistency") or {}
    if not isinstance(cons, dict):
        cons = {}

    stages = {
        "collection": {
            "collection_round": clocks.get("collection_round"),
            "env_step": clocks.get("env_step"),
            "agent_transition": clocks.get("agent_transition"),
            "replay_size": rs.get("size"),
            "replay_capacity": rs.get("capacity"),
            "replay_utilization": rs.get("utilization"),
            "sample_age_mean": rs.get("sample_age_mean"),
            "draw_count_mean": rs.get("draw_count_mean"),
            "draw_count_max": rs.get("draw_count_max"),
            "overwritten": rs.get("overwritten"),
        },
        "batch": {
            "size": batch.get("size"),
            "sample_id_range": batch.get("sample_id_range"),
            "n_collection_rounds": len(
                batch.get("collection_round_counts") or {}),
            "collection_round_counts": batch.get(
                "collection_round_counts"),
            "n_policies": len(
                batch.get("policy_fingerprint_counts") or {}),
            "terminated_rows": batch.get("terminated_rows"),
            "truncated_rows": batch.get("truncated_rows"),
        },
        "target": {
            "channels": target_channels,
            "reward_scale": sac_params.get("reward_scale"),
            "regularizer_mode": sac_params.get(
                "regularizer_mode", "shannon"),
        },
        "critic": {
            "channels": critic_channels,
            "critic_loss": us.get("critic_loss"),
            "td_abs_max": max(
                ((pc[c] or {}).get("td_abs_max") or 0.0 for c in pc),
                default=None),
        },
        "actor": {
            "actor_loss": us.get("actor_loss"),
            "entropy_proxy_mean": us.get("entropy_proxy_mean"),
            "log_prob_mean": us.get("log_prob_mean"),
            "valid_count": us.get("actor_valid_count"),
            "valid_weight": us.get("actor_valid_weight"),
            "pair1_frac": us.get("actor_pair1_frac"),
            "grad_norm": us.get("grad_norm_actor"),
            "channel_weights": actor_weights,
        },
        "regularizer": {
            "mode": sac_params.get("regularizer_mode", "shannon"),
            "alpha": us.get("alpha"),
            "alpha_loss": us.get("alpha_loss"),
            "regularizer_shannon": us.get("regularizer_shannon"),
            "u_floor": sac_params.get("u_floor"),
            "reg_lambda": sac_params.get("reg_lambda"),
        },
        "tickline": {
            "clocks": clocks,
            "critic_tick": ins.get("critic_tick"),
        },
        "postupdate": {
            "param_delta": pd_out,
        },
        "drift": {
            "consistency_checked": cons.get("checked"),
            "n_violations": len(cons.get("violations") or {}),
            "violations": cons.get("violations"),
        },
    }
    return {
        "dump_dir": str(dump_dir),
        "schema_version": ins.get("schema_version"),
        "kind": manifest.get("kind"),
        "evidence_level": manifest.get("evidence_level"),
        "files": manifest.get("files"),
        "critic_tick": ins.get("critic_tick"),
        "clocks": clocks,
        "hypothesis": ins.get("hypothesis") or "",
        "consistency": cons,
        "stages": stages,
    }


def dump_trace(
    dump_dir: Path,
    *,
    sample_id: Optional[int] = None,
    source_key: Optional[str] = None,
) -> Dict[str, Any]:
    from .debugkit import load_dump

    if sample_id is None and source_key is None:
        raise ValueError("trace requires --sample-id or --source-key")
    dump = load_dump(Path(dump_dir))
    batch, forward = dump["batch"], dump["forward"]
    ids = batch["sample_ids"].cpu().numpy()
    if sample_id is not None:
        matches = np.nonzero(ids == int(sample_id))[0]
    else:
        matches = np.array(
            [i for i, k in enumerate(batch["source_keys"])
             if str(k) == str(source_key)]
        )
    if len(matches) != 1:
        raise ValueError(
            f"sample matches={len(matches)} (expected exactly 1)"
        )
    i = int(matches[0])
    channels = _channels_of(dump)
    meta = (batch.get("metadata") or [{}] * len(ids))[i] or {}
    tparts = _target_parts(dump)
    aparts = _actor_parts(dump)

    def _row(t: Any) -> Any:
        return _jsonable(t[i]) if isinstance(t, torch.Tensor) else t

    per_channel = {}
    for c, ch in enumerate(channels):
        per_channel[ch] = {
            "reward": float(batch["rewards"][i, c]),
            "channel_valid": bool(batch["channel_valid"][i, c]),
            "actor_gate": float(batch["actor_gate"][i, c]),
            "actor_weight": float(batch["actor_weight"][i, c]),
            "actor_gate_next": float(batch["actor_gate_next"][i, c]),
            "actor_weight_next": float(batch["actor_weight_next"][i, c]),
            "q1_pred": float(forward["q1_pred"][ch][i]),
            "q2_pred": float(forward["q2_pred"][ch][i]),
            "target": float(forward["targets"][ch][i]),
            "td_q1": (
                float(_field(dump, "td_q1")[ch][i])
                if isinstance(_field(dump, "td_q1"), Mapping)
                else "unavailable"
            ),
            "td_q2": (
                float(_field(dump, "td_q2")[ch][i])
                if isinstance(_field(dump, "td_q2"), Mapping)
                else "unavailable"
            ),
            "next_cand_q": (
                _jsonable(forward["next_cand_q"][ch][i])
                if isinstance(_field(dump, "next_cand_q"), Mapping)
                else "unavailable"
            ),
            "actor_cand_q": (
                _jsonable(forward["actor_cand_q"][ch][i])
                if isinstance(_field(dump, "actor_cand_q"), Mapping)
                else "unavailable"
            ),
            "target_parts": {
                k: float(v[i])
                for k, v in tparts["channels"][ch].items()
            },
            "actor_q_term": float(aparts["per_channel_q"][ch][i]),
        }
    return {
        "dump_dir": str(Path(dump_dir)),
        "critic_tick": dump["manifest"].get("critic_tick"),
        "sample_id": int(ids[i]),
        "row": i,
        "source_key": str(batch["source_keys"][i]),
        "provenance": parse_source_key(str(batch["source_keys"][i])),
        "metadata": _jsonable(meta),
        "terminated": bool(batch["terminated"][i]),
        "truncated": bool(batch["truncated"][i]),
        "bootstrap": float(batch["bootstrap"][i]),
        "sample_weight": float(batch["sample_weight"][i]),
        "draw_count": (
            int(batch["draw_counts"][i])
            if "draw_counts" in batch else "unavailable"
        ),
        "per_channel": per_channel,
        "target_side": {
            "pair_index": _row(forward.get("target_pair_index")),
            "cand_f": _row(_field(dump, "next_cand_f")),
            "log_probs": _row(forward.get("next_log_probs")),
            "integration_weights": _row(
                forward.get("next_integration_weights")
            ),
        },
        "actor_side": {
            "pair_index": _row(forward.get("actor_pair_index")),
            "cand_f": _row(_field(dump, "actor_cand_f")),
            "log_probs": _row(forward.get("new_log_probs")),
            "integration_weights": _row(
                forward.get("actor_integration_weights")
            ),
            "weighted_q": _row(forward.get("weighted_q")),
            "alpha_pre": _jsonable(forward.get("alpha_pre")),
            "entropy_term": float(aparts["entropy_term"][i]),
            "reg_term": float(aparts["reg_term"][i]),
        },
        "task_facts": _jsonable(
            {k: v[i] for k, v in (batch.get("task_facts") or {}).items()}
        ),
    }


def _target_parts(dump: Mapping[str, Any]) -> Dict[str, Any]:
    """Decompose each channel's Bellman target into parts per row.

    ``target_c = reward_scale*r_c + gamma_c*bootstrap*(softq_c + reg)`` where
    ``softq_c = Σ_km w_km·Q_c(sel pair)`` and ``reg`` is −α·logp̄ (shannon)
    or the per-state U term (u_bonus/u_floor, from ``reg_next``).
    """
    batch, forward = dump["batch"], dump["forward"]
    spec = dump["spec"]
    channels = _channels_of(dump)
    gamma = {c["name"]: float(c["gamma"]) for c in spec["channels"]}
    reward_scale = float(spec["sac_params"]["reward_scale"])
    mode = spec["sac_params"].get("regularizer_mode", "shannon")

    integ = forward["next_integration_weights"].float()
    cand_q = _field(dump, "next_cand_q")
    softq: Dict[str, np.ndarray] = {}
    for c, ch in enumerate(channels):
        q_sel = cand_q[ch].float().gather(        # [B,K,M]
            -1, forward["target_pair_index"][..., None],
        ).squeeze(-1)
        softq[ch] = (integ * q_sel).sum(dim=(1, 2)).numpy()
    if mode == "shannon":
        alpha = forward["alpha_pre"].float()
        reg = (
            -alpha * (integ * forward["next_log_probs"].float())
            .sum(dim=(1, 2))
        ).numpy()
    else:
        reg = forward["reg_next"].float().numpy()
    bootstrap = batch["bootstrap"].float().numpy()
    rewards = batch["rewards"].float().numpy()
    parts: Dict[str, Any] = {"mode": mode, "channels": {}}
    for c, ch in enumerate(channels):
        r_term = rewards[:, c] * reward_scale
        boot = gamma[ch] * bootstrap
        parts["channels"][ch] = {
            "reward_term": r_term,
            "softq_term": boot * softq[ch],
            "reg_term": boot * reg,
            "total": r_term + boot * (softq[ch] + reg),
        }
    return parts


def _actor_parts(dump: Mapping[str, Any]) -> Dict[str, Any]:
    """Decompose each row's actor objective into per-channel Q + reg parts.

    ``objective_i = Σ_km w_km·(α·logp − Σ_c w_c·Q_c(sel)) + reg_i`` for
    shannon; U modes drop the α·logp term inside the candidate integral.
    """
    batch, forward = dump["batch"], dump["forward"]
    spec = dump["spec"]
    channels = _channels_of(dump)
    mode = spec["sac_params"].get("regularizer_mode", "shannon")
    integ = forward["actor_integration_weights"].float()
    pair = forward["actor_pair_index"]
    cand_q = _field(dump, "actor_cand_q")
    aw = batch["actor_weight"].float()
    share = torch.tensor(
        [1.0 if c["actor_weight_share"] else 0.0
         for c in spec["channels"]]
    )
    per_ch: Dict[str, np.ndarray] = {}
    for c, ch in enumerate(channels):
        q_sel = cand_q[ch].float().gather(-1, pair[..., None]).squeeze(-1)
        per_ch[ch] = (
            integ * (-aw[:, c, None, None] * share[c] * q_sel)
        ).sum(dim=(1, 2)).numpy()
    parts: Dict[str, Any] = {"mode": mode, "per_channel_q": per_ch}
    if mode == "shannon":
        alpha = forward["alpha_pre"].float()
        parts["entropy_term"] = (
            alpha
            * (integ * forward["new_log_probs"].float()).sum(dim=(1, 2))
        ).numpy()
    else:
        parts["entropy_term"] = np.zeros(
            int(batch["sample_ids"].numel())
        )
    parts["reg_term"] = (
        forward["reg_actor"].float().numpy()
        if isinstance(forward.get("reg_actor"), torch.Tensor)
        else np.zeros(int(batch["sample_ids"].numel()))
    )
    parts["total"] = (
        sum(per_ch.values()) + parts["entropy_term"] + parts["reg_term"]
    )
    return parts


def objective_decomp(
    dump_dir: Path,
    *,
    sample_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Part-wise decomposition of the dump's targets and actor objective.

    Every reported ``*_term`` sums back to the recorded target/objective
    within float tolerance — the decomposition is accounting, not a new
    estimate. ``sample_id`` selects one row for full per-row parts.
    """
    from .debugkit import load_dump

    dump = load_dump(Path(dump_dir))
    tparts = _target_parts(dump)
    aparts = _actor_parts(dump)
    batch = dump["batch"]
    recorded_targets = {
        ch: dump["forward"]["targets"][ch].float().numpy()
        for ch in tparts["channels"]
    }
    recorded_obj = (
        dump["forward"]["weighted_q"].float().numpy()
        + (
            dump["forward"]["reg_actor"].float().numpy()
            if isinstance(dump["forward"].get("reg_actor"), torch.Tensor)
            else 0.0
        )
    )
    rows: Optional[Dict[str, Any]] = None
    if sample_id is not None:
        ids = batch["sample_ids"].cpu().numpy()
        matches = np.nonzero(ids == int(sample_id))[0]
        if len(matches) != 1:
            raise ValueError(
                f"sample_id={sample_id} matches={len(matches)} in dump"
            )
        i = int(matches[0])
        rows = {
            "sample_id": int(sample_id),
            "target": {
                ch: {k: float(v[i]) for k, v in parts.items()}
                for ch, parts in tparts["channels"].items()
            },
            "actor": {
                "per_channel_q": {
                    ch: float(v[i]) for ch, v in aparts["per_channel_q"].items()
                },
                "entropy_term": float(aparts["entropy_term"][i]),
                "reg_term": float(aparts["reg_term"][i]),
                "total": float(aparts["total"][i]),
            },
        }
    summary = {
        "target": {
            ch: {
                k: {"mean": float(v.mean()), "std": float(v.std())}
                for k, v in parts.items()
            }
            for ch, parts in tparts["channels"].items()
        },
        "actor": {
            "per_channel_q": {
                ch: {
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                }
                for ch, v in aparts["per_channel_q"].items()
            },
            "entropy_term": {
                "mean": float(aparts["entropy_term"].mean()),
            },
            "reg_term": {"mean": float(aparts["reg_term"].mean())},
            "total": {"mean": float(aparts["total"].mean())},
        },
    }
    check = {
        "target_max_abs_err": float(max(
            np.abs(parts["total"] - recorded_targets[ch]).max()
            for ch, parts in tparts["channels"].items()
        )),
        "objective_max_abs_err": float(
            np.abs(aparts["total"] - recorded_obj).max()
        ),
    }
    return {
        "dump_dir": str(Path(dump_dir)),
        "critic_tick": dump["manifest"].get("critic_tick"),
        "regularizer_mode": tparts["mode"],
        "summary": summary,
        "row": rows,
        "check": check,
    }


def _flat_norm(grads: Iterable[Optional[torch.Tensor]]) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum().item())
    return float(total ** 0.5)


def _flat_dot(
    a: Iterable[Optional[torch.Tensor]],
    b: Iterable[Optional[torch.Tensor]],
) -> float:
    total = 0.0
    for ga, gb in zip(a, b):
        if ga is not None and gb is not None:
            total += float((ga.detach() * gb.detach()).sum().item())
    return total


def grad_attribution(
    dump_dir: Path,
    *,
    group_by: str = "collection_round",
) -> Dict[str, Any]:
    """Heavy on-demand gradient attribution for one critic_tick dump.

    Rebuilds the pre-update actor/critic from the dump spec, recomputes
    the losses with autograd enabled (captured u draws freeze the
    candidates, so the graph matches the recorded update), then takes
    per-group ``autograd.grad``:

    - **critic**: one gradient vector per provenance group per channel —
      answers "which behavior version's TD formed this channel's update";
    - **actor**: one gradient vector per objective *term* (Q /
      entropy / reg) — the SAC analogue of PPO's gradsig decomposition;
    - **alpha**: scalar per-group contributions (they sum to the total
      alpha gradient exactly).

    Offline only — never runs in the training path. Cost ≈
    (n_groups × C + 3) backwards over one batch.
    """
    from .debugkit import _build_actor, _build_critic, load_dump
    from .experiment import SACParams, SACRewardChannel
    from .trainer import (
        _RegularizerArgs,
        _forward_all_candidates,
        _state_regularizer,
    )

    if group_by not in _GROUP_KEYS:
        raise ValueError(
            f"unknown group_by {group_by!r}; allowed={sorted(_GROUP_KEYS)}"
        )
    dump = load_dump(Path(dump_dir))
    spec = dump["spec"]
    sp = SACParams(**spec["sac_params"])
    channels = tuple(SACRewardChannel(**item) for item in spec["channels"])
    batch, forward = dump["batch"], dump["forward"]
    pre, post = dump["trainer_pre"], dump["trainer_post"]
    actor = _build_actor(spec, pre)
    pre_critic = _build_critic(spec, channels, pre)
    post_critic = _build_critic(spec, channels, post)
    for p in post_critic.all_parameters():
        p.requires_grad_(False)

    device = batch["obs"].device
    obs = batch["obs"].float()
    actions = batch["actions"].float()
    channel_valid = batch["channel_valid"].bool()
    sample_weight = batch["sample_weight"].float()
    actor_weight = batch["actor_weight"].float()
    B = int(obs.shape[0])
    key_fn = _GROUP_KEYS[group_by]
    metas = batch.get("metadata") or [{}] * B
    group_names = []
    for i in range(B):
        g = key_fn(metas[i] or {})
        if g not in group_names:
            group_names.append(g)
    group_rows = {
        g: torch.tensor(
            [i for i in range(B) if key_fn(metas[i] or {}) == g],
            device=device,
        )
        for g in group_names
    }

    # ---- critic: per-channel, per-provenance-group gradient attribution
    critic_out: Dict[str, Any] = {}
    for c, ch in enumerate(channels):
        mask = channel_valid[:, c].float() * sample_weight
        mask_sum = mask.sum()
        if float(mask_sum.item()) <= 0:
            continue
        target = forward["targets"][ch.name].float().to(device)
        params = list(pre_critic.channel_parameters(ch.name))
        q1 = pre_critic.q1_forward(obs, actions, ch.name)
        q2 = pre_critic.q2_forward(obs, actions, ch.name)
        per_row = mask * ((q1 - target).pow(2) + (q2 - target).pow(2))
        total_grad = torch.autograd.grad(
            per_row.sum() / mask_sum, params, retain_graph=True,
        )
        total_norm = _flat_norm(total_grad)
        groups = []
        for g, rows in group_rows.items():
            g_loss = per_row[rows].sum() / mask_sum
            g_grad = torch.autograd.grad(
                g_loss, params, retain_graph=True,
            )
            g_norm = _flat_norm(g_grad)
            cos = (
                _flat_dot(g_grad, total_grad) / (g_norm * total_norm)
                if g_norm > 0 and total_norm > 0 else None
            )
            groups.append({
                "key": g,
                "rows": int(rows.numel()),
                "grad_norm": g_norm,
                "cos_to_total": cos,
            })
        critic_out[ch.name] = {
            "total_grad_norm": total_norm,
            "groups": groups,
        }

    # ---- actor: per-term gradient decomposition
    u_actor = forward["u_actor"].float().to(device)
    new_actions, new_log_probs, integ_w = actor.expectation_samples(
        obs, u_actor,
    )
    q1_all = torch.stack(
        [
            _forward_all_candidates(
                lambda o, a: post_critic.q1_forward(o, a, ch.name),
                obs, new_actions,
            )
            for ch in channels
        ],
        dim=-1,
    )
    q2_all = torch.stack(
        [
            _forward_all_candidates(
                lambda o, a: post_critic.q2_forward(o, a, ch.name),
                obs, new_actions,
            )
            for ch in channels
        ],
        dim=-1,
    )
    q_online_pairs = torch.stack((q1_all, q2_all), dim=-1)
    share = torch.tensor(
        [1.0 if ch.actor_weight_share else 0.0 for ch in channels],
        dtype=actor_weight.dtype, device=device,
    )
    w_actor = actor_weight[:, None, None, :, None] * share[None, None, None, :, None]
    f_actor = (q_online_pairs * w_actor).sum(dim=3)
    pair_actor = f_actor.detach().argmin(dim=-1)
    f_selected = f_actor.gather(-1, pair_actor[..., None]).squeeze(-1)
    alpha = pre["log_alpha"].float().exp().to(device)
    is_shannon = sp.regularizer_mode == "shannon"
    if is_shannon:
        q_term = -(integ_w * f_selected).sum(dim=(1, 2))
        ent_term = (integ_w * new_log_probs).sum(dim=(1, 2)) * alpha
    else:
        q_term = -(integ_w * f_selected).sum(dim=(1, 2))
        ent_term = torch.zeros_like(q_term)
    reg_term = torch.zeros_like(q_term)
    if not is_shannon:
        reg_term = -_state_regularizer(
            actor, obs,
            _RegularizerArgs(
                regularizer_mode=sp.regularizer_mode,
                reg_lambda=sp.reg_lambda,
                u_floor=sp.u_floor,
                u_kind=sp.u_kind,
            ),
        )
    actor_rows = channel_valid.all(dim=1).float() * sample_weight
    actor_den = actor_rows.sum()
    actor_params = list(actor.parameters())
    terms = {
        "q": q_term,
        "entropy": ent_term,
        "reg": reg_term,
    }
    total_obj = (actor_rows * sum(terms.values())).sum() / actor_den
    total_actor_grad = torch.autograd.grad(
        total_obj, actor_params, retain_graph=True,
    )
    total_actor_norm = _flat_norm(total_actor_grad)
    term_rows = []
    for name, term in terms.items():
        loss = (actor_rows * term).sum() / actor_den
        if not loss.requires_grad:
            # identically-zero term (reg in shannon, entropy in U modes)
            term_rows.append({
                "term": name, "loss": 0.0,
                "grad_norm": 0.0, "cos_to_total": None,
            })
            continue
        g = torch.autograd.grad(loss, actor_params, retain_graph=True)
        g_norm = _flat_norm(g)
        cos = (
            _flat_dot(g, total_actor_grad)
            / (g_norm * total_actor_norm)
            if g_norm > 0 and total_actor_norm > 0 else None
        )
        term_rows.append({
            "term": name,
            "loss": float(loss.item()),
            "grad_norm": g_norm,
            "cos_to_total": cos,
        })

    # ---- alpha: exact scalar per-group contributions
    alpha_out: Dict[str, Any] = {"groups": []}
    logp_bar = (integ_w.detach() * new_log_probs.detach()).sum(dim=(1, 2))
    target_entropy = (
        -float(spec["actor"]["action_dim"])
        if sp.target_entropy is None else float(sp.target_entropy)
    )
    per_row_alpha = -actor_rows * (logp_bar + target_entropy) / actor_den
    alpha_out["total_grad"] = float(per_row_alpha.sum().item())
    for g, rows in group_rows.items():
        alpha_out["groups"].append({
            "key": g,
            "rows": int(rows.numel()),
            "grad_contrib": float(per_row_alpha[rows].sum().item()),
        })

    return {
        "dump_dir": str(Path(dump_dir)),
        "critic_tick": dump["manifest"].get("critic_tick"),
        "group_by": group_by,
        "regularizer_mode": sp.regularizer_mode,
        "n_groups": len(group_names),
        "critic": critic_out,
        "actor": {
            "total_grad_norm": total_actor_norm,
            "terms": term_rows,
        },
        "alpha": alpha_out,
    }


def _log_prob_of(actor: Any, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "log_prob_of"):
        return actor.log_prob_of(obs, act)
    return actor.evaluate_actions(obs, act)[0]


def policy_drift(dump_dir: Path) -> Dict[str, Any]:
    """Fixed-input before/after drift for one critic_tick dump.

    Rebuilds pre/post actor and critic, then measures on the dump's own
    batch states: deterministic-action drift, logπ drift on the recorded
    behavior actions, per-channel Q drift at (s,a) and at (s,π_det(s)),
    and the alpha displacement — the SAC analogue of PPO's delta view.
    """
    from .debugkit import _build_actor, _build_critic, load_dump
    from .experiment import SACRewardChannel

    dump = load_dump(Path(dump_dir))
    spec = dump["spec"]
    channels = tuple(SACRewardChannel(**item) for item in spec["channels"])
    pre, post = dump["trainer_pre"], dump["trainer_post"]
    actor_pre = _build_actor(spec, pre)
    actor_post = _build_actor(spec, post)
    critic_pre = _build_critic(spec, channels, pre)
    critic_post = _build_critic(spec, channels, post)
    batch = dump["batch"]
    obs = batch["obs"].float()
    actions = batch["actions"].float()

    with torch.no_grad():
        det_pre = actor_pre.deterministic_action(obs)
        det_post = actor_post.deterministic_action(obs)
        action_diff = (det_post - det_pre)
        logp_pre = _log_prob_of(actor_pre, obs, actions).float()
        logp_post = _log_prob_of(actor_post, obs, actions).float()
        logp_diff = logp_post - logp_pre

        per_channel: Dict[str, Any] = {}
        for ch in channels:
            q_sa_pre = (
                critic_pre.q1_forward(obs, actions, ch.name)
                + critic_pre.q2_forward(obs, actions, ch.name)
            ) / 2.0
            q_sa_post = (
                critic_post.q1_forward(obs, actions, ch.name)
                + critic_post.q2_forward(obs, actions, ch.name)
            ) / 2.0
            q_det_pre = (
                critic_pre.q1_forward(obs, det_post, ch.name)
                + critic_pre.q2_forward(obs, det_post, ch.name)
            ) / 2.0
            q_det_post = (
                critic_post.q1_forward(obs, det_post, ch.name)
                + critic_post.q2_forward(obs, det_post, ch.name)
            ) / 2.0
            d_sa = (q_sa_post - q_sa_pre)
            d_det = (q_det_post - q_det_pre)
            per_channel[ch.name] = {
                "q_drift_sa": {
                    "mean": float(d_sa.mean()),
                    "abs_max": float(d_sa.abs().max()),
                },
                "q_drift_at_det_action": {
                    "mean": float(d_det.mean()),
                    "abs_max": float(d_det.abs().max()),
                },
            }
        alpha_delta = float(
            post["log_alpha"].float() - pre["log_alpha"].float()
        )
    return {
        "dump_dir": str(Path(dump_dir)),
        "critic_tick": dump["manifest"].get("critic_tick"),
        "n_probe_rows": int(obs.shape[0]),
        "det_action_drift": {
            "l2_mean": float(action_diff.norm(dim=-1).mean()),
            "l2_max": float(action_diff.norm(dim=-1).max()),
            "abs_max": float(action_diff.abs().max()),
        },
        "logp_drift": {
            "mean": float(logp_diff.mean()),
            "abs_max": float(logp_diff.abs().max()),
        },
        "per_channel": per_channel,
        "alpha_delta": alpha_delta,
    }


# ---------------------------------------------------------- attribution


_GROUP_KEYS = {
    "collection_round": lambda meta: str(
        (meta.get("collection") or {}).get("collection_round", "unknown")
    ),
    "policy_fingerprint": lambda meta: str(
        (meta.get("collection") or {}).get("policy_fingerprint", "unknown")
    )[:12],
    "explore_factor": lambda meta: str(
        (meta.get("behavior") or {}).get("explore_factor", "unknown")
    ),
    "random_start": lambda meta: str(
        bool((meta.get("behavior") or {}).get("random_start", False))
    ),
}


def batch_attribution(
    dump_dir: Path,
    *,
    group_by: str = "collection_round",
) -> Dict[str, Any]:
    """Attribute one dump's update contribution to sample provenance.

    Groups the batch rows by a metadata key, then for each group reports
    row counts, per-channel critic-loss share (``mask * td^2 / Σ``) and
    actor-objective share — the joint provenance × contribution view that
    answers "which behavior version's samples drove this update".
    """
    from .debugkit import load_dump

    if group_by not in _GROUP_KEYS:
        raise ValueError(
            f"unknown group_by {group_by!r}; allowed={sorted(_GROUP_KEYS)}"
        )
    dump = load_dump(Path(dump_dir))
    batch, forward = dump["batch"], dump["forward"]
    channels = _channels_of(dump)
    B = int(batch["sample_ids"].numel())
    key_fn = _GROUP_KEYS[group_by]

    td2: Dict[str, np.ndarray] = {}
    td1_map, td2_map = _field(dump, "td_q1"), _field(dump, "td_q2")
    for ch in channels:
        if isinstance(td1_map, Mapping) and ch in td1_map:
            t1 = td1_map[ch].float().numpy()
            t2 = td2_map[ch].float().numpy()
        else:
            t = forward["targets"][ch].float().numpy()
            t1 = forward["q1_pred"][ch].float().numpy() - t
            t2 = forward["q2_pred"][ch].float().numpy() - t
        td2[ch] = (t1 ** 2 + t2 ** 2) / 2.0
    valid = batch["channel_valid"].float().numpy()
    sw = batch["sample_weight"].float().numpy()
    mask = valid * sw[:, None]                              # [B,C]
    contrib = {
        ch: mask[:, c] * td2[ch] for c, ch in enumerate(channels)
    }
    contrib_totals = {
        ch: float(contrib[ch].sum()) for ch in channels
    }

    actor_rows = batch["channel_valid"].all(dim=1).float().numpy() * sw
    objective = (
        forward["weighted_q"].float().numpy()
        if "weighted_q" in forward else np.zeros(B)
    )
    if isinstance(forward.get("reg_actor"), torch.Tensor):
        objective = objective + forward["reg_actor"].float().numpy()
    actor_contrib = actor_rows * objective
    actor_total = float(actor_contrib.sum())

    metas = (batch.get("metadata") or [{}] * B)
    groups: Dict[str, Dict[str, Any]] = {}
    for i in range(B):
        g = key_fn(metas[i] or {})
        entry = groups.setdefault(g, {
            "key": g,
            "rows": 0,
            "actor_valid_rows": 0,
            "td_abs_sum": {ch: 0.0 for ch in channels},
            "critic_loss_sum": {ch: 0.0 for ch in channels},
            "actor_objective_sum": 0.0,
        })
        entry["rows"] += 1
        entry["actor_valid_rows"] += int(actor_rows[i] > 0)
        for ch in channels:
            entry["td_abs_sum"][ch] += float(np.sqrt(td2[ch][i]))
            entry["critic_loss_sum"][ch] += float(contrib[ch][i])
        entry["actor_objective_sum"] += float(actor_contrib[i])

    out_groups = []
    for g in sorted(groups, key=lambda k: -groups[k]["rows"]):
        entry = groups[g]
        out_groups.append({
            "key": g,
            "rows": entry["rows"],
            "batch_share": entry["rows"] / max(B, 1),
            "actor_valid_rows": entry["actor_valid_rows"],
            "critic_loss_share": {
                ch: (
                    entry["critic_loss_sum"][ch] / contrib_totals[ch]
                    if contrib_totals[ch] > 0 else None
                )
                for ch in channels
            },
            "td_abs_mean": {
                ch: (
                    entry["td_abs_sum"][ch] / max(entry["rows"], 1)
                )
                for ch in channels
            },
            "actor_objective_share": (
                entry["actor_objective_sum"] / actor_total
                if abs(actor_total) > 0 else None
            ),
        })
    return {
        "dump_dir": str(Path(dump_dir)),
        "critic_tick": dump["manifest"].get("critic_tick"),
        "group_by": group_by,
        "n_rows": B,
        "groups": out_groups,
    }


# ---------------------------------------------------------------- replay


def replay_stats_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Offline replay demographics from a persisted replay state dict."""
    size = int(state["size"])
    next_id = int(state["next_sample_id"])
    ids = np.asarray(state["sample_ids"][:size], dtype=np.int64)
    ages = next_id - ids if size else np.empty(0, dtype=np.int64)
    round_counts: Dict[str, int] = {}
    policy_fp: Dict[str, int] = {}
    explore: Dict[str, int] = {}
    random_start = 0
    for meta in list(state.get("metadata") or [])[:size]:
        if meta is None:
            continue
        collection = meta.get("collection") or {}
        behavior = meta.get("behavior") or {}
        r = str(collection.get("collection_round", "?"))
        round_counts[r] = round_counts.get(r, 0) + 1
        fp = str(collection.get("policy_fingerprint", "?"))[:12]
        policy_fp[fp] = policy_fp.get(fp, 0) + 1
        e = str(behavior.get("explore_factor", "?"))
        explore[e] = explore.get(e, 0) + 1
        if behavior.get("random_start"):
            random_start += 1
    draw_counts = state.get("draw_counts")
    out: Dict[str, Any] = {
        "schema": state.get("schema"),
        "size": size,
        "capacity": int(state["capacity"]),
        "total_inserted": int(state["total_inserted"]),
        "overwritten": int(state["overwritten"]),
        "next_sample_id": next_id,
        "sample_age": {
            "min": int(ages.min()) if size else None,
            "mean": float(ages.mean()) if size else None,
            "max": int(ages.max()) if size else None,
            "quantiles": (
                np.quantile(ages, [0.5, 0.9, 0.99]).tolist()
                if size else []
            ),
        },
        "collection_round_counts": round_counts,
        "policy_fingerprint_counts": policy_fp,
        "explore_factor_counts": explore,
        "random_start_rows": random_start,
    }
    if draw_counts is not None:
        dc = np.asarray(draw_counts[:size], dtype=np.int64)
        out["reuse"] = {
            "draw_count_min": int(dc.min()) if size else None,
            "draw_count_mean": float(dc.mean()) if size else None,
            "draw_count_max": int(dc.max()) if size else None,
        }
    else:
        out["reuse"] = "unavailable"
    return out


def slice_context(
    path: Path,
    *,
    source_key: Optional[str] = None,
    slice_id: Optional[str] = None,
    sample_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Rebuild the live neighbourhood of one slice inside a replay state.

    Loads ``replay.pt`` (from a run dir, checkpoint dir, or the file
    itself), locates the target slice (via ``source_key`` prefix,
    explicit ``slice_id``, or a ``sample_id`` lookup), then returns the
    still-live frames of that slice ordered by frame index, the frame
    gaps left by FIFO eviction, and the live slices of the same episode.
    """
    path = Path(path)
    if path.is_dir() and (path / "checkpoints").is_dir():
        ckpts = sorted((path / "checkpoints").iterdir())
        if not ckpts:
            raise ValueError(f"no checkpoints under {path}")
        path = ckpts[-1]
    if path.is_dir():
        path = path / "replay.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    size = int(state["size"])
    # Live slots are those ever written and not yet overwritten — after
    # the write pointer wraps they are NOT the first `size` entries, so
    # select by sample_id >= 0 rather than slicing.
    all_ids = np.asarray(state["sample_ids"])
    live = np.nonzero(all_ids >= 0)[0][:size]
    source_keys = [state["source_keys"][i] for i in live]
    slice_ids = [state["slice_ids"][i] for i in live]
    frame_indices = np.asarray(state["frame_indices"])[live]
    sample_ids = all_ids[live]
    draw_counts = np.asarray(
        state.get("draw_counts", np.zeros(len(all_ids))), dtype=np.int64,
    )[live]
    rewards = np.asarray(state["rewards"])[live]
    terminated_arr = np.asarray(state["terminated"])[live]
    truncated_arr = np.asarray(state["truncated"])[live]

    target_slice = slice_id
    episode_prefix = None
    if target_slice is None and source_key is not None:
        parsed = parse_source_key(str(source_key))
        if parsed.get("parseable"):
            episode_prefix = (
                f"{parsed['run_id']}/r{parsed['collection_round']:06d}/"
                f"j{parsed['job_index']:06d}/{parsed['agent_id']}/"
                f"e{parsed['episode_seed']}"
            )
        # direct hit: the row carrying this source_key names its slice
        for i, k in enumerate(source_keys):
            if str(k) == str(source_key):
                target_slice = slice_ids[i]
                break
        # frame evicted: fall back to any live slice in the same episode
        if target_slice is None and episode_prefix is not None:
            for sid in slice_ids:
                if sid and str(sid).startswith(episode_prefix + "/"):
                    target_slice = sid
                    break
    if target_slice is None and sample_id is not None:
        matches = np.nonzero(sample_ids == int(sample_id))[0]
        if len(matches) == 1:
            target_slice = slice_ids[int(matches[0])]
    if target_slice is None:
        raise ValueError(
            "cannot resolve target slice — provide source_key, slice_id, "
            "or a live sample_id"
        )
    episode_prefix = episode_prefix or str(target_slice).rsplit("/slice", 1)[0]

    slots = [
        i for i, sid in enumerate(slice_ids) if str(sid) == str(target_slice)
    ]
    slots.sort(key=lambda i: int(frame_indices[i]))
    frames = [{
        "live_index": int(i),
        "sample_id": int(sample_ids[i]),
        "frame_index": int(frame_indices[i]),
        "draw_count": int(draw_counts[i]),
        "reward_sum": float(rewards[i].sum()),
        "terminated": bool(terminated_arr[i]),
        "truncated": bool(truncated_arr[i]),
    } for i in slots]
    fi = [f["frame_index"] for f in frames]
    gaps = (
        sorted(set(range(fi[0], fi[-1] + 1)) - set(fi))
        if len(fi) > 1 else []
    )
    same_episode: Dict[str, int] = {}
    for sid in slice_ids:
        if sid and str(sid).startswith(episode_prefix + "/"):
            same_episode[str(sid)] = same_episode.get(str(sid), 0) + 1
    return {
        "source": str(path),
        "slice_id": str(target_slice),
        "episode_prefix": episode_prefix,
        "live_frames": frames,
        "live_count": len(frames),
        "frame_index_range": [fi[0], fi[-1]] if fi else None,
        "missing_frame_indices": gaps,
        "same_episode_live_slices": same_episode,
        "note": (
            "replay is FIFO — frames absent here were evicted, not "
            "dropped; use the earliest checkpoint for older context"
        ),
    }


def replay_report(path: Path) -> Dict[str, Any]:
    """Report replay demographics for a run dir, checkpoint dir, or
    a raw ``replay.pt`` state-dict file."""
    path = Path(path)
    if path.is_dir() and (path / "checkpoints").is_dir():
        ckpts = sorted((path / "checkpoints").iterdir())
        if not ckpts:
            raise ValueError(f"no checkpoints under {path}")
        path = ckpts[-1]
    if path.is_dir():
        path = path / "replay.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    out = replay_stats_from_state(state)
    out["source"] = str(path)
    return out


# ---------------------------------------------------------------- generic


def query_payload(payload: Any, path: str) -> Any:
    """Dot-path query into a loaded JSON-able artifact ('a.b.0.c')."""
    cur = payload
    for part in [p for p in str(path).split(".") if p != ""]:
        if isinstance(cur, Mapping):
            if part not in cur:
                raise KeyError(f"key {part!r} not found (have {list(cur)[:12]})")
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {part!r}")
    return cur


def load_artifact(path: Path) -> Any:
    """Load a run/dump/json artifact into a JSON-able payload for `query`."""
    path = Path(path)
    if path.is_dir() and (path / "manifest.json").exists():
        from .debugkit import load_dump
        return _jsonable(load_dump(path))
    if path.is_dir() and (path / "metrics" / "events.jsonl").exists():
        return {"run": run_summary(path)}
    if path.is_file() and path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported artifact path: {path}")
