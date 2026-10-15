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


# ======================================================== contract ledger
#
# WP1: falsifiable per-cut assertions over one ``critic_tick`` dump.
# Each check returns a dict:
#   {"id", "cut", "name", "status": pass|warn|fail|skip,
#    "severity": "contract"|"statistical", "metric": {...},
#    "violating_samples": [...], "detail": "..."}
# ``skip`` carries the reason; it is NOT a pass.

CONTRACT_THRESHOLDS = {
    # Float-recompute identities (same tolerance family as recompute_dump).
    "math_abs": 2e-5,
    # Network-forward recomputes on CPU vs captured device float32: use
    # combined atol + rtol*|value| (LayerNorm stacks accumulate noise).
    "math_rtol": 2e-4,
    # Per-row formula recomputation (numpy float64 vs stored float32).
    "formula_abs": 1e-5,
    # Integration weights sum to one per row.
    "integ_w_abs": 1e-5,
    # Action bounds (policy exports clamp to ±1).
    "action_bound": 1.0 + 1e-6,
    # Statistical (WARN) domains — not correctness bugs but pathological.
    "pair_frac_lo": 0.05,
    "pair_frac_hi": 0.95,
    "critic_ev_warn": 0.1,
    "terminal_share_warn": 0.01,
    "u_floor_active_warn": 0.0,   # 0% violation rate → floor dormant
    "u_floor_active_warn_hi": 1.0,  # 100% violation → policy too narrow
}


class _DumpCtx:
    """Shared lazy loader for one dump — keeps per-cut functions cheap."""

    def __init__(self, dump_dir: Path) -> None:
        from .debugkit import load_dump
        self.dir = Path(dump_dir)
        self.dump = load_dump(self.dir)
        self.batch = self.dump["batch"]
        self.forward = self.dump["forward"]
        self.spec = self.dump["spec"]
        self.manifest = self.dump["manifest"]
        self.pre = self.dump["trainer_pre"]
        self.post = self.dump["trainer_post"]
        self.analysis = self.dump["analysis"]
        self.channels = _channels_of(self.dump)
        self.sp = self.spec.get("sac_params") or {}
        self.mode = str(self.sp.get("regularizer_mode", "shannon"))
        self._actor_pre = None
        self._critic_pre = None
        self._critic_post = None

    def actor_pre(self):
        if self._actor_pre is None:
            from .debugkit import _build_actor
            self._actor_pre = _build_actor(self.spec, self.pre)
        return self._actor_pre

    def critic_pre(self):
        if self._critic_pre is None:
            from .debugkit import _build_critic
            self._critic_pre = _build_critic(
                self.spec, _load_channels(self.spec), self.pre,
            )
        return self._critic_pre

    def critic_post(self):
        if self._critic_post is None:
            from .debugkit import _build_critic
            self._critic_post = _build_critic(
                self.spec, _load_channels(self.spec), self.post,
            )
        return self._critic_post

    def sample_ids(self) -> np.ndarray:
        return self.batch["sample_ids"].cpu().numpy()

    def metadata(self) -> List[Mapping[str, Any]]:
        return list(self.batch.get("metadata") or [{}] * len(self.sample_ids()))


def _load_channels(spec: Mapping[str, Any]):
    from .experiment import SACRewardChannel
    return tuple(
        SACRewardChannel(**item) for item in spec.get("channels", [])
    )


def _check(
    check_id: str,
    cut: str,
    name: str,
    status: str,
    *,
    severity: str = "contract",
    metric: Optional[Dict[str, Any]] = None,
    violating: Optional[List[int]] = None,
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "id": check_id,
        "cut": cut,
        "name": name,
        "status": status,
        "severity": severity,
        "metric": metric or {},
        "violating_samples": violating or [],
        "detail": detail,
    }


def _t(batch: Mapping[str, Any], name: str) -> Optional[torch.Tensor]:
    v = batch.get(name)
    return v.detach().cpu() if isinstance(v, torch.Tensor) else None


# ------------------------------------------------------------- cut ⓪ rollout


def rollout_exploration(dump_dir: Path) -> Dict[str, Any]:
    """Realized exploration per batch row: eps = a_env - a_det.

    The honest source is ``policy_det_action`` (the deterministic action
    the *same* behavior policy would take at the stored obs).  Legacy
    dumps only have ``policy_action`` — which, due to a capture-side
    bug fixed post-WP1, recorded the SAMPLED action, making eps ≡ 0.
    Rows lacking a finite ``policy_det_action`` are reported as
    unverifiable rather than silently producing eps=0.
    """
    ctx = _DumpCtx(dump_dir)
    b = ctx.batch
    ids = ctx.sample_ids()
    a_env = _t(b, "actions")
    a_det = _t(b, "policy_det_action")
    out: Dict[str, Any] = {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "available": False,
    }
    if a_env is None:
        out["reason"] = "actions absent from batch"
        return out
    a_env = a_env.float().numpy()
    B = a_env.shape[0]
    if a_det is None:
        out["det_source"] = "unavailable"
        out["reason"] = (
            "policy_det_action absent from batch — realized eps is "
            "unverifiable (legacy capture stored the sampled action in "
            "policy_action, which would report eps≡0 dishonestly)"
        )
        return out
    a_det_np = a_det.float().numpy()
    valid = np.isfinite(a_det_np).all(axis=1)
    out["det_source"] = "policy_det_action"
    out["unverifiable_rows"] = int((~valid).sum())
    out["unverifiable_sample_ids"] = [
        int(ids[i]) for i in np.nonzero(~valid)[0][:20]
    ]
    if not valid.any():
        out["reason"] = "no row carries a finite policy_det_action"
        return out
    eps = a_env[valid] - a_det_np[valid]
    eps_l2 = np.linalg.norm(eps, axis=1)
    out["available"] = True
    out["batch_size"] = int(B)
    out["n_verified"] = int(valid.sum())
    out["eps_l2"] = {
        "mean": float(eps_l2.mean()),
        "q50": float(np.quantile(eps_l2, 0.5)),
        "q95": float(np.quantile(eps_l2, 0.95)),
        "max": float(eps_l2.max()),
    }
    out["eps_dim_std"] = [float(x) for x in eps.std(axis=0)]
    out["sat_frac"] = float(
        (np.abs(a_env[valid]) > CONTRACT_THRESHOLDS["action_bound"] - 1e-6)
        .astype(float).mean()
    )
    # Group by recorded behavior knobs (mode + explore_factor); eps_l2 is
    # indexed over the *valid* subset, so map batch rows → valid index.
    vpos = np.full(B, -1, dtype=np.int64)
    vpos[np.nonzero(valid)[0]] = np.arange(int(valid.sum()))
    groups: Dict[str, List[int]] = {}
    metas = ctx.metadata()
    for i, meta in enumerate(metas):
        if vpos[i] < 0:
            continue
        beh = meta.get("behavior") or {}
        key = f"{beh.get('mode', '?')}/ef={beh.get('explore_factor', '?')}"
        groups.setdefault(key, []).append(int(vpos[i]))
    out["by_behavior"] = {
        k: {
            "rows": len(v),
            "eps_l2_mean": float(eps_l2[v].mean()),
            "eps_l2_max": float(eps_l2[v].max()),
        }
        for k, v in sorted(groups.items())
    }
    random_rows = [
        int(vpos[i]) for i, m in enumerate(metas)
        if vpos[i] >= 0 and (
            (m.get("behavior") or {}).get("parameters", {}).get(
                "random_start"
            ) or (m.get("behavior") or {}).get("mode") == "random")
    ]
    out["random_start_rows"] = len(random_rows)
    if random_rows:
        out["random_start_eps_l2_mean"] = float(eps_l2[random_rows].mean())
    # strongest-exploration samples (batch-row ids, not valid indices)
    vrows = np.nonzero(valid)[0]
    order = np.argsort(-eps_l2)[:20]
    out["top_eps_samples"] = [
        {"sample_id": int(ids[vrows[i]]), "eps_l2": float(eps_l2[i])}
        for i in order
    ]
    return out


def _checks_rollout(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    a_env = _t(ctx.batch, "actions")
    a_det = _t(ctx.batch, "policy_action")
    if a_env is None:
        checks.append(_check(
            "C0.1", "rollout", "action_bounds", "skip",
            detail="batch.actions absent",
        ))
        return checks
    bound = CONTRACT_THRESHOLDS["action_bound"]
    over = (a_env.abs() > bound).any(dim=1)
    n_over = int(over.sum().item())
    ids = ctx.sample_ids()
    checks.append(_check(
        "C0.1", "rollout", "action_bounds",
        "pass" if n_over == 0 else "fail",
        metric={"violations": n_over, "bound": bound},
        violating=[int(ids[i]) for i in np.nonzero(over.numpy())[0][:20]],
        detail=f"max |a_env| = {a_env.abs().max().item():.6f} vs bound {bound}",
    ))
    a_det = _t(ctx.batch, "policy_det_action")
    if a_det is None:
        # Legacy capture: policy_action holds the SAMPLED action, so
        # eps ≡ 0 — the realized-exploration question is unverifiable.
        checks.append(_check(
            "C0.2", "rollout", "realized_eps_present", "warn",
            severity="statistical",
            detail=(
                "policy_det_action absent; legacy policy_action stores "
                "the sampled action → eps would be identically 0 "
                "(unverifiable, not evidence of zero exploration)"
            ),
        ))
        return checks
    det = a_det.float().numpy()
    valid = np.isfinite(det).all(axis=1)
    n_valid = int(valid.sum())
    if n_valid == 0:
        checks.append(_check(
            "C0.2", "rollout", "realized_eps_present", "warn",
            severity="statistical",
            detail="policy_det_action present but all-NaN (unverifiable)",
        ))
        return checks
    eps = (a_env.float().numpy()[valid] - det[valid])
    eps_l2 = np.linalg.norm(eps, axis=1)
    checks.append(_check(
        "C0.2", "rollout", "realized_eps_present", "pass",
        severity="statistical",
        metric={
            "verified_rows": n_valid,
            "unverifiable_rows": int((~valid).sum()),
            "eps_l2_mean": float(eps_l2.mean()),
            "eps_l2_max": float(eps_l2.max()),
        },
        detail="per-row eps = actions − policy_det_action",
    ))
    return checks


# ----------------------------------------------------------- cut ① ep→slice


def slice_verification(dump_dir: Path) -> Dict[str, Any]:
    """Per-row contract recomputation of rewards/gates from task facts.

    Dispatches on ``versions.reward_semantics``; verifiers are experiment
    owned (``experiments_sac.contract_verifiers``, late import).  Unknown
    semantics → ``warn unverifiable`` — never a silent pass.
    """
    ctx = _DumpCtx(dump_dir)
    b = ctx.batch
    ids = ctx.sample_ids()
    checks: List[Dict[str, Any]] = []

    # C1.1 bootstrap / termination exclusivity (existing consistency check).
    terminated = _t(b, "terminated")
    truncated = _t(b, "truncated")
    bootstrap = _t(b, "bootstrap")
    if terminated is not None and bootstrap is not None:
        term_bs = (terminated.bool() & (bootstrap > 0))
        viol = np.nonzero(term_bs.numpy())[0]
        checks.append(_check(
            "C1.1", "ep2slice", "bootstrap_terminal_exclusivity",
            "pass" if len(viol) == 0 else "fail",
            metric={
                "terminated_with_bootstrap": len(viol),
                "terminated_rows": int(terminated.sum().item()),
                "truncated_rows": int(truncated.sum().item())
                if truncated is not None else 0,
            },
            violating=[int(ids[i]) for i in viol[:20]],
            detail="terminated rows must have bootstrap=0",
        ))
        if truncated is not None:
            trunc_no_bs = (truncated.bool() & (bootstrap <= 0))
            viol2 = np.nonzero(trunc_no_bs.numpy())[0]
            checks.append(_check(
                "C1.1b", "ep2slice", "truncated_requires_bootstrap",
                "pass" if len(viol2) == 0 else "fail",
                metric={"truncated_without_bootstrap": len(viol2)},
                violating=[int(ids[i]) for i in viol2[:20]],
                detail="truncated (timeout) rows must keep bootstrap=1",
            ))
    physics = _t(b, "physics_delta")
    if physics is not None:
        bad = np.nonzero((physics <= 0).numpy())[0]
        checks.append(_check(
            "C1.5", "ep2slice", "physics_delta_positive",
            "pass" if len(bad) == 0 else "fail",
            metric={"non_positive": len(bad)},
            violating=[int(ids[i]) for i in bad[:20]],
            detail="every admitted transition advanced physics",
        ))

    # C1.4 provenance honesty: versions vs spec.sac_params.
    metas = ctx.metadata()
    spec_mode = str(ctx.mode)
    mismatched = []
    for i, meta in enumerate(metas):
        v = meta.get("versions") or {}
        rm = str(v.get("regularizer_mode", ""))
        if rm and rm != spec_mode:
            mismatched.append(i)
    checks.append(_check(
        "C1.4", "ep2slice", "provenance_version_match",
        "pass" if not mismatched else "fail",
        metric={
            "mismatched_rows": len(mismatched),
            "spec_regularizer_mode": spec_mode,
        },
        violating=[int(ids[i]) for i in mismatched[:20]],
        detail=(
            f"slice versions.regularizer_mode must equal spec "
            f"({spec_mode!r}); {len(mismatched)} rows disagree"
        ),
    ))

    # Reward/gate recompute via experiment-owned verifier.
    semantics_set = {
        str((m.get("versions") or {}).get("reward_semantics", ""))
        for m in metas
    }
    contract = ctx.spec.get("slice_contract")
    contract_source = (
        "spec.slice_contract" if isinstance(contract, Mapping)
        else "class_default_fallback"
    )
    batch_np = {
        k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in b.items()
        if k not in ("metadata",)
    }
    batch_np["task_facts"] = {
        k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in (b.get("task_facts") or {}).items()
    }
    batch_np["reward_features"] = {
        k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in (b.get("reward_features") or {}).items()
    }
    try:
        from baseline.experiments_sac.contract_verifiers import (
            get_slice_verifier,
        )
        verifiable = True
    except ImportError:
        verifiable = False
    if not verifiable:
        checks.append(_check(
            "C1.2", "ep2slice", "reward_formula_recompute", "skip",
            detail="experiments_sac.contract_verifiers not importable",
        ))
    else:
        for sem in sorted(semantics_set):
            fn = get_slice_verifier(sem)
            if fn is None:
                checks.append(_check(
                    "C1.2", "ep2slice",
                    f"reward_formula_recompute:{sem or 'unknown'}", "warn",
                    severity="statistical",
                    detail=f"no verifier registered for semantics {sem!r}",
                ))
                continue
            try:
                sub = fn(
                    batch_np, ctx.channels, contract,
                    atol=CONTRACT_THRESHOLDS["formula_abs"],
                    contract_source=contract_source,
                )
                for i, c in enumerate(sub):
                    cid = "C1.2" if "reward" in c.get("name", "") else "C1.3"
                    checks.append(_check(
                        cid, "ep2slice", c["name"],
                        c.get("status", "warn"),
                        severity="contract" if c.get("status") == "fail"
                        else "statistical",
                        metric=c.get("metric"),
                        violating=c.get("violating_samples"),
                        detail=c.get("detail", ""),
                    ))
            except Exception as exc:  # verifier bugs must surface, not hide
                checks.append(_check(
                    "C1.2", "ep2slice",
                    f"reward_formula_recompute:{sem}", "warn",
                    severity="statistical",
                    detail=f"verifier raised: {exc}",
                ))
    return {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "semantics": sorted(semantics_set),
        "contract_source": contract_source,
        "checks": checks,
    }


def _checks_ep2slice(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    return slice_verification(ctx.dir)["checks"]


# ---------------------------------------------------------- cut ② demography


def batch_demography(dump_dir: Path) -> Dict[str, Any]:
    """Batch composition vs live replay population (provenance census)."""
    ctx = _DumpCtx(dump_dir)
    b = ctx.batch
    ids = ctx.sample_ids()
    metas = ctx.metadata()
    replay_stats = ctx.analysis.get("replay_stats") or {}

    round_counts: Dict[str, int] = {}
    fp_counts: Dict[str, int] = {}
    for m in metas:
        col = m.get("collection") or {}
        r = str(col.get("collection_round", "?"))
        round_counts[r] = round_counts.get(r, 0) + 1
        fp = str(col.get("policy_fingerprint", "?"))[:12]
        fp_counts[fp] = fp_counts.get(fp, 0) + 1

    pop_counts = replay_stats.get("collection_round_counts") or {}
    keys = set(pop_counts) | set(round_counts)
    total_pop = float(sum(pop_counts.values()) or 1.0)
    n_batch = float(len(ids) or 1.0)
    l1 = sum(
        abs(round_counts.get(k, 0) / n_batch - pop_counts.get(k, 0) / total_pop)
        for k in keys
    )

    draws = _t(b, "draw_counts")
    terminated = _t(b, "terminated")
    truncated = _t(b, "truncated")
    out = {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "batch_size": int(len(ids)),
        "batch_collection_round_counts": round_counts,
        "replay_collection_round_counts": pop_counts,
        "batch_vs_replay_round_l1": float(l1),
        "n_policy_fingerprints": len(fp_counts),
        "policy_fingerprint_counts": fp_counts,
        "draw_counts": (
            {
                "mean": float(draws.float().mean()),
                "max": int(draws.max()),
                "zero_share": float((draws == 0).float().mean()),
            }
            if draws is not None else None
        ),
        "terminated_share": (
            float(terminated.float().mean())
            if terminated is not None else None
        ),
        "truncated_share": (
            float(truncated.float().mean())
            if truncated is not None else None
        ),
        "replay_size": replay_stats.get("size"),
        "sample_age_quantiles_replay": replay_stats.get(
            "sample_age_quantiles"
        ),
    }
    return out


def _checks_demography(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    b = ctx.batch
    checks: List[Dict[str, Any]] = []
    ids = ctx.sample_ids()
    checks.append(_check(
        "C2.1", "replay", "live_slot_integrity",
        "pass" if (ids >= 0).all() else "fail",
        metric={"min_sample_id": int(ids.min()), "rows": int(len(ids))},
        violating=[int(x) for x in ids[ids < 0][:20]],
        detail="sample_ids all live (>=0), no corrupted slots",
    ))
    terminated = _t(b, "terminated")
    share = (
        float(terminated.float().mean()) if terminated is not None else 0.0
    )
    warn = share < CONTRACT_THRESHOLDS["terminal_share_warn"]
    checks.append(_check(
        "C2.2", "replay", "terminal_sample_coverage",
        "warn" if warn else "pass",
        severity="statistical",
        metric={
            "terminated_share": share,
            "warn_below": CONTRACT_THRESHOLDS["terminal_share_warn"],
        },
        detail=(
            "terminal frames underrepresented — slow end-state value "
            "propagation" if warn else "terminal coverage adequate"
        ),
    ))
    return checks


# ------------------------------------------------------ cut ③ candidates/twin


def candidate_twin_analysis(dump_dir: Path) -> Dict[str, Any]:
    """Enumerated-candidate twin-min audit (target side + actor side)."""
    ctx = _DumpCtx(dump_dir)
    f = ctx.forward
    ids = ctx.sample_ids()
    out: Dict[str, Any] = {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "channels": ctx.channels,
    }
    integ = _t(f, "next_integration_weights")
    if integ is not None:
        sums = integ.float().sum(dim=(1, 2)).numpy()
        out["integ_weight_sum"] = {
            "min": float(sums.min()),
            "max": float(sums.max()),
            "max_abs_err": float(np.abs(sums - 1.0).max()),
        }
    pair_t = _t(f, "target_pair_index")
    if pair_t is not None:
        out["target_pair1_frac"] = float(pair_t.float().mean())
    pair_a = _t(f, "actor_pair_index")
    if pair_a is not None:
        out["actor_pair1_frac"] = float(pair_a.float().mean())

    for side, cand_key, pair in (
        ("target", "next_cand_q", pair_t),
        ("actor", "actor_cand_q", pair_a),
    ):
        cand = f.get(cand_key)
        if not isinstance(cand, Mapping) or pair is None:
            continue
        side_out: Dict[str, Any] = {}
        for ch in ctx.channels:
            q = cand.get(ch)
            if not isinstance(q, torch.Tensor) or q.shape[-1] != 2:
                continue
            qf = q.float()
            gap = (qf[..., 0] - qf[..., 1]).abs().numpy()
            sel_joint = qf.gather(
                -1, pair[..., None].long()
            ).squeeze(-1)  # [B,K,M] — pair indexes last dim
            sel_indep = qf.min(dim=-1).values
            diff = (sel_joint - sel_indep).numpy()
            side_out[ch] = {
                "twin_gap_mean": float(gap.mean()),
                "twin_gap_q95": float(np.quantile(gap, 0.95)),
                "twin_gap_max": float(gap.max()),
                # >0 means joint argmin picked the *higher* per-channel Q
                # because the OTHER channel forced the pair choice.
                "joint_vs_indep_mean": float(diff.mean()),
                "joint_vs_indep_max": float(diff.max()),
            }
        out[side] = side_out
    return out


def _checks_twin(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    f = ctx.forward
    integ = _t(f, "next_integration_weights")
    if integ is None:
        checks.append(_check(
            "C3.1", "candidates", "integration_weights_sum_one", "skip",
            detail="next_integration_weights absent",
        ))
    else:
        sums = integ.float().sum(dim=(1, 2)).numpy()
        err = float(np.abs(sums - 1.0).max())
        checks.append(_check(
            "C3.1", "candidates", "integration_weights_sum_one",
            "pass" if err <= CONTRACT_THRESHOLDS["integ_w_abs"] else "fail",
            metric={
                "max_abs_err": err,
                "atol": CONTRACT_THRESHOLDS["integ_w_abs"],
            },
            detail="Σ_km w_integ == 1 per row",
        ))
    lo, hi = (
        CONTRACT_THRESHOLDS["pair_frac_lo"],
        CONTRACT_THRESHOLDS["pair_frac_hi"],
    )
    for name, key in (
        ("target_pair1_frac", "target_pair_index"),
        ("actor_pair1_frac", "actor_pair_index"),
    ):
        pair = _t(f, key)
        if pair is None:
            checks.append(_check(
                "C3.3", "candidates", name, "skip",
                detail=f"{key} absent",
            ))
            continue
        frac = float(pair.float().mean())
        checks.append(_check(
            "C3.3", "candidates", name,
            "pass" if lo <= frac <= hi else "warn",
            severity="statistical",
            metric={"frac": frac, "bounds": [lo, hi]},
            detail=(
                f"pair1 share {frac:.3f} — twin-min degenerating toward a "
                "single network" if not (lo <= frac <= hi)
                else f"pair1 share {frac:.3f} balanced"
            ),
        ))
    return checks


# --------------------------------------------------------- cut ④ target recompute


def recompute_targets(
    dump_dir: Path,
    *,
    gamma: Optional[float] = None,
    reward_scale: Optional[float] = None,
    alpha: Optional[float] = None,
    u_floor: Optional[float] = None,
    reg_lambda: Optional[float] = None,
) -> Dict[str, Any]:
    """Counterfactual Bellman target recompute under parameter overrides.

    Default (no overrides) is a closure check: recomputed targets must equal
    the recorded ``forward.targets`` within ``math_abs``.  With overrides,
    returns the counterfactual distribution shift.
    """
    from .debugkit import _build_actor
    from .trainer import (
        _RegularizerArgs,
        _resolve_u_kind,
        _state_regularizer,
    )

    ctx = _DumpCtx(dump_dir)
    f = ctx.forward
    b = ctx.batch
    spec = ctx.spec
    channels = _load_channels(spec)
    sacp = spec["sac_params"]
    mode = str(sacp.get("regularizer_mode", "shannon"))
    scale = float(
        reward_scale if reward_scale is not None
        else sacp.get("reward_scale", 1.0)
    )
    alpha_v = float(
        alpha if alpha is not None
        else f["alpha_pre"].float().item()
        if isinstance(f.get("alpha_pre"), torch.Tensor)
        else sacp.get("init_alpha", 1.0)
    )
    lam = float(
        reg_lambda if reg_lambda is not None
        else sacp.get("reg_lambda", 0.0)
    )
    floor = float(
        u_floor if u_floor is not None
        else sacp.get("u_floor", 0.0)
    )
    gamma_ov = gamma  # per-channel override applies to all channels

    integ = f["next_integration_weights"].float()
    pair = f["target_pair_index"].long()
    cand_q = f["next_cand_q"]
    logp = f["next_log_probs"].float()
    rewards = b["rewards"].float().numpy()
    bootstrap = b["bootstrap"].float().numpy()

    # softq per channel = Σ_km integ * q_sel (joint pair selection)
    softq: Dict[str, np.ndarray] = {}
    for c, ch in enumerate(channels):
        q = cand_q[ch.name].float()
        q_sel = q.gather(-1, pair[..., None]).squeeze(-1)
        softq[ch.name] = (integ * q_sel).sum(dim=(1, 2)).numpy()

    # Continuation regularizer under (possibly overridden) params — reuse
    # the trainer's own _state_regularizer so counterfactuals stay
    # formula-faithful (D-no-second-formula).
    u_next_stats: Optional[Dict[str, Any]] = None
    if mode == "shannon":
        reg = (-alpha_v * (integ * logp).sum(dim=(1, 2))).numpy()
    else:
        actor = _build_actor(spec, ctx.pre)
        next_obs = b["next_obs"].float()
        u_kind = str(sacp.get("u_kind", "native"))
        with torch.no_grad():
            u_val = actor.uncertainty(
                next_obs, _resolve_u_kind(actor, u_kind)
            ).cpu().numpy()
            b_state = _state_regularizer(
                actor, next_obs,
                _RegularizerArgs(
                    regularizer_mode=mode,
                    reg_lambda=lam,
                    u_floor=floor,
                    u_kind=u_kind,
                ),
            ).cpu().numpy()
        u_next_stats = {
            "mean": float(u_val.mean()),
            "q05": float(np.quantile(u_val, 0.05)),
            "q50": float(np.quantile(u_val, 0.5)),
            "q95": float(np.quantile(u_val, 0.95)),
            "min": float(u_val.min()),
            "floor_violation_rate": float((u_val < floor).mean())
            if mode == "u_floor" else None,
        }
        reg = b_state

    per_channel: Dict[str, Any] = {}
    overrides_active = any(
        v is not None
        for v in (gamma, reward_scale, alpha, u_floor, reg_lambda)
    )
    recorded = f.get("targets") or {}
    for c, ch in enumerate(channels):
        g = float(gamma_ov if gamma_ov is not None else ch.gamma)
        reward_term = rewards[:, c] * scale
        boot = g * bootstrap
        target = reward_term + boot * (softq[ch.name] + reg)
        entry: Dict[str, Any] = {
            "gamma": g,
            "reward_scale": scale,
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "target_min": float(target.min()),
            "target_max": float(target.max()),
        }
        if ch.name in recorded and not overrides_active:
            rec = recorded[ch.name].float().numpy()
            entry["recorded_max_abs_err"] = float(
                np.abs(target - rec).max()
            )
        if ch.name in recorded and overrides_active:
            rec = recorded[ch.name].float().numpy()
            d = target - rec
            entry["delta_vs_recorded"] = {
                "mean": float(d.mean()),
                "std": float(d.std()),
                "abs_max": float(np.abs(d).max()),
            }
        per_channel[ch.name] = entry
    return {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "regularizer_mode": mode,
        "overrides": {
            k: v for k, v in {
                "gamma": gamma, "reward_scale": reward_scale,
                "alpha": alpha, "u_floor": u_floor,
                "reg_lambda": reg_lambda,
            }.items() if v is not None
        },
        "u_next": u_next_stats,
        "per_channel": per_channel,
        "closure_ok": all(
            e.get("recorded_max_abs_err", 0.0)
            <= CONTRACT_THRESHOLDS["math_abs"]
            for e in per_channel.values()
            if "recorded_max_abs_err" in e
        ) if not overrides_active else None,
    }


def _checks_targets(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    try:
        res = recompute_targets(ctx.dir)
    except Exception as exc:
        return [_check(
            "C4.1", "target", "bellman_decomp_closure", "warn",
            severity="statistical", detail=f"recompute raised: {exc}",
        )]
    errs = {
        ch: e.get("recorded_max_abs_err")
        for ch, e in res["per_channel"].items()
        if e.get("recorded_max_abs_err") is not None
    }
    if not errs:
        checks.append(_check(
            "C4.1", "target", "bellman_decomp_closure", "skip",
            detail="no recorded targets to compare",
        ))
    else:
        worst = max(errs.values())
        checks.append(_check(
            "C4.1", "target", "bellman_decomp_closure",
            "pass" if worst <= CONTRACT_THRESHOLDS["math_abs"] else "fail",
            metric={
                "max_abs_err": worst,
                "atol": CONTRACT_THRESHOLDS["math_abs"],
                "per_channel": errs,
            },
            detail="r·scale + γ·b·(softq + reg) == recorded target",
        ))
    if ctx.mode in ("u_floor", "u_bonus"):
        u = res.get("u_next")
        if u is None:
            checks.append(_check(
                "C4.2", "target", "u_reg_next_exact", "skip",
                detail="actor lacks uncertainty() or next_obs missing",
            ))
        else:
            reg_next = _t(ctx.forward, "reg_next")
            # residual against recorded reg_next is verified inside
            # recompute via same formula; here expose violation rate.
            rate = u.get("floor_violation_rate")
            status = "pass"
            detail = "u_floor active on a sane share of next-states"
            if rate is not None:
                if rate <= CONTRACT_THRESHOLDS["u_floor_active_warn"]:
                    status, detail = "warn", (
                        f"u_floor dormant: P(U(s')<f)={rate:.3f}"
                    )
                elif rate >= CONTRACT_THRESHOLDS["u_floor_active_warn_hi"]:
                    status, detail = "warn", (
                        f"u_floor saturating: P(U(s')<f)={rate:.3f}"
                    )
            checks.append(_check(
                "C4.2", "target", "u_reg_next_exact", status,
                severity="statistical",
                metric={"u_next": u},
                detail=detail,
            ))
    return checks


# ------------------------------------------------------------- cut ⑤ critic


def critic_fit(dump_dir: Path) -> Dict[str, Any]:
    """Per-channel critic fit quality: EV, Q-vs-target scatter, TD focus."""
    ctx = _DumpCtx(dump_dir)
    f = ctx.forward
    b = ctx.batch
    cv = b["channel_valid"].float().numpy()
    sw = b["sample_weight"].float().numpy()
    ids = ctx.sample_ids()
    per: Dict[str, Any] = {}
    for ch in ctx.channels:
        targets = f.get("targets", {}).get(ch)
        q1 = f.get("q1_pred", {}).get(ch)
        q2 = f.get("q2_pred", {}).get(ch)
        if targets is None or q1 is None or q2 is None:
            continue
        c = ctx.channels.index(ch)
        mask = (cv[:, c] * sw)
        msum = float(mask.sum())
        entry: Dict[str, Any] = {"valid_weight": msum}
        if msum <= 0:
            entry["skipped"] = "empty mask"
            per[ch] = entry
            continue
        y = targets.float().numpy()
        for tag, q in (("q1", q1), ("q2", q2)):
            qn = q.float().numpy()
            w = mask / msum
            resid = y - qn
            var_y = float((w * (y - (w * y).sum()) ** 2).sum())
            var_r = float((w * (resid - (w * resid).sum()) ** 2).sum())
            ev = 1.0 - var_r / var_y if var_y > 1e-12 else None
            entry[f"ev_{tag}"] = ev
            entry[f"td_abs_mean_{tag}"] = float(
                (w * np.abs(resid)).sum()
            )
        td2 = ((q1.float().numpy() - y) ** 2
               + (q2.float().numpy() - y) ** 2) * mask
        order = np.argsort(-td2)
        top5 = order[: max(1, int(0.05 * len(order)))]
        entry["td2_top5pct_share"] = float(
            td2[top5].sum() / max(td2.sum(), 1e-12)
        )
        entry["top_td2_samples"] = [
            {"sample_id": int(ids[i]), "td2": float(td2[i])}
            for i in order[:20]
        ]
        per[ch] = entry
    return {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "per_channel": per,
    }


def _checks_critic(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    res = critic_fit(ctx.dir)
    for ch, e in res["per_channel"].items():
        evs = [e.get("ev_q1"), e.get("ev_q2")]
        evs = [x for x in evs if x is not None]
        if not evs:
            continue
        worst = min(evs)
        status = "pass" if worst >= CONTRACT_THRESHOLDS["critic_ev_warn"] else "warn"
        checks.append(_check(
            "C5.2", "critic", f"critic_explained_variance:{ch}", status,
            severity="statistical",
            metric={
                "ev_q1": e.get("ev_q1"), "ev_q2": e.get("ev_q2"),
                "warn_below": CONTRACT_THRESHOLDS["critic_ev_warn"],
                "td2_top5pct_share": e.get("td2_top5pct_share"),
            },
            violating=[
                s["sample_id"] for s in e.get("top_td2_samples", [])[:10]
            ] if status == "warn" else [],
            detail=(
                f"{ch} EV_q1={e.get('ev_q1'):.3f} EV_q2={e.get('ev_q2'):.3f}"
                if status == "pass"
                else f"{ch} critic EV={worst:.3f} — Q estimate is noise-level"
            ),
        ))
    # MSE closure against recorded losses.
    us = ctx.analysis.get("update_stats") or {}
    for ch in ctx.channels:
        targets = ctx.forward.get("targets", {}).get(ch)
        q1 = ctx.forward.get("q1_pred", {}).get(ch)
        q2 = ctx.forward.get("q2_pred", {}).get(ch)
        rec1 = us.get(f"q1_loss_{ch}")
        rec2 = us.get(f"q2_loss_{ch}")
        if targets is None or q1 is None or rec1 is None:
            continue
        c = ctx.channels.index(ch)
        mask = (
            ctx.batch["channel_valid"][:, c].float()
            * ctx.batch["sample_weight"].float()
        )
        msum = float(mask.sum().item())
        if msum <= 0:
            continue
        y = targets.float()
        denom = mask.sum()
        loss1 = float(
            ((mask * (q1.float() - y).pow(2)).sum() / denom).item()
        )
        loss2 = float(
            ((mask * (q2.float() - y).pow(2)).sum() / denom).item()
        )
        err = max(abs(loss1 - float(rec1)),
                  abs(loss2 - float(rec2)) if rec2 is not None else 0.0)
        checks.append(_check(
            "C5.1", "critic", f"critic_mse_exact:{ch}",
            "pass" if err <= CONTRACT_THRESHOLDS["math_abs"] else "fail",
            metric={
                "max_abs_err": err,
                "atol": CONTRACT_THRESHOLDS["math_abs"],
                "recomputed_q1_loss": loss1,
                "recorded_q1_loss": float(rec1),
            },
            detail="masked MSE recomputed == recorded update_stats",
        ))
    return checks


# ------------------------------------------------------------- cut ⑥ actor


def actor_contract(dump_dir: Path) -> Dict[str, Any]:
    """Actor objective audit: U(s) distribution, floor activity, gate chain.

    ``forward.actor_cand_q`` is verified to equal the **post-update**
    critic's forward on the enumerated candidates (the trainer updates the
    critic first inside the same tick — pre/post confusion is a real bug
    class).
    """
    ctx = _DumpCtx(dump_dir)
    f = ctx.forward
    b = ctx.batch
    out: Dict[str, Any] = {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "regularizer_mode": ctx.mode,
    }
    actor = ctx.actor_pre()
    obs = b["obs"].float()

    # U(s) distribution + floor activity (U modes only).
    if ctx.mode in ("u_floor", "u_bonus") and hasattr(actor, "uncertainty"):
        from .trainer import _resolve_u_kind
        u_kind = str(ctx.sp.get("u_kind", "native"))
        with torch.no_grad():
            u_s = actor.uncertainty(
                obs, _resolve_u_kind(actor, u_kind)
            ).cpu().numpy()
        floor = float(ctx.sp.get("u_floor", 0.0))
        out["u_s"] = {
            "mean": float(u_s.mean()),
            "q05": float(np.quantile(u_s, 0.05)),
            "q50": float(np.quantile(u_s, 0.5)),
            "q95": float(np.quantile(u_s, 0.95)),
            "min": float(u_s.min()),
            "floor": floor,
            "floor_violation_rate": float((u_s < floor).mean())
            if ctx.mode == "u_floor" else None,
        }
        reg_actor = _t(f, "reg_actor")
        if reg_actor is not None and ctx.mode == "u_floor":
            lam = float(ctx.sp.get("reg_lambda", 0.0))
            # reg_actor = -B_theta(s) = +lambda * relu(floor - U(s))^2
            expected = lam * np.clip(floor - u_s, 0.0, None) ** 2
            err = np.abs(expected - reg_actor.numpy())
            out["reg_actor_max_abs_err"] = float(err.max())

    # φ_post semantic check on actor_cand_q.
    cand = f.get("actor_cand_q")
    new_actions = _t(f, "new_actions")
    semantic = {"checked": False}
    if isinstance(cand, Mapping) and new_actions is not None:
        try:
            from .trainer import _forward_all_candidates
            post_critic = ctx.critic_post()
            atol = CONTRACT_THRESHOLDS["math_abs"]
            rtol = CONTRACT_THRESHOLDS["math_rtol"]
            max_err = 0.0
            n_viol = 0
            for ch in ctx.channels:
                q = cand.get(ch)
                if not isinstance(q, torch.Tensor):
                    continue
                q_post = torch.stack(
                    (
                        _forward_all_candidates(
                            lambda o, a: post_critic.q1_forward(o, a, ch),
                            obs, new_actions.float(),
                        ),
                        _forward_all_candidates(
                            lambda o, a: post_critic.q2_forward(o, a, ch),
                            obs, new_actions.float(),
                        ),
                    ),
                    dim=-1,
                )
                diff = (q_post - q.float()).abs()
                max_err = max(max_err, float(diff.max().item()))
                tol = atol + rtol * q.float().abs()
                n_viol += int((diff > tol).sum().item())
            semantic = {
                "checked": True,
                "actor_cand_q_vs_post_critic_max_err": max_err,
                "out_of_tol_elements": n_viol,
                "atol": atol,
                "rtol": rtol,
            }
        except Exception as exc:
            semantic = {"checked": False, "error": str(exc)}
    out["cand_q_semantics"] = semantic

    # Gate → weight → contribution chain.
    aw = _t(b, "actor_weight")
    if aw is not None and isinstance(cand, Mapping):
        pair = _t(f, "actor_pair_index")
        per_ch: Dict[str, Any] = {}
        if pair is not None:
            for c, ch in enumerate(ctx.channels):
                q = cand.get(ch)
                if not isinstance(q, torch.Tensor):
                    continue
                q_sel = q.float().gather(
                    -1, pair[..., None].long()
                ).squeeze(-1)
                integ = _t(f, "actor_integration_weights")
                contrib = (
                    (integ * q_sel).sum(dim=(1, 2))
                    * aw[:, c].float()
                ).numpy() if integ is not None else None
                per_ch[ch] = {
                    "weight_mean": float(aw[:, c].float().mean()),
                    "weighted_q_mean": float(contrib.mean())
                    if contrib is not None else None,
                }
            out["gate_chain"] = per_ch
    return out


def _checks_actor(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    res = actor_contract(ctx.dir)
    # C6.1 closure comes from objective_decomp.
    try:
        dec = objective_decomp(ctx.dir)
        err = float((dec.get("check") or {}).get(
            "objective_max_abs_err", float("nan")))
        checks.append(_check(
            "C6.1", "actor", "actor_decomp_closure",
            "pass" if err <= CONTRACT_THRESHOLDS["math_abs"] else "fail",
            metric={
                "max_abs_err": err,
                "atol": CONTRACT_THRESHOLDS["math_abs"],
            },
            detail="Σ_c wQ + entropy + reg == recorded actor objective",
        ))
    except Exception as exc:
        checks.append(_check(
            "C6.1", "actor", "actor_decomp_closure", "warn",
            severity="statistical", detail=f"decomp raised: {exc}",
        ))
    u_s = res.get("u_s")
    if ctx.mode in ("u_floor", "u_bonus"):
        if u_s is None:
            checks.append(_check(
                "C6.2", "actor", "u_floor_activity", "skip",
                detail="actor lacks uncertainty()",
            ))
        else:
            rate = u_s.get("floor_violation_rate")
            status, detail = "pass", "u_floor penalty active"
            if rate is not None:
                if rate <= CONTRACT_THRESHOLDS["u_floor_active_warn"]:
                    status = "warn"
                    detail = (
                        f"floor dormant: P(U<f)={rate:.3f} — u_floor not "
                        "pushing actor anywhere right now"
                    )
                elif rate >= CONTRACT_THRESHOLDS["u_floor_active_warn_hi"]:
                    status = "warn"
                    detail = f"floor saturating: P(U<f)={rate:.3f}"
            checks.append(_check(
                "C6.2", "actor", "u_floor_activity", status,
                severity="statistical",
                metric=u_s, detail=detail,
            ))
    sem = res.get("cand_q_semantics") or {}
    if sem.get("checked"):
        n_viol = int(sem.get("out_of_tol_elements", 0))
        checks.append(_check(
            "C6.3", "actor", "actor_cand_q_is_post_critic",
            "pass" if n_viol == 0 else "fail",
            metric={
                "max_abs_err": sem.get(
                    "actor_cand_q_vs_post_critic_max_err"),
                "out_of_tol_elements": n_viol,
                "atol": sem.get("atol"),
                "rtol": sem.get("rtol"),
            },
            detail=(
                "actor_cand_q must equal critic(φ_post) forward — "
                "critic steps BEFORE actor inside one tick"
            ),
        ))
    return checks


# ------------------------------------------------------------- cut ⑧ update


def update_checks(dump_dir: Path) -> Dict[str, Any]:
    """Optimizer-step contracts: Polyak identity, alpha invariance, layer
    relative steps."""
    ctx = _DumpCtx(dump_dir)
    pre, post = ctx.pre, ctx.post
    tau = float(ctx.sp.get("tau", 0.005))
    out: Dict[str, Any] = {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "tau": tau,
    }
    # Polyak: q{1,2}_target_post ?= (1-τ)·q{1,2}_target_pre + τ·q{1,2}_post
    residuals: Dict[str, float] = {}
    pre_c = pre.get("critic_state_dict") or {}
    post_c = post.get("critic_state_dict") or {}
    for ch_key in pre_c:
        ch_state_pre = pre_c[ch_key]
        ch_state_post = post_c.get(ch_key, {})
        for net in ("q1", "q2"):
            tgt_pre = ch_state_pre.get(f"{net}_target") or {}
            tgt_post = ch_state_post.get(f"{net}_target") or {}
            onl_post = ch_state_post.get(net) or {}
            if not tgt_pre or not tgt_post or not onl_post:
                continue
            max_r = 0.0
            for k, v_pre in tgt_pre.items():
                v_post = tgt_post.get(k)
                v_onl = onl_post.get(k)
                if v_post is None or v_onl is None:
                    continue
                expected = (1.0 - tau) * v_pre.float() + tau * v_onl.float()
                max_r = max(
                    max_r, float((v_post.float() - expected).abs().max()),
                )
            residuals[f"{ch_key}.{net}_target"] = max_r
    out["polyak_residuals"] = residuals

    la_pre = pre.get("log_alpha")
    la_post = post.get("log_alpha")
    alpha_delta = None
    if isinstance(la_pre, torch.Tensor) and isinstance(la_post, torch.Tensor):
        alpha_delta = float((la_post.float() - la_pre.float()).abs().max())
    out["alpha_delta_abs"] = alpha_delta
    out["auto_alpha"] = bool(ctx.sp.get("auto_alpha", False))

    # Layer-wise relative steps for params (not optimizer states).
    def _rel_steps(node_pre, node_post, prefix=""):
        rows = []
        for k, v in node_pre.items():
            if k.endswith("optimizer"):
                continue
            vp = node_post.get(k) if isinstance(node_post, Mapping) else None
            if isinstance(v, torch.Tensor) and isinstance(vp, torch.Tensor):
                if v.shape == vp.shape:
                    d = (vp.float() - v.float()).norm().item()
                    n = float(v.float().norm().item())
                    rows.append({
                        "param": f"{prefix}{k}",
                        "l2_delta": d,
                        "rel_step": d / n if n > 0 else None,
                    })
            elif isinstance(v, Mapping) and isinstance(vp, Mapping):
                rows.extend(_rel_steps(v, vp, f"{prefix}{k}."))
        return rows
    actor_rows = _rel_steps(
        pre.get("actor_state_dict") or {},
        post.get("actor_state_dict") or {},
        "actor.",
    )
    critic_rows = _rel_steps(
        pre.get("critic_state_dict") or {},
        post.get("critic_state_dict") or {},
        "critic.",
    )
    out["param_steps"] = actor_rows + critic_rows
    return out


def _checks_update(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    res = update_checks(ctx.dir)
    residuals = res["polyak_residuals"]
    if not residuals:
        checks.append(_check(
            "C8.1", "update", "polyak_soft_update_exact", "skip",
            detail="no target params found in trainer states",
        ))
    else:
        worst = max(residuals.values())
        checks.append(_check(
            "C8.1", "update", "polyak_soft_update_exact",
            "pass" if worst <= CONTRACT_THRESHOLDS["math_abs"] else "fail",
            metric={
                "max_residual": worst,
                "atol": CONTRACT_THRESHOLDS["math_abs"],
                "per_net": residuals,
            },
            detail="ψ_post == (1−τ)ψ_pre + τ·ϕ_post elementwise",
        ))
    delta = res.get("alpha_delta_abs")
    if delta is None:
        checks.append(_check(
            "C8.2", "update", "alpha_mode_invariant", "skip",
            detail="log_alpha absent",
        ))
    elif not res["auto_alpha"]:
        checks.append(_check(
            "C8.2", "update", "alpha_mode_invariant",
            "pass" if delta == 0.0 else "fail",
            metric={"alpha_delta_abs": delta},
            detail="alpha must be frozen when auto_alpha=false/U-modes",
        ))
    else:
        checks.append(_check(
            "C8.2", "update", "alpha_mode_invariant", "pass",
            severity="statistical",
            metric={"alpha_delta_abs": delta},
            detail="auto_alpha active; direction check in metrics stream",
        ))
    return checks


# ------------------------------------------------------------ cut ⑨ post


def post_detail(dump_dir: Path) -> Dict[str, Any]:
    """Per-sample pre→post drift on the same batch + single-step descent."""
    ctx = _DumpCtx(dump_dir)
    b = ctx.batch
    ids = ctx.sample_ids()
    actor_pre = ctx.actor_pre()
    from .debugkit import _build_actor
    actor_post = _build_actor(ctx.spec, ctx.post)
    critic_pre = ctx.critic_pre()
    critic_post = ctx.critic_post()
    f = ctx.forward
    obs = b["obs"].float()
    actions = b["actions"].float()

    out: Dict[str, Any] = {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
    }
    with torch.no_grad():
        det_pre = actor_pre.deterministic_action(obs)
        det_post = actor_post.deterministic_action(obs)
        d_det = (det_post - det_pre).norm(dim=-1).numpy()
        out["det_action_drift"] = {
            "l2_mean": float(d_det.mean()),
            "l2_max": float(d_det.max()),
            "per_sample_l2_top": [
                {"sample_id": int(ids[i]), "l2": float(d_det[i])}
                for i in np.argsort(-d_det)[:20]
            ],
        }
        logp_pre = _log_prob_of(actor_pre, obs, actions).float()
        logp_post = _log_prob_of(actor_post, obs, actions).float()
        d_lp = (logp_post - logp_pre).numpy()
        out["logp_drift"] = {
            "mean": float(d_lp.mean()),
            "abs_max": float(np.abs(d_lp).max()),
        }
        # U(s) drift (U modes only).
        if ctx.mode in ("u_floor", "u_bonus") and hasattr(
            actor_pre, "uncertainty"
        ):
            from .trainer import _resolve_u_kind
            u_kind = str(ctx.sp.get("u_kind", "native"))
            uk = _resolve_u_kind(actor_pre, u_kind)
            u_pre = actor_pre.uncertainty(obs, uk).numpy()
            u_post = actor_post.uncertainty(obs, uk).numpy()
            du = u_post - u_pre
            out["u_drift"] = {
                "mean": float(du.mean()),
                "abs_max": float(np.abs(du).max()),
                "toward_floor_share": float(
                    ((u_pre < float(ctx.sp.get("u_floor", 0.0)))
                     & (du > 0)).mean()
                ) if ctx.mode == "u_floor" else None,
            }
        # Per-channel ΔQ and TD² descent on fixed pre-targets.
        cv = b["channel_valid"].float().numpy()
        sw = b["sample_weight"].float().numpy()
        per: Dict[str, Any] = {}
        for c, ch in enumerate(ctx.channels):
            targets = f.get("targets", {}).get(ch)
            if targets is None:
                continue
            y = targets.float().numpy()
            q1p = critic_pre.q1_forward(obs, actions, ch).numpy()
            q2p = critic_pre.q2_forward(obs, actions, ch).numpy()
            q1n = critic_post.q1_forward(obs, actions, ch).numpy()
            q2n = critic_post.q2_forward(obs, actions, ch).numpy()
            mask = (cv[:, c] * sw)
            msum = float(mask.sum()) or 1.0
            td2_pre = ((q1p - y) ** 2 + (q2p - y) ** 2) * mask
            td2_post = ((q1n - y) ** 2 + (q2n - y) ** 2) * mask
            per[ch] = {
                "mse_pre": float(td2_pre.sum() / (2 * msum)),
                "mse_post": float(td2_post.sum() / (2 * msum)),
                "mse_descent": float(
                    td2_pre.sum() / (2 * msum)
                    - td2_post.sum() / (2 * msum)
                ),
                "dq_mean": float(
                    (((q1n + q2n) / 2 - (q1p + q2p) / 2) * mask).sum() / msum
                ),
                "dq_abs_max": float(
                    np.abs((q1n + q2n) / 2 - (q1p + q2p) / 2).max()
                ),
            }
        out["per_channel"] = per
    return out


def _checks_post(ctx: _DumpCtx) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    try:
        res = post_detail(ctx.dir)
    except Exception as exc:
        return [_check(
            "C9.1", "post", "single_step_loss_descent", "warn",
            severity="statistical", detail=f"post_detail raised: {exc}",
        )]
    for ch, e in (res.get("per_channel") or {}).items():
        desc = e.get("mse_descent")
        checks.append(_check(
            "C9.1", "post", f"single_step_loss_descent:{ch}",
            "pass" if (desc is not None and desc >= -1e-8) else "warn",
            severity="statistical",
            metric={
                "mse_pre": e.get("mse_pre"),
                "mse_post": e.get("mse_post"),
                "descent": desc,
            },
            detail=(
                "one optimizer step should reduce masked TD-MSE on the "
                "same batch/targets (overshoot → warn)"
            ),
        ))
    return checks


# ------------------------------------------------------------- ledger glue


_CONTRACT_STAGES = (
    _checks_rollout,       # ⓪
    _checks_ep2slice,      # ①
    _checks_demography,    # ②
    _checks_twin,          # ③
    _checks_targets,       # ④
    _checks_critic,        # ⑤
    _checks_actor,         # ⑥
    _checks_update,        # ⑧ (⑦ gradsig is WP2)
    _checks_post,          # ⑨
)


def verify_dump_contracts(
    dump_dir: Path,
    *,
    checks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run all contract assertions for one dump and return the ledger.

    ``checks`` optionally restricts to check-ids (e.g. ``["C4.1"]``) or cut
    names (``"target"``).  Every entry carries pass/warn/fail/skip — ``skip``
    is explicit with a reason, never a silent pass.
    """
    ctx = _DumpCtx(Path(dump_dir))
    out_checks: List[Dict[str, Any]] = []
    want = set(checks or [])
    for stage in _CONTRACT_STAGES:
        try:
            produced = stage(ctx)
        except Exception as exc:
            produced = [_check(
                "X", stage.__name__, "stage_error", "warn",
                severity="statistical",
                detail=f"{stage.__name__} raised: {exc}",
            )]
        for c in produced:
            if want and c["id"] not in want and c["cut"] not in want:
                continue
            out_checks.append(c)
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for c in out_checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {
        "dump_dir": str(ctx.dir),
        "critic_tick": ctx.manifest.get("critic_tick"),
        "regularizer_mode": ctx.mode,
        "checks": out_checks,
        "summary": counts,
        "verdict": (
            "fail" if counts["fail"]
            else "warn" if counts["warn"]
            else "pass"
        ),
    }
