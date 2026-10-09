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
    return {
        "run_dir": str(run_dir),
        "metric": key,
        "event": event,
        "count": len(points),
        "points": points,
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


def param_delta(dump: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-tensor pre→post update delta summary (actor + critic + alpha)."""
    out: Dict[str, Any] = {}
    for scope in ("actor_state_dict", "critic_state_dict"):
        pre = dump["trainer_pre"].get(scope) or {}
        post = dump["trainer_post"].get(scope) or {}
        total_l2 = 0.0
        max_abs = 0.0
        n = 0
        for key, pre_t in pre.items():
            post_t = post.get(key)
            if not isinstance(pre_t, torch.Tensor) or post_t is None:
                continue
            delta = (post_t.float() - pre_t.float()).abs()
            total_l2 += float((delta ** 2).sum().item())
            max_abs = max(max_abs, float(delta.max().item()))
            n += int(delta.numel())
        out[scope] = {
            "l2": float(total_l2 ** 0.5),
            "max_abs_delta": max_abs,
            "elements": n,
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
        },
        "task_facts": _jsonable(
            {k: v[i] for k, v in (batch.get("task_facts") or {}).items()}
        ),
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
