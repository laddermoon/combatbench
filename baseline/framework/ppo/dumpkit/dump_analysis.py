"""Shared dump-analysis layer — the single computation source behind the
viewer API, the frontend, and ``debug.py`` CLI commands.

All functions are pure and read-only: they take a ``DumpDataset``
(``dumpkit.frame_access``) and return JSON-safe dicts (no NaN/Inf
literals, no numpy types).  All dump file access — member naming,
dict-of-array unwrapping, frame↔trajectory↔episode index math — lives
in the DumpDataset layer; this module only computes.

Semantic contract every caller must preserve:

- ``proj``/``cos``/``grad_norm`` come from a *sampled* per-frame
  diagnostic at θ_old — at most ``DUMP_GRADSIG_SAMPLE_SIZE`` frames —
  against the full-buffer aggregate gradient G.  Sampled statistics are
  never full-buffer statistics.
- ``proj < 0`` means the frame's gradient points against G; it does NOT
  mean the frame is "bad" — opposition can be exactly what keeps an
  update sane.
- ``dtheta_norm``/``dtheta_cos_descent`` describe the *realized* Adam
  step; ``proj``/``cos`` describe the *pre-update* gradient intent.
  They are different points in time.
- Missing fields surface as absent keys or ``None`` — never fabricated
  zeros.  Unmappable frames surface as ``mapped: false``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.dumpkit.frame_access import (
    DumpDataset,
    parse_frame_id,  # re-exported: older callers import it from here
)


# ---------------------------------------------------------------------------
# Small local helpers (kept dependency-free so this module never imports the
# viewer — the viewer imports this, not the other way round).
# ---------------------------------------------------------------------------

def _f(v: Any) -> Optional[float]:
    """float or None for non-finite/missing."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _i(v: Any) -> Optional[int]:
    x = _f(v)
    return int(x) if x is not None else None


def _nan_to_null(arr: np.ndarray) -> List[Any]:
    out: List[Any] = []
    for v in np.asarray(arr, dtype=np.float64).tolist():
        out.append(v if math.isfinite(v) else None)
    return out


def _quantiles(arr: np.ndarray, levels=(0.01, 0.05, 0.5, 0.95, 0.99)) -> Dict[str, Optional[float]]:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{int(l*100)}": None for l in levels}
    qs = np.quantile(a, levels)
    return {f"p{int(l*100)}": float(q) for l, q in zip(levels, qs)}


def _summ(arr: np.ndarray) -> Dict[str, Any]:
    """Compact distribution summary for a float array."""
    a = np.asarray(arr, dtype=np.float64)
    n_total = int(a.size)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n_total": n_total, "n_valid": 0}
    out: Dict[str, Any] = {
        "n_total": n_total,
        "n_valid": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }
    out.update(_quantiles(a))
    return out


# ---------------------------------------------------------------------------
# Timeline — overview with every captured field + derived key steps
# ---------------------------------------------------------------------------

_TIMELINE_ARRAY_KEYS = (
    "epoch_idx", "mb_idx", "actor_active", "kl", "clip_frac",
    "clip_frac_hi", "clip_frac_lo",
    "ratio_mean", "ratio_max", "ratio_min", "policy_loss",
    "actor_grad", "running_mean_kl", "window_mean_kl",
    # Extended causal-chain fields (NaN on actor-stopped steps).
    "adv_mean", "adv_std", "adv_min", "adv_max",
    "argmax_ratio_bufidx", "argmax_ratio_adv", "argmax_ratio_logr",
    "argmin_ratio_bufidx", "argmin_ratio_adv", "argmin_ratio_logr",
    "n_ratio_gt2", "n_ratio_lt05",
    "dtheta_norm", "dtheta_cos_descent",
    "floor_loss", "mb_size", "dual_clip_frac",
)


def timeline_overview(ds: DumpDataset) -> Dict[str, Any]:
    """Full per-minibatch timeline incl. fields the trainer captures but
    the old API dropped, plus a ``key_steps`` index for fast navigation."""
    t = ds.timeline
    if not ds.npz("timeline").available:
        return {"available": False, "reason": "timeline.npz not found"}

    result: Dict[str, Any] = {
        "available": True,
        "n_epochs": _i(t.scalar("n_epochs")) or 0,
        "n_batches": _i(t.scalar("n_batches")) or 0,
        "n_steps": _i(t.scalar("n_steps")) or 0,
        "early_stop_step": _i(t.scalar("early_stop_step"))
        if t.col("early_stop_step") is not None else -1,
        "target_kl": _f(t.scalar("target_kl")),
        "clip_eps": _f(t.scalar("clip_eps")) or 0.2,
    }

    for key in _TIMELINE_ARRAY_KEYS:
        arr = t.col(key)
        if arr is None:
            continue
        if arr.dtype == bool:
            result[key] = [bool(x) for x in arr]
        elif arr.dtype == np.int64 or arr.dtype == np.int32:
            result[key] = [int(x) for x in arr]
        else:
            result[key] = _nan_to_null(arr)

    # Per-channel critic_loss / critic_grad dicts — channel names come
    # from the dict keys themselves (trajectories.npz may be absent).
    for src, dst in (("critic_loss", "critic_loss_"),
                     ("critic_grad", "critic_grad_")):
        for col in t.columns:
            if col.startswith(src + "."):
                ch = col[len(src) + 1:]
                a = t.col(col)
                if a is not None:
                    result[dst + str(ch)] = _nan_to_null(np.asarray(a))

    result["key_steps"] = _key_steps(result)
    return result


def _key_steps(ov: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Indices of the steps worth opening first — all derived from
    recorded arrays, no recomputation of the update itself."""
    n = ov.get("n_steps") or 0
    if n == 0:
        return []

    def _arr(key):
        a = ov.get(key)
        if not isinstance(a, list):
            return None
        return np.array(
            [x if x is not None else np.nan for x in a], dtype=np.float64)

    def _entry(kind: str, step: int, value_key: Optional[str] = None):
        e: Dict[str, Any] = {
            "kind": kind,
            "step": int(step),
            "epoch": ov["epoch_idx"][step] if ov.get("epoch_idx") else None,
            "mb_idx": ov["mb_idx"][step] if ov.get("mb_idx") else None,
        }
        if value_key:
            v = ov.get(value_key)
            e["value"] = v[step] if isinstance(v, list) and step < len(v) else None
        return e

    steps: List[Dict[str, Any]] = []

    es = ov.get("early_stop_step")
    if es is not None and es >= 0 and es < n:
        steps.append(_entry("early_stop", es, "window_mean_kl"))

    wmk = _arr("window_mean_kl")
    target = ov.get("target_kl")
    if wmk is not None and target is not None:
        breach = np.where(wmk > target)[0]
        if breach.size and (es is None or int(breach[0]) != es):
            steps.append(_entry("kl_breach", int(breach[0]), "window_mean_kl"))

    for kind, key in (
        ("max_actor_grad", "actor_grad"),
        ("max_dtheta", "dtheta_norm"),
        ("max_kl", "kl"),
        ("max_ratio", "ratio_max"),
        ("min_dtheta_cos", "dtheta_cos_descent"),
    ):
        a = _arr(key)
        if a is None:
            continue
        finite = np.isfinite(a)
        if not finite.any():
            continue
        idx = int(np.nanargmax(np.where(finite, a, -np.inf))) \
            if kind != "min_dtheta_cos" \
            else int(np.nanargmin(np.where(finite, a, np.inf)))
        if not any(s["step"] == idx and s["kind"] == kind for s in steps):
            steps.append(_entry(kind, idx, key))
    return steps


def timeline_step(ds: DumpDataset, step: int) -> Tuple[int, Dict[str, Any]]:
    """One minibatch's full record."""
    ov = timeline_overview(ds)
    if not ov.get("available"):
        return 404, {"error": ov.get("reason", "timeline not available")}
    n = ov["n_steps"]
    if step < 0 or step >= n:
        return 404, {"error": f"step {step} out of range (n_steps={n})"}
    result: Dict[str, Any] = {
        "step": step,
        "early_stop_step": ov.get("early_stop_step"),
        "target_kl": ov.get("target_kl"),
        "clip_eps": ov.get("clip_eps"),
    }
    for key, val in ov.items():
        if isinstance(val, list) and key != "key_steps" and len(val) == n:
            result[key] = val[step]
    return 200, result


# ---------------------------------------------------------------------------
# inspect — the dump-level overview payload
# ---------------------------------------------------------------------------

def inspect_dump(ds: DumpDataset) -> Dict[str, Any]:
    """The dump home payload: provenance, capabilities, per-stage
    summaries, and drill-down links."""
    try:
        m = ds.manifest
    except (FileNotFoundError, TypeError):
        m = {}

    out: Dict[str, Any] = {
        "meta": {
            "update": _i(m.get("update")),
            "experiment": m.get("experiment_name"),
            "dump_source": m.get("dump_source"),
            "hypothesis": m.get("hypothesis") or "",
            "timestamp_utc": m.get("timestamp_utc"),
            "git_commit": m.get("git_commit"),
            "n_episodes": _i(m.get("n_episodes")),
            "n_trajectories": _i(m.get("n_trajectories")),
            "total_frames": _i(m.get("total_frames")),
        },
        "capabilities": ds.capabilities(),
        "semantics": {
            "gradsig": "per-frame gradient diagnostics are a <=2000-frame "
                       "sample at theta_old against the full-buffer "
                       "aggregate gradient G; proj<0 = opposes G, not "
                       "'bad sample'; dtheta_* = realized Adam step",
            "timeline": "per-minibatch record; loss/ratio of step i and "
                        "dtheta of step i share the step but bracket the "
                        "same parameter transition",
        },
    }

    # -- sampling summary ---------------------------------------------------
    tm = ds.traj_map or []
    ep_lens = [e.get("num_frames") for e in tm if e.get("num_frames")]
    out["sampling"] = {
        "n_episodes": len(tm),
        "ep_len": _summ(np.asarray(ep_lens)) if ep_lens else None,
    }

    # -- advantage signal ---------------------------------------------------
    f = ds.frames
    adv = f.col("combined_adv")
    if adv is not None:
        adv = np.asarray(adv, dtype=np.float64)
        finite = adv[np.isfinite(adv)]
        adv_blk: Dict[str, Any] = {"combined_adv": _summ(adv)}
        if finite.size:
            adv_blk["frac_pos"] = float((finite > 0).mean())
            adv_blk["frac_neg"] = float((finite < 0).mean())
        cb = ds.npz("combine")
        adv_blk["winsorize_clip_frac"] = _f(cb.scalar("adv_winsorize_clip_frac"))
        raw = f.col("combined_adv_raw")
        adv_blk["winsorized"] = raw is not None
        if raw is not None:
            adv_blk["combined_adv_raw"] = _summ(
                np.asarray(raw, dtype=np.float64))
        out["adv"] = adv_blk

    # -- gradient signal ----------------------------------------------------
    gs = ds.npz("gradsig")
    if gs.available:
        valid = gs.get("valid")
        n_sampled = _i(gs.scalar("n_sampled"))
        if n_sampled is None and valid is not None:
            n_sampled = int(np.asarray(valid).shape[0])
        n_valid = _i(gs.scalar("n_valid"))
        if n_valid is None and valid is not None:
            n_valid = int(np.asarray(valid).sum())
        g_blk: Dict[str, Any] = {
            "partial": "hist" not in gs.keys(),
            "n_sampled": n_sampled,
            "n_valid": n_valid,
        }
        for k in ("gnorm", "coherence", "dir_cos", "proj_mean",
                  "proj_std", "frac_neg", "n_nonfinite", "n_excluded"):
            v = gs.scalar(k)
            if v is not None:
                g_blk[k] = _i(v) if k.startswith("n_") else _f(v)
        out["gradsig"] = g_blk

    # -- update outcome -----------------------------------------------------
    up = ds.update
    if up.available:
        out["update"] = {
            k: _f(up.get(k)) for k in (
                "kl_mean", "kl_max", "early_stop_kl_mean",
                "clip_frac_mean", "ratio_mean", "ratio_max", "ratio_min",
                "policy_loss_mean", "grad_norm_actor_mean",
            ) if up.get(k) is not None
        }
        for k in ("epochs_done", "actor_epochs_done", "n_batches"):
            if up.get(k) is not None:
                out["update"][k] = _i(up.get(k))

    # -- timeline summary ---------------------------------------------------
    tlo = timeline_overview(ds)
    if tlo.get("available"):
        n = tlo["n_steps"]
        active = tlo.get("actor_active") or []
        out["timeline"] = {
            "n_steps": n,
            "n_epochs": tlo.get("n_epochs"),
            "early_stop_step": tlo.get("early_stop_step"),
            "actor_steps_done": sum(1 for x in active if x),
            "target_kl": tlo.get("target_kl"),
            "key_steps": tlo.get("key_steps"),
        }

    out["links"] = {
        "samples": "gradsig/samples?sort=abs_proj&limit=20",
        "timeline": "timeline/overview",
        "trace": "trace/<buffer_idx>",
    }
    return out


# ---------------------------------------------------------------------------
# samples — the sampled-gradient table
# ---------------------------------------------------------------------------

_SAMPLE_SORTS = ("abs_proj", "proj", "neg_proj", "grad_norm", "w_adv")


def gradsig_samples(
    ds: DumpDataset,
    sort: str = "abs_proj",
    sign: str = "all",
    limit: int = 50,
    offset: int = 0,
    group_by: Optional[str] = None,
    include_invalid: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """Ranked sampled-frame table joined with provenance.

    Returns (status, body).  All statistics are *sampled* — the meta
    block says so explicitly.
    """
    g = ds.gradsig
    if not ds.npz("gradsig").available:
        return 404, {"available": False, "reason": "gradsig.npz not found"}
    sel = g.col("sampled_idx")
    if sel is None:
        return 404, {"available": False,
                     "reason": "gradsig.npz has no per-frame arrays"}
    if sort not in _SAMPLE_SORTS:
        return 400, {"error": f"sort must be one of {_SAMPLE_SORTS}"}
    if sign not in ("all", "pos", "neg"):
        return 400, {"error": "sign must be all|pos|neg"}
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    sel = np.asarray(sel, dtype=np.int64)
    valid = np.asarray(g.col("valid"), dtype=bool)

    def _col(name):
        a = g.col(name)
        return np.asarray(a, dtype=np.float64) if a is not None \
            else np.full(len(sel), np.nan)

    proj = _col("proj")
    cos = _col("cos")
    gnorm_a = _col("grad_norm")
    w_adv = _col("w_adv")
    floor_pen = _col("floor_pen")

    n = len(sel)
    keep = np.ones(n, dtype=bool) if include_invalid else valid.copy()
    if sign == "pos":
        keep &= proj > 0
    elif sign == "neg":
        keep &= proj < 0
    keep &= np.isfinite(proj) if sort in ("abs_proj", "proj", "neg_proj") \
        else np.ones(n, dtype=bool)

    key = {
        "abs_proj": -np.abs(proj),
        "proj": -proj,
        "neg_proj": proj,
        "grad_norm": -gnorm_a,
        "w_adv": -w_adv,
    }[sort]
    order = np.argsort(np.where(keep, key, np.inf), kind="stable")
    order = order[keep[order]]

    if group_by == "episode":
        return 200, _samples_by_episode(
            ds, sel, valid, proj, cos, gnorm_a, w_adv, order)

    rows: List[Dict[str, Any]] = []
    for i in order[offset:offset + limit]:
        bi = int(sel[i])
        row: Dict[str, Any] = {
            "buffer_idx": bi,
            "valid": bool(valid[i]),
            "grad_norm": _f(gnorm_a[i]),
            "cos": _f(cos[i]),
            "proj": _f(proj[i]),
            "w_adv": _f(w_adv[i]),
            "floor_pen": _f(floor_pen[i]),
        }
        row.update(ds.frame_provenance(bi))
        rows.append(row)

    n_sampled = ds.npz("gradsig").scalar("n_sampled")
    return 200, {
        "available": True,
        "meta": {
            "scope": "sampled",
            "n_sampled": int(n_sampled) if n_sampled is not None else n,
            "n_valid": _i(ds.npz("gradsig").scalar("n_valid")),
            "n_after_filter": int(order.size),
            "offset": offset,
            "limit": limit,
            "sort": sort,
            "sign": sign,
            "semantics": "rows are the <=2000 sampled frames at "
                         "theta_old; proj<0 opposes the aggregate "
                         "gradient direction (not 'bad data')",
        },
        "rows": rows,
    }


def _samples_by_episode(ds: DumpDataset, sel, valid, proj, cos, gnorm_a,
                        w_adv, order) -> Dict[str, Any]:
    """Aggregate the sampled frames per episode — positive and negative
    projections summed separately (a near-zero net hides opposition)."""
    groups: Dict[int, Dict[str, Any]] = {}
    unmapped = 0
    for i in order:
        bi = int(sel[i])
        loc = ds.frame_provenance(bi)
        if not loc.get("mapped"):
            unmapped += 1
            continue
        ep = loc["episode"]
        g = groups.setdefault(ep, {
            "episode": ep, "n_sampled": 0,
            "sum_pos_proj": 0.0, "sum_neg_proj": 0.0,
            "max_abs_proj": 0.0, "max_abs_proj_bufidx": None,
        })
        g["n_sampled"] += 1
        p = proj[i]
        if math.isfinite(p):
            if p > 0:
                g["sum_pos_proj"] += float(p)
            else:
                g["sum_neg_proj"] += float(p)
            if abs(p) > abs(g["max_abs_proj"]):
                g["max_abs_proj"] = float(p)
                g["max_abs_proj_bufidx"] = bi
    rows = sorted(groups.values(),
                  key=lambda g: -(abs(g["sum_pos_proj"])
                                  + abs(g["sum_neg_proj"])))
    return {
        "available": True,
        "meta": {
            "scope": "sampled",
            "group_by": "episode",
            "n_episodes": len(rows),
            "n_unmapped_frames": unmapped,
            "semantics": "per-episode sums of sampled-frame projections; "
                         "pos/neg kept separate — net alone hides "
                         "cancellation",
        },
        "episodes": rows,
    }


# ---------------------------------------------------------------------------
# trace — one buffer frame across every stage
# ---------------------------------------------------------------------------

def trace_frame(ds: DumpDataset, buffer_idx: int) -> Tuple[int, Dict[str, Any]]:
    """Everything recorded about one flat buffer index, joined across
    buffer → GAE → combine → gradsig → epoch_frames → timeline."""
    f = ds.frames
    n = f.n_rows
    i = int(buffer_idx)
    if i < 0 or i >= n:
        return 404, {"error": f"buffer_idx {i} out of range (n={n})"}

    out: Dict[str, Any] = {"buffer_idx": i}
    out["location"] = ds.frame_provenance(i)

    # -- rollout-time record -------------------------------------------------
    rec: Dict[str, Any] = {}
    for k in ("log_probs", "sample_weights", "explore_factor",
              "floor_weight", "uncertainty"):
        a = f.col(k)
        if a is not None and i < len(a):
            rec[k] = _f(a[i])
    out["buffer"] = rec

    # -- GAE stage (per channel) ---------------------------------------------
    gae = ds.npz("gae")
    if gae.available:
        per_ch: Dict[str, Any] = {}
        for src in ("advs_all", "rets_all", "values_all"):
            d = gae.get_dict(src)
            if not isinstance(d, dict):
                continue
            for ch, a in d.items():
                a = np.asarray(a)
                if i < a.size:
                    per_ch.setdefault(str(ch), {})[
                        {"advs_all": "adv", "rets_all": "ret",
                         "values_all": "value"}[src]] = _f(a[i])
        kfm = gae.get_dict("key_frame_mask")
        if isinstance(kfm, dict):
            for ch, a in kfm.items():
                a = np.asarray(a, dtype=bool)
                if i < a.size:
                    per_ch.setdefault(str(ch), {})["active"] = bool(a[i])
        out["gae"] = per_ch

    # -- combine stage --------------------------------------------------------
    cb = ds.npz("combine")
    if cb.available:
        cblk: Dict[str, Any] = {}
        for k in ("combined_adv", "combined_adv_raw", "aw_l1_sum"):
            a = f.col(k)
            if a is not None and i < len(a):
                cblk[k] = _f(a[i])
        for src, dst in (("normed_advs", "normed_adv"),
                         ("key_actor_weight_frame", "actor_weight")):
            d = cb.get_dict(src)
            if isinstance(d, dict):
                for ch, a in d.items():
                    a = np.asarray(a)
                    if i < a.size:
                        cblk.setdefault("per_channel", {}).setdefault(
                            str(ch), {})[dst] = _f(a[i])
        conf = cb.get_dict("confidences")
        if isinstance(conf, dict):
            cblk["confidences"] = {str(k): _f(v) for k, v in conf.items()}
        out["combine"] = cblk

    # -- gradient diagnostic (sampled only) -----------------------------------
    g = ds.gradsig
    sel = g.col("sampled_idx")
    if sel is not None:
        sel = np.asarray(sel, dtype=np.int64)
        hit = np.where(sel == i)[0]
        if hit.size:
            j = int(hit[0])
            out["gradsig"] = {
                "sampled": True,
                "valid": bool(np.asarray(g.col("valid"))[j])
                if g.col("valid") is not None else None,
                "grad_norm": _f(g.col("grad_norm")[j])
                if g.col("grad_norm") is not None else None,
                "cos": _f(g.col("cos")[j])
                if g.col("cos") is not None else None,
                "proj": _f(g.col("proj")[j])
                if g.col("proj") is not None else None,
                "w_adv": _f(g.col("w_adv")[j])
                if g.col("w_adv") is not None else None,
                "floor_pen": _f(g.col("floor_pen")[j])
                if g.col("floor_pen") is not None else None,
            }
        else:
            out["gradsig"] = {
                "sampled": False,
                "reason": "frame not in the <=2000 sampled subset — "
                          "not a zero gradient",
            }

    # -- epoch snapshots -------------------------------------------------------
    ef = ds.npz("epoch_frames")
    if ef.available:
        epochs: Dict[str, Any] = {}
        for k in ef.keys():
            parts = k.split(".")
            if len(parts) < 2:
                continue
            name, e = parts[0], parts[1]
            if name in ("ratio", "clip_mask", "new_log_prob"):
                a = np.asarray(ef.get(k))
                if i < a.size:
                    v = a[i]
                    epochs.setdefault(e, {})[name] = (
                        bool(v) if name == "clip_mask" else _f(v))
            elif name == "new_value" and len(parts) >= 3:
                a = np.asarray(ef.get(k))
                if i < a.size:
                    epochs.setdefault(e, {}).setdefault(
                        "new_value", {})[parts[2]] = _f(a[i])
        if epochs:
            out["epoch_frames"] = dict(
                sorted(epochs.items(), key=lambda kv: int(kv[0])))
        ase = ef.scalar("actor_stopped_epoch")
        if ase is not None:
            out["actor_stopped_epoch"] = _i(ase)

    # -- timeline reverse refs ---------------------------------------------------
    t = ds.timeline
    if ds.npz("timeline").available:
        refs = []
        for k in ("argmax_ratio_bufidx", "argmin_ratio_bufidx"):
            arr = t.col(k)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            for s in np.where(arr == i)[0]:
                refs.append({
                    "step": int(s),
                    "role": "argmax_ratio" if "max" in k else "argmin_ratio",
                    "epoch": _i(t.col("epoch_idx")[s])
                    if t.col("epoch_idx") is not None else None,
                    "mb_idx": _i(t.col("mb_idx")[s])
                    if t.col("mb_idx") is not None else None,
                })
        if refs:
            out["timeline_refs"] = refs

    # -- deep links ---------------------------------------------------------------
    loc = out["location"]
    links: Dict[str, Any] = {}
    if loc.get("traj_idx") is not None:
        links["trajectory"] = f"trajectory/{loc['traj_idx']}"
        if loc.get("traj_frame") is not None:
            links["trajectory_frame"] = (
                f"trajectory/{loc['traj_idx']}/frame/{loc['traj_frame']}")
    if loc.get("mapped"):
        links["episode_frame"] = (
            f"episode/{loc['episode']}/frame/{loc['env_frame']}")
    out["links"] = links
    return 200, out


# ---------------------------------------------------------------------------
# ADV distribution histograms — the transformation chain as chart data
# ---------------------------------------------------------------------------

def _hist(arr: np.ndarray, bins: int = 64) -> Optional[Dict[str, Any]]:
    """Histogram + compact stats for one float array (own x-range)."""
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    counts, edges = np.histogram(a, bins=bins)
    out = _summ(a)
    out["edges"] = [float(e) for e in edges]
    out["counts"] = [int(c) for c in counts]
    return out


def adv_histograms(ds: DumpDataset, bins: int = 64) -> Dict[str, Any]:
    """Per-stage ADV distributions, ordered along the transformation chain:

        gae.advs_all (raw) → normed_advs[ch] → aw_normed[ch]
        → combined_adv_raw (pre-winsorize) → combined_adv (final)

    Each stage gets its own bin edges — shapes are meant to be compared,
    not x-ranges (normalization deliberately changes scale).
    """
    f = ds.frames
    stages: List[Tuple[str, np.ndarray]] = []
    advs = ds.npz("gae").get_dict("advs_all")
    if isinstance(advs, dict):
        for ch, a in advs.items():
            stages.append((f"raw adv:{ch}", np.asarray(a)))
    elif advs is not None:
        stages.append(("raw adv", np.asarray(advs)))
    for dict_key, label in (("normed_advs", "normed"),
                            ("aw_normed", "actor-weighted")):
        d = ds.npz("combine").get_dict(dict_key)
        if isinstance(d, dict):
            for ch, a in d.items():
                stages.append((f"{label}:{ch}", np.asarray(a)))
    raw = f.col("combined_adv_raw")
    if raw is not None:
        stages.append(("combined (pre-winsorize)", np.asarray(raw)))
    fin = f.col("combined_adv")
    if fin is not None:
        stages.append(("combined (final)", np.asarray(fin)))

    out: Dict[str, Any] = {"available": False, "stages": []}
    for label, arr in stages:
        h = _hist(arr, bins)
        if h is not None:
            out["stages"].append({"label": label, **h})
    out["available"] = bool(out["stages"])
    if not out["available"]:
        out["reason"] = "no ADV arrays in gae.npz/combine.npz"
    return out
