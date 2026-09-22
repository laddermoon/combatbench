"""Shared dump-analysis layer — the single computation source behind the
viewer API, the frontend, and ``debug.py`` CLI commands.

All functions are pure and read-only: they take a ``DumpData`` (or any
object exposing the same lazy accessors — ``manifest``, ``buffer_npz``,
``gae_npz``, ``combine_npz``, ``timeline_npz``, ``epoch_frames_npz``,
``gradsig_npz``, ``channel_names``) and return JSON-safe dicts
(no NaN/Inf literals, no numpy types).

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


def _dict_item(arr: np.ndarray) -> Any:
    """Extract dict from a 0-d object array (npz-serialized dict)."""
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        return arr.item()
    return arr


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
# Frame ↔ trajectory/episode mapping
# ---------------------------------------------------------------------------

def _frame_map(dd) -> Optional[Dict[str, Any]]:
    """Per-buffer-frame provenance lookup.

    Returns dict with:
      frame_id:     object array of 'epNNNN:agent:t' or 'flat:*'
      traj_lengths: int array
      traj_starts:  int array — flat offset where each trajectory begins
    or None when buffer.npz/frame_id is unavailable.
    """
    buf = dd.buffer_npz
    if buf is None or "frame_id" not in buf:
        return None
    fid = buf["frame_id"]
    tl = buf.get("traj_lengths")
    starts = None
    if tl is not None and len(tl) > 0:
        tl = np.asarray(tl, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(tl)[:-1]]).astype(np.int64)
    return {"frame_id": fid, "traj_lengths": tl, "traj_starts": starts}


def parse_frame_id(fid: Any) -> Optional[Dict[str, Any]]:
    """'ep0007:robot_a:123' → {episode, agent, env_frame}; 'flat:*' → None."""
    s = str(fid)
    if s.startswith("flat:"):
        return None
    parts = s.split(":")
    if len(parts) != 3 or not parts[0].startswith("ep"):
        return None
    try:
        return {
            "episode": int(parts[0][2:]),
            "agent": parts[1],
            "env_frame": int(parts[2]),
        }
    except ValueError:
        return None


def _traj_of(fm: Dict[str, Any], buf_idx: int) -> Tuple[Optional[int], Optional[int]]:
    """buffer flat index → (traj_idx, frame offset inside that traj)."""
    starts = fm.get("traj_starts")
    tl = fm.get("traj_lengths")
    if starts is None or tl is None:
        return None, None
    if buf_idx < 0 or buf_idx >= int(tl.sum()):
        return None, None
    t = int(np.searchsorted(starts, buf_idx, side="right") - 1)
    return t, buf_idx - int(starts[t])


def _locate(dd, fm: Optional[Dict[str, Any]], buf_idx: int) -> Dict[str, Any]:
    """Full provenance for one buffer index."""
    loc: Dict[str, Any] = {"buffer_idx": buf_idx, "mapped": False}
    if fm is None or buf_idx < 0 or buf_idx >= len(fm["frame_id"]):
        loc["reason"] = "buffer.npz/frame_id unavailable or index out of range"
        return loc
    parsed = parse_frame_id(fm["frame_id"][buf_idx])
    traj_idx, traj_frame = _traj_of(fm, buf_idx)
    if traj_idx is not None:
        loc["traj_idx"] = traj_idx
        loc["traj_frame"] = traj_frame
    if parsed is None:
        loc["reason"] = "frame_id is flat:* or unparsable — not an env episode frame"
        return loc
    loc.update({
        "mapped": True,
        "episode": parsed["episode"],
        "agent": parsed["agent"],
        "env_frame": parsed["env_frame"],
    })
    return loc


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


def timeline_overview(dd) -> Dict[str, Any]:
    """Full per-minibatch timeline incl. fields the trainer captures but
    the old API dropped, plus a ``key_steps`` index for fast navigation."""
    tl = dd.timeline_npz
    if tl is None:
        return {"available": False, "reason": "timeline.npz not found"}

    result: Dict[str, Any] = {
        "available": True,
        "n_epochs": _i(tl.get("n_epochs")) or 0,
        "n_batches": _i(tl.get("n_batches")) or 0,
        "n_steps": _i(tl.get("n_steps")) or 0,
        "early_stop_step": _i(tl.get("early_stop_step")) if "early_stop_step" in tl else -1,
        "target_kl": _f(tl.get("target_kl")),
        "clip_eps": _f(tl.get("clip_eps")) or 0.2,
    }

    for key in _TIMELINE_ARRAY_KEYS:
        if key not in tl:
            continue
        arr = tl[key]
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
        if src in tl:
            d = _dict_item(tl[src])
            if isinstance(d, dict):
                for ch, a in d.items():
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


def timeline_step(dd, step: int) -> Tuple[int, Dict[str, Any]]:
    """One minibatch's full record."""
    ov = timeline_overview(dd)
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

def inspect_dump(dd) -> Dict[str, Any]:
    """The dump home payload: provenance, capabilities, per-stage
    summaries, factual 'worth inspecting' flags, and drill-down links."""
    try:
        m = dd.manifest
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
        "capabilities": _capabilities(dd),
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
    tm = getattr(dd, "traj_map", []) or []
    ep_lens = [e.get("num_frames") for e in tm if e.get("num_frames")]
    out["sampling"] = {
        "n_episodes": len(tm),
        "ep_len": _summ(np.asarray(ep_lens)) if ep_lens else None,
    }

    # -- advantage signal ---------------------------------------------------
    cb = dd.combine_npz
    if cb is not None and "combined_adv" in cb:
        adv = np.asarray(cb["combined_adv"], dtype=np.float64)
        finite = adv[np.isfinite(adv)]
        adv_blk: Dict[str, Any] = {"combined_adv": _summ(adv)}
        if finite.size:
            adv_blk["frac_pos"] = float((finite > 0).mean())
            adv_blk["frac_neg"] = float((finite < 0).mean())
        adv_blk["winsorize_clip_frac"] = _f(cb.get("adv_winsorize_clip_frac"))
        adv_blk["winsorized"] = "combined_adv_raw" in cb
        if "combined_adv_raw" in cb:
            adv_blk["combined_adv_raw"] = _summ(
                np.asarray(cb["combined_adv_raw"], dtype=np.float64))
        # Per-channel normed ADV — flags channels that carry no signal.
        na = _dict_item(cb["normed_advs"]) if "normed_advs" in cb else {}
        km = _dict_item(cb["key_frame_mask"]) if "key_frame_mask" in cb else {}
        ch_zero: List[str] = []
        if isinstance(na, dict):
            for ch, a in na.items():
                a = np.asarray(a, dtype=np.float64)
                active = None
                if isinstance(km, dict) and ch in km:
                    active = np.asarray(km[ch], dtype=bool)
                    a_use = a[active] if active.size == a.size else a
                else:
                    a_use = a
                a_use = a_use[np.isfinite(a_use)]
                if a_use.size and float(np.abs(a_use).max()) < 1e-9:
                    ch_zero.append(str(ch))
        if ch_zero:
            adv_blk["zero_signal_channels"] = ch_zero
        out["adv"] = adv_blk

    # -- gradient signal ----------------------------------------------------
    gs = dd.gradsig_npz
    if gs is not None:
        g_blk: Dict[str, Any] = {
            "partial": "hist" not in gs,
            "n_sampled": _i(gs.get("n_sampled")) or int(gs["valid"].shape[0]),
            "n_valid": _i(gs.get("n_valid")) if "n_valid" in gs else int(
                np.asarray(gs["valid"]).sum()),
        }
        for k in ("gnorm", "coherence", "dir_cos", "proj_mean",
                  "proj_std", "frac_neg", "n_nonfinite", "n_excluded"):
            if k in gs:
                v = _f(gs[k])
                g_blk[k] = _i(gs[k]) if k.startswith("n_") else v
        out["gradsig"] = g_blk

    # -- update outcome -----------------------------------------------------
    up = dd.update_npz
    if up is not None:
        out["update"] = {
            k: _f(up[k]) for k in (
                "kl_mean", "kl_max", "early_stop_kl_mean",
                "clip_frac_mean", "ratio_mean", "ratio_max", "ratio_min",
                "policy_loss_mean", "grad_norm_actor_mean",
            ) if k in up
        }
        for k in ("epochs_done", "actor_epochs_done", "n_batches"):
            if k in up:
                out["update"][k] = _i(up[k])

    # -- timeline summary ---------------------------------------------------
    tlo = timeline_overview(dd)
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

    out["flags"] = _flags(out, tlo)
    out["links"] = {
        "samples": "gradsig/samples?sort=abs_proj&limit=20",
        "timeline": "timeline/overview",
        "trace": "trace/<buffer_idx>",
    }
    return out


def _capabilities(dd) -> Dict[str, Any]:
    caps: Dict[str, Any] = {}
    for name, attr in (
        ("episodes", "episodes_npz"), ("trajectories", "trajectories_npz"),
        ("buffer", "buffer_npz"), ("gae", "gae_npz"),
        ("combine", "combine_npz"), ("update", "update_npz"),
        ("timeline", "timeline_npz"), ("epoch_frames", "epoch_frames_npz"),
    ):
        caps[name] = getattr(dd, attr) is not None
    gs = dd.gradsig_npz
    caps["gradsig"] = (
        False if gs is None else ("full" if "hist" in gs else "partial")
    )
    return caps


def _flags(out: Dict[str, Any], tlo: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Factual 'worth inspecting' hints — observable facts with evidence
    pointers, never causal claims."""
    flags: List[Dict[str, Any]] = []
    tl = out.get("timeline") or {}
    es = tl.get("early_stop_step")
    if es is not None and es >= 0:
        ep = mb = wmk = None
        if tlo.get("epoch_idx") and es < len(tlo["epoch_idx"]):
            ep = tlo["epoch_idx"][es]
            mb = tlo["mb_idx"][es]
        if tlo.get("window_mean_kl") and es < len(tlo["window_mean_kl"]):
            wmk = tlo["window_mean_kl"][es]
        flags.append({
            "kind": "early_stop",
            "text": f"actor early-stopped at step {es} "
                    f"(epoch {ep}, minibatch {mb}); "
                    f"window_mean_kl={wmk} vs target={tl.get('target_kl')}",
            "evidence": {"endpoint": "timeline", "step": es},
        })
    adv = out.get("adv") or {}
    wcf = adv.get("winsorize_clip_frac")
    if wcf:
        flags.append({
            "kind": "winsorize",
            "text": f"combined_adv winsorized: {wcf:.1%} of frames clipped",
            "evidence": {"endpoint": "inspect", "section": "adv"},
        })
    for ch in adv.get("zero_signal_channels") or []:
        flags.append({
            "kind": "zero_signal_channel",
            "text": f"channel '{ch}' has active frames but zero normed ADV "
                    f"— contributes no actor signal this update",
            "evidence": {"endpoint": "inspect", "section": "adv"},
        })
    g = out.get("gradsig") or {}
    if g.get("frac_neg") is not None and g["frac_neg"] > 0.5:
        flags.append({
            "kind": "frac_neg_majority",
            "text": f"{g['frac_neg']:.0%} of sampled frames project "
                    f"against the aggregate gradient direction",
            "evidence": {"endpoint": "gradsig/samples",
                         "query": "sort=neg_proj&limit=20"},
        })
    if (g.get("coherence") is not None and g["coherence"] < 0.05
            and g.get("gnorm")):
        flags.append({
            "kind": "low_coherence",
            "text": f"coherence={g['coherence']:.3f}: aggregate ‖G‖ is a "
                    f"small residue of opposing per-frame gradients",
            "evidence": {"endpoint": "gradsig/samples",
                         "query": "sort=abs_proj&limit=20"},
        })
    # Sampling-representativeness: same z-test the trainer logs.
    pm, ps, nv = g.get("proj_mean"), g.get("proj_std"), g.get("n_valid")
    gn = g.get("gnorm")
    if None not in (pm, ps, nv, gn) and nv and nv > 0:
        se = ps / math.sqrt(nv)
        if se > 0 and abs(pm - gn) / se > 4.0:
            flags.append({
                "kind": "sample_unrepresentative",
                "text": f"mean(proj)={pm:.4g} deviates >4σ from "
                        f"‖G‖={gn:.4g} — sampled frames under-represent "
                        f"the full-buffer gradient",
                "evidence": {"endpoint": "gradsig/samples",
                             "query": "sort=abs_proj&limit=20"},
            })
    return flags


# ---------------------------------------------------------------------------
# samples — the sampled-gradient table
# ---------------------------------------------------------------------------

_SAMPLE_SORTS = ("abs_proj", "proj", "neg_proj", "grad_norm", "w_adv")


def gradsig_samples(
    dd,
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
    gs = dd.gradsig_npz
    if gs is None:
        return 404, {"available": False, "reason": "gradsig.npz not found"}
    if "sampled_idx" not in gs:
        return 404, {"available": False,
                     "reason": "gradsig.npz has no per-frame arrays"}
    if sort not in _SAMPLE_SORTS:
        return 400, {"error": f"sort must be one of {_SAMPLE_SORTS}"}
    if sign not in ("all", "pos", "neg"):
        return 400, {"error": "sign must be all|pos|neg"}
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    sel = np.asarray(gs["sampled_idx"], dtype=np.int64)
    valid = np.asarray(gs["valid"], dtype=bool)

    def _col(name):
        a = gs.get(name)
        return np.asarray(a, dtype=np.float64) if a is not None \
            else np.full(len(sel), np.nan)

    proj = _col("proj")
    cos = _col("cos")
    gnorm_a = _col("grad_norm")
    w_adv = _col("w_adv")
    floor_pen = _col("floor_pen")

    fm = _frame_map(dd)
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
            dd, fm, sel, valid, proj, cos, gnorm_a, w_adv, order)

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
        row.update(_locate(dd, fm, bi))
        rows.append(row)

    return 200, {
        "available": True,
        "meta": {
            "scope": "sampled",
            "n_sampled": int(gs["n_sampled"]) if "n_sampled" in gs else n,
            "n_valid": _i(gs.get("n_valid")),
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


def _samples_by_episode(dd, fm, sel, valid, proj, cos, gnorm_a, w_adv,
                        order) -> Dict[str, Any]:
    """Aggregate the sampled frames per episode — positive and negative
    projections summed separately (a near-zero net hides opposition)."""
    groups: Dict[int, Dict[str, Any]] = {}
    unmapped = 0
    for i in order:
        bi = int(sel[i])
        loc = _locate(dd, fm, bi)
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

def trace_frame(dd, buffer_idx: int) -> Tuple[int, Dict[str, Any]]:
    """Everything recorded about one flat buffer index, joined across
    buffer → GAE → combine → gradsig → epoch_frames → timeline."""
    buf = dd.buffer_npz
    if buf is None:
        return 404, {"available": False, "reason": "buffer.npz not found"}
    n = int(buf["frame_id"].shape[0]) if "frame_id" in buf else (
        int(buf["log_probs"].shape[0]) if "log_probs" in buf else 0)
    i = int(buffer_idx)
    if i < 0 or i >= n:
        return 404, {"error": f"buffer_idx {i} out of range (n={n})"}

    fm = _frame_map(dd)
    out: Dict[str, Any] = {"buffer_idx": i}
    out["location"] = _locate(dd, fm, i)

    # -- rollout-time record -------------------------------------------------
    rec: Dict[str, Any] = {}
    for k in ("log_probs", "sample_weights", "explore_factor",
              "floor_weight", "uncertainty"):
        if k in buf:
            rec[k] = _f(buf[k][i])
    out["buffer"] = rec

    # -- GAE stage (per channel) ---------------------------------------------
    gae = dd.gae_npz
    if gae is not None:
        per_ch: Dict[str, Any] = {}
        for src in ("advs_all", "rets_all", "values_all"):
            d = _dict_item(gae[src]) if src in gae else {}
            if not isinstance(d, dict):
                continue
            for ch, a in d.items():
                a = np.asarray(a)
                if i < a.size:
                    per_ch.setdefault(str(ch), {})[
                        {"advs_all": "adv", "rets_all": "ret",
                         "values_all": "value"}[src]] = _f(a[i])
        kfm = _dict_item(gae["key_frame_mask"]) \
            if "key_frame_mask" in gae else {}
        if isinstance(kfm, dict):
            for ch, a in kfm.items():
                a = np.asarray(a, dtype=bool)
                if i < a.size:
                    per_ch.setdefault(str(ch), {})["active"] = bool(a[i])
        out["gae"] = per_ch

    # -- combine stage --------------------------------------------------------
    cb = dd.combine_npz
    if cb is not None:
        cblk: Dict[str, Any] = {}
        for k in ("combined_adv", "combined_adv_raw", "aw_l1_sum"):
            if k in cb:
                cblk[k] = _f(cb[k][i])
        for src, dst in (("normed_advs", "normed_adv"),
                         ("key_actor_weight_frame", "actor_weight")):
            d = _dict_item(cb[src]) if src in cb else {}
            if isinstance(d, dict):
                for ch, a in d.items():
                    a = np.asarray(a)
                    if i < a.size:
                        cblk.setdefault("per_channel", {}).setdefault(
                            str(ch), {})[dst] = _f(a[i])
        conf = _dict_item(cb["confidences"]) \
            if "confidences" in cb else {}
        if isinstance(conf, dict):
            cblk["confidences"] = {str(k): _f(v) for k, v in conf.items()}
        out["combine"] = cblk

    # -- gradient diagnostic (sampled only) -----------------------------------
    gs = dd.gradsig_npz
    if gs is not None and "sampled_idx" in gs:
        sel = np.asarray(gs["sampled_idx"], dtype=np.int64)
        hit = np.where(sel == i)[0]
        if hit.size:
            j = int(hit[0])
            out["gradsig"] = {
                "sampled": True,
                "valid": bool(np.asarray(gs["valid"])[j]),
                "grad_norm": _f(gs["grad_norm"][j])
                if "grad_norm" in gs else None,
                "cos": _f(gs["cos"][j]) if "cos" in gs else None,
                "proj": _f(gs["proj"][j]) if "proj" in gs else None,
                "w_adv": _f(gs["w_adv"][j]) if "w_adv" in gs else None,
                "floor_pen": _f(gs["floor_pen"][j])
                if "floor_pen" in gs else None,
            }
        else:
            out["gradsig"] = {
                "sampled": False,
                "reason": "frame not in the <=2000 sampled subset — "
                          "not a zero gradient",
            }

    # -- epoch snapshots -------------------------------------------------------
    ef = dd.epoch_frames_npz
    if ef is not None:
        epochs: Dict[str, Any] = {}
        for k in ef:
            parts = k.split(".")
            if len(parts) < 2:
                continue
            name, e = parts[0], parts[1]
            if name in ("ratio", "clip_mask", "new_log_prob"):
                a = np.asarray(ef[k])
                if i < a.size:
                    v = a[i]
                    epochs.setdefault(e, {})[name] = (
                        bool(v) if name == "clip_mask" else _f(v))
            elif name == "new_value" and len(parts) >= 3:
                a = np.asarray(ef[k])
                if i < a.size:
                    epochs.setdefault(e, {}).setdefault(
                        "new_value", {})[parts[2]] = _f(a[i])
        if epochs:
            out["epoch_frames"] = dict(
                sorted(epochs.items(), key=lambda kv: int(kv[0])))
        if "actor_stopped_epoch" in ef:
            out["actor_stopped_epoch"] = _i(ef["actor_stopped_epoch"])

    # -- timeline reverse refs ---------------------------------------------------
    tl = dd.timeline_npz
    if tl is not None:
        refs = []
        for k in ("argmax_ratio_bufidx", "argmin_ratio_bufidx"):
            if k not in tl:
                continue
            arr = np.asarray(tl[k], dtype=np.float64)
            for s in np.where(arr == i)[0]:
                refs.append({
                    "step": int(s),
                    "role": "argmax_ratio" if "max" in k else "argmin_ratio",
                    "epoch": _i(tl["epoch_idx"][s])
                    if "epoch_idx" in tl else None,
                    "mb_idx": _i(tl["mb_idx"][s])
                    if "mb_idx" in tl else None,
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
