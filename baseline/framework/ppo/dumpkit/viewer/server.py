"""HTTP server for the debug viewer.

Serves a single-page frontend plus JSON API endpoints that read dump
NPZ files lazily and cache them in memory.

Usage::

    PYTHONPATH=. python3 baseline/framework/ppo/debug.py viewer <dump_dir>
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.dumpkit.dump_request import (
    DumpRequest,
    SENTINEL_FILENAME,
)
from baseline.framework.ppo.dumpkit.frame_access import DumpDataset
from baseline.framework.ppo.dumpkit.metric_catalog import (
    catalog as _metric_catalog,
    metric_doc,
)
from baseline.framework.ppo.dumpkit import dump_analysis as _da

_HERE = Path(__file__).resolve().parent
_BUNDLED_HTML = _HERE / "index.html"


# ---------------------------------------------------------------------------
# Episode render job — one at a time, runs `debug.py render` as a subprocess
# ---------------------------------------------------------------------------

_RENDER_LOCK = threading.Lock()
_RENDER_JOB: Optional[Dict[str, Any]] = None


def _render_status() -> Dict[str, Any]:
    """Current render job as a JSON-safe dict (job=None when idle)."""
    with _RENDER_LOCK:
        job = _RENDER_JOB
        if job is None:
            return {"job": None}
        rc = job["proc"].poll()
        if rc is not None and job.get("log_f") is not None:
            job["log_f"].close()
            job["log_f"] = None
        status = "running" if rc is None else ("done" if rc == 0 else "error")
        return {
            "job": {
                "kind": job["kind"],
                "dump": job["dump"],
                "episode": job["episode"],
                "status": status,
                "returncode": rc,
                "started": job["started"],
                "log": job["log"],
            }
        }


_JOB_CMD = {
    "render": ("render", "render_ep{:05d}.log"),
    "delta": ("delta", "delta_ep{:05d}.log"),
}


def _start_job(
    kind: str,
    dump_dir: Path,
    dump_name: str,
    episode: int,
    gens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Spawn `debug.py <kind>` in the background; refuses a second job.

    One background job at a time for the whole viewer — renders and
    deltas share the slot so heavy work never stacks up.
    """
    global _RENDER_JOB
    if episode < 0:
        return 400, {"error": "episode must be >= 0"}
    if not (dump_dir / "episodes.npz").exists():
        return 404, {"error": f"not a valid dump: {dump_dir}"}
    sub, log_tmpl = _JOB_CMD[kind]
    cmd = [
        sys.executable, "-B", "-m", "baseline.framework.ppo.debug",
        sub, str(dump_dir), "--episode", str(episode),
    ]
    if kind == "delta":
        if gens is None or not 1 <= gens <= 10:
            return 400, {"error": "gens must be an int in [1, 10]"}
        cmd += ["--gens", str(gens)]
    with _RENDER_LOCK:
        if _RENDER_JOB is not None and _RENDER_JOB["proc"].poll() is None:
            j = _RENDER_JOB
            return 409, {
                "error": f"a {j['kind']} job is already running "
                         f"({j['dump']} episode {j['episode']})",
            }
        # Repo root = parents[5] of .../baseline/framework/ppo/dumpkit/viewer/
        repo_root = Path(__file__).resolve().parents[5]
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        )
        log_path = dump_dir / log_tmpl.format(episode)
        log_f = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(repo_root),
                env=env,
            )
        except OSError as e:
            log_f.close()
            return 500, {"error": f"failed to spawn {kind}: {e}"}
        _RENDER_JOB = {
            "kind": kind,
            "dump": dump_name,
            "episode": episode,
            "proc": proc,
            "log_f": log_f,
            "started": time.time(),
            "log": str(log_path),
        }
        return 200, {
            "ok": True, "kind": kind, "dump": dump_name,
            "episode": episode, "log": str(log_path),
        }


# ---------------------------------------------------------------------------
# RunData — training-run level data (dump list + train.log metrics)
# ---------------------------------------------------------------------------

# Keys historically contributed via ActorEval.stats (truncated_normal_mlp).
# Used to re-classify stats.* keys in old logs that predate the
# raw["policy_stats"] provenance field.
_LEGACY_POLICY_STAT_KEYS = frozenset({
    "std_mean", "eff_std_mean", "std_min", "std_max", "mean_abs",
})

# Stats keys the framework owns even when a policy also emits them inside
# ActorEval.stats: uncertainty is aggregated from the contract field
# ActorEval.uncertainty (consumed by the floor loss), so it classifies as
# stats.uncertainty_mean, never policy.uncertainty_mean.
_FRAMEWORK_STAT_KEYS = frozenset({"uncertainty_mean"})


def _nan_to_null(arr: np.ndarray) -> List[Any]:
    """float array → JSON-safe list; non-finite values become ``None``.

    ``json.dumps`` emits a bare ``NaN``/``Infinity`` literal for
    non-finite floats — invalid JSON that makes ``fetch().json()``
    throw client-side.  Explicit nulls keep the payload parseable.
    """
    return [
        (float(v) if np.isfinite(v) else None)
        for v in np.asarray(arr, dtype=np.float64).ravel()
    ]


def _grad_sig_json(d: Dict[str, np.ndarray], update: int) -> Dict[str, Any]:
    """gradsig npz dict → JSON-safe payload.

    Shared by the legacy run-level artifact (``gradsig/uNNNNN.npz``)
    and the dump artifact (``dumps/uNNNNN/gradsig.npz``) — both carry
    the same hist + scalar + per-frame keys.
    """
    return {
        "available": True,
        "update": int(update),
        # hist[norm_bin, cos_bin] — frame counts (per-frame
        # norm bins × cos bins vs the aggregate direction).
        "hist": d["hist"].tolist(),
        "cos_edges": d["cos_edges"].tolist(),
        "norm_edges": d["norm_edges"].tolist(),
        "n_sampled": int(d["n_sampled"]),
        "n_valid": int(d["n_valid"]),
        "n_excluded": int(d["n_excluded"]),
        "n_frames_in_hist": int(d["n_frames_in_hist"]),
        # Under/overflow edge rows — cosine profiles of
        # frames outside the frozen quantile norm axis.
        "hist_under": d["hist_under"].tolist(),
        "hist_over": d["hist_over"].tolist(),
        "n_under": int(d["n_under"]),
        "n_over": int(d["n_over"]),
        # Aggregate-direction scalars.
        "gnorm": float(d["gnorm"]),
        "coherence": float(d["coherence"]),
        "dir_cos": float(d["dir_cos"]),
        "proj_mean": float(d["proj_mean"]),
        "proj_std": float(d["proj_std"]),
        "frac_neg": float(d["frac_neg"]),
        # Raw per-frame arrays — the frontend bins the
        # projection histogram itself.  Non-finite values
        # (invalid frames) become null: a literal NaN or
        # Infinity in the JSON would break fetch().json().
        "grad_norm": _nan_to_null(d["grad_norm"]),
        "cos": _nan_to_null(d["cos"]),
        "proj": _nan_to_null(d["proj"]),
        "valid": d["valid"].tolist(),
        "norm_quantiles": d["norm_quantiles"].tolist(),
        "norm_quantile_levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "norm_edges_derived": bool(d["norm_edges_derived"]),
    }


def _dump_gradsig(ds: DumpDataset) -> Optional[Dict[str, Any]]:
    """Dump-level ``gradsig.npz`` → JSON-safe payload.

    The dump npz merges the run-level histogram payload with the
    per-frame dump detail (``sampled_idx``/``w_adv``/``floor_pen``).
    When the diagnostic bailed before the histogram (zero aggregate
    gradient, no valid frames) only the raw arrays exist — return a
    degraded ``partial`` payload instead of KeyError → 404.
    """
    g = ds.npz("gradsig")
    if not g.available:
        return None
    d = {k: g.get(k) for k in g.keys()}
    try:
        upd = int(ds.manifest.get("update", -1))
    except (FileNotFoundError, TypeError, ValueError):
        upd = -1
    extras = (
        "sampled_idx", "w_adv", "floor_pen", "n_params",
    )
    try:
        out = _grad_sig_json(d, update=upd)
    except KeyError:
        out = {
            "available": True,
            "partial": True,
            "update": upd,
            "n_sampled": int(d["valid"].shape[0]),
            "n_valid": int(d["valid"].sum()),
            "grad_norm": _nan_to_null(d["grad_norm"]),
            "valid": d["valid"].tolist(),
        }
    for k in extras + ("proj", "cos"):
        if k in d and k not in out:
            out[k] = (
                _nan_to_null(d[k]) if d[k].dtype.kind == "f"
                else d[k].tolist()
            )
    return out


class RunData:
    """Training run directory: scan dumps and parse __RAW_STATS__ from train.log."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_dir():
            raise NotADirectoryError(f"run dir not found: {self.run_dir}")
        self._dump_apis: Dict[str, "ViewerAPI"] = {}
        self._metrics_cache: Optional[List[Dict[str, Any]]] = None
        self._metrics_offset: int = 0

    # -- dump discovery -----------------------------------------------------

    def dumps(self) -> List[Dict[str, Any]]:
        """List dumps under run_dir/dumps/, sorted by update number."""
        dumps_dir = self.run_dir / "dumps"
        result: List[Dict[str, Any]] = []
        if not dumps_dir.is_dir():
            return result
        for d in sorted(dumps_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("u"):
                continue
            manifest_path = d / "manifest.json"
            entry: Dict[str, Any] = {"name": d.name}
            if manifest_path.exists():
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        m = json.load(f)
                    entry["update"] = m.get("update")
                    entry["n_episodes"] = m.get("n_episodes")
                    entry["n_trajectories"] = m.get("n_trajectories")
                    entry["total_frames"] = m.get("total_frames")
                    entry["experiment_name"] = m.get("experiment_name")
                except (json.JSONDecodeError, OSError):
                    pass
            # Captured time from dir mtime
            entry["mtime"] = d.stat().st_mtime
            result.append(entry)
        result.sort(key=lambda e: e.get("update") or 0)
        return result

    def get_dump_api(self, name: str) -> Optional["ViewerAPI"]:
        """Get (or lazily create) a ViewerAPI for dump <name>."""
        if name in self._dump_apis:
            return self._dump_apis[name]
        dump_dir = self.run_dir / "dumps" / name
        if not dump_dir.is_dir() or not (dump_dir / "manifest.json").exists():
            return None
        api = ViewerAPI(DumpDataset(dump_dir))
        self._dump_apis[name] = api
        return api

    def dump_image_path(self, name: str, ep_pos: int, frame: int) -> Optional[Path]:
        api = self.get_dump_api(name)
        if api is None:
            return None
        return api.data.image_path(ep_pos, frame)

    # -- eval videos --------------------------------------------------------

    _VIDEO_NAME = re.compile(r"^u(\d+)\.mp4$")

    def videos(self) -> List[Dict[str, Any]]:
        """Eval videos under videos/, newest first.  Each uNNNNN.mp4 may
        carry a uNNNNN.log sidecar (steps/termination/health/seed);
        eval.* metrics are joined in by update number."""
        vdir = self.run_dir / "videos"
        if not vdir.is_dir():
            return []
        by_update = {m["update"]: m for m in self.metrics()}
        out: List[Dict[str, Any]] = []
        for f in sorted(vdir.iterdir()):
            m = self._VIDEO_NAME.match(f.name)
            if not m or not f.is_file():
                continue
            upd = int(m.group(1))
            meta: Dict[str, Any] = {}
            side = f.with_suffix(".log")
            if side.exists():
                try:
                    # The sidecar is captured stdout: warnings + a
                    # "Video saved to ..." line, then a JSON result block.
                    txt = side.read_text(encoding="utf-8")
                    i = txt.find("\n{")
                    i = 0 if txt.startswith("{") else (i + 1 if i >= 0 else -1)
                    if i >= 0:
                        meta, _ = json.JSONDecoder().raw_decode(txt[i:])
                except (json.JSONDecodeError, OSError):
                    meta = {}
            try:
                st = f.stat()
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                size, mtime = 0, 0.0
            term = meta.get("termination_reasons") or {}
            entry: Dict[str, Any] = {
                "name": f.name,
                "update": upd,
                "size": size,
                "mtime": mtime,
                "steps": meta.get("steps"),
                "seed": meta.get("seed"),
                "health_a": meta.get("health_a"),
                "health_b": meta.get("health_b"),
                "term_a": (term.get("robot_a") or [None])[0],
                "term_b": (term.get("robot_b") or [None])[0],
            }
            mu = by_update.get(upd)
            if mu is not None:
                entry["eval_success"] = mu.get("eval.success")
                entry["eval_pot"] = mu.get("eval.final_pot")
                entry["eval_max_pot"] = mu.get("eval.max_pot")
            out.append(entry)
        out.sort(key=lambda e: e["update"], reverse=True)
        return out

    def video_path(self, name: str) -> Optional[Path]:
        if not self._VIDEO_NAME.match(name):
            return None
        vdir = (self.run_dir / "videos").resolve()
        p = (vdir / name).resolve()
        if p.parent != vdir or not p.is_file():
            return None
        return p

    # -- gradsig artifacts --------------------------------------------------

    def grad_sig(self, update: int) -> Optional[Dict[str, Any]]:
        """Load ``gradsig/uNNNNN.npz`` for one update → JSON-safe dict.

        Legacy run-level artifact: the diagnostic is now dump-only, so
        this file only exists in runs written before the dump-only
        switch.  Missing files and artifacts written by a different
        format generation (KeyError on a required key) return ``None``
        → the API reports ``available: false``.
        """
        p = self.run_dir / "gradsig" / f"u{update:05d}.npz"
        if not p.is_file():
            return None
        try:
            with np.load(p) as d:
                return _grad_sig_json(dict(d), update=int(update))
        except (OSError, KeyError, ValueError):
            return None

    # -- train.log metrics --------------------------------------------------

    @property
    def train_log_path(self) -> Path:
        return self.run_dir / "train.log"

    def metrics(self) -> List[Dict[str, Any]]:
        """Parse __RAW_STATS__ JSON lines from train.log.

        Incremental: tracks byte offset so a growing log is cheap to re-parse.
        Each entry is {update, stats: {...flat scalars...}, timing: {...}}.
        """
        log = self.train_log_path
        if not log.exists():
            return []
        size = log.stat().st_size
        if self._metrics_cache is not None and size == self._metrics_offset:
            return self._metrics_cache
        if self._metrics_cache is None or size < self._metrics_offset:
            # First parse, or log was truncated/rotated — start over.
            self._metrics_cache = []
            self._metrics_offset = 0
        with open(log, "rb") as f:
            f.seek(self._metrics_offset)
            tail = f.read()
        # Only consume up to the last newline — a partially-written trailing
        # line gets another chance on the next call.
        last_nl = tail.rfind(b"\n")
        if last_nl < 0:
            return self._metrics_cache
        complete = tail[: last_nl + 1]
        self._metrics_offset += len(complete)
        for line in complete.decode("utf-8", errors="replace").splitlines():
            marker = "__RAW_STATS__"
            idx = line.find(marker)
            if idx < 0:
                continue
            try:
                raw = json.loads(line[idx + len(marker):])
            except json.JSONDecodeError:
                continue
            entry = self._flatten_update(raw)
            if entry is not None:
                self._metrics_cache.append(entry)
        return self._metrics_cache

    def summary(self) -> Dict[str, Any]:
        """Per-metric digest of the flattened update series.

        For every flat key present in metrics(): presence stats (n,
        first/latest update), latest value, and min/max with their
        update indices — each annotated with zone + hint from
        metric_catalog so the output is self-describing.  Sparse keys
        (eval.*) report n < n_updates honestly.
        """
        metrics = self.metrics()
        per_key: Dict[str, Any] = {}
        keys = sorted(
            {k for u in metrics for k in u if k != "update"}
        )
        for k in keys:
            pts = [
                (u["update"], u[k]) for u in metrics
                if isinstance(u.get(k), (int, float))
            ]
            if not pts:
                continue
            vals = [v for _, v in pts]
            v_min, v_max = min(vals), max(vals)
            doc = metric_doc(k)
            per_key[k] = {
                "zone": doc["zone"],
                "hint": doc["hint"],
                "n": len(pts),
                "first_update": pts[0][0],
                "latest": vals[-1],
                "latest_update": pts[-1][0],
                "min": v_min,
                "min_update": pts[vals.index(v_min)][0],
                "max": v_max,
                "max_update": pts[vals.index(v_max)][0],
            }
        info = self.run_info()
        return {
            "run": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "experiment_name": info["experiment_name"],
            "status": info["status"],
            "n_updates": len(metrics),
            "n_dumps": info["n_dumps"],
            "dumps": [d["name"] for d in self.dumps()],
            "metrics": per_key,
        }

    @staticmethod
    def _flatten_update(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Flatten one __RAW_STATS__ record into scalar fields for charting."""
        update = raw.get("update")
        if update is None:
            return None
        out: Dict[str, Any] = {"update": int(update)}

        # Channel names (for grouping per-channel metrics in stats.*)
        buf = raw.get("buffer_stats") or {}
        pc = buf.get("per_channel") or {}
        channels = [c for c in pc.keys() if isinstance(c, str)] if isinstance(pc, dict) else []

        # Policy-contributed stats get their own policy.* namespace so the
        # UI can group them separately from framework-guaranteed stats.*.
        # Newer logs carry raw["policy_stats"]; for older logs fall back
        # to the key set truncated_normal_mlp historically emitted.
        pol = raw.get("policy_stats")
        if isinstance(pol, dict):
            policy_keys = set(pol.keys()) - _FRAMEWORK_STAT_KEYS
            for k, v in pol.items():
                if k in policy_keys and isinstance(v, (int, float)):
                    out[f"policy.{k}"] = float(v)
        else:
            policy_keys = _LEGACY_POLICY_STAT_KEYS

        stats = raw.get("stats") or {}
        for k, v in stats.items():
            if not isinstance(v, (int, float)):
                continue  # skip non-scalars (e.g. epoch_kl_stats list)
            # Backward compat: stats.n_episodes in older logs actually
            # counted buffer trajectories (len(buf.traj_lengths)), not
            # episodes.  Renamed to n_trajectories upstream; remap here
            # so old logs display the honest name.
            if k == "n_episodes":
                k = "n_trajectories"
            # stats.ep_len_* in older logs were trajectory lengths
            # (computed from buf.traj_lengths); renamed upstream.
            if k.startswith("ep_len_"):
                k = "traj_" + k[3:]
            if k in policy_keys:
                # Policy-contributed (spread into stats for legacy
                # consumers); emit under policy.* if not already there.
                out.setdefault(f"policy.{k}", float(v))
                continue
            # Re-classify per-channel keys like "vloss_r_potential" → pc.vloss.r_potential
            grouped = False
            for ch in channels:
                if k.endswith("_" + ch):
                    out[f"pc.{k[:-len(ch) - 1]}.{ch}"] = float(v)
                    grouped = True
                    break
            if not grouped:
                out[f"stats.{k}"] = float(v)

        ev = raw.get("eval_info") or {}
        for k, v in ev.items():
            if isinstance(v, (int, float)):
                out[f"eval.{k}"] = float(v)

        ep = raw.get("episode_stats") or {}
        for k, v in ep.items():
            if isinstance(v, (int, float)):
                out[f"ep.{k}"] = float(v)

        # Experiment-defined metrics (on_update() return value) get
        # their own exp.* namespace — parallel to policy.*, absent in
        # logs written before this field existed.
        exp = raw.get("experiment")
        if isinstance(exp, dict):
            for k, v in exp.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    out[f"exp.{k}"] = float(v)
        # per-channel buffer stats → pc.<field>.<ch>
        if isinstance(pc, dict):
            for ch, fields in pc.items():
                if not isinstance(fields, dict):
                    continue
                for k, v in fields.items():
                    if isinstance(v, (int, float)):
                        out[f"pc.{k}.{ch}"] = float(v)

        timing = raw.get("timing") or {}
        for k, v in timing.items():
            if isinstance(v, (int, float)):
                out[f"time.{k}"] = float(v)

        # param_overrides: per-update effective parameter overrides —
        # row-level metadata for Update Detail (config-change evidence),
        # never a chartable metric.
        po = raw.get("param_overrides")
        if isinstance(po, dict) and po:
            out["param_overrides"] = po
        return out

    def run_info(self) -> Dict[str, Any]:
        config_path = self.run_dir / "config.json"
        exp_name = None
        cfg: Optional[Dict[str, Any]] = None
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                exp_name = cfg.get("experiment", {}).get("name")
            except (json.JSONDecodeError, OSError):
                pass
        snapshot = None
        snap_path = self.run_dir / "code_snapshot.json"
        if snap_path.exists():
            try:
                with open(snap_path, encoding="utf-8") as f:
                    snapshot = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "run_name": self.run_dir.name,
            "experiment_name": exp_name,
            "run_dir": str(self.run_dir),
            "config": cfg,
            "code_snapshot": snapshot,
            "has_train_log": self.train_log_path.exists(),
            "n_dumps": len(self.dumps()),
            "status": self.status(),
            "dump_pending": (self.run_dir / SENTINEL_FILENAME).exists(),
        }

    def status(self) -> str:
        """Live status — same rules as the runs index."""
        max_updates = None
        cfg_path = self.run_dir / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                max_updates = (
                    (cfg.get("experiment") or {})
                    .get("common_params") or {}
                ).get("max_updates")
            except (json.JSONDecodeError, OSError):
                pass
        log = self.train_log_path
        last, _ = (
            _tail_raw_stats(log) if log.exists() else (None, None)
        )
        update = last.get("update") if isinstance(last, dict) else None
        try:
            activity = log.stat().st_mtime
        except OSError:
            activity = self.run_dir.stat().st_mtime
        return _determine_status(self.run_dir, update, max_updates, activity)


# ---------------------------------------------------------------------------
# Runs-root mode: index of every run under a parent directory
# ---------------------------------------------------------------------------

_RUN_NAME_TS = re.compile(r"_(\d{8})_(\d{6})$")


def _tail_raw_stats(
    log_path: Path, tail_bytes: int = 262144,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(latest stats, latest stats with eval_info) from train.log's tail.

    Index view needs only the latest update — tail-read avoids parsing
    whole logs for every run on every listing.  eval_info only exists on
    eval_interval updates, so the scan keeps going backwards until it
    also finds one line carrying eval results (or exhausts the tail).
    """
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            tail = f.read()
    except OSError:
        return None, None
    latest: Optional[Dict[str, Any]] = None
    latest_eval: Optional[Dict[str, Any]] = None
    for line in reversed(tail.decode("utf-8", errors="replace").splitlines()):
        i = line.find("__RAW_STATS__")
        if i < 0:
            continue
        try:
            d = json.loads(line[i + len("__RAW_STATS__"):])
        except json.JSONDecodeError:
            continue
        if latest is None:
            latest = d
        if latest_eval is None and d.get("eval_info"):
            latest_eval = d
        if latest is not None and latest_eval is not None:
            break
    return latest, latest_eval


def _pid_alive(pid_path: Path) -> Optional[bool]:
    """True/False when a pid file exists; None when absent or unreadable.

    Guards against PID reuse: a live pid is only trusted when the
    process started no later than the pid file was written — the
    launcher forks first and writes the file immediately after, so a
    process that started *after* the write must be a recycled pid
    belonging to someone else (e.g. a multiprocessing worker of an
    unrelated run), and the original training process is gone.
    """
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return _pid_start_matches(pid, pid_path)


def _pid_start_matches(pid: int, pid_path: Path) -> bool:
    """True when /proc/<pid> started no later than pid_path's mtime.

    Linux-only probe; falls back to trusting the live pid when the
    start time cannot be determined.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # field 22 (starttime, clock ticks since boot) follows the comm
        # field, which may itself contain ')' — split on the last one.
        start_ticks = int(stat.rsplit(")", 1)[1].split()[19])
        btime = next(
            int(line.split()[1])
            for line in open("/proc/stat")
            if line.startswith("btime")
        )
        start_epoch = btime + start_ticks / os.sysconf("SC_CLK_TCK")
        return start_epoch <= pid_path.stat().st_mtime + 1.0
    except (OSError, ValueError, IndexError, StopIteration):
        return True


def _run_created_ts(
    name: str, run_dir: Path, cfg: Optional[Dict[str, Any]],
) -> float:
    """Best-effort creation time: config saved_at → name ts → dir mtime."""
    if cfg:
        saved = cfg.get("saved_at")
        if isinstance(saved, str):
            try:
                return time.mktime(
                    time.strptime(saved, "%Y-%m-%d %H:%M:%S")
                )
            except ValueError:
                pass
    m = _RUN_NAME_TS.search(name)
    if m:
        try:
            return time.mktime(
                time.strptime(m.group(0).lstrip("_"), "%Y%m%d_%H%M%S")
            )
        except ValueError:
            pass
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return 0.0


def _determine_status(
    run_dir: Path,
    update: Optional[int],
    max_updates: Optional[int],
    activity: float,
) -> str:
    """Run status from pid liveness, completion, and log freshness."""
    alive = _pid_alive(run_dir / "pid")
    fresh = (time.time() - activity) < 90.0
    if alive:
        return "running"
    if update is not None and max_updates and update >= max_updates:
        return "finished"
    if alive is None and fresh:
        # No pid file but the log moved within the last 90s.
        return "running"
    if update is None:
        return "unknown"
    return "stopped"


def _scan_run_summary(run_dir: Path) -> Dict[str, Any]:
    """Lightweight per-run summary for the runs index."""
    cfg: Optional[Dict[str, Any]] = None
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cfg = None
    log_path = run_dir / "train.log"
    last, last_eval = (
        _tail_raw_stats(log_path) if log_path.exists() else (None, None)
    )
    exp = (cfg or {}).get("experiment") or {}
    common = exp.get("common_params") or {}
    max_updates = common.get("max_updates")
    update = last.get("update") if isinstance(last, dict) else None
    eval_info = (last_eval or {}).get("eval_info") or {}
    try:
        activity = log_path.stat().st_mtime
    except OSError:
        activity = run_dir.stat().st_mtime
    status = _determine_status(run_dir, update, max_updates, activity)
    dumps_dir = run_dir / "dumps"
    n_dumps = 0
    if dumps_dir.is_dir():
        n_dumps = sum(
            1 for c in dumps_dir.iterdir()
            if c.is_dir() and (c / "manifest.json").exists()
        )
    videos_dir = run_dir / "videos"
    n_videos = 0
    if videos_dir.is_dir():
        n_videos = sum(
            1 for c in videos_dir.iterdir()
            if c.is_file() and c.suffix == ".mp4"
        )
    return {
        "name": run_dir.name,
        "experiment": exp.get("name"),
        "algo": (cfg or {}).get("algorithm"),
        "status": status,
        "update": update,
        "max_updates": max_updates,
        "eval_success": eval_info.get("success"),
        "eval_pot": eval_info.get("final_pot") or eval_info.get("max_pot"),
        "created": _run_created_ts(run_dir.name, run_dir, cfg),
        "activity": activity,
        "n_dumps": n_dumps,
        "n_videos": n_videos,
    }


def list_runs(root: Path) -> List[Dict[str, Any]]:
    """All run dirs under root (anything containing config.json or
    train.log), unsorted."""
    out: List[Dict[str, Any]] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        return out
    for d in children:
        if not d.is_dir():
            continue
        if not ((d / "config.json").exists() or (d / "train.log").exists()):
            continue
        out.append(_scan_run_summary(d))
    return out


def query_runs_index(
    runs: List[Dict[str, Any]],
    *,
    q: str = "",
    sort: str = "created",
    order: str = "desc",
    page: int = 1,
    size: int = 20,
) -> Dict[str, Any]:
    """Filter / sort / paginate a list of run summaries (pure function —
    the /api/runs handler is a thin wrapper over this)."""
    page = max(1, page)
    size = min(200, max(1, size))
    q = q.strip().lower()
    if q:
        runs = [
            r for r in runs
            if q in r["name"].lower()
            or q in str(r.get("experiment") or "").lower()
            or q in str(r.get("algo") or "").lower()
        ]
    else:
        runs = list(runs)
    sort_keys = {
        "name": lambda r: (r["name"] or "").lower(),
        "experiment": lambda r: (r.get("experiment") or "").lower(),
        "status": lambda r: r["status"],
        "update": lambda r: (
            r["update"] if r["update"] is not None else -1),
        "success": lambda r: (
            r["eval_success"]
            if r["eval_success"] is not None else -1),
        "created": lambda r: r.get("created") or 0,
        "activity": lambda r: r.get("activity") or 0,
    }
    runs.sort(
        key=sort_keys.get(sort, sort_keys["created"]),
        reverse=(order != "asc"),
    )
    total = len(runs)
    start = (page - 1) * size
    return {
        "total": total,
        "page": page,
        "size": size,
        "runs": runs[start:start + size],
    }


# ---------------------------------------------------------------------------
# Experiment introspection — static AST scan of experiments_ppo/_sac.
# No imports: keeps the viewer dependency-free so it can serve logs on
# machines without the training stack.
# ---------------------------------------------------------------------------

_BASELINE_DIR = Path(__file__).resolve().parents[4]  # .../baseline


def _literal_scalar(node: ast.AST) -> Tuple[bool, Any]:
    """(True, value) for literal scalar assigns; (False, None) otherwise."""
    try:
        v = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False, None
    return isinstance(v, (bool, int, float, str)), v


def _parse_experiment_file(path: Path) -> Dict[str, Any]:
    """Static parse of one experiment/base file.

    Returns {doc, experiment_class, classes: {ClassName: {bases, attrs,
    has_kwargs_init}}}.  attrs = public scalar class attributes
    (the ``--set``-overridable knob space by convention).
    """
    out: Dict[str, Any] = {"doc": "", "experiment_class": None, "classes": {}}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return out
    out["doc"] = ast.get_docstring(tree) or ""
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            info: Dict[str, Any] = {"bases": [], "attrs": {}, "has_kwargs_init": False}
            for b in node.bases:
                if isinstance(b, ast.Name):
                    info["bases"].append(b.id)
                elif isinstance(b, ast.Attribute):
                    info["bases"].append(b.attr)
            for item in node.body:
                target: Optional[ast.Name] = None
                value_node: Optional[ast.AST] = None
                annotation: Optional[ast.AST] = None
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__" and item.args.kwarg is not None:
                        info["has_kwargs_init"] = True
                    continue
                elif (isinstance(item, ast.AnnAssign)
                      and isinstance(item.target, ast.Name)):
                    target = item.target
                    value_node = item.value
                    annotation = item.annotation
                elif (isinstance(item, ast.Assign)
                      and len(item.targets) == 1
                      and isinstance(item.targets[0], ast.Name)):
                    target = item.targets[0]
                    value_node = item.value
                if target is None or target.id.startswith("_") or value_node is None:
                    continue
                ok, v = _literal_scalar(value_node)
                if not ok:
                    continue
                typ = annotation.id if isinstance(annotation, ast.Name) else type(v).__name__
                info["attrs"][target.id] = {"type": typ, "default": v}
            out["classes"][node.name] = info
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Name) and t.id == "EXPERIMENT_CLASS"
                        and isinstance(node.value, ast.Name)):
                    out["experiment_class"] = node.value.id
    return out


def scan_experiments() -> List[Dict[str, Any]]:
    """Discover all PPO/SAC experiments with their tunable attributes."""
    results: List[Dict[str, Any]] = []
    for algo in ("ppo", "sac"):
        pkg_dir = _BASELINE_DIR / f"experiments_{algo}"
        if not pkg_dir.is_dir():
            continue
        # First pass: class table across the whole package (exp_*.py +
        # base.py) so subclass→base attribute resolution works.
        class_map: Dict[str, Dict[str, Any]] = {}
        parsed_files: List[Tuple[Path, Dict[str, Any]]] = []
        for f in sorted(pkg_dir.glob("*.py")):
            if f.name.startswith("__"):
                continue
            parsed = _parse_experiment_file(f)
            parsed_files.append((f, parsed))
            class_map.update(parsed["classes"])
        for f, parsed in parsed_files:
            cls_name = parsed["experiment_class"]
            if cls_name is None or cls_name not in parsed["classes"]:
                continue
            # Merge attrs over the resolvable MRO (subclass wins by
            # visiting it first).
            attrs: Dict[str, Dict[str, Any]] = {}
            has_kwargs_init = False
            seen: set = set()
            stack = [cls_name]
            while stack:
                cname = stack.pop()
                if cname in seen:
                    continue
                seen.add(cname)
                cinfo = class_map.get(cname)
                if cinfo is None:
                    continue
                has_kwargs_init = has_kwargs_init or cinfo["has_kwargs_init"]
                for k, v in cinfo["attrs"].items():
                    attrs.setdefault(k, v)
                stack.extend(cinfo["bases"])
            exp_name = attrs.get("name", {}).get("default") or cls_name
            doc = parsed["doc"]
            results.append({
                "name": exp_name,
                "class": cls_name,
                "algo": algo,
                "title": doc.strip().splitlines()[0] if doc.strip() else "",
                "doc": doc,
                "source": str(f.relative_to(_BASELINE_DIR.parent)),
                "tunable": has_kwargs_init,
                "tunables": [
                    {"key": k, "type": v["type"], "default": v["default"]}
                    for k, v in sorted(attrs.items())
                    if k != "name"
                ],
            })
    results.sort(key=lambda e: (e["algo"], e["name"]))
    return results


def _experiment_of(run: Dict[str, Any], names: set) -> Optional[str]:
    """Which experiment a run belongs to: config name first, then the
    train_<exp>_<algo>_<ts> naming convention as fallback."""
    exp = run.get("experiment")
    if exp in names:
        return exp
    for n in names:
        if run["name"].startswith(f"train_{n}_"):
            return n
    return None


def _run_group_stats(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    latest = max(rs, key=lambda r: r.get("activity") or 0, default=None)
    return {
        "n_runs": len(rs),
        "n_running": sum(1 for r in rs if r.get("status") == "running"),
        "latest_activity": latest.get("activity") if latest else None,
        "latest_run": latest["name"] if latest else None,
        "latest_eval_pot": latest.get("eval_pot") if latest else None,
        "latest_eval_success": latest.get("eval_success") if latest else None,
    }


def experiments_index(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """GET /api/experiments — every discovered experiment + run stats."""
    exps = scan_experiments()
    names = {e["name"] for e in exps}
    by_exp: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for r in runs:
        by_exp.setdefault(_experiment_of(r, names), []).append(r)
    out: List[Dict[str, Any]] = []
    for e in exps:
        rs = by_exp.pop(e["name"], [])
        out.append({k: e[k] for k in ("name", "algo", "title", "tunable", "source")}
                    | _run_group_stats(rs))
    if by_exp.get(None):
        out.append({"name": "(unassigned)", "algo": "",
                    "title": "runs whose experiment is unregistered or has no config",
                    "tunable": False, "source": "",
                    **_run_group_stats(by_exp[None])})
    return out


def experiment_detail(
    name: str, runs: List[Dict[str, Any]], root: Path,
) -> Optional[Dict[str, Any]]:
    """GET /api/experiment/<name> — dashboard payload."""
    exps = {e["name"]: e for e in scan_experiments()}
    info = exps.get(name)
    if info is None:
        return None
    exp_runs = [r for r in runs if _experiment_of(r, set(exps)) == name]
    exp_runs.sort(key=lambda r: r.get("created") or 0, reverse=True)

    # Flatten each run's config params; a key becomes a "diff column"
    # when its value varies across runs.
    param_maps: Dict[str, Dict[str, Any]] = {}
    schema: Optional[Dict[str, Any]] = None
    for r in exp_runs:
        cfg_path = root / r["name"] / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        exp_cfg = cfg.get("experiment") or {}
        if schema is None and exp_cfg:
            schema = {
                k: exp_cfg[k]
                for k in ("common_params", "ppo_params", "sac_params",
                          "initial_exploration", "initial_lr_schedule")
                if exp_cfg.get(k) is not None
            }
        flat: Dict[str, Any] = {}
        for sect in ("common_params", "ppo_params", "sac_params",
                     "initial_exploration"):
            sect_v = exp_cfg.get(sect)
            if isinstance(sect_v, dict):
                flat.update(sect_v)
        param_maps[r["name"]] = flat
    all_keys: set = set()
    for m in param_maps.values():
        all_keys.update(m)
    diff_keys = sorted(
        k for k in all_keys if k != "name"
        if len({json.dumps(m.get(k), sort_keys=True, default=str)
                for m in param_maps.values()}) > 1
    )
    for r in exp_runs:
        m = param_maps.get(r["name"])
        r["params"] = {k: m.get(k) for k in diff_keys} if m else {}

    # Latest checkpoint per run — feeds the resume-from dropdown.
    checkpoints: List[Dict[str, Any]] = []
    for r in exp_runs:
        ck_dir = root / r["name"] / "checkpoints"
        if not ck_dir.is_dir():
            continue
        best: Optional[Tuple[int, Path]] = None
        for f in ck_dir.glob("checkpoint_u*.pt"):
            m = re.match(r"checkpoint_u(\d+)\.pt$", f.name)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), f)
        if best is not None:
            try:
                rel = str(best[1].relative_to(root.parent.parent))
            except ValueError:
                rel = str(best[1])
            checkpoints.append({"run": r["name"], "update": best[0], "path": rel})
    checkpoints.sort(key=lambda c: c["update"], reverse=True)

    return {
        **info,
        "schema": schema,
        "diff_params": diff_keys,
        "runs": exp_runs,
        "checkpoints": checkpoints,
        "cwd": str(_BASELINE_DIR.parent),
    }


def resolve_run(
    root: Path, name: str, cache: Dict[str, "RunData"],
) -> Optional["RunData"]:
    """Resolve a run name to a cached RunData. Only direct children of
    root that look like run dirs are accepted (blocks ../ etc.)."""
    if not name or "/" in name or "\\" in name:
        return None
    d = (root / name).resolve()
    if d.parent != root or not d.is_dir():
        return None
    if not ((d / "config.json").exists() or (d / "train.log").exists()):
        return None
    rd = cache.get(name)
    if rd is None:
        rd = RunData(d)
        cache[name] = rd
    return rd


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

def _arr_to_list(arr: np.ndarray) -> Any:
    """Convert numpy array to JSON-serializable list."""
    if arr.dtype == object:
        return [str(x) for x in arr]
    return arr.tolist()


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class ViewerAPI:
    """Handles all /api/* endpoints.  Each method returns (status, json_body)."""

    def __init__(self, data: DumpDataset):
        self.data = data

    def handle(self, path: str, query: str = "") -> Tuple[int, Any]:
        """Route an /api/... path and return (HTTP status, response body).

        ``query`` is the raw query string — ``do_GET`` strips it before
        dispatch, so endpoints that take parameters receive it here.
        """
        parts = path.strip("/").split("/")
        # parts[0] == "api"
        if len(parts) < 2:
            return 404, {"error": "unknown endpoint"}

        endpoint = parts[1]

        try:
            if endpoint == "manifest":
                return 200, self._manifest()
            elif endpoint == "episode_list":
                return 200, self._episode_list()
            elif endpoint == "traj_map":
                return 200, self.data.traj_map
            elif endpoint == "episode" and len(parts) >= 4 and parts[3] == "frame":
                return self._episode_frame(int(parts[2]), int(parts[4]))
            elif endpoint == "episode" and len(parts) >= 4 and parts[3] == "delta":
                return self._episode_delta(int(parts[2]))
            elif endpoint == "image" and len(parts) >= 4:
                return self._image(int(parts[2]), int(parts[3]))
            elif endpoint == "trajectory" and len(parts) >= 3:
                traj_idx = int(parts[2])
                if len(parts) >= 5 and parts[3] == "frame":
                    return self._traj_frame(traj_idx, int(parts[4]))
                elif len(parts) >= 6 and parts[3] == "epoch" and parts[5] == "overview":
                    return self._traj_epoch_overview(traj_idx, int(parts[4]))
                elif len(parts) >= 7 and parts[3] == "epoch" and parts[5] == "frame":
                    return self._traj_epoch_frame(traj_idx, int(parts[4]), int(parts[6]))
                elif len(parts) >= 4 and parts[3] == "epoch_compare":
                    return self._traj_epoch_compare(traj_idx)
                elif len(parts) >= 3:
                    return self._traj_overview(traj_idx)
            elif endpoint == "timeline" and len(parts) >= 3:
                if parts[2] == "overview":
                    return 200, self._timeline_overview()
                elif parts[2] == "step" and len(parts) >= 4:
                    return self._timeline_step(int(parts[3]))
            elif endpoint == "gradsig" and len(parts) >= 3 \
                    and parts[2] == "samples":
                return self._gradsig_samples(query)
            elif endpoint == "gradsig":
                gs = _dump_gradsig(self.data)
                if gs is None:
                    return 404, {"available": False}
                return 200, gs
            elif endpoint == "inspect":
                return 200, _da.inspect_dump(self.data)
            elif endpoint == "adv" and len(parts) >= 3 \
                    and parts[2] == "hist":
                return 200, _da.adv_histograms(self.data)
            elif endpoint == "trace" and len(parts) >= 3:
                return _da.trace_frame(self.data, int(parts[2]))
            return 404, {"error": f"unknown endpoint: {endpoint}"}
        except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
            return 404, {"error": str(e)}

    # -- /api/manifest ------------------------------------------------------

    def _manifest(self) -> Dict[str, Any]:
        m = dict(self.data.manifest)
        m["channel_names"] = self.data.channel_names
        m["observer_keys"] = self.data.observer_keys
        m["agent_ids"] = self.data.agent_ids
        m["has_images"] = self.data.has_images
        m["has_timeline"] = self.data.has("timeline")
        m["has_epoch_frames"] = self.data.has("epoch_frames")
        m["has_gradsig"] = self.data.has("gradsig")
        # clip_eps and target_kl from timeline
        t = self.data.timeline
        if self.data.has("timeline"):
            m["clip_eps"] = float(t.scalar("clip_eps") or 0.2)
            tgt = t.scalar("target_kl")
            m["target_kl"] = float(tgt) if tgt is not None else float("nan")
            m["n_epochs"] = int(t.scalar("n_epochs") or 0)
            m["n_steps"] = int(t.scalar("n_steps") or 0)
            es = t.scalar("early_stop_step")
            m["early_stop_step"] = int(es) if es is not None else -1
        else:
            m["clip_eps"] = 0.2
            m["target_kl"] = float("nan")
            m["n_epochs"] = 0
            m["n_steps"] = 0
            m["early_stop_step"] = -1
        return m

    # -- /api/episode_list --------------------------------------------------

    def _episode_list(self) -> List[Dict[str, Any]]:
        tm = self.data.traj_map
        result = []
        for ep in tm:
            result.append({
                "list_pos": ep["list_pos"],
                "seed": ep.get("seed"),
                "num_frames": ep.get("num_frames"),
                "n_trajectories": len(ep.get("trajectories", [])),
                "rendered": self.data.episode_rendered(ep["list_pos"]),
            })
        return result

    # -- /api/episode/<pos>/frame/<f> --------------------------------------

    def _episode_frame(self, ep_pos: int, frame: int) -> Tuple[int, Dict[str, Any]]:
        ds = self.data
        if not ds.has("episodes"):
            return 404, {"error": "episodes.npz not found"}
        try:
            ev = ds.episodes[ep_pos]
        except IndexError:
            return 404, {"error": f"episode {ep_pos} out of range"}
        if frame < 0 or frame >= ev.n_frames:
            return 404, {"error": f"frame {frame} out of range for episode {ep_pos}"}

        result: Dict[str, Any] = {
            "episode_pos": ep_pos,
            "frame": frame,
            "has_image": ds.image_path(ep_pos, frame) is not None,
        }

        # Per-agent obs/actions
        for aid in ds.agent_ids:
            obs = ev.col(f"obs.{aid}", frame)
            if obs is not None:
                result[f"obs_{aid}"] = _arr_to_list(np.asarray(obs))
            act = ev.col(f"actions.{aid}", frame)
            if act is not None:
                result[f"actions_{aid}"] = _arr_to_list(np.asarray(act))
            ef = ev.col(f"explore_factors.{aid}", frame)
            if ef is not None:
                result[f"explore_factor_{aid}"] = float(ef)

        # Observer outputs for this frame — EpisodeView handles both
        # flat and per-episode object-array layouts.
        observer_data: Dict[str, Any] = {}
        for cname in ev.observer_columns:
            val = ev.col(cname, frame)
            if val is None:
                continue
            out_key = cname[len("observer."):]
            if np.isscalar(val) or np.ndim(val) == 0:
                observer_data[out_key] = float(val)
            else:
                observer_data[out_key] = _arr_to_list(np.asarray(val))
        result["observer_outputs"] = observer_data

        # Termination info
        term = ev.termination
        if term is not None:
            result["termination"] = term

        # Associated trajectories with per-frame channel data
        f = ds.frames
        tr = ds.trajs
        enriched_trajs = []
        for t in ev.trajectories:
            traj_idx = t["traj_idx"]
            t_start = t["t_start"]
            t_len = t["length"]
            # The trajectory frame corresponding to this episode frame
            traj_frame = frame - t_start
            covered = 0 <= traj_frame < t_len

            entry: Dict[str, Any] = {
                "traj_idx": traj_idx,
                "agent_id": t["agent_id"],
                "t_start": t_start,
                "length": t_len,
                "covered": covered,
                "traj_frame": traj_frame if covered else -1,
            }

            if covered and traj_idx < ds.n_trajectories:
                flat_idx = int(ds.seg_offsets[traj_idx]) + traj_frame
                for ch in ds.channel_names:
                    rew = f.col(f"reward.{ch}")
                    if rew is not None:
                        entry[f"reward_{ch}"] = float(rew[flat_idx])
                    aw = f.col(f"actor_weight.{ch}")
                    if aw is not None:
                        entry[f"actor_weight_{ch}"] = float(aw[flat_idx])
                for name in ("floor_weight", "explore_factor"):
                    a = f.col(name)
                    if a is not None:
                        entry[name] = float(a[flat_idx])
                # Per-channel is_terminated (per-trajectory scalar)
                for ch in ds.channel_names:
                    it = tr.col(f"is_terminated.{ch}")
                    if it is not None:
                        entry[f"is_terminated_{ch}"] = bool(it[traj_idx])
                imp = tr.col("importance")
                if imp is not None:
                    entry["importance"] = float(imp[traj_idx])
                fid = f.col("frame_id")
                if fid is not None:
                    entry["frame_id"] = str(fid[flat_idx])

            enriched_trajs.append(entry)

        result["trajectories"] = enriched_trajs

        return 200, result

    # -- /api/episode/<pos>/delta --------------------------------------------

    def _episode_delta(self, ep_pos: int) -> Tuple[int, Dict[str, Any]]:
        """Policy-drift delta data produced by ``debug.py delta``.

        Returns per-generation deterministic actions for each trained
        agent: ``agents[aid]["actions"][g][t][dim]`` where g=0 is the
        rollout policy (update u) and g≥1 are reference generations.
        """
        delta_dir = self.data.delta_dir(ep_pos)
        npz_path = delta_dir / "delta.npz"
        meta_path = delta_dir / "meta.json"
        if not npz_path.exists() or not meta_path.exists():
            return 200, {"available": False, "episode_pos": ep_pos}

        meta = json.loads(meta_path.read_text())
        npz = np.load(npz_path)
        agents: Dict[str, Any] = {}
        for aid in meta.get("agents", []):
            key = f"actions.{aid}"
            if key in npz:
                agents[aid] = {"actions": _arr_to_list(npz[key])}
        return 200, {
            "available": True,
            "episode_pos": ep_pos,
            "update": meta["update"],
            "gen_updates": meta["gen_updates"],
            "missing_updates": meta.get("missing_updates", []),
            "agents": agents,
        }

    # -- /api/image/<pos>/<f> -----------------------------------------------

    def _image(self, ep_pos: int, frame: int) -> Tuple[int, Any]:
        png_path = self.data.image_path(ep_pos, frame)
        if png_path is None or not png_path.exists():
            return 404, {"error": "image not found"}
        return 200, png_path  # signal: serve as binary

    # -- /api/trajectory/<idx>/overview ------------------------------------

    def _find_traj_provenance(self, traj_idx: int) -> Tuple[Optional[int], Optional[int]]:
        """(episode_pos, t_start) for a trajectory, or (None, None)."""
        p = self.data.provenance(traj_idx)
        if p is None:
            return None, None
        return p.get("ep_pos"), p.get("t_start")

    def _traj_overview(self, traj_idx: int) -> Tuple[int, Dict[str, Any]]:
        ds = self.data
        if traj_idx < 0 or traj_idx + 1 >= len(ds.seg_offsets):
            return 404, {"error": f"trajectory {traj_idx} out of range"}
        start, end = ds.frames.traj_slice(traj_idx)
        t = ds.frames.traj(traj_idx)
        T = end - start

        result: Dict[str, Any] = {
            "traj_idx": traj_idx,
            "length": T,
            "channels": ds.channel_names,
        }

        # Episode provenance (for image lookup)
        ep_pos, t_start = self._find_traj_provenance(traj_idx)
        result["episode_pos"] = ep_pos
        result["t_start"] = t_start

        # Frame IDs
        fids = t.col("frame_id")
        if fids is not None:
            result["frame_ids"] = [str(x) for x in fids]

        # Per-channel reward / actor_weight / GAE value+adv+ret
        for ch in ds.channel_names:
            for pub, out_key in (
                    ("reward", f"reward_{ch}"),
                    ("actor_weight", f"actor_weight_{ch}"),
                    ("value", f"values_{ch}"),
                    ("adv", f"advs_{ch}"),
                    ("ret", f"rets_{ch}")):
                a = t.col(f"{pub}.{ch}")
                if a is not None:
                    result[out_key] = _arr_to_list(
                        np.asarray(a, dtype=np.float32))

        # GAE: key masks + bootstrap (dict members, sliced by traj range)
        gae = ds.npz("gae")
        if gae.available:
            d = gae.get_dict("key_frame_mask")
            if isinstance(d, dict):
                result["key_frame_mask"] = {
                    ch: [bool(x) for x in np.asarray(v[start:end])]
                    for ch, v in d.items()
                }
            # key_seg_active / key_seg_terminated (per-trajectory bool)
            for seg_key in ("key_seg_active", "key_seg_terminated"):
                d = gae.get_dict(seg_key)
                if isinstance(d, dict):
                    result[seg_key] = {
                        ch: bool(np.asarray(v)[traj_idx])
                        if traj_idx < len(v) else False
                        for ch, v in d.items()
                    }
            d = gae.get_dict("bootstrap_values")
            if isinstance(d, dict):
                result["bootstrap_values"] = {
                    ch: float(np.asarray(v)[traj_idx])
                    for ch, v in d.items()
                }
            bi = gae.get("bootstrap_indices")
            if bi is not None:
                result["bootstrap_indices"] = _arr_to_list(np.asarray(bi))
                if traj_idx in np.asarray(bi).tolist():
                    result["is_bootstrap_seg"] = True

        # Combine stage
        combine = ds.npz("combine")
        if combine.available:
            for name in ("combined_adv", "aw_l1_sum"):
                a = t.col(name)
                if a is not None:
                    result[name] = _arr_to_list(a)
            d = combine.get_dict("confidences")
            if isinstance(d, dict):
                result["confidences"] = {
                    ch: float(v) for ch, v in d.items()}
            d = combine.get_dict("explained_variances")
            if isinstance(d, dict):
                # Strip "ev_" prefix if present so keys match channel names
                result["explained_variances"] = {
                    (ch[3:] if ch.startswith("ev_") else ch): float(v)
                    for ch, v in d.items()
                }
            d = combine.get_dict("key_actor_weight_frame")
            if isinstance(d, dict):
                result["actor_weights"] = {
                    ch: _arr_to_list(np.asarray(v[start:end],
                                                dtype=np.float32))
                    for ch, v in d.items()
                }
            for member, pub in (("normed_advs", "normed_adv"),
                                ("aw_normed", "aw_normed")):
                d = combine.get_dict(member)
                if isinstance(d, dict):
                    for ch, v in d.items():
                        result[f"{pub}_{ch}"] = _arr_to_list(
                            np.asarray(v[start:end], dtype=np.float32))
            for k in ("uncertainty_floor", "uncertainty_coef"):
                v = combine.scalar(k)
                if v is not None:
                    result[k] = float(v)

        # Trajectory-level + buffer columns
        for name in ("floor_weight", "explore_factor"):
            a = t.col(name)
            if a is not None:
                result[name] = _arr_to_list(a)
        imp = ds.trajs.col("importance")
        if imp is not None and traj_idx < len(imp):
            result["importance"] = float(imp[traj_idx])
        for ch in ds.channel_names:
            it = ds.trajs.col(f"is_terminated.{ch}")
            if it is not None and traj_idx < len(it):
                result[f"is_terminated_{ch}"] = bool(it[traj_idx])

        lp = t.col("log_probs")
        if lp is not None:
            result["old_log_prob"] = _arr_to_list(lp)
        un = t.col("uncertainty")
        if un is not None and len(un) > 0:
            result["uncertainty"] = _arr_to_list(
                np.asarray(un, dtype=np.float32))

        return 200, result

    # -- /api/trajectory/<idx>/frame/<f> -----------------------------------

    def _traj_frame(self, traj_idx: int, frame: int) -> Tuple[int, Dict[str, Any]]:
        status, overview = self._traj_overview(traj_idx)
        if status != 200:
            return status, overview
        T = overview["length"]
        if frame < 0 or frame >= T:
            return 404, {"error": f"frame {frame} out of range (len={T})"}

        result: Dict[str, Any] = {"traj_idx": traj_idx, "frame": frame, "length": T}

        # Extract single-frame values from overview
        for ch in self.data.channel_names:
            for prefix in ["reward", "actor_weight", "values", "advs", "rets"]:
                key = f"{prefix}_{ch}"
                if key in overview:
                    result[key] = overview[key][frame]

        if "combined_adv" in overview:
            result["combined_adv"] = overview["combined_adv"][frame]
        if "old_log_prob" in overview:
            result["old_log_prob"] = overview["old_log_prob"][frame]
        if "frame_ids" in overview:
            result["frame_id"] = overview["frame_ids"][frame]

        # Bootstrap info for this trajectory
        if "bootstrap_indices" in overview:
            bi = overview["bootstrap_indices"]
            # Find if this trajectory has a bootstrap entry
            # bootstrap_indices maps seg_idx → flattened_frame_idx
            traj_start = int(self.data.seg_offsets[traj_idx])
            for i, val in enumerate(bi):
                if int(val) == traj_start + frame or int(val) == traj_start:
                    result["is_bootstrap_frame"] = True
                    if "bootstrap_values" in overview:
                        result["bootstrap_value"] = overview["bootstrap_values"]
                    break

        return 200, result

    # -- /api/trajectory/<idx>/epoch/<e>/overview --------------------------

    def _traj_epoch_overview(self, traj_idx: int, epoch: int) -> Tuple[int, Dict[str, Any]]:
        ds = self.data
        ef = ds.npz("epoch_frames")
        if not ef.available:
            return 404, {"error": "epoch_frames.npz not found"}
        if traj_idx < 0 or traj_idx + 1 >= len(ds.seg_offsets):
            return 404, {"error": f"trajectory {traj_idx} out of range"}
        start, end = ds.frames.traj_slice(traj_idx)
        t = ds.frames.traj(traj_idx)

        n_epochs = int(ef.scalar("n_epochs") or 0)
        if epoch < 0 or epoch >= n_epochs:
            return 404, {"error": f"epoch {epoch} out of range (n_epochs={n_epochs})"}

        result: Dict[str, Any] = {
            "traj_idx": traj_idx,
            "epoch": epoch,
            "length": end - start,
        }

        # Episode provenance (for image lookup)
        ep_pos, t_start = self._find_traj_provenance(traj_idx)
        result["episode_pos"] = ep_pos
        result["t_start"] = t_start

        a = t.col(f"epoch.{epoch}.ratio")
        if a is not None:
            result["ratio"] = _arr_to_list(a)
        a = t.col(f"epoch.{epoch}.clip_mask")
        if a is not None:
            result["clip_mask"] = [bool(x) for x in a]
        a = t.col(f"epoch.{epoch}.log_prob")
        if a is not None:
            result["new_log_prob"] = _arr_to_list(a)

        # Per-channel new_value
        for ch in ds.channel_names:
            a = t.col(f"epoch.{epoch}.value.{ch}")
            if a is not None:
                result[f"new_value_{ch}"] = _arr_to_list(a)

        # Cross-reference: old_log_prob, old_value, return, combined_adv
        a = t.col("log_probs")
        if a is not None:
            result["old_log_prob"] = _arr_to_list(a)
        for ch in ds.channel_names:
            a = t.col(f"value.{ch}")
            if a is not None:
                result[f"old_value_{ch}"] = _arr_to_list(
                    np.asarray(a, dtype=np.float32))
            a = t.col(f"ret.{ch}")
            if a is not None:
                result[f"return_{ch}"] = _arr_to_list(
                    np.asarray(a, dtype=np.float32))
        a = t.col("combined_adv")
        if a is not None:
            result["combined_adv"] = _arr_to_list(a)

        # clip_eps from timeline or update
        ce = ds.timeline.scalar("clip_eps")
        result["clip_eps"] = float(ce) if ce is not None else 0.2

        ase = ef.scalar("actor_stopped_epoch")
        if ase is not None:
            result["actor_stopped_epoch"] = int(ase)

        return 200, result

    # -- /api/trajectory/<idx>/epoch/<e>/frame/<f> -------------------------

    def _traj_epoch_frame(self, traj_idx: int, epoch: int, frame: int) -> Tuple[int, Dict[str, Any]]:
        status, overview = self._traj_epoch_overview(traj_idx, epoch)
        if status != 200:
            return status, overview
        T = overview["length"]
        if frame < 0 or frame >= T:
            return 404, {"error": f"frame {frame} out of range (len={T})"}

        result: Dict[str, Any] = {"traj_idx": traj_idx, "epoch": epoch, "frame": frame}
        for key in ["ratio", "clip_mask", "new_log_prob"]:
            if key in overview:
                result[key] = overview[key][frame]
        for ch in self.data.channel_names:
            nv_key = f"new_value_{ch}"
            if nv_key in overview:
                result[nv_key] = overview[nv_key][frame]

        return 200, result

    # -- /api/trajectory/<idx>/epoch_compare --------------------------------

    def _traj_epoch_compare(self, traj_idx: int) -> Tuple[int, Dict[str, Any]]:
        ds = self.data
        ef = ds.npz("epoch_frames")
        if not ef.available:
            return 404, {"error": "epoch_frames.npz not found"}
        if traj_idx < 0 or traj_idx + 1 >= len(ds.seg_offsets):
            return 404, {"error": f"trajectory {traj_idx} out of range"}
        t = ds.frames.traj(traj_idx)
        T = t.n_rows

        n_epochs = int(ef.scalar("n_epochs") or 0)
        result: Dict[str, Any] = {
            "traj_idx": traj_idx,
            "length": T,
            "n_epochs": n_epochs,
            "epochs": [],
        }

        for e in range(n_epochs):
            entry: Dict[str, Any] = {"epoch": e}
            a = t.col(f"epoch.{e}.ratio")
            if a is not None:
                entry["ratio"] = _arr_to_list(a)
            a = t.col(f"epoch.{e}.clip_mask")
            if a is not None:
                entry["clip_mask"] = [bool(x) for x in a]
            result["epochs"].append(entry)

        ase = ef.scalar("actor_stopped_epoch")
        if ase is not None:
            result["actor_stopped_epoch"] = int(ase)

        ce = ds.timeline.scalar("clip_eps")
        result["clip_eps"] = float(ce) if ce is not None else 0.2

        return 200, result

    # -- /api/timeline/overview ---------------------------------------------

    def _timeline_overview(self) -> Dict[str, Any]:
        return _da.timeline_overview(self.data)

    # -- /api/timeline/step/<s> --------------------------------------------

    def _timeline_step(self, step: int) -> Tuple[int, Dict[str, Any]]:
        return _da.timeline_step(self.data, step)

    # -- /api/gradsig/samples ----------------------------------------------

    def _gradsig_samples(self, query: str) -> Tuple[int, Any]:
        """Ranked sampled-frame table; params via query string."""
        qs = urllib.parse.parse_qs(query)
        def _one(name, default=None):
            v = qs.get(name)
            return v[0] if v else default
        try:
            return _da.gradsig_samples(
                self.data,
                sort=_one("sort", "abs_proj"),
                sign=_one("sign", "all"),
                limit=int(_one("limit", "50")),
                offset=int(_one("offset", "0")),
                group_by=_one("group_by"),
                include_invalid=_one("include_invalid") in ("1", "true"),
            )
        except ValueError as e:
            return 400, {"error": f"bad query parameter: {e}"}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/±Inf) with None.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` literals which are not
    valid JSON — ``JSON.parse`` in the browser rejects the whole response.
    Dump arrays legitimately contain NaN (e.g. timeline steps after an
    actor early-stop), so sanitize here for every API response.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class _ViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the frontend + API + static images.

    Two modes:

    - dump mode: ``server.api`` is a ViewerAPI; routes ``/api/...``, ``/img/...``.
    - run mode: ``server.run_data`` is a RunData; routes ``/api/run/...``,
      ``/api/dump/<name>/...``, ``/img/dump/<name>/...``, and ``/dump/<name>/...``
      serves the SPA for client-side routing.
    - runs mode: ``server.runs_root`` is a Path whose children are run dirs;
      ``/`` serves the runs index, ``/api/runs`` lists summaries, and
      ``/run/<name>/...`` prefixes scope all run/dump routes to that run.
    """

    # Set when the request is scoped to /run/<name>/... in runs mode.
    _run_override: Optional["RunData"] = None

    @property
    def api(self) -> Optional[ViewerAPI]:
        return getattr(self.server, "api", None)  # type: ignore[attr-defined]

    @property
    def run_data(self) -> Optional[RunData]:
        if self._run_override is not None:
            return self._run_override
        return getattr(self.server, "run_data", None)  # type: ignore[attr-defined]

    @property
    def runs_root(self) -> Optional[Path]:
        return getattr(self.server, "runs_root", None)  # type: ignore[attr-defined]

    @property
    def viewer_dir(self) -> Path:
        return self.server.viewer_dir  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002
        return  # quiet

    def _run_for_name(self, name: str) -> Optional["RunData"]:
        root = self.runs_root
        if root is None:
            return None
        return resolve_run(
            root, name, self.server.run_cache  # type: ignore[attr-defined]
        )

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query

        # Runs-root mode: /run/<name>/... scopes everything below it.
        if self.runs_root is not None:
            m = re.match(r"^/run/([^/]+)(/.*)?$", path)
            if m:
                rd = self._run_for_name(urllib.parse.unquote(m.group(1)))
                if rd is None:
                    self.send_error(404)
                    return
                self._run_override = rd
                path = m.group(2) or "/"

        # API routes
        if path.startswith("/api/"):
            self._handle_api(path)
            return

        # Image route (binary)
        if path.startswith("/img/"):
            self._handle_image(path)
            return

        # Eval video route (binary, Range-capable)
        if path.startswith("/video/"):
            self._handle_video(path)
            return

        # Static files / SPA
        self._handle_static(path)

    def do_POST(self):
        path = self.path.split("?")[0]  # strip query

        # Runs-root mode: /run/<name>/... scopes everything below it.
        if self.runs_root is not None:
            m = re.match(r"^/run/([^/]+)(/.*)?$", path)
            if m:
                rd = self._run_for_name(urllib.parse.unquote(m.group(1)))
                if rd is None:
                    self.send_error(404)
                    return
                self._run_override = rd
                path = m.group(2) or "/"

        if self.run_data is not None and path == "/api/run/dump-request":
            self._handle_dump_request()
            return

        # Render / delta endpoints — run mode: /api/dump/<name>/<kind>;
        # single-dump mode: /api/<kind>.  Both only need the dump
        # directory, so they are allowed regardless of run status.
        m = re.match(r"^/api/dump/([^/]+)/(render|delta)$", path)
        if m and self.run_data is not None:
            name = urllib.parse.unquote(m.group(1))
            api = self.run_data.get_dump_api(name)
            if api is None:
                self.send_error(404)
                return
            self._handle_job_request(m.group(2), api.data, name)
            return
        if path in ("/api/render", "/api/delta") and self.api is not None:
            data = self.api.data
            self._handle_job_request(
                path.rsplit("/", 1)[-1], data, data.dump_dir.name,
            )
            return

        self.send_error(404)

    def _handle_job_request(
        self, kind: str, data: DumpDataset, dump_name: str,
    ):
        """POST .../<render|delta> {"episode": N, "gens": G?}."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            body = {}
        try:
            episode = int(body.get("episode"))
        except (TypeError, ValueError):
            self._reply_json(400, {"error": "episode (int) is required"})
            return
        gens = body.get("gens")
        try:
            gens = int(gens) if gens is not None else None
        except (TypeError, ValueError):
            self._reply_json(400, {"error": "gens must be an int in [1, 10]"})
            return
        status, resp = _start_job(
            kind, data.dump_dir, dump_name, episode, gens,
        )
        self._reply_json(status, resp)

    def _handle_dump_request(self):
        """POST /api/run/dump-request — write the dump sentinel file.

        Body: {"hypothesis": str (optional), "full_grad": bool}.
        Only accepted while the run is alive and no request is pending.
        """
        rd = self.run_data
        assert rd is not None
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            body = {}
        req = DumpRequest(
            hypothesis=str(body.get("hypothesis") or "").strip(),
            include_full_grad=bool(body.get("full_grad")),
        )
        if rd.status() != "running":
            self._reply_json(409, {"error": "run is not running"})
            return
        sentinel = rd.run_dir / SENTINEL_FILENAME
        if sentinel.exists():
            self._reply_json(409, {
                "error": "a dump request is already pending — "
                         "it will be consumed at the next update",
            })
            return
        payload = {
            "hypothesis": req.hypothesis,
            "include_full_grad": req.include_full_grad,
            "requested_via": "viewer",
        }
        sentinel.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._reply_json(200, {"ok": True})

    def _reply_json(self, status: int, body: Any):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(_json_safe(body), ensure_ascii=False, allow_nan=False)
            .encode("utf-8")
        )

    # -- run-mode routing ---------------------------------------------------

    def _run_api(self, path: str) -> Tuple[int, Any]:
        """Handle run-scoped API paths (run mode only)."""
        rd = self.run_data
        assert rd is not None
        if path == "/api/run/info":
            return 200, rd.run_info()
        if path == "/api/run/dumps":
            return 200, rd.dumps()
        if path == "/api/run/metrics":
            return 200, rd.metrics()
        if path == "/api/run/videos":
            return 200, rd.videos()
        m = re.match(r"^/api/run/gradsig/(\d+)$", path)
        if m:
            gs = rd.grad_sig(int(m.group(1)))
            if gs is None:
                return 404, {"available": False}
            return 200, gs
        if path.startswith("/api/dump/"):
            # /api/dump/<name>/<endpoint...>
            rest = path[len("/api/dump/"):]
            name, _, sub = rest.partition("/")
            api = rd.get_dump_api(name)
            if api is None:
                return 404, {"error": f"dump not found: {name}"}
            return api.handle(
                "/api/" + sub,
                urllib.parse.urlparse(self.path).query)
        return 404, {"error": f"unknown run api: {path}"}

    def _runs_index(self) -> Dict[str, Any]:
        """GET /api/runs?q=&page=&size=&sort=&order= — paginated summaries."""
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        try:
            page = int((qs.get("page") or ["1"])[0])
            size = int((qs.get("size") or ["20"])[0])
        except ValueError:
            page, size = 1, 20
        return query_runs_index(
            list_runs(self.runs_root),  # type: ignore[arg-type]
            q=(qs.get("q") or [""])[0],
            sort=(qs.get("sort") or ["created"])[0],
            order=(qs.get("order") or ["desc"])[0],
            page=page,
            size=size,
        )

    def _handle_api(self, path: str):
        if path == "/api/catalog":
            # Metric semantics catalog — global, mode-independent.
            status, body = 200, _metric_catalog()
        elif path == "/api/render-status" or (
            path.startswith("/api/dump/") and path.endswith("/render-status")
        ):
            # Global single-job render status — works in every mode.
            status, body = 200, _render_status()
        elif path == "/api/mode":
            mode = (
                "runs" if self.runs_root is not None
                else "run" if self.run_data is not None else "dump"
            )
            status, body = 200, {"mode": mode}
        elif self.run_data is not None:
            status, body = self._run_api(path)
        elif self.runs_root is not None and path == "/api/runs":
            status, body = 200, self._runs_index()
        elif self.runs_root is not None and path == "/api/experiments":
            status, body = 200, experiments_index(list_runs(self.runs_root))
        elif self.runs_root is not None and path.startswith("/api/experiment/"):
            exp_name = urllib.parse.unquote(path[len("/api/experiment/"):])
            detail = experiment_detail(
                exp_name, list_runs(self.runs_root), self.runs_root)
            if detail is not None:
                status, body = 200, detail
            else:
                status, body = 404, {"error": f"unknown experiment: {exp_name}"}
        elif self.runs_root is not None:
            status, body = 404, {"error": f"unknown api: {path}"}
        else:
            status, body = self.api.handle(  # type: ignore[union-attr]
                path, urllib.parse.urlparse(self.path).query)
        if status == 200 and isinstance(body, Path):
            # Image binary response
            self._serve_file(body, "image/png")
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(_json_safe(body), ensure_ascii=False, allow_nan=False)
            .encode("utf-8")
        )

    def _handle_image(self, path: str):
        rd = self.run_data
        if rd is not None:
            # /img/dump/<name>/<ep_pos>/<frame>
            parts = path.strip("/").split("/")
            if len(parts) != 5 or parts[1] != "dump":
                self.send_error(404)
                return
            name = parts[2]
            try:
                ep_pos = int(parts[3])
                frame = int(parts[4])
            except ValueError:
                self.send_error(404)
                return
            png_path = rd.dump_image_path(name, ep_pos, frame)
        else:
            # /img/<ep_pos>/<frame>
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self.send_error(404)
                return
            try:
                ep_pos = int(parts[1])
                frame = int(parts[2])
            except ValueError:
                self.send_error(404)
                return
            png_path = self.api.data.image_path(ep_pos, frame)  # type: ignore[union-attr]
        if png_path is None or not png_path.exists():
            self.send_error(404)
            return
        self._serve_file(png_path, "image/png")

    def _handle_video(self, path: str):
        rd = self.run_data
        p = rd.video_path(path[len("/video/"):]) if rd is not None else None
        if p is None:
            self.send_error(404)
            return
        self._serve_file_range(p, "video/mp4")

    def _serve_file_range(self, path: Path, content_type: str):
        """Serve a file with minimal HTTP Range support (video seeking)."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_hdr = self.headers.get("Range")
        if range_hdr:
            m = re.match(r"bytes=(\d*)-(\d*)$", range_hdr.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = int(m.group(2))
                elif m.group(2):  # suffix range: bytes=-N
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_error(416)
                    return
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _handle_static(self, path: str):
        if path == "/" or path == "":
            path = "/index.html"
        elif self.run_data is not None and (
            path == "/dump" or path.startswith("/dump/")
        ):
            # SPA deep-link into a dump page — serve the app shell.
            path = "/index.html"
        elif "." not in path.rsplit("/", 1)[-1]:
            # SPA deep-link without a file suffix (e.g. /episode/3 in
            # dump mode) — serve the app shell.
            path = "/index.html"
        # Security: only serve files from viewer_dir
        file_path = (self.viewer_dir / path.lstrip("/")).resolve()
        try:
            file_path.relative_to(self.viewer_dir)
        except ValueError:
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return
        content_type = "application/octet-stream"
        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript"
        elif file_path.suffix == ".css":
            content_type = "text/css"
        self._serve_file(file_path, content_type)

    def _serve_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(
    path: Path,
    port: int = 8766,
    open_browser: bool = True,
) -> None:
    """Start the viewer HTTP server.

    Args:
        path: Path to a dump directory (runs/.../dumps/u00008/), a training
            run directory (runs/.../ containing dumps/ and/or train.log),
            or a runs-root directory whose children are run dirs.
        port: HTTP port (default 8766).
        open_browser: Auto-open the browser (default True).
    """
    path = Path(path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"dir not found: {path}")

    if not _BUNDLED_HTML.exists():
        raise FileNotFoundError(f"bundled index.html not found: {_BUNDLED_HTML}")

    is_dump = (path / "manifest.json").exists()
    is_run = not is_dump and (
        (path / "dumps").is_dir() or (path / "train.log").exists()
        or (path / "config.json").exists()
    )
    if not is_dump and not is_run and not list_runs(path):
        raise FileNotFoundError(
            f"not a dump / run / runs-root dir "
            f"(no manifest.json / dumps/ / train.log / run children): {path}"
        )

    with _ThreadingServer(("", port), _ViewerHandler) as httpd:
        # Inject api (dump mode) or run_data (run mode) or runs_root
        # (runs index mode) plus viewer_dir into the server instance so
        # handlers can access them via self.server.
        httpd.runs_root = None  # type: ignore[attr-defined]
        if is_dump:
            data = DumpDataset(path)
            _ = data.manifest  # verify manifest exists
            httpd.api = ViewerAPI(data)  # type: ignore[attr-defined]
            httpd.run_data = None  # type: ignore[attr-defined]
            print(f"[viewer] serving dump: {path}", flush=True)
        elif is_run:
            httpd.api = None  # type: ignore[attr-defined]
            httpd.run_data = RunData(path)  # type: ignore[attr-defined]
            print(f"[viewer] serving run: {path}", flush=True)
        else:
            httpd.api = None  # type: ignore[attr-defined]
            httpd.run_data = None  # type: ignore[attr-defined]
            httpd.runs_root = path  # type: ignore[attr-defined]
            httpd.run_cache = {}  # type: ignore[attr-defined]
            print(
                f"[viewer] serving runs root: {path} "
                f"({len(list_runs(path))} runs)", flush=True
            )
        httpd.viewer_dir = _HERE  # type: ignore[attr-defined]
        url = f"http://localhost:{port}/"
        print(f"[viewer] url: {url}", flush=True)
        print(f"[viewer] press Ctrl+C to stop", flush=True)
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[viewer] shutting down", flush=True)
            httpd.shutdown()


__all__ = ["RunData", "ViewerAPI", "serve"]
