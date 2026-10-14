"""Run-level data machinery for the SAC debug viewer.

SAC-owned counterpart of the run-facing half of the PPO dumpkit
viewer server: status probing, train.log byte paging, the flat
round-row table that feeds the sectioned dashboard, dump/video
listings, the runs index, and the experiments registry scan.

Everything here is read-only over run artifacts on disk.  Heavier
dump-level computation lives in ``analysis.py`` / ``debugkit.py``.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .metric_catalog import TCK_SUM_KEYS

EVENTS_NAME = Path("metrics") / "events.jsonl"


def iter_events(run_dir: Path) -> Iterable[Dict[str, Any]]:
    path = Path(run_dir) / EVENTS_NAME
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# train.log tail: crash/interrupt detection + backward byte paging
# ---------------------------------------------------------------------------

_TRAIN_MARK = re.compile(
    r"^\[(round|eval|update|dump|param_patch|checkpoint|done|resumed|"
    r"pruned|signal|config|warn|early)")
_INTERRUPT_EXCS = {"KeyboardInterrupt", "SystemExit", "GeneratorExit"}
_SHUTDOWN_NOISE = {"BrokenPipeError", "BlockingIOError"}


def _tail_failure(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Terminal-traceback detection on decoded log-tail lines.

    A traceback only counts as a failure when it is terminal — i.e. no
    training markers appear after its block.  KeyboardInterrupt /
    SystemExit means user-stopped; shutdown noise is ignored.
    """
    tbi = [
        i for i, l in enumerate(lines)
        if "Traceback (most recent call last)" in l
    ]
    if not tbi:
        return None

    def exc_of(i: int) -> Optional[str]:
        for j in range(i + 1, len(lines)):
            l = lines[j]
            if "Traceback (most recent call last)" in l:
                return None
            if not l.strip() or l.startswith((" ", "\t")):
                continue
            return l.strip()
        return None

    if any(_TRAIN_MARK.search(l) for l in lines[tbi[-1] + 1:]):
        return None
    excs = [exc_of(i) or "" for i in tbi]
    if any(
        e.split(":")[0].strip() in _INTERRUPT_EXCS or "CancelledError" in e
        for e in excs
    ):
        return {"interrupted": True}
    if excs and all(
        e.split(":")[0].strip() in _SHUTDOWN_NOISE for e in excs
    ):
        return None
    root = tbi[-1]
    for i in reversed(tbi[:-1]):
        if any(_TRAIN_MARK.search(l) for l in lines[i + 1:root]):
            break
        root = i
    exc = exc_of(root)
    return {"crash": True,
            "exception": (exc or "traceback at end of log")[:240]}


def _read_log_tail_lines(
    log_path: Path, tail_bytes: int = 262144,
) -> List[str]:
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            tail = f.read()
    except OSError:
        return []
    return tail.decode("utf-8", errors="replace").splitlines()


def read_log_window(
    log_path: Path, n_lines: int, before: Optional[int],
) -> Dict[str, Any]:
    """Backward byte-window read of train.log (ported from the PPO
    viewer — same no-gap/no-overlap contract)."""
    try:
        size = log_path.stat().st_size
        f = open(log_path, "rb")
    except OSError:
        return {"lines": [], "start": 0, "end": 0, "total_bytes": 0,
                "has_more": False}
    CHUNK, CAP = 256 * 1024, 4 * 1024 * 1024
    with f:
        end = size if before is None else max(0, min(int(before), size))
        start, buf = end, b""
        while start > 0 and buf.count(b"\n") <= n_lines and len(buf) < CAP:
            step = min(CHUNK, start)
            start -= step
            f.seek(start)
            buf = f.read(step) + buf
    parts = buf.split(b"\n")
    if buf.endswith(b"\n"):
        parts = parts[:-1]
    if start > 0 and parts:
        start += len(parts[0]) + 1
        parts = parts[1:]
    dropped = max(0, len(parts) - n_lines)
    keep = parts[dropped:]
    start += sum(len(p) + 1 for p in parts[:dropped])
    return {
        "lines": [p.decode("utf-8", errors="replace").rstrip("\r")
                  for p in keep],
        "start": start,
        "end": end,
        "total_bytes": size,
        "has_more": start > 0,
    }


# ---------------------------------------------------------------------------
# Status probing (SAC rules — terminal event + pid + freshness + tail)
# ---------------------------------------------------------------------------


def _pid_alive(pid_path: Path) -> Optional[bool]:
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
    """Linux-only PID-reuse guard (same rule as the PPO viewer)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
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


_TERMINAL_STATUS_MAP = {
    "completed": "finished",
    "done": "finished",
    "signal_stop": "stopped",
    "user": "stopped",
    "stopped": "stopped",
    "error": "failed",
    "diverged": "failed",
}


def _tail_events(
    run_dir: Path, tail_bytes: int = 262144,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(last_event, terminal_event) scanned from the events.jsonl tail."""
    path = Path(run_dir) / EVENTS_NAME
    lines = _read_log_tail_lines(path, tail_bytes)
    last = None
    terminal = None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if last is None:
            last = ev
        if terminal is None and ev.get("event_type") == "terminal":
            terminal = ev
        if last is not None and terminal is not None:
            break
    return last, terminal


def _max_env_steps(run_dir: Path) -> Optional[int]:
    cfg_path = Path(run_dir) / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            v = ((cfg.get("experiment") or {})
                 .get("common_params") or {}).get("max_env_steps")
            return int(v) if v is not None else None
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None
    return None


def run_status(run_dir: Path) -> Tuple[str, Optional[str]]:
    """(status, error) for a run dir — SAC rules mirroring PPO's five.

    running: pid alive, or no pid file but train.log fresh (<90s);
    finished: terminal event status=completed (or env_step reached
    max_env_steps when no terminal event exists); stopped: terminal
    signal_stop/user, or interrupted tail; failed: terminal
    error/diverged, or a terminal traceback in the log tail;
    otherwise stopped when events exist, unknown when not.
    """
    run_dir = Path(run_dir)
    log = run_dir / "train.log"
    last_ev, terminal = _tail_events(run_dir)
    failure = _tail_failure(_read_log_tail_lines(log)) if log.exists() else None

    alive = _pid_alive(run_dir / "pid")
    if alive:
        return "running", (failure or {}).get("exception")

    if terminal is not None:
        st = str((terminal.get("context") or {}).get("status") or "")
        mapped = _TERMINAL_STATUS_MAP.get(st, "stopped")
        err = None
        if mapped == "failed":
            err = str((terminal.get("context") or {}).get("reason") or st)
        return mapped, err or (failure or {}).get("exception")

    env_step = None
    if last_ev is not None:
        env_step = (last_ev.get("clocks") or {}).get("env_step")
    max_steps = _max_env_steps(run_dir)
    if env_step is not None and max_steps and env_step >= max_steps:
        return "finished", None

    try:
        activity = log.stat().st_mtime if log.exists() else (
            run_dir / EVENTS_NAME).stat().st_mtime
    except OSError:
        activity = run_dir.stat().st_mtime
    # Freshness only implies "running" when the dir actually looks like
    # a live run (log or events exist) — an empty dir is unknown.
    has_artifacts = log.exists() or (run_dir / EVENTS_NAME).exists()
    if (alive is None and has_artifacts
            and (time.time() - activity) < 90.0):
        return "running", (failure or {}).get("exception")
    if failure and failure.get("crash"):
        return "failed", failure.get("exception")
    if failure and failure.get("interrupted"):
        return "stopped", None
    if last_ev is None:
        return "unknown", None
    return "stopped", None


# ---------------------------------------------------------------------------
# Flat round-row table — the dashboard payload
# ---------------------------------------------------------------------------

_META_ROW_KEYS = ("round", "env_step", "critic_tick", "param_patches",
                  "_ctx")


def run_events_table(run_dir: Path) -> List[Dict[str, Any]]:
    """One flat row per ``round`` event — the SAC analog of the PPO
    per-update stats table.

    Row fields:
    - ``round``, ``env_step``, ``critic_tick`` (row identity, not charted);
    - the round event's own metrics verbatim (``collection.*``,
      ``replay.*``, ``behavior.*``, round-level ``timing.*``);
    - ``eval.*`` metrics merged by ``collection_round`` (sparse);
    - ``tck.<metric>`` = per-round aggregate of the tick metric
      (mean, except TCK_SUM_KEYS which sum);
    - ``param_patches`` — the round event's context patch list as a
      ``{field: value}`` dict (metadata; Round Detail renders it);
    - ``_ctx`` — the remaining round-event context (utd_credit,
      replay_stats, termination_reasons…) for the detail panel.
    """
    run_dir = Path(run_dir)
    rows: "dict[int, Dict[str, Any]]" = {}
    order: List[int] = []
    tick_acc: "dict[int, Dict[str, List[float]]]" = {}
    eval_rows: "dict[int, Dict[str, float]]" = {}

    for ev in iter_events(run_dir):
        etype = ev.get("event_type")
        clocks = ev.get("clocks") or {}
        rnd = clocks.get("collection_round")
        if rnd is None:
            continue
        metrics = ev.get("metrics") or {}
        if etype == "round":
            row = dict(metrics)
            ctx = ev.get("context") or {}
            patches = ctx.get("param_patches")
            row["param_patches"] = {
                str(p["field"]): p.get("value")
                for p in patches
            } if isinstance(patches, list) else {}
            row["round"] = int(rnd)
            row["env_step"] = int(clocks.get("env_step", 0))
            row["critic_tick"] = int(clocks.get("critic_tick", 0))
            row["_ctx"] = {
                k: v for k, v in ctx.items() if k != "param_patches"
            }
            rows[int(rnd)] = row
            order.append(int(rnd))
        elif etype == "tick":
            acc = tick_acc.setdefault(int(rnd), {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    acc.setdefault(k, []).append(float(v))
        elif etype == "eval":
            dst = eval_rows.setdefault(int(rnd), {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    dst[k] = float(v)

    out: List[Dict[str, Any]] = []
    for rnd in order:
        row = rows[rnd]
        for k, vals in (tick_acc.get(rnd) or {}).items():
            tk = f"tck.{k}"
            row[tk] = (
                sum(vals) if k in TCK_SUM_KEYS
                else sum(vals) / len(vals)
            )
        for k, v in (eval_rows.get(rnd) or {}).items():
            row.setdefault(k, v)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Run info / dumps / videos / runs index / experiments
# ---------------------------------------------------------------------------

_RUN_NAME_TS = re.compile(r"_(\d{8})_(\d{6})$")


def _run_created_ts(name: str, run_dir: Path, cfg: Optional[Dict]) -> float:
    if cfg:
        saved = cfg.get("saved_at")
        if isinstance(saved, str):
            try:
                return time.mktime(
                    time.strptime(saved, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
    m = _RUN_NAME_TS.search(name)
    if m:
        try:
            return time.mktime(
                time.strptime(m.group(0).lstrip("_"), "%Y%m%d_%H%M%S"))
        except ValueError:
            pass
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return 0.0


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = Path(run_dir) / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _last_eval_metrics(run_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ev in iter_events(run_dir):
        if ev.get("event_type") == "eval":
            for k, v in (ev.get("metrics") or {}).items():
                if isinstance(v, (int, float)):
                    out[k] = float(v)
    return out


def dumps_list(run_dir: Path) -> List[Dict[str, Any]]:
    root = Path(run_dir) / "debug_dumps"
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        man = d / "manifest.json"
        if not d.is_dir() or not man.exists():
            continue
        info: Dict[str, Any] = {"name": d.name}
        try:
            info["mtime"] = man.stat().st_mtime
            m = json.loads(man.read_text(encoding="utf-8"))
            info["critic_tick"] = m.get("critic_tick")
            info["kind"] = m.get("kind")
            info["evidence_level"] = m.get("evidence_level")
            info["schema_version"] = m.get("schema_version")
            info["files"] = sorted((m.get("files") or {}).keys())
        except (json.JSONDecodeError, OSError):
            pass
        try:
            clocks = json.loads(
                (d / "clocks.json").read_text(encoding="utf-8"))
            info["collection_round"] = clocks.get("collection_round")
            info["env_step"] = clocks.get("env_step")
        except (json.JSONDecodeError, OSError):
            pass
        out.append(info)
    return out


def videos_list(run_dir: Path) -> List[Dict[str, Any]]:
    root = Path(run_dir) / "videos"
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for f in sorted(root.iterdir()):
        if f.suffix != ".mp4":
            continue
        env_step = None
        m = re.match(r"s(\d+)", f.stem)
        if m:
            env_step = int(m.group(1))
        out.append({"name": f.name, "env_step": env_step,
                    "mtime": f.stat().st_mtime,
                    "size": f.stat().st_size})
    out.sort(key=lambda v: (v.get("env_step") or 0), reverse=True)
    return out


def run_info(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = _load_config(run_dir)
    status, error = run_status(run_dir)
    exp = cfg.get("experiment") or {}
    snap_path = run_dir / "code_snapshot.json"
    snapshot: Dict[str, Any] = {}
    if snap_path.exists():
        try:
            snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snapshot = {}
    dump_req = run_dir / "dump_request.json"
    _, terminal = _tail_events(run_dir)
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "experiment_name": exp.get("name"),
        "algorithm": cfg.get("algorithm"),
        "config": cfg,
        "code_snapshot": snapshot,
        "status": status,
        "error": error,
        "knobs": exp.get("knobs") or {},
        "n_dumps": len(dumps_list(run_dir)),
        "dump_pending": dump_req.exists(),
        "terminal": (terminal or {}).get("context"),
        "created": _run_created_ts(run_dir.name, run_dir, cfg),
    }


def scan_run_summary(run_dir: Path) -> Dict[str, Any]:
    """Lightweight per-run summary for the runs index."""
    run_dir = Path(run_dir)
    cfg = _load_config(run_dir)
    status, error = run_status(run_dir)
    last_ev, terminal = _tail_events(run_dir)
    clocks = (last_ev or {}).get("clocks") or {}
    exp = cfg.get("experiment") or {}
    common = exp.get("common_params") or {}
    params = dict(common)
    params.update(exp.get("knobs") or {})
    last_eval = _last_eval_metrics(run_dir)
    log = run_dir / "train.log"
    try:
        activity = log.stat().st_mtime
    except OSError:
        activity = run_dir.stat().st_mtime
    ev_path = run_dir / EVENTS_NAME
    n_dumps = 0
    dd = run_dir / "debug_dumps"
    if dd.is_dir():
        n_dumps = sum(
            1 for c in dd.iterdir()
            if c.is_dir() and (c / "manifest.json").exists())
    vid = run_dir / "videos"
    n_videos = (
        sum(1 for c in vid.iterdir() if c.suffix == ".mp4")
        if vid.is_dir() else 0)
    return {
        "name": run_dir.name,
        "experiment": exp.get("name"),
        "algo": cfg.get("algorithm"),
        "status": status,
        "error": error,
        "env_step": clocks.get("env_step"),
        "critic_tick": clocks.get("critic_tick"),
        "collection_round": clocks.get("collection_round"),
        "max_env_steps": common.get("max_env_steps"),
        "eval_success": last_eval.get("eval.success"),
        "eval_survival_rate": last_eval.get("eval.survival_rate"),
        "eval_max_pot": last_eval.get("eval.max_pot"),
        "terminal_status": (
            (terminal.get("context") or {}).get("status")
            if terminal else None),
        "created": _run_created_ts(run_dir.name, run_dir, cfg),
        "activity": activity,
        "n_dumps": n_dumps,
        "n_videos": n_videos,
        "has_events": ev_path.exists(),
        "params": params,
    }


def list_runs(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    root = Path(root)
    try:
        children = sorted(root.iterdir())
    except OSError:
        return out
    for d in children:
        if not d.is_dir():
            continue
        if not ((d / "config.json").exists() or (d / EVENTS_NAME).exists()
                or (d / "train.log").exists()):
            continue
        try:
            out.append(scan_run_summary(d))
        except Exception as exc:  # one bad run must not sink the index
            out.append({"name": d.name, "status": "unknown",
                        "error": str(exc)})
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
    """Filter / sort / paginate run summaries (pure function)."""
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
        "env_step": lambda r: (
            r["env_step"] if r["env_step"] is not None else -1),
        "success": lambda r: (
            r["eval_success"] if r.get("eval_success") is not None
            else r["eval_survival_rate"]
            if r.get("eval_survival_rate") is not None else -1),
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
# Experiments registry scan (ast-based; experiments_sac package)
# ---------------------------------------------------------------------------


def _literal_scalar(node: ast.AST) -> Tuple[bool, Any]:
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None or isinstance(v, (bool, int, float, str)):
            return True, v
    if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd)):
        ok, v = _literal_scalar(node.operand)
        if ok and isinstance(v, (int, float)):
            return True, -v if isinstance(node.op, ast.USub) else v
    return False, None


def _parse_experiment_file(path: Path) -> Dict[str, Any]:
    """Class attr table {ClassName: {attr: scalar}} — public scalars."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return {}
    classes: Dict[str, Dict[str, Any]] = {}
    module_doc = ast.get_docstring(tree) or ""
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs: Dict[str, Any] = {}
        for st in node.body:
            target = None
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                t = st.targets[0]
                if isinstance(t, ast.Name):
                    target, val = t.id, st.value
                else:
                    continue
            elif isinstance(st, ast.AnnAssign) and isinstance(
                    st.target, ast.Name) and st.value is not None:
                target, val = st.target.id, st.value
            else:
                continue
            if target.startswith("_"):
                continue
            ok, v = _literal_scalar(val)
            if ok:
                attrs[target] = v
        classes[node.name] = {
            "attrs": attrs,
            "docstring": ast.get_docstring(node) or "",
            "bases": [
                b.id if isinstance(b, ast.Name)
                else getattr(b, "attr", "")
                for b in node.bases
            ],
        }
    return {"classes": classes, "module_doc": module_doc,
            "module": path.stem}


def scan_experiments(pkg_dir: Path) -> List[Dict[str, Any]]:
    """Every experiment class under ``pkg_dir`` (exp_*.py + base.py).

    Attrs merge over the resolvable MRO within the package (subclass
    wins) — same rule as the PPO experiments index.
    """
    pkg_dir = Path(pkg_dir)
    files = [pkg_dir / "base.py"] + sorted(pkg_dir.glob("exp_*.py"))
    tables: Dict[str, Dict[str, Any]] = {}
    for f in files:
        if not f.exists():
            continue
        parsed = _parse_experiment_file(f)
        for cname, info in parsed["classes"].items():
            tables[cname] = {**info, "file": f.name,
                             "module_doc": parsed["module_doc"]}

    def merged_attrs(cname: str, seen: frozenset = frozenset()) -> Dict:
        if cname in seen or cname not in tables:
            return {}
        info = tables[cname]
        base_attrs: Dict[str, Any] = {}
        for b in info["bases"]:
            for k, v in merged_attrs(b, seen | {cname}).items():
                base_attrs.setdefault(k, v)
        base_attrs.update(info["attrs"])
        return base_attrs

    out: List[Dict[str, Any]] = []
    for cname, info in tables.items():
        attrs = merged_attrs(cname)
        name = attrs.get("name")
        if not isinstance(name, str):
            continue
        tunables = {
            k: v for k, v in sorted(attrs.items())
            if k not in ("name", "EXPERIMENT_CLASS")
        }
        out.append({
            "name": name,
            "class": cname,
            "file": info["file"],
            "docstring": info["docstring"] or tables[cname].get(
                "module_doc", ""),
            "attrs": attrs,
            "tunables": tunables,
        })
    out.sort(key=lambda e: e["name"])
    return out


def experiments_index(
    pkg_dir: Path, runs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Experiments + per-experiment run stats."""
    by_exp: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_exp.setdefault(str(r.get("experiment") or ""), []).append(r)
    out = []
    for exp in scan_experiments(pkg_dir):
        rs = by_exp.get(exp["name"], [])
        running = sum(1 for r in rs if r.get("status") == "running")
        best = None
        for r in rs:
            for key in ("eval_success", "eval_survival_rate"):
                v = r.get(key)
                if isinstance(v, (int, float)) and (
                        best is None or v > best):
                    best = v
        out.append({
            **exp,
            "n_runs": len(rs), "n_running": running,
            "best_eval": best,
        })
    return out
