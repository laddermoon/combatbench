"""SAC debug viewer server — three-mode HTTP JSON + SPA host.

Read-only over run artifacts (the only writes are the
``dump_request.json`` sentinel — the protocol the training loop
already consumes — and, in later work packages, single-job-slot
compute tasks).

    python -m baseline.framework.sac.debugserver \
        <runs_root|run_dir|dump_dir> [--port 8766]

API surface (PPO-shaped, SAC semantics):

    GET /api/mode | /api/catalog | /api/job-status
    GET /api/runs?q=&page=&size=&sort=&order=
    GET /api/experiments | /api/experiment/<name>
    GET /run/<n>/api/run/{info,metrics,dumps,videos,log,replay,
                         slice-ctx,metrics-series}
    GET /run/<n>/train.log            (streamed file)
    GET /run/<n>/videos/<file>        (mp4)
    GET /api/run/{...}                (run mode — root IS the run)
    GET /api/dump/<d>/<endpoint>      (dump endpoints; d resolves under
                                       run root or, in dump mode, root)
    POST /run/<n>/api/run/dump-request {hypothesis}
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import analysis, runview
from .debugkit import recompute_dump
from .metric_catalog import catalog as _catalog

_EXPERIMENTS_DIR = (
    Path(__file__).resolve().parents[2] / "experiments_sac"
)


def _is_run_dir(path: Path) -> bool:
    path = Path(path)
    return (path / "metrics" / "events.jsonl").exists() or (
        path / "config.json").exists()


def _is_dump_dir(path: Path) -> bool:
    man = Path(path) / "manifest.json"
    if not man.exists():
        return False
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
        return str(m.get("schema_version", "")).startswith("sac_dump")
    except (json.JSONDecodeError, OSError):
        return False


def _root_mode(root: Path) -> str:
    if _is_dump_dir(root):
        return "dump"
    if _is_run_dir(root):
        return "run"
    return "runs"


def _resolve_run(root: Path, name: str) -> Path:
    if not name or "/" in name or ".." in name:
        raise KeyError(f"bad run name {name!r}")
    path = Path(root) / name
    if not path.is_dir():
        raise KeyError(f"run {name!r} not found under {root}")
    return path


def _resolve_dump(root: Path, query: Dict[str, list]) -> Path:
    if query.get("path"):
        return Path(query["path"][0])
    if _is_dump_dir(root) and not query.get("name"):
        return root
    run = _resolve_run(root, query["run"][0]) if query.get("run") else root
    name = query.get("name")
    tick = query.get("tick")
    if tick and not name:
        name = f"critic_tick_{int(tick[0]):08d}"
    elif name:
        name = name[0]
    if not name:
        raise KeyError("dump query requires name=|tick=|path=")
    path = Path(run) / "debug_dumps" / name
    if not path.is_dir():
        raise KeyError(f"dump not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Dispatch — one table of (path → handler), used by both run modes.
# ---------------------------------------------------------------------------

def _dump_endpoint(path: str, root: Path, q: Dict[str, list]) -> Any:
    def _opt(name: str, default=None):
        v = q.get(name)
        return v[0] if v else default

    if path == "inspect":
        return analysis.dump_inspect(_resolve_dump(root, q))
    if path == "pipeline":
        return analysis.pipeline_summary(_resolve_dump(root, q))
    if path == "manifest":
        dump = _resolve_dump(root, q)
        return json.loads(
            (dump / "manifest.json").read_text(encoding="utf-8"))
    if path == "spec":
        dump = _resolve_dump(root, q)
        spec_path = dump / "spec.json"
        if not spec_path.exists():
            raise KeyError("dump has no spec.json")
        return json.loads(spec_path.read_text(encoding="utf-8"))
    if path == "samples":
        return analysis.dump_samples(
            _resolve_dump(root, q),
            sort=_opt("sort", "td_abs"),
            channel=_opt("channel"),
            limit=int(_opt("limit", 50)),
        )
    if path == "trace":
        return analysis.dump_trace(
            _resolve_dump(root, q),
            sample_id=(
                int(_opt("sample_id")) if _opt("sample_id") else None),
            source_key=_opt("source_key"),
        )
    if path == "recompute":
        return recompute_dump(_resolve_dump(root, q))
    if path == "attribute":
        return analysis.batch_attribution(
            _resolve_dump(root, q),
            group_by=_opt("group_by", "collection_round"),
        )
    if path == "decomp":
        return analysis.objective_decomp(
            _resolve_dump(root, q),
            sample_id=(
                int(_opt("sample_id")) if _opt("sample_id") else None),
        )
    if path == "drift":
        return analysis.policy_drift(_resolve_dump(root, q))
    if path == "grad-attr":
        return analysis.grad_attribution(
            _resolve_dump(root, q),
            group_by=_opt("group_by", "collection_round"),
        )
    raise KeyError(f"unknown dump endpoint {path!r}")


def _run_endpoint(
    path: str, run_dir: Path, q: Dict[str, list],
) -> Any:
    def _opt(name: str, default=None):
        v = q.get(name)
        return v[0] if v else default

    if path == "info":
        return runview.run_info(run_dir)
    if path == "metrics":
        return runview.run_events_table(run_dir)
    if path == "metrics-series":
        return analysis.metric_series(
            run_dir,
            key=_opt("key", ""),
            event=_opt("event"),
            max_points=(
                int(_opt("max_points")) if _opt("max_points") else None),
        )
    if path == "dumps":
        return runview.dumps_list(run_dir)
    if path == "videos":
        return runview.videos_list(run_dir)
    if path == "log":
        return runview.read_log_window(
            run_dir / "train.log",
            int(_opt("lines", 300)),
            int(_opt("before")) if _opt("before") else None,
        )
    if path == "replay":
        return analysis.replay_report(run_dir)
    if path == "slice-ctx":
        target = (
            Path(_opt("replay")) if _opt("replay") else run_dir
        )
        return analysis.slice_context(
            target,
            source_key=_opt("source_key"),
            slice_id=_opt("slice_id"),
            sample_id=(
                int(_opt("sample_id")) if _opt("sample_id") else None),
        )
    if path.startswith("dump/"):
        sub = q
        sub.setdefault("name", [path.split("/", 2)[1]])
        return _dump_endpoint(path.split("/", 2)[2], run_dir, sub)
    raise KeyError(f"unknown run endpoint {path!r}")


def _experiment_by_name(name: str) -> Dict[str, Any]:
    for e in runview.scan_experiments(_EXPERIMENTS_DIR):
        if e["name"] == name:
            return e
    raise KeyError(f"experiment {name!r} not found")


def _dispatch(
    root: Path, mode: str, path: str, q: Dict[str, list],
) -> Any:
    def _opt(name: str, default=None):
        v = q.get(name)
        return v[0] if v else default

    if path == "/api/mode":
        return {"root_mode": mode, "root": str(root)}
    if path == "/api/catalog":
        return _catalog()
    if path == "/api/job-status":
        return {"job": None}  # single-job slot lands in R3-W4
    if path == "/api/runs":
        return runview.query_runs_index(
            runview.list_runs(root),
            q=_opt("q", ""),
            sort=_opt("sort", "created"),
            order=_opt("order", "desc"),
            page=int(_opt("page", 1)),
            size=int(_opt("size", 20)),
        )
    if path == "/api/experiments":
        return runview.experiments_index(
            _EXPERIMENTS_DIR, runview.list_runs(root))
    if path.startswith("/api/experiment/"):
        name = path.rsplit("/", 1)[-1]
        exp = _experiment_by_name(name)
        runs = [
            r for r in runview.list_runs(root)
            if r.get("experiment") == name
        ]
        checkpoints = []
        for r in runs:
            ck_dir = root / r["name"] / "checkpoints"
            if ck_dir.is_dir():
                cks = sorted(ck_dir.iterdir())
                if cks:
                    checkpoints.append({
                        "run": r["name"], "name": cks[-1].name,
                        "path": str(cks[-1].relative_to(root.parent))})
        return {**exp, "algo": "sac", "runs": runs,
                "checkpoints": checkpoints,
                "cwd": str(Path(__file__).resolve().parents[2])}
    # /run/<name>/api/... — run-scoped endpoints under a runs root
    if path.startswith("/run/"):
        rest = path[len("/run/"):]
        run_name, _, tail = rest.partition("/")
        run_dir = _resolve_run(root, run_name)
        if tail == "train.log":
            raise _StreamFile(run_dir / "train.log", "text/plain")
        if tail.startswith("videos/"):
            raise _StreamFile(
                run_dir / "videos" / tail.split("/", 1)[1], "video/mp4")
        if tail.startswith("api/run/"):
            return _run_endpoint(tail[len("api/run/"):], run_dir, q)
        if tail.startswith("api/dump/"):
            sub = tail[len("api/dump/"):]
            name_part, _, ep = sub.partition("/")
            if not ep:
                # /api/dump/<endpoint>?name=
                return _dump_endpoint(name_part, run_dir, q)
            qq = dict(q)
            qq.setdefault("name", [name_part])
            return _dump_endpoint(ep, run_dir, qq)
        raise KeyError(f"unknown run route {tail!r}")
    if path == "/train.log" and mode == "run":
        raise _StreamFile(root / "train.log", "text/plain")
    if path.startswith("/videos/") and mode == "run":
        raise _StreamFile(
            root / "videos" / path.split("/", 2)[2], "video/mp4")
    if path.startswith("/api/run/"):
        if mode != "run":
            raise KeyError("/api/run/* requires run mode root")
        return _run_endpoint(path[len("/api/run/"):], root, q)
    if path.startswith("/api/dump/"):
        sub = path[len("/api/dump/"):]
        name_part, _, ep = sub.partition("/")
        if not ep:
            return _dump_endpoint(name_part, root, q)
        qq = dict(q)
        qq.setdefault("name", [name_part])
        return _dump_endpoint(ep, root, qq)
    # legacy flat query-string form kept for debugkit query parity
    if path == "/api/dump":
        return analysis.dump_inspect(_resolve_dump(root, q))
    raise KeyError(f"unknown endpoint {path!r}")


class _StreamFile(Exception):
    def __init__(self, path: Path, ctype: str) -> None:
        super().__init__(str(path))
        self.path = Path(path)
        self.ctype = ctype


def _post_dump_request(run_dir: Path, body: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise KeyError(f"bad json body: {exc}")
    hypothesis = str(payload.get("hypothesis") or "")
    req_path = run_dir / "dump_request.json"
    if req_path.exists():
        raise KeyError("dump request already pending")
    req = {
        "hypothesis": hypothesis,
        "source": "viewer",
        "requested_at_unix": time.time(),
    }
    tmp = req_path.with_name(req_path.name + ".tmp")
    tmp.write_text(
        json.dumps(req, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(tmp, req_path)
    return {"ok": True, "request": req}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

_VIEWER_HTML = Path(__file__).with_name("sac_viewer.html")


def _json_safe(obj: Any) -> Any:
    import math

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def make_server(
    root: Path, port: int, host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    root = Path(root).resolve()
    mode = _root_mode(root)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: Any) -> None:
            body = json.dumps(
                _json_safe(payload), indent=2, sort_keys=True,
                default=str,
            ).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self) -> None:
            if not _VIEWER_HTML.exists():
                return self._send(404, {"error": "viewer not built"})
            body = _VIEWER_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, ctype: str) -> None:
            try:
                resolved = path.resolve()
                if root not in resolved.parents or not resolved.is_file():
                    raise KeyError(f"not under served root: {path}")
                size = resolved.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with resolved.open("rb") as f:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except KeyError as exc:
                self._send(404, {"error": str(exc)})

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            is_api = path.startswith("/api") or "/api/" in path
            is_file = path.endswith("/train.log") or "/videos/" in path
            if not is_api and not is_file:
                # SPA routes — every non-API path serves the viewer page
                if "." not in Path(path).name or path == "/index.html":
                    return self._send_html()
            try:
                q = parse_qs(parsed.query)
                out = _dispatch(root, mode, path, q)
                self._send(200, out)
            except _StreamFile as sf:
                self._send_file(sf.path, sf.ctype)
            except KeyError as exc:
                self._send(404, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {
                    "error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                if path.endswith("/api/run/dump-request"):
                    if mode == "run":
                        run_dir = root
                    else:
                        run_name = path.split("/")[2]
                        run_dir = _resolve_run(root, run_name)
                    return self._send(
                        200, _post_dump_request(run_dir, body))
                return self._send(404, {"error": f"unknown POST {path}"})
            except KeyError as exc:
                self._send(409, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {
                    "error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer((host, int(port)), Handler)
    return server


def serve(
    root: Path, port: int = 8766, host: str = "127.0.0.1",
) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    server = make_server(root, port, host)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SAC debug viewer server")
    parser.add_argument("root", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(args.root, args.port, args.host)
    url = f"http://{args.host}:{args.port}/"
    print(f"[sac-debug] serving {args.root} ({_root_mode(args.root)}) on {url}")
    if not args.no_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
