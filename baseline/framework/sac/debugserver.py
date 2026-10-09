"""Read-only HTTP JSON server for SAC debug artifacts.

Every endpoint delegates to ``baseline.framework.sac.analysis`` — the single
computation layer (D51).  The server never mutates run artifacts and does
not attach to a live training process; it reads what is on disk.

    python -m baseline.framework.sac.debugserver <runs_root|run_dir> \
        [--port 8766]
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import analysis
from .debugkit import recompute_dump


def _resolve_run(root: Path, name: str) -> Path:
    path = Path(root) / name
    if not path.is_dir():
        raise KeyError(f"run {name!r} not found under {root}")
    return path


def _resolve_dump(root: Path, query: Dict[str, list]) -> Path:
    if query.get("path"):
        return Path(query["path"][0])
    run = _resolve_run(root, query["run"][0])
    tick = query.get("tick")
    name = query.get("name")
    if name:
        name = name[0]
    if tick:
        name = f"critic_tick_{int(tick[0]):08d}"
    if not name:
        raise KeyError("dump query requires path= or run= + (tick=|name=)")
    path = run / "debug_dumps" / name
    if not path.is_dir():
        raise KeyError(f"dump not found: {path}")
    return path


def _is_run_dir(path: Path) -> bool:
    return (path / "metrics" / "events.jsonl").exists() or (
        path / "config.json"
    ).exists()


def _dispatch(root: Path, path: str, q: Dict[str, list]) -> Any:
    def _opt(name: str, default=None):
        v = q.get(name)
        return v[0] if v else default

    if path == "/api/runs":
        if _is_run_dir(root):
            return [analysis.run_summary(root)]
        return analysis.runs_index(root)
    if path == "/api/run":
        return analysis.run_summary(_resolve_run(root, _opt("name")))
    if path == "/api/run/metrics":
        return analysis.metric_series(
            _resolve_run(root, _opt("name")),
            key=_opt("key", ""),
            event=_opt("event"),
        )
    if path == "/api/run/dumps":
        return analysis.run_summary(
            _resolve_run(root, _opt("name"))
        )["dumps"]
    if path == "/api/run/replay":
        return analysis.replay_report(_resolve_run(root, _opt("name")))
    if path == "/api/dump/inspect":
        return analysis.dump_inspect(_resolve_dump(root, q))
    if path == "/api/dump/samples":
        return analysis.dump_samples(
            _resolve_dump(root, q),
            sort=_opt("sort", "td_abs"),
            channel=_opt("channel"),
            limit=int(_opt("limit", 50)),
        )
    if path == "/api/dump/trace":
        return analysis.dump_trace(
            _resolve_dump(root, q),
            sample_id=(
                int(_opt("sample_id")) if _opt("sample_id") else None
            ),
            source_key=_opt("source_key"),
        )
    if path == "/api/dump/recompute":
        return recompute_dump(_resolve_dump(root, q))
    if path == "/api/catalog":
        return analysis.metric_catalog(prefix=_opt("prefix"))
    raise KeyError(f"unknown endpoint {path!r}")


def make_server(
    root: Path, port: int, host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    root = Path(root).resolve()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: Any) -> None:
            body = json.dumps(
                payload, indent=2, sort_keys=True, default=str,
            ).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, path: Path) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/viewer", "/index.html"):
                page = Path(__file__).with_name("sac_viewer.html")
                if page.exists():
                    return self._send_html(page)
            try:
                out = _dispatch(
                    root, parsed.path, parse_qs(parsed.query),
                )
                self._send(200, out)
            except KeyError as exc:
                self._send(404, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args: Any) -> None:  # quiet
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

    parser = argparse.ArgumentParser(description="SAC debug JSON server")
    parser.add_argument("root", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = make_server(args.root, args.port, args.host)
    print(f"[sac-debug] serving {args.root} on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
