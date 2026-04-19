"""Web viewer for ``BaseFrameRecorder`` output.

Usage::

    python -m envs.framework.recorder_viewer <recording_dir> [--port 8765] [--no-browser]

The command:

1. Copies the bundled ``viewer.html`` into ``<recording_dir>/viewer.html`` (if
   missing or older than the bundled one).
2. Starts an ``http.server`` rooted at ``<recording_dir>``.
3. Opens ``http://localhost:<port>/viewer.html`` in the default browser.

The viewer fetches ``index.json`` for the episode list, then per-episode
``manifest.json`` for the step list, then per-step ``step_XXXXX.png`` /
``step_XXXXX.json`` as the user navigates.
"""
from __future__ import annotations

import argparse
import http.server
import shutil
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path


_BUNDLED_VIEWER = Path(__file__).with_name("_recorder_viewer.html")


def ensure_viewer_html(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = target_dir / "viewer.html"
    if not _BUNDLED_VIEWER.exists():
        raise FileNotFoundError(f"bundled viewer not found: {_BUNDLED_VIEWER}")
    if (
        not viewer_path.exists()
        or viewer_path.stat().st_mtime < _BUNDLED_VIEWER.stat().st_mtime
    ):
        shutil.copy2(_BUNDLED_VIEWER, viewer_path)
    return viewer_path


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - http.server signature
        return


def serve(directory: Path, port: int, open_browser: bool = True) -> None:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"recording dir not found: {directory}")
    ensure_viewer_html(directory)

    handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(directory), **kwargs)
    with socketserver.ThreadingTCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/viewer.html"
        print(f"[recorder_viewer] serving {directory} at {url}", flush=True)
        print("[recorder_viewer] press Ctrl+C to stop", flush=True)
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[recorder_viewer] shutting down", flush=True)
            httpd.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a BaseFrameRecorder recording for interactive viewing.")
    parser.add_argument("directory", type=Path, help="Directory produced by BaseFrameRecorder (contains index.json).")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765).")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    args = parser.parse_args(argv)
    try:
        serve(args.directory, args.port, open_browser=not args.no_browser)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        print(f"[recorder_viewer] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
