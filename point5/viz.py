"""Offline geometric visualization for point5 clips (v2).

Two outputs, both simulator-unaware:

- ``export_mp4``: PIL-rendered 3-view frames piped to ffmpeg.
- ``export_html``: single self-contained HTML canvas player.

Input is a list of world-frame position samples (``(N,5,3)``), e.g.
from ``trajectory.sample_clip``.

CLI::

    PYTHONPATH=. python3 -m point5.viz --clip c.json --out c.mp4
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .contracts import (
    CLIP_FORMAT,
    POINT_NAMES,
    P5Clip,
    TaskValidationError,
)

_POINT_COLORS = {
    "head": "#e0483e",
    "hand_right": "#3ba53b",
    "hand_left": "#3b7fd4",
    "foot_right": "#e0a02a",
    "foot_left": "#b56ae0",
}
_BONES = [("head", "hand_right"), ("head", "hand_left"),
          ("head", "foot_right"), ("head", "foot_left"),
          ("foot_right", "foot_left"), ("hand_right", "hand_left")]

_VIEWS = (("top  x-y", (0, 1)),
          ("front x-z", (0, 2)),
          ("side  y-z", (1, 2)))

_W, _H = 960, 480


def _view_img(pos: np.ndarray, axes: Tuple[int, int],
              bounds: np.ndarray, title: str,
              size: Tuple[int, int]) -> Image.Image:
    (x0, y0), (x1, y1) = bounds
    w, h = size
    img = Image.new("RGB", (w, h), (17, 17, 20))
    dr = ImageDraw.Draw(img)
    pad = 20
    sx = (w - 2 * pad) / max(x1 - x0, 1e-6)
    sy = (h - 2 * pad) / max(y1 - y0, 1e-6)
    sc = min(sx, sy)

    def px(p):
        return (pad + (p[axes[0]] - x0) * sc,
                h - pad - (p[axes[1]] - y0) * sc)

    for a, b in _BONES:
        pa, pb = px(pos[_i(a)]), px(pos[_i(b)])
        dr.line([pa, pb], fill=(90, 90, 100), width=2)
    for i, n in enumerate(POINT_NAMES):
        x, y = px(pos[i])
        c = _POINT_COLORS[n]
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=c, width=2)
        dr.text((x + 8, y - 5), n.split("_")[0][0] + ("r" if "right" in n else "l" if "left" in n else ""), fill=c)
    dr.text((4, 4), title, fill=(150, 150, 160))
    return img


def _i(n: str) -> int:
    return POINT_NAMES.index(n)


def _bounds(positions: np.ndarray) -> np.ndarray:
    lo = positions.reshape(-1, 3).min(axis=0) - 0.15
    hi = positions.reshape(-1, 3).max(axis=0) + 0.15
    return np.stack([lo, hi])[:2, :2]


def _frames_img(world_pos: np.ndarray, bounds: np.ndarray,
                fidx: int) -> Image.Image:
    img = Image.new("RGB", (_W, _H), (17, 17, 20))
    cw, ch = _W // 2, _H // 2
    cells = [
        _view_img(world_pos, _VIEWS[0][1], bounds, _VIEWS[0][0], (cw, ch)),
        _view_img(world_pos, _VIEWS[1][1], bounds, _VIEWS[1][0], (cw, ch)),
        _view_img(world_pos, _VIEWS[2][1], bounds, _VIEWS[2][0], (_W, ch)),
    ]
    img.paste(cells[0], (0, 0))
    img.paste(cells[1], (cw, 0))
    img.paste(cells[2], (0, ch))
    d = ImageDraw.Draw(img)
    d.text((6, _H - 16), f"#{fidx}", fill=(255, 255, 0))
    return img


def export_mp4(world_samples: np.ndarray, out_path: str,
               fps: int = 30) -> None:
    """world_samples: (N,5,3) world positions."""
    pos = np.asarray(world_samples, dtype=np.float64)
    bounds = _bounds(pos)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{_W}x{_H}", "-r", str(fps), "-i", "-",
         "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        stdin=subprocess.PIPE)
    try:
        for i, p in enumerate(pos):
            ff.stdin.write(_frames_img(p, bounds, i).tobytes())
    finally:
        ff.stdin.close()
        ff.wait()


def export_html(world_samples: np.ndarray, out_path: str,
                fps: int = 30) -> None:
    """Self-contained HTML canvas player of the 3-view animation."""
    pos = np.asarray(world_samples, dtype=np.float64)
    bounds = _bounds(pos)
    imgs = [_frames_img(p, bounds, i) for i, p in enumerate(pos)]
    import base64, io
    b64 = []
    for im in imgs:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64.append(base64.b64encode(buf.getvalue()).decode())
    page = (f"<!doctype html><meta charset=utf-8><title>p5 clip</title>"
            f"<img id=v style='image-rendering:pixelated'>"
            f"<script>const F={json.dumps(b64)},fps={fps};let i=0;"
            f"const v=document.getElementById('v');"
            f"setInterval(()=>{{v.src='data:image/png;base64,'+F[i];"
            f"i=(i+1)%F.length}},1000/fps);v.src='data:image/png;base64,'+F[0]</script>")
    Path(out_path).write_text(page)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-step", type=float, default=0.01)
    ap.add_argument("--html", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from point5.trajectory import sample_clip  # noqa: E402

    d = json.loads(Path(a.clip).read_text())
    clip = P5Clip.from_dict(d)
    world = np.stack([s.world_pos() for s in sample_clip(clip,
                                                       max_step=a.max_step)])
    if a.html:
        export_html(world, a.out, a.fps)
    else:
        export_mp4(world, a.out, a.fps)
    print(f"wrote {a.out} ({len(world)} frames)")


if __name__ == "__main__":
    main()
