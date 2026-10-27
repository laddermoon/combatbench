"""Offline visualization for point5 reference trajectories.

Two outputs, both simulator-unaware:

- ``export_mp4``: PIL-rendered 3-view frames piped to ffmpeg.
- ``export_html``: single self-contained HTML canvas player.

Frame sampling is distance-adaptive (``adaptive_sample``): between consecutive
frames no point moves more than ``max_step`` metres.

CLI::

    PYTHONPATH=. python3 -m mocap.point5.viz --task t.json --out t.mp4
    PYTHONPATH=. python3 -m mocap.point5.viz --task t.json --ctx start.json \
        --out t.mp4 --fps 30
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
    POINT_NAMES,
    Point5State,
    Point5Task,
    StaticStartResolver,
    TaskValidationError,
)
from .trajectory import (
    Point5Reference,
    ReferenceFrames,
    adaptive_sample,
    build_reference,
    frames_payload,
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
_PANEL = 300          # drawing area per view (square)
_MARGIN_X = 15
_TOP = 55


# ---------------------------------------------------------------------------
# shared bounds / projection
# ---------------------------------------------------------------------------

def _bounds(*pos_arrays: np.ndarray) -> Tuple[np.ndarray, float]:
    """Shared (center, half-span) across frames so views don't wobble."""
    allpos = np.concatenate(
        [p.reshape(-1, 3) for p in pos_arrays if p is not None])
    lo, hi = allpos.min(axis=0), allpos.max(axis=0)
    lo[2], hi[2] = min(lo[2], 0.0), max(hi[2], 1.6)
    ctr = (lo + hi) / 2.0
    span = float(max(hi - lo)) * 1.15
    return ctr, max(span, 0.3)


def _xy(ctr: np.ndarray, scale: float, ax: Tuple[int, int],
        p3: np.ndarray, ox: int, oy: int) -> Tuple[float, float]:
    s = _PANEL / scale
    return (ox + _PANEL / 2 + (p3[ax[0]] - ctr[ax[0]]) * s,
            oy + _PANEL / 2 - (p3[ax[1]] - ctr[ax[1]]) * s)


# ---------------------------------------------------------------------------
# PIL rendering
# ---------------------------------------------------------------------------

def render_frame(pos: np.ndarray, info: str,
                 ctr: np.ndarray, scale: float,
                 measured: Optional[np.ndarray] = None) -> Image.Image:
    """One 960x480 frame: three orthographic views of the five points."""
    img = Image.new("RGB", (_W, _H), (17, 17, 17))
    dr = ImageDraw.Draw(img)
    dr.text((_MARGIN_X, 8), info, fill=(200, 200, 200))
    dr.text((_MARGIN_X, 24),
            " ".join(f"{n}" for n in POINT_NAMES), fill=(120, 120, 120))
    for i, n in enumerate(POINT_NAMES):
        c = _POINT_COLORS[n]
        dr.text((150 + i * 95, 24), n, fill=tuple(
            int(c[j:j + 2], 16) for j in (1, 3, 5)))
    for vi, (label, ax) in enumerate(_VIEWS):
        ox = _MARGIN_X + vi * (_PANEL + 20)
        oy = _TOP
        dr.rectangle([ox, oy, ox + _PANEL, oy + _PANEL],
                     outline=(60, 60, 60))
        dr.text((ox + 6, oy + 4), label, fill=(136, 136, 136))
        if ax[1] == 2:  # ground line on elevation views
            xa, ya = _xy(ctr, scale, ax, [-1e3, 0, 0], ox, oy)
            xb, yb = _xy(ctr, scale, ax, [1e3, 0, 0], ox, oy)
            dr.line([xa, ya, xb, yb], fill=(42, 74, 42), width=1)
        if measured is not None:
            for a, b in _BONES:
                i, j = POINT_NAMES.index(a), POINT_NAMES.index(b)
                dr.line([*_xy(ctr, scale, ax, measured[i], ox, oy),
                         *_xy(ctr, scale, ax, measured[j], ox, oy)],
                        fill=(70, 70, 70), width=1)
            for i in range(len(POINT_NAMES)):
                x, y = _xy(ctr, scale, ax, measured[i], ox, oy)
                dr.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5],
                           fill=(90, 90, 90))
        for a, b in _BONES:
            i, j = POINT_NAMES.index(a), POINT_NAMES.index(b)
            dr.line([*_xy(ctr, scale, ax, pos[i], ox, oy),
                     *_xy(ctr, scale, ax, pos[j], ox, oy)],
                    fill=(100, 100, 100), width=2)
        for i, n in enumerate(POINT_NAMES):
            x, y = _xy(ctr, scale, ax, pos[i], ox, oy)
            c = _POINT_COLORS[n]
            dr.ellipse([x - 5, y - 5, x + 5, y + 5],
                       fill=tuple(int(c[k:k + 2], 16) for k in (1, 3, 5)))
    return img


# ---------------------------------------------------------------------------
# MP4 export
# ---------------------------------------------------------------------------

def export_mp4(reference: Point5Reference, out_path,
               *, max_step: float = 0.01, fps: int = 30,
               measured: Optional[np.ndarray] = None,
               ffmpeg: str = "ffmpeg") -> Path:
    """Render the reference to an MP4 via ffmpeg.

    Frames come from ``adaptive_sample(max_step)`` — distance-uniform, so
    playback speed follows motion density rather than wall time.
    """
    if fps <= 0:
        raise TaskValidationError("fps must be positive")
    frames = adaptive_sample(reference, max_step=max_step)
    arrays = [frames.pos] + ([measured] if measured is not None else [])
    ctr, span = _bounds(*arrays)
    out = Path(out_path)
    cmd = [ffmpeg, "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{_W}x{_H}", "-r", str(fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    try:
        n = len(frames.states)
        for k, st in enumerate(frames.states):
            m = st.metadata
            info = (f"{reference.task.name}  {k + 1}/{n}  "
                    f"t={m['t']:.3f}s seg{m['seg']} s={m['s']:.2f} "
                    f"[{m['keyframe_label']}]")
            img = render_frame(st.pos, info, ctr, span, measured)
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return out


# ---------------------------------------------------------------------------
# HTML export (kept: zero-dependency alternative)
# ---------------------------------------------------------------------------

def export_html(reference: Point5Reference, out_path,
                *, max_step: float = 0.01,
                measured: Optional[np.ndarray] = None,
                frames: Optional[ReferenceFrames] = None,
                title: Optional[str] = None) -> Path:
    if frames is None:
        frames = adaptive_sample(reference, max_step=max_step)
    payload = frames_payload(reference, frames)
    payload["point_colors"] = _POINT_COLORS
    payload["bones"] = _BONES
    payload["max_step"] = float(max_step)
    if measured is not None:
        payload["measured"] = np.asarray(measured,
                                         dtype=np.float64).tolist()
    data = json.dumps(payload, separators=(",", ":"))
    name = title or f"point5: {reference.task.name}"
    out = Path(out_path)
    out.write_text(_TEMPLATE.replace("__TITLE__", html.escape(name))
                    .replace("__DATA__", data), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_context(path: Optional[str]) -> StaticStartResolver:
    """Start context for anchor resolution.

    JSON: {"initial": [[x,y,z] x5], "root_pos": [x,y,z],
           "root_rot": [[..],[..],[..]]}
    Without ``--ctx``: zero initial state + identity root at origin
    (sufficient for tasks that only use WorldPoint anchors).
    """
    if path is None:
        return StaticStartResolver(
            initial=Point5State(pos=np.zeros((5, 3))),
            root_pos=np.zeros(3), root_rot=np.eye(3))
    d = json.loads(Path(path).read_text())
    return StaticStartResolver(
        initial=Point5State(pos=np.asarray(d["initial"], dtype=np.float64)),
        root_pos=np.asarray(d.get("root_pos", [0, 0, 0]), dtype=np.float64),
        root_rot=np.asarray(d.get("root_rot", np.eye(3).tolist()),
                            dtype=np.float64))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="point5-viz")
    ap.add_argument("--task", required=True, help="point5.task.v1 JSON")
    ap.add_argument("--ctx", help="start context JSON for non-world anchors")
    ap.add_argument("--out", required=True, help="output .mp4 (or .html)")
    ap.add_argument("--max-step", type=float, default=0.01)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(argv)

    task = Point5Task.from_dict(json.loads(Path(args.task).read_text()))
    resolver = _load_context(args.ctx)
    ref = build_reference(task, resolver)
    measured = (resolver.initial.pos if args.ctx else None)
    out = Path(args.out)
    if out.suffix == ".html":
        path = export_html(ref, out, max_step=args.max_step,
                           measured=measured)
    else:
        path = export_mp4(ref, out, max_step=args.max_step, fps=args.fps,
                          measured=measured)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{font-family:monospace;background:#111;color:#ddd;margin:16px}
canvas{background:#1a1a1a;border:1px solid #333;margin-right:8px}
#bar{margin:10px 0}button{background:#333;color:#ddd;border:1px solid #555;
padding:4px 12px;cursor:pointer}input[type=range]{width:420px}
.legend span{margin-right:14px}
</style></head><body>
<h3>__TITLE__</h3>
<div class="legend" id="legend"></div>
<div>
<canvas id="top" width="360" height="360"></canvas>
<canvas id="front" width="360" height="360"></canvas>
<canvas id="side" width="360" height="360"></canvas>
</div>
<div id="bar">
<button id="play">&#9654;</button>
<input id="scrub" type="range" min="0" max="0" value="0" step="1">
<span id="info"></span>
</div>
<div id="meta"></div>
<script>
const D = __DATA__;
const N = D.point_names, C = D.point_colors, F = D.frames;
const VIEWS = [
  {id:"top",   axes:[0,1], label:"top  x-y"},
  {id:"front", axes:[0,2], label:"front x-z"},
  {id:"side",  axes:[1,2], label:"side  y-z"},
];
document.getElementById("legend").innerHTML =
  N.map(n=>`<span style="color:${C[n]}">&#9679; ${n}</span>`).join("");
document.getElementById("meta").textContent =
  `frames=${F.length} duration=${D.duration.toFixed(3)}s ` +
  `max_step=${D.max_step}m  (左: top x-y, 中: front x-z, 右: side y-z)`;
let lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
for (const f of F) for (const p of f.pos)
  for (let a=0;a<3;a++){lo[a]=Math.min(lo[a],p[a]);hi[a]=Math.max(hi[a],p[a]);}
if (D.measured) for (const p of D.measured)
  for (let a=0;a<3;a++){lo[a]=Math.min(lo[a],p[a]);hi[a]=Math.max(hi[a],p[a]);}
lo[2]=Math.min(lo[2],0);hi[2]=Math.max(hi[2],1.6);
const ctr=lo.map((v,i)=>(v+hi[i])/2);
const span=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],0.3)*1.15;
const ctx={};VIEWS.forEach(v=>ctx[v.id]=document.getElementById(v.id).getContext("2d"));
let k=0,playing=false,timer=null;
const scrub=document.getElementById("scrub");scrub.max=F.length-1;
function px(v,u,w){const S=340/span;return [
  180+(u-ctr[v.axes[0]])*S, 180-(w-ctr[v.axes[1]])*S];}
function draw(){
  const f=F[k];
  for (const v of VIEWS){const g=ctx[v.id];
    g.clearRect(0,0,360,360);
    g.fillStyle="#888";g.font="11px monospace";g.fillText(v.label,6,14);
    if(v.axes[1]===2){g.strokeStyle="#2a4a2a";g.beginPath();
      const[xa,ya]=px(v,-10,0),[xb,yb]=px(v,10,0);
      g.moveTo(xa,ya);g.lineTo(xb,yb);g.stroke();}
    if(D.measured){g.strokeStyle="#555";g.lineWidth=1;
      for(const b of D.bones){const i=N.indexOf(b[0]),j=N.indexOf(b[1]);
        const[x1,y1]=px(v,D.measured[i][v.axes[0]],D.measured[i][v.axes[1]]);
        const[x2,y2]=px(v,D.measured[j][v.axes[0]],D.measured[j][v.axes[1]]);
        g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();}
      for(let i=0;i<N.length;i++){const[x,y]=px(v,D.measured[i][v.axes[0]],
        D.measured[i][v.axes[1]]);g.fillStyle="#555";g.beginPath();
        g.arc(x,y,3,0,7);g.fill();}}
    g.strokeStyle="#666";g.lineWidth=1.5;
    for(const b of D.bones){const i=N.indexOf(b[0]),j=N.indexOf(b[1]);
      const[x1,y1]=px(v,f.pos[i][v.axes[0]],f.pos[i][v.axes[1]]);
      const[x2,y2]=px(v,f.pos[j][v.axes[0]],f.pos[j][v.axes[1]]);
      g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();}
    for(let i=0;i<N.length;i++){const[x,y]=px(v,f.pos[i][v.axes[0]],
      f.pos[i][v.axes[1]]);g.fillStyle=C[N[i]];g.beginPath();
      g.arc(x,y,5,0,7);g.fill();}
  }
  scrub.value=k;
  document.getElementById("info").textContent=
    `${k+1}/${F.length} t=${f.t.toFixed(3)}s seg${f.seg} s=${f.s.toFixed(2)} [${f.label}]`;
}
scrub.oninput=e=>{k=+e.target.value;draw();};
document.getElementById("play").onclick=()=>{
  playing=!playing;
  document.getElementById("play").innerHTML=playing?"&#10074;&#10074;":"&#9654;";
  if(playing)timer=setInterval(()=>{k=(k+1)%F.length;draw();},50);
  else clearInterval(timer);};
draw();
</script></body></html>
"""
