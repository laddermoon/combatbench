"""Standalone HTML visualization for a point5 reference trajectory.

Produces a single self-contained .html file: frames are embedded as JSON and
drawn on three orthographic canvases (top XY / front XZ / side YZ) with plain
JavaScript. No external dependencies, works offline.

Frame sampling is distance-adaptive (``adaptive_sample``): the tool guarantees
that between consecutive frames no point moves more than ``max_step`` metres.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .contracts import POINT_NAMES
from .trajectory import (
    Point5Reference,
    ReferenceFrames,
    adaptive_sample,
    frames_payload,
)

_POINT_COLORS = {
    "head": "#e0483e",
    "hand_right": "#3ba53b",
    "hand_left": "#3b7fd4",
    "foot_right": "#e0a02a",
    "foot_left": "#b56ae0",
}
# star skeleton: which pairs get a connecting line
_BONES = [("head", "hand_right"), ("head", "hand_left"),
          ("head", "foot_right"), ("head", "foot_left"),
          ("foot_right", "foot_left"), ("hand_right", "hand_left")]


def export_html(reference: Point5Reference,
                out_path,
                *,
                max_step: float = 0.01,
                measured: Optional[np.ndarray] = None,
                frames: Optional[ReferenceFrames] = None,
                title: Optional[str] = None) -> Path:
    """Write a self-contained HTML player for ``reference``.

    ``measured`` (optional (5,3) world positions) is drawn as a grey ghost
    for comparison with the reference.
    """
    if frames is None:
        frames = adaptive_sample(reference, max_step=max_step)
    payload = frames_payload(reference, frames)
    payload["point_colors"] = _POINT_COLORS
    payload["bones"] = _BONES
    payload["max_step"] = float(max_step)
    if measured is not None:
        payload["measured"] = np.asarray(
            measured, dtype=np.float64).tolist()
    data = json.dumps(payload, separators=(",", ":"))
    name = title or f"point5: {reference.task.name}"
    out = Path(out_path)
    out.write_text(_TEMPLATE.replace("__TITLE__", html.escape(name))
                    .replace("__DATA__", data), encoding="utf-8")
    return out


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
// shared scale across all frames and views
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
