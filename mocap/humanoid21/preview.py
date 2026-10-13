"""Render source-skeleton stick-figure previews of AMC clips (headless).

For human eyeball review of clip content (M0 W3 / W6 minimal-set selection).
Draws bone segments (parent joint -> child joint), hands highlighted,
feet contact-ish tinted, ground plane at y=0.

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.preview --clips 13_17 13_18 \
        --outdir /data1/dev/combatbench_mocap/preview [--every 4] [--fps 30]
    PYTHONPATH=. python3 -m mocap.humanoid21.preview --all --outdir ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baseline.humanoid21.mocap.asf_parser import parse_asf
from baseline.humanoid21.mocap.amc_parser import parse_amc
from baseline.humanoid21.mocap.fk import compute_fk_batch

RAW = PROJECT_ROOT / "baseline/humanoid21/mocap/data/raw"
FK_TO_M = 0.0254 / 0.45  # see content_scan.py for derivation
FPS_SRC = 120.0


def bone_edges(skel):
    """(parent, child) joint pairs for stick-figure drawing."""
    edges = []
    for name, bone in skel.bones.items():
        for ch in bone.children:
            edges.append((name, ch))
    return edges


def render_clip(clip: str, outdir: Path, every: int = 4, fps: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    subj = clip.split("_")[0]
    skel = parse_asf(str(RAW / f"{subj}.asf"))
    frames = parse_amc(str(RAW / f"{clip}.amc"))
    transforms = compute_fk_batch(skel, frames)
    edges = bone_edges(skel)

    T = len(transforms)
    idxs = list(range(0, T, every))

    # world positions (m), CMU y-up
    P = {n: np.array([tr[n][1] for tr in transforms]) * FK_TO_M
         for n in skel.bones if n in transforms[0]}

    # fixed axis limits from global extents
    allp = np.concatenate([v[idxs] for v in P.values()], axis=0)
    cx, cz = allp[:, 0].mean(), allp[:, 2].mean()
    span = max(1.2, float(allp[:, 1].max()) * 0.65)

    desc = ""
    try:
        d = json.loads((Path(__file__).parent / "cmu_motion_descriptions.json").read_text())
        desc = d["subjects"][subj]["trials"].get(str(int(clip.split("_")[1])), "")
    except Exception:
        pass

    out = outdir / f"{clip}.mp4"
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.2), dpi=90)
    writer = imageio.get_writer(out, fps=fps)
    markers = (("lhand", "red"), ("rhand", "darkred"),
               ("lfoot", "green"), ("rfoot", "darkgreen"),
               ("head", "black"))
    # (view horizontal axis index, label): front=x, side=z, top=(x,z)
    views = [(0, "front (x)"), (2, "side (z)")]
    try:
        for t in idxs:
            for vi, (hax, label) in enumerate(views):
                ax = axes[vi]
                ax.clear()
                ax.set_xlim(-span, span)
                ax.set_ylim(-0.05, 1.9)
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.axhline(0, color="gray", lw=0.6)
                for a, b in edges:
                    if a in P and b in P:
                        pa, pb = P[a][t], P[b][t]
                        ax.plot([pa[hax] - (cx if hax == 0 else cz),
                                 pb[hax] - (cx if hax == 0 else cz)],
                                [pa[1], pb[1]], color="steelblue", lw=1.6)
                for h, c in markers:
                    if h in P:
                        ax.scatter(P[h][t, hax] - (cx if hax == 0 else cz),
                                   P[h][t, 1], color=c, s=24, zorder=5)
                ax.set_title(label, fontsize=8)
            ax = axes[2]
            ax.clear()
            ax.set_xlim(-span, span); ax.set_ylim(-span, span)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
            for a, b in edges:
                if a in P and b in P:
                    pa, pb = P[a][t], P[b][t]
                    ax.plot([pa[0] - cx, pb[0] - cx],
                            [pa[2] - cz, pb[2] - cz], color="steelblue", lw=1.2)
            for h, c in markers:
                if h in P:
                    ax.scatter(P[h][t, 0] - cx, P[h][t, 2] - cz,
                               color=c, s=20, zorder=5)
            ax.set_title("top (x,z)", fontsize=8)
            fig.suptitle(f"{clip}  frame {t}/{T}   [{desc}]", fontsize=9)
            fig.tight_layout()
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    finally:
        writer.close()
        plt.close(fig)
    print(f"wrote {out} ({len(idxs)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.all:
        clips = sorted(p.stem for p in RAW.glob("*.amc"))
    else:
        clips = args.clips
    if not clips:
        raise SystemExit("no clips specified")

    for c in clips:
        render_clip(c, outdir, every=args.every, fps=args.fps)


if __name__ == "__main__":
    main()
