"""Render source-skeleton stick-figure previews of AMC clips (headless).

For human eyeball review of clip content (M0 W3 / W6 minimal-set selection).
Draws bone segments (joint -> distal endpoint), hands highlighted,
feet contact-ish tinted, ground line at y=0.

FK uses mocap.humanoid21.ref_fk — the reference implementation faithful to
CMU's official ASF/AMC viewer (the legacy baseline fk.py produced wrong
kinematics; see ref_fk.py docstring and M0_AUDIT.md).

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.preview --clips 13_17 13_18 \
        --outdir /data1/dev/combatbench_mocap/preview [--every 4] [--fps 30]
    PYTHONPATH=. python3 -m mocap.humanoid21.preview --all --outdir ...

Raw dirs searched for <subj>.asf / <clip>.amc: repo raw/ then
/data1/dev/combatbench_mocap/raw/.
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
from mocap.humanoid21.ref_fk import RefSkeleton, fk_batch

RAW_DIRS = [
    PROJECT_ROOT / "baseline/humanoid21/mocap/data/raw",
    Path("/data1/dev/combatbench_mocap/raw"),
]
FPS_SRC = 120.0
# ASF `:units length 0.45` => positions/lengths in units of 0.45 inch.
# Display scale is arbitrary; we normalize plots to body height.
UNIT_TO_INCH = 0.45


def _find_raw(name: str) -> Path:
    for d in RAW_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found in {RAW_DIRS}")


def render_clip(clip: str, outdir: Path, every: int = 4, fps: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    subj = clip.split("_")[0]
    skel = RefSkeleton.from_asf(parse_asf(str(_find_raw(f"{subj}.asf"))))
    frames = parse_amc(str(_find_raw(f"{clip}.amc")))
    J, D, names = fk_batch(skel, frames)          # (T,N,3) joint & distal positions

    T = len(frames)
    idxs = list(range(0, T, every))

    # distal endpoints for highlighted markers
    idx = {n: i for i, n in enumerate(names)}
    marker_bones = (("lhand", "lwrist", "red"), ("rhand", "rwrist", "darkred"),
                    ("lfoot", "lfoot", "green"), ("rfoot", "rfoot", "darkgreen"),
                    ("head", "head", "black"))

    # fixed axis limits from global extents; normalize around body height
    allp = np.concatenate([J[idxs].reshape(-1, 3), D[idxs].reshape(-1, 3)], axis=0)
    body_h = float(allp[:, 1].max() - allp[:, 1].min())
    span = max(0.35 * body_h, float(allp[:, 0].std()) * 2.5, 0.3 * body_h)
    ymax = allp[:, 1].max() * 1.08

    # Smoothed facing per frame (CMU coords, y-up): facing = up x shoulder-line.
    # Panels are facing-aligned so punches always extend screen-right in the
    # sagittal view — fixed world-axis projections read as "backward" whenever
    # the subject faces a diagonal direction.
    lsh = idx["lhumerus"]; rsh = idx["rhumerus"]
    up = np.array([0.0, 1.0, 0.0])
    sl = J[:, rsh] - J[:, lsh]
    fwd3 = np.cross(np.broadcast_to(up, sl.shape), sl)
    fwd3[:, 1] = 0.0
    fwd3 = fwd3 / np.maximum(np.linalg.norm(fwd3, axis=1, keepdims=True), 1e-9)
    zc = np.exp(1j * np.arctan2(fwd3[:, 2], fwd3[:, 0]))
    zc = np.convolve(zc, np.ones(91) / 91, mode="same")
    ang = np.unwrap(np.angle(zc))
    fwd_xz = np.stack([np.cos(ang), np.sin(ang)], axis=1)          # (T,2) x,z
    left_xz = np.stack([np.sin(ang), -np.cos(ang)], axis=1)        # up x fwd
    root_xz = J[:, idx["root"], [0, 2]].copy()

    desc = ""
    try:
        d = json.loads((Path(__file__).parent / "cmu_motion_descriptions.json").read_text())
        desc = d["subjects"][subj]["trials"].get(str(int(clip.split("_")[1])), "")
    except Exception:
        pass

    # bone segments: each bone drawn from joint[b] to distal[b]
    segs = [n for n in names if n != "root"]

    out = outdir / f"{clip}.mp4"
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.2), dpi=90)
    writer = imageio.get_writer(out, fps=fps)
    views = [(fwd_xz, "sagittal (facing ->)"), (left_xz, "coronal (left+)")]
    try:
        for t in idxs:
            for vi, (axs, label) in enumerate(views):
                ax = axes[vi]
                ax.clear()
                ax.set_xlim(-span, span)
                ax.set_ylim(-0.05 * ymax, ymax)
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.axhline(0, color="gray", lw=0.6)
                ax.annotate("", xy=(0.8 * span, 0.08 * ymax),
                            xytext=(0.4 * span, 0.08 * ymax),
                            arrowprops=dict(arrowstyle="->", color="orange", lw=2))
                h = axs[t]
                for b in segs:
                    i = idx[b]
                    ax.plot([np.dot(J[t, i, [0, 2]] - root_xz[t], h),
                             np.dot(D[t, i, [0, 2]] - root_xz[t], h)],
                            [J[t, i, 1], D[t, i, 1]], color="steelblue", lw=1.6)
                for label_b, draw_b, c in marker_bones:
                    if draw_b in idx:
                        ax.scatter(np.dot(D[t, idx[draw_b], [0, 2]] - root_xz[t], h),
                                   D[t, idx[draw_b], 1], color=c, s=24, zorder=5)
                ax.set_title(label, fontsize=8)
            ax = axes[2]
            ax.clear()
            ax.set_xlim(-span, span); ax.set_ylim(-span, span)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
            for b in segs:
                i = idx[b]
                ax.plot([J[t, i, 0] - root_xz[t, 0], D[t, i, 0] - root_xz[t, 0]],
                        [J[t, i, 2] - root_xz[t, 1], D[t, i, 2] - root_xz[t, 1]],
                        color="steelblue", lw=1.2)
            for label_b, draw_b, c in marker_bones:
                if draw_b in idx:
                    ax.scatter(D[t, idx[draw_b], 0] - root_xz[t, 0],
                               D[t, idx[draw_b], 2] - root_xz[t, 1],
                               color=c, s=20, zorder=5)
            ax.annotate("", xy=(root_xz[t] + fwd_xz[t] * 0.5 * span - root_xz[t]),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="orange", lw=2))
            ax.set_title("top (x,z) + facing arrow", fontsize=8)
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
        seen = {}
        for d in RAW_DIRS:
            for p in sorted(d.glob("*.amc")):
                seen.setdefault(p.stem, p)
        clips = sorted(seen)
    else:
        clips = args.clips
    if not clips:
        raise SystemExit("no clips specified")

    for c in clips:
        render_clip(c, outdir, every=args.every, fps=args.fps)


if __name__ == "__main__":
    main()
