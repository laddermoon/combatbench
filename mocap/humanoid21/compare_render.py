"""M1-W5: side-by-side comparison videos.

LEFT:  source CMU skeleton stick figure (ref_fk, 2D front+side projections)
RIGHT: humanoid21 kinematic replay of the retargeted joint trajectory —
       MuJoCo mj_forward + offscreen render. This is KINEMATIC playback only
       (state overwrite per frame), NOT physics validation (that is M2).

Requires MUJOCO_GL=egl for headless rendering.

Usage:
    MUJOCO_GL=egl PYTHONPATH=. python3 -m mocap.humanoid21.compare_render \
        --clips 13_17 07_01 --outdir /data1/dev/combatbench_mocap/compare \
        [--every 4] [--fps 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco

from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS

LIBRARY = Path("/data1/dev/combatbench_mocap")
XML = PROJECT_ROOT / "envs/humanoid21/battle_circular_v2.xml"

# robot_A controlled joints qpos addresses (freejoint = 7, then 21 hinges)
SRC_MARKER_COLORS = {"lhand": "red", "rhand": "darkred",
                     "lfoot": "green", "rfoot": "darkgreen",
                     "head": "black"}


def render_compare(clip_id: str, outdir: Path, every: int = 4, fps: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    src = np.load(LIBRARY / "clips" / f"{clip_id}_src.npz")
    rob = np.load(LIBRARY / "clips" / f"{clip_id}.npz")

    kp_s = src["keypoints"]            # (T,29,3) m, z-up
    kn_s = list(src["keypoint_names"])
    bi = {b: i for i, b in enumerate(kn_s)}
    # bone segments: each bone joint-> its parent's distal == joint pos of the
    # bone itself to the bone's own distal? We only have joint positions;
    # connect parent joint -> child joint for a readable figure.
    PARENT = {  # enough for a humanoid stick figure
        "lhipjoint": "root", "rhipjoint": "root",
        "lfemur": "lhipjoint", "rfemur": "rhipjoint",
        "ltibia": "lfemur", "rtibia": "rfemur",
        "lfoot": "ltibia", "rfoot": "rtibia",
        "ltoes": "lfoot", "rtoes": "rfoot",
        "lowerback": "root", "upperback": "lowerback", "thorax": "upperback",
        "lowerneck": "thorax", "upperneck": "lowerneck", "head": "upperneck",
        "lclavicle": "thorax", "rclavicle": "thorax",
        "lhumerus": "lclavicle", "rhumerus": "rclavicle",
        "lradius": "lhumerus", "rradius": "rhumerus",
        "lwrist": "lradius", "rwrist": "rradius",
        "lhand": "lwrist", "rhand": "rwrist",
    }
    segs = [(bi[c], bi[p]) for c, p in PARENT.items()
            if c in bi and p in bi]

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    rend = mujoco.Renderer(m, 432, 512)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.2
    cam.azimuth = 115.0
    cam.elevation = -12.0

    qang = rob["joint_angles"]
    rpos = rob["root_pos"]
    rquat = rob["root_quat"]
    qadr = {}
    for i, n in enumerate(CONTROLLED_JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_a")
        qadr[n] = m.jnt_qposadr[jid]
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_a")
    d.qpos[28:31] = [0.0, 0.0, -10.0]   # park robot B under the floor

    T = len(qang)
    idxs = list(range(0, T, every))
    out = outdir / f"{clip_id}.mp4"

    # 1280 x 432: src front | src side | robot render   (all /16 divisible)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.32), dpi=100)
    writer = imageio.get_writer(out, fps=fps, macro_block_size=16)
    span = 1.4
    try:
        for t in idxs:
            # --- robot replay ---
            d.qpos[:3] = rpos[t]
            d.qpos[3:7] = rquat[t]
            for i, n in enumerate(CONTROLLED_JOINTS):
                d.qpos[qadr[n]] = qang[t, i]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.xpos[pelvis_id]
            rend.update_scene(d, camera=cam)
            img = rend.render()

            cx, cy = rpos[t, 0], rpos[t, 1]
            for vi, (hx, label) in enumerate(((0, "src front (x,z)"),
                                              (1, "src side (y,z)"))):
                ax = axes[vi]
                ax.clear()
                ax.set_xlim(cx - span, cx + span)
                ax.set_ylim(-0.1, 2.0)
                ax.set_aspect("equal")
                ax.axhline(0, color="gray", lw=0.6)
                for ci, pi in segs:
                    ax.plot([kp_s[t, ci, hx], kp_s[t, pi, hx]],
                            [kp_s[t, ci, 2], kp_s[t, pi, 2]],
                            color="steelblue", lw=1.6)
                for mb, c in SRC_MARKER_COLORS.items():
                    if mb in bi:
                        ax.scatter(kp_s[t, bi[mb], hx], kp_s[t, bi[mb], 2],
                                   color=c, s=22, zorder=5)
                ax.set_title(label, fontsize=8)
            axes[2].clear()
            axes[2].imshow(img)
            axes[2].set_title("robot replay (mj_forward)", fontsize=8)
            axes[2].set_xticks([]); axes[2].set_yticks([])
            axes[0].set_title("src front (x,z)", fontsize=8)
            fig.suptitle(f"{clip_id}  frame {t}/{T}", fontsize=9)
            fig.tight_layout()
            fig.canvas.draw()
            writer.append_data(
                np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    finally:
        writer.close()
        plt.close(fig)
        rend.close()
    print(f"wrote {out} ({len(idxs)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for c in args.clips:
        cid = c if c.startswith("cmu_") else f"cmu_{c}"
        render_compare(cid, outdir, args.every, args.fps)


if __name__ == "__main__":
    main()
