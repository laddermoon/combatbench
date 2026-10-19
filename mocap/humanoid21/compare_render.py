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


def _smooth_facing(kp: np.ndarray, bi: dict, half: int = 45) -> np.ndarray:
    """Per-frame smoothed horizontal facing dir (T,2) in MJ xy.

    Facing = up x (rshoulder - lshoulder). Box-smoothed over `2*half+1` frames
    (~0.75 s at 120 Hz) via circular averaging so the view does not jitter.
    """
    up = np.array([0.0, 0.0, 1.0])
    sl = kp[:, bi["rhumerus"], :] - kp[:, bi["lhumerus"], :]
    fwd = np.cross(np.broadcast_to(up, sl.shape), sl)
    fwd[:, 2] = 0.0
    n = np.linalg.norm(fwd, axis=1, keepdims=True)
    fwd = fwd / np.maximum(n, 1e-9)
    z = np.exp(1j * np.arctan2(fwd[:, 1], fwd[:, 0]))
    k = np.ones(2 * half + 1) / (2 * half + 1)
    zs = np.convolve(z, k, mode="same")
    ang = np.unwrap(np.angle(zs))
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def render_compare(clip_id: str, outdir: Path, every: int = 4, fps: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    if clip_id.endswith("_m") or "_m_" in clip_id:
        # 镜像片段无自有 _src：取父片段源骨架做 mj 系 y->-y 镜像显示
        parent_id = clip_id[:-2] if clip_id.endswith("_m") \
            else clip_id.replace("_m_", "_", 1)
        parent_src = np.load(LIBRARY / "clips" / f"{parent_id}_src.npz")
        M = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        from scipy.spatial.transform import Rotation as _R
        mirrored = {}
        for k in parent_src.files:
            arr = parent_src[k]
            if arr.ndim >= 2 and arr.shape[-1] == 3:
                arr = arr @ M.T
            elif k == "root_quat":
                # wxyz -> xyzw, MRM 共轭, 回 wxyz
                q = _R.from_quat(arr[:, [1, 2, 3, 0]])
                qm = (_R.from_matrix(M[None] @ q.as_matrix() @ M[None])
                      ).as_quat()
                arr = qm[:, [3, 0, 1, 2]]
            mirrored[k] = arr
        src = mirrored
    else:
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

    # Smoothed facing per frame; panels are facing-aligned so a punch always
    # extends screen-right in the sagittal view (fixes misreading world-axis
    # projections when the subject faces diagonally).
    fwd2 = _smooth_facing(kp_s, bi)
    left2 = np.stack([-fwd2[:, 1], fwd2[:, 0]], axis=1)  # up x fwd, z-up

    # 1280 x 432: src sagittal | src coronal | robot render  (all /16 divisible)
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
            fv, lv = fwd2[t], left2[t]
            # panel 0: sagittal (facing-aligned), panel 1: top-down + facing arrow
            for vi, (axis2, label) in enumerate(((fv, "sagittal (facing ->)"),
                                                 (None, "top (x,y)"))):
                ax = axes[vi]
                ax.clear()
                ax.set_aspect("equal")
                ax.axhline(0, color="gray", lw=0.6)
                if axis2 is not None:
                    ax.set_xlim(-span, span)
                    ax.set_ylim(-0.1, 2.0)
                    ax.annotate("", xy=(0.9 * span, 0.12), xytext=(0.3 * span, 0.12),
                                arrowprops=dict(arrowstyle="->", color="orange", lw=2))
                    for ci, pi in segs:
                        hx_c = np.dot(kp_s[t, ci, :2] - [cx, cy], axis2)
                        hx_p = np.dot(kp_s[t, pi, :2] - [cx, cy], axis2)
                        ax.plot([hx_p, hx_c],
                                [kp_s[t, pi, 2], kp_s[t, ci, 2]],
                                color="steelblue", lw=1.6)
                    for mb, c in SRC_MARKER_COLORS.items():
                        if mb in bi:
                            ax.scatter(np.dot(kp_s[t, bi[mb], :2] - [cx, cy], axis2),
                                       kp_s[t, bi[mb], 2], color=c, s=22, zorder=5)
                else:
                    ax.set_xlim(-span, span)
                    ax.set_ylim(-span, span)
                    for ci, pi in segs:
                        ax.plot([kp_s[t, pi, 0] - cx, kp_s[t, ci, 0] - cx],
                                [kp_s[t, pi, 1] - cy, kp_s[t, ci, 1] - cy],
                                color="steelblue", lw=1.6)
                    for mb, c in SRC_MARKER_COLORS.items():
                        if mb in bi:
                            ax.scatter(kp_s[t, bi[mb], 0] - cx, kp_s[t, bi[mb], 1] - cy,
                                       color=c, s=22, zorder=5)
                    ax.annotate("", xy=(fv[0] * span, fv[1] * span),
                                xytext=(0.0, 0.0),
                                arrowprops=dict(arrowstyle="->", color="orange", lw=2.2))
                ax.set_title(label, fontsize=8)
            axes[2].clear()
            axes[2].imshow(img)
            axes[2].set_title("robot replay (mj_forward)", fontsize=8)
            axes[2].set_xticks([]); axes[2].set_yticks([])
            fig.suptitle(f"{clip_id}  frame {t}/{T}   facing=({fv[0]:+.2f},{fv[1]:+.2f})",
                         fontsize=9)
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
        cid = c if c.startswith(("cmu_", "hdm05_")) else f"cmu_{c}"
        render_compare(cid, outdir, args.every, args.fps)


if __name__ == "__main__":
    main()
