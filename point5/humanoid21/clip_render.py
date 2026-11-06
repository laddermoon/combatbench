"""Render a P5Clip to MP4 with the real Humanoid21 body.

Per sampled frame: interpolate base-frame poses (Base.pos + delta_i,
change-adaptive via per_segment lerp), re-anchor each to its own local
frame, solve_equilibrium for the joint pose, lift the solved pose back
into the base frame, render front + side views with target overlays.

    MUJOCO_GL=egl PYTHONPATH=. python3 -m point5.humanoid21.clip_render \
        --clip clip.json --out clip.mp4 [--per-segment 15] [--robot robot_a]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from point5.contracts import POINT_NAMES, P5Clip, to_local  # noqa: E402
from point5.humanoid21.pose import Pose, _project  # noqa: E402
from point5.humanoid21.mechanics import (  # noqa: E402
    Skeleton, solve_equilibrium)
from scipy.spatial.transform import Rotation as _R  # noqa: E402

PT_COLOR = {"head": (224, 72, 62), "hand_right": (59, 165, 59),
            "hand_left": (59, 127, 212), "foot_right": (224, 160, 42),
            "foot_left": (181, 106, 224)}


def _samples(clip: P5Clip, per_segment: int = 15):
    """Yield (raw (5,3), yaw) interpolated in the BASE frame."""
    raws = [clip.Base.pos] + [clip.Base.pos + t.delta_pos
                              for t in clip.Transfers]
    yaws = [0.0] + [t.delta_yaw for t in clip.Transfers]
    for k in range(len(raws) - 1):
        for j in range(per_segment):
            t = j / per_segment
            yield (raws[k] + (raws[k + 1] - raws[k]) * t,
                   yaws[k] + (yaws[k + 1] - yaws[k]) * t)
    yield raws[-1], yaws[-1]


def _ankle_mid(raw):
    return 0.5 * (raw[3, :2] + raw[4, :2])


def _set_pose(model, data, r, root_pos, quat_wxyz, joints):
    adr = int(r["root_qpos_adr"])
    data.qpos[adr:adr + 3] = np.asarray(root_pos, np.float64)
    data.qpos[adr + 3:adr + 7] = np.asarray(quat_wxyz, np.float64)
    data.qpos[np.asarray(r["qpos_indices"])] = np.asarray(
        joints, np.float64)
    mujoco.mj_kinematics(model, data)


def render_one(model, data, r, rd, cam, img_size, root_pos, quat,
               joints, target, sat, failed, fidx):
    _set_pose(model, data, r, root_pos, quat, joints)
    rd.update_scene(data, camera=cam)
    img = Image.fromarray(rd.render()).convert("RGB")
    dr = ImageDraw.Draw(img)
    for i, n in enumerate(POINT_NAMES):
        px = _project(cam, 60.0, img_size, target[i])
        if px is None:
            continue
        x, y = px
        c = PT_COLOR[n]
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline=c, width=3)
    if not sat:
        dr.text((8, 8), f"ILLEGAL {failed}", fill=(255, 40, 40))
    dr.text((8, img_size[1] - 18), f"#{fidx}", fill=(255, 255, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-segment", type=int, default=15)
    ap.add_argument("--robot", default="robot_a")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=640)
    args = ap.parse_args()

    clip = P5Clip.from_dict(json.loads(Path(args.clip).read_text()))
    frames = list(_samples(clip, args.per_segment))
    print(f"{len(clip.p5)} frames -> {len(frames)} samples", flush=True)

    model, data, robots = Pose._kinematics()
    r = robots[args.robot]
    sk = Skeleton()
    other = "robot_b" if args.robot == "robot_a" else "robot_a"
    _set_pose(model, data, robots[other], np.array([0., 0., -20.]),
              np.array([1., 0, 0, 0]), np.zeros(21))
    size = (args.w, args.h)
    rd = mujoco.Renderer(model, args.h, args.w)
    cams = []
    for az in (0, 90):
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 3.0, az, 0.0
        cams.append(cam)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.w * 2}x{args.h}", "-r", "10",
         "-i", "-", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE)

    for fi, (raw, yaw) in enumerate(frames):
        mid = _ankle_mid(raw)
        local = to_local(raw, mid, float(np.degrees(yaw)))
        res = solve_equilibrium(local, skeleton=sk)
        # lift the solved local pose into the base frame at (mid, yaw)
        rz = _R.from_euler("z", yaw)
        rp = np.asarray(res["root_pos"], float).copy()
        rp[:2] = rz.as_matrix()[:2, :2] @ rp[:2] + mid
        ql = res["root_quat_wxyz"]
        qw = (rz * _R.from_quat([ql[1], ql[2], ql[3], ql[0]])).as_quat()
        quat = np.array([qw[3], qw[0], qw[1], qw[2]])
        cells = []
        for cam in cams:
            cam.lookat[:] = rp + np.array([0, 0, -0.35])
            cells.append(render_one(model, data, r, rd, cam, size,
                                    rp, quat, res["q_rad"], raw,
                                    res["satisfied"],
                                    res["failed_stage"], fi))
        img = Image.new("RGB", (args.w * 2, args.h))
        img.paste(cells[0], (0, 0)); img.paste(cells[1], (args.w, 0))
        ff.stdin.write(img.tobytes())
        if fi % 20 == 0:
            print(f"{fi}/{len(frames)}", flush=True)

    ff.stdin.close(); ff.wait()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
