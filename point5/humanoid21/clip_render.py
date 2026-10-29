"""Render a P5 clip to MP4 with the Humanoid21 robot body.

Per frame: sample the clip (change-adaptive — max per-point displacement
bounded), run the two-stage equilibrium solve for the joint pose, render
front + side views with the real model geoms, and overlay the five target
points on top of the image (always visible, even inside the body).

    MUJOCO_GL=egl PYTHONPATH=. python3 -m point5.humanoid21.clip_render \
        --clip clip.json --out clip.mp4 [--max-step 0.01] [--robot robot_a]

Illegal frames keep their best-effort pose and get a red label.
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

from point5.contracts import (  # noqa: E402
    POINT_NAMES, P5Clip, to_world)
from point5.trajectory import sample_clip  # noqa: E402
from point5.humanoid21.fk import Humanoid21FK  # noqa: E402
from point5.humanoid21.mechanics import (  # noqa: E402
    Skeleton, solve_equilibrium)
from point5.humanoid21.render import _project  # noqa: E402
from scipy.spatial.transform import Rotation as _R  # noqa: E402

PT_COLOR = {"head": (224, 72, 62), "hand_right": (59, 165, 59),
            "hand_left": (59, 127, 212), "foot_right": (224, 160, 42),
            "foot_left": (181, 106, 224)}


def _world_pose(res, origin_xy, yaw_deg):
    """Lift a local-frame solve result into world under the frame pose."""
    rz = _R.from_euler("z", np.deg2rad(yaw_deg))
    rp = np.asarray(res["root_pos"], float).copy()
    rp[:2] = rz.as_matrix()[:2, :2] @ rp[:2] + np.asarray(origin_xy)
    ql = res["root_quat_wxyz"]          # wxyz
    qw = (rz * _R.from_quat([ql[1], ql[2], ql[3], ql[0]])).as_quat()
    return {"root_pos": rp,
            "root_quat_wxyz": np.array([qw[3], qw[0], qw[1], qw[2]]),
            "q_rad": res["q_rad"]}


def render_one(fk, rd, cam, img_size, pose, target, sat, failed, fidx):
    fk.set_pose(robot, pose["root_pos"], pose["root_quat_wxyz"],
                pose["q_rad"])
    rd.update_scene(fk.data, camera=cam)
    img = Image.fromarray(rd.render()).convert("RGB")
    dr = ImageDraw.Draw(img)
    for i, n in enumerate(POINT_NAMES):
        px = _project(cam, 60.0, img_size, target[i])
        if px is None:
            continue
        x, y = px
        c = PT_COLOR[n]
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline=c, width=3)
        dr.text((x + 9, y - 6), n.split("_")[0][0] + ("r" if "right" in n else "l" if "left" in n else ""), fill=c)
    if not sat:
        dr.text((8, 8), f"ILLEGAL {failed}", fill=(255, 40, 40))
    dr.text((8, img_size[1] - 18), f"#{fidx}", fill=(255, 255, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-step", type=float, default=0.01)
    ap.add_argument("--robot", default="robot_a")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=640)
    args = ap.parse_args()

    global robot
    robot = args.robot
    clip = P5Clip.from_dict(json.loads(Path(args.clip).read_text()))
    frames = sample_clip(clip, max_step=args.max_step)
    print(f"{len(clip.keyframes)} keyframes -> {len(frames)} frames",
          flush=True)

    fk = Humanoid21FK()
    sk = Skeleton()
    # park the opponent model out of view
    other = "robot_b" if args.robot == "robot_a" else "robot_a"
    fk.set_pose(other, np.array([0.0, 0.0, -20.0]),
                np.array([1., 0, 0, 0]), np.zeros(21))
    size = (args.w, args.h)
    rd = mujoco.Renderer(fk.model, args.h, args.w)
    cams = []
    for az in (0, 90):                      # front, side
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 3.0, az, 0.0
        cams.append(cam)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.w * 2}x{args.h}", "-r", "10",
         "-i", "-", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE)

    for fi, sm in enumerate(frames):
        res = solve_equilibrium(sm.pos, skeleton=sk)
        wp = _world_pose(res, sm.origin_xy, sm.yaw_deg)
        target = sm.world_pos()
        cells = []
        for cam in cams:
            cam.lookat[:] = wp["root_pos"] + np.array([0, 0, -0.35])
            cells.append(render_one(fk, rd, cam, size, wp, target,
                                    res["satisfied"],
                                    res["failed_stage"], fi))
        img = Image.new("RGB", (args.w * 2, args.h))
        img.paste(cells[0], (0, 0)); img.paste(cells[1], (args.w, 0))
        ff.stdin.write(img.tobytes())
        if fi % 10 == 0:
            print(f"{fi}/{len(frames)}", flush=True)

    ff.stdin.close(); ff.wait()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
