"""Render a Humanoid21 pose to PNG(s) for review (natural-pose iteration).

Uses the real arena model + MuJoCo offscreen renderer. Poses are raw qpos
radians; root is (pos, quat_wxyz).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .fk import Humanoid21FK
from .landmarks import LANDMARKS

_POINT_RGBA = {
    "head": (224, 72, 62, 255),
    "hand_right": (59, 165, 59, 255),
    "hand_left": (59, 127, 212, 255),
    "foot_right": (224, 160, 42, 255),
    "foot_left": (181, 106, 224, 255),
}


def _project(cam: mujoco.MjvCamera, fovy: float,
             size: tuple, p3: np.ndarray):
    """Project world point to pixel using a free mjvCamera."""
    az = np.deg2rad(cam.azimuth)
    el = np.deg2rad(cam.elevation)
    fwd = np.array([-np.cos(el) * np.cos(az),
                    -np.cos(el) * np.sin(az), -np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(right, fwd)
    pos = cam.lookat - fwd * cam.distance
    rel = p3 - pos
    x, y = rel @ right, rel @ up
    z = -(rel @ fwd)
    if z <= 0:
        return None
    f = 0.5 * size[1] / np.tan(np.deg2rad(fovy) / 2)
    return (size[0] / 2 + f * x / z, size[1] / 2 - f * y / z)


def render_pose(fk: Humanoid21FK, out_path,
                robot_id: str = "robot_a",
                root_pos=(0.0, 0.0, 1.15),
                root_quat_wxyz=(1.0, 0, 0, 0),
                joint_pos: Optional[np.ndarray] = None,
                views: Sequence = (("front", 0.0, 0.0),
                                   ("side", 90.0, 0.0),
                                   ("oblique", 45.0, 18.0)),
                size=(640, 480),
                mark_points: bool = True) -> list:
    """Render the posed robot from each (label, azimuth, elevation)."""
    if joint_pos is None:
        joint_pos = np.zeros(len(fk.controlled_joints(robot_id)))
    fk.set_pose(robot_id, root_pos, root_quat_wxyz, joint_pos)
    lm = fk.landmark_positions(robot_id)
    model, data = fk.model, fk.data
    fovy = float(model.vis.global_.fovy) or 45.0
    outs = []
    for label, az, el in views:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = np.asarray(root_pos, dtype=np.float64) + \
            np.array([0, 0, -0.35])
        cam.distance, cam.azimuth, cam.elevation = 3.0, az, el
        with mujoco.Renderer(model, size[1], size[0]) as rd:
            rd.update_scene(data, camera=cam)
            img = rd.render()
        im = Image.fromarray(img)
        if mark_points:
            dr = ImageDraw.Draw(im)
            from mocap.point5.contracts import POINT_NAMES
            for i, n in enumerate(POINT_NAMES):
                px = _project(cam, fovy, size, lm[i])
                if px:
                    c = _POINT_RGBA[n]
                    dr.ellipse([px[0] - 6, px[1] - 6, px[0] + 6, px[1] + 6],
                               outline=(0, 0, 0), fill=c[:3])
        p = Path(str(out_path).format(label=label))
        im.save(p)
        outs.append(p)
    return outs
