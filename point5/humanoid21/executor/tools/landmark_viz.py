"""Tool 5: LandMark sequence -> MP4 walkthrough (MuJoCo render).

Each anchor is held for ``hold`` frames so the planned support event /
posture can be read off:

- blue ball at ``lm.com`` + drop line to the ground
- foot patch at ``feet[f].foot_pos`` — solid when loaded
  (force_ratio>0), small floating marker when airborne
- red dot at ``feet[f].cop`` when a CoP is given
- cyan arrow along ``lm.vel`` when a velocity constraint exists
- fading com trail through previous anchors, caption with index

    render_landmarks(landmarks, out_mp4)
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import mujoco
import numpy as np

_FPS = 24
_HOLD = 14                  # frames each anchor is shown
_TRAIL = 40
_BALL_R = 0.06
_SIZE = (960, 720)

# foot patch half-extents from the arena XML (x fwd, y lateral)
_PATCH_HX, _PATCH_HY = 0.105, 0.057

_XML = """<mujoco>
  <option gravity="0 0 -9.81"/>
  <visual>
    <quality shadowsize="2048" offsamples="4"/>
    <rgba haze="0.15 0.15 0.15 1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" mark="edge"
             markrgb=".2 .2 .2" width="300" height="300"
             rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"/>
    <material name="grid" texture="grid" texrepeat="1 1"
              texuniform="true" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2"
           pos="0.4 -0.6 2.2" dir="-0.3 0.4 -1"/>
    <light directional="true" diffuse=".35 .35 .4"
           pos="-0.6 0.4 1.8" dir="0.4 -0.3 -1"/>
    <geom type="plane" size="0 0 .05" material="grid"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="{r}" rgba="0.15 0.45 0.95 1"/>
    </body>
  </worldbody>
</mujoco>"""

_FOOT_RGBA = {"right": np.array([0.95, 0.55, 0.10, 0.85], np.float32),
              "left":  np.array([0.10, 0.75, 0.35, 0.85], np.float32)}


def _z2dir(d: np.ndarray) -> np.ndarray:
    """Rotation matrix taking local +z to unit vector d."""
    d = np.asarray(d, np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3)
    d = d / n
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if d[2] > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 - d[2]) / (s * s)


def _geom(scn, type_, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, type_, np.asarray(size, np.float64),
        np.asarray(pos, np.float32),
        np.asarray(mat, np.float32).flatten(),
        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _project(cam, fovy_deg, world, size):
    a = np.radians(cam.azimuth)
    e = np.radians(cam.elevation)
    fwd = np.array([np.cos(e) * np.cos(a),
                    np.cos(e) * np.sin(a), np.sin(e)])
    campos = cam.lookat - cam.distance * fwd
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    right /= (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, fwd)
    rel = np.asarray(world) - campos
    z = float(np.dot(rel, fwd))
    if z <= 0.01:
        return None
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    w, h = size
    nx = np.dot(rel, right) * f / z / (w / h)
    ny = np.dot(rel, up) * f / z
    return int((nx + 1) * 0.5 * w), int((1 - ny) * 0.5 * h)


def render_landmarks(landmarks: List, out_mp4: str,
                     hold: int = _HOLD) -> Path:
    """Render an ordered LandMark sequence to MP4."""
    cs = np.array([lm.com for lm in landmarks], np.float64)

    model = mujoco.MjModel.from_xml_string(_XML.format(r=_BALL_R))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = _SIZE

    cam = mujoco.MjvCamera()
    cx = cs.mean(axis=0)
    span = float(np.ptp(cs[:, :2], axis=0).max())
    cam.lookat[:] = [cx[0], cx[1], max(0.5, cx[2] - 0.2)]
    cam.distance = max(1.4, 1.0 + span * 1.6)
    cam.azimuth = 145
    cam.elevation = -24

    w, h = _SIZE
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(_FPS), "-i", "-",
         "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         out_mp4], stdin=subprocess.PIPE)

    ball_qpos = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball")]
    I3 = np.eye(3)

    with mujoco.Renderer(model, h, w) as rd:
        for k, lm in enumerate(landmarks):
            c = np.asarray(lm.com, np.float64)
            for _ in range(hold):
                data.qpos[ball_qpos:ball_qpos + 3] = c
                mujoco.mj_forward(model, data)
                rd.update_scene(data, camera=cam)
                scn = rd.scene

                # fading com trail through previous anchors
                for i in range(max(0, k - _TRAIL), k):
                    a = 0.10 + 0.45 * (i - max(0, k - _TRAIL) + 1) \
                        / max(k - max(0, k - _TRAIL), 1)
                    _geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                          [0.022, 0, 0], cs[i], I3,
                          [0.2, 0.5, 1.0, a])

                # feet patches / airborne markers + CoP
                for f in ("right", "left"):
                    fs = lm.feet[f]
                    fp = np.asarray(fs.foot_pos, np.float64)
                    rgba = _FOOT_RGBA[f].copy()
                    rgba[3] *= 0.25 + 0.75 * float(fs.force_ratio)
                    if fs.force_ratio > 0.05:
                        _geom(scn, mujoco.mjtGeom.mjGEOM_BOX,
                              [_PATCH_HX, _PATCH_HY, 0.006],
                              [fp[0], fp[1], 0.006], I3, rgba)
                    else:
                        _geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                              [0.05, 0, 0], [fp[0], fp[1], 0.07],
                              I3, rgba)
                    if fs.cop is not None and fs.force_ratio > 0.05:
                        _geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                              [0.018, 0, 0],
                              [fs.cop[0], fs.cop[1], 0.02], I3,
                              [0.9, 0.1, 0.1, 0.95])

                # velocity constraint arrow (cyan capsule)
                if lm.vel is not None:
                    v = np.asarray(lm.vel, np.float64)
                    vn = np.linalg.norm(v)
                    if vn > 1e-6:
                        _geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE,
                              [0.012, vn / 2, 0],
                              c + v / 2, _z2dir(v),
                              [0.0, 0.85, 0.85, 0.9])

                # drop line ground -> ball
                _geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE,
                      [0.004, c[2] / 2, 0],
                      [c[0], c[1], c[2] / 2], I3,
                      [0, 0, 0, 0.45])

                frame = rd.render()
                try:
                    from PIL import Image, ImageDraw
                    img = Image.fromarray(frame)
                    dr = ImageDraw.Draw(img)
                    try:
                        from PIL import ImageFont
                        font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/"
                            "DejaVuSans.ttf", 24)
                    except Exception:
                        font = ImageFont.load_default()
                    fr_ = lm.feet
                    frame_i = getattr(lm, "frame", None)
                    dr.text((22, h - 44),
                            f"lm {k}/{len(landmarks) - 1}"
                            + (f"  kf={frame_i}" if frame_i is not None
                               else "")
                            + f"  frR={fr_['right'].force_ratio:.2f}"
                            + f" frL={fr_['left'].force_ratio:.2f}",
                            fill=(30, 30, 30), font=font,
                            stroke_width=2,
                            stroke_fill=(255, 255, 255))
                    px = _project(cam, model.vis.global_.fovy,
                                  c + np.array([0, 0, _BALL_R + 0.04]),
                                  _SIZE)
                    if px:
                        dr.text((px[0], px[1] - 14), f"z={c[2]:.2f}m",
                                fill=(20, 20, 60), font=font,
                                stroke_width=2,
                                stroke_fill=(255, 255, 255))
                    frame = np.asarray(img)
                except ImportError:
                    pass
                ffmpeg.stdin.write(frame.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    return Path(out_mp4)
