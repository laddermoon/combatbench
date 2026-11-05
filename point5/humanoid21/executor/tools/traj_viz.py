"""Tool 5: CentroidalTrajectory -> 3D MP4 animation (MuJoCo render).

A shaded ball follows com c(t); linear momentum shows as motion.
Three surface spot geoms are attached to the ball body — they are
physically part of it — and the ball integrates w = L / I_REF about
the L axis, so angular momentum reads as spin.  A fading trail, a
drop line to the ground, a ground shadow and a z-height caption
complete the picture.

    render_traj(traj_json, out_mp4)

Caller supplies both paths.  Fixed style — nothing user-adjustable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np

from ..centroidal import CTRL_DT, load_traj

I_REF = 0.53            # kg·m^2 nominal inertia (standing Izz)
_TRAIL = 40             # steps kept in the fading trail
_BALL_R = 0.06
_FPS = 20
_SIZE = (960, 720)

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
      <geom type="sphere" size="{s}" pos="{r} 0 0"
            rgba="1 0.15 0.15 1"/>
      <geom type="sphere" size="{s}" pos="0 {r} 0"
            rgba="0.1 0.8 0.2 1"/>
      <geom type="sphere" size="{s}" pos="0 0 {r}"
            rgba="0.1 0.1 0.9 1"/>
      <geom type="sphere" size="{s}" pos="-{r} 0 0"
            rgba="1 0.55 0.05 1"/>
    </body>
  </worldbody>
</mujoco>"""


def _rot(axis, ang):
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _R2q(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2
        q = np.empty(4)
        q[i + 1] = s / 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _mat2quat(R):
    return _R2q(R)


def _project(cam, fovy_deg, world, size):
    """World -> pixel under an mjvCamera."""
    a = np.radians(cam.azimuth)
    e = np.radians(cam.elevation)
    fwd = np.array([np.cos(e) * np.cos(a),
                    np.cos(e) * np.sin(a), np.sin(e)])
    campos = cam.lookat - cam.distance * fwd
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    right = right / (np.linalg.norm(right) + 1e-12)
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


def render_traj(traj_json: str, out_mp4: str) -> Path:
    """Render a saved CentroidalTrajectory JSON to MP4."""
    traj = load_traj(traj_json)
    states = traj.states
    cs = np.array([s.com for s in states])
    Ls = np.array([s.ang_mom for s in states])

    spot_r = _BALL_R * 0.28
    model = mujoco.MjModel.from_xml_string(
        _XML.format(r=_BALL_R, s=spot_r))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = _SIZE

    cam = mujoco.MjvCamera()
    cx = cs.mean(axis=0)
    cam.lookat[:] = [cx[0], cx[1], max(0.55, cx[2] - 0.15)]
    cam.distance = 1.25
    cam.azimuth = 145
    cam.elevation = -22

    w, h = _SIZE
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(_FPS), "-i", "-",
         "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         out_mp4], stdin=subprocess.PIPE)

    R_orient = np.eye(3)
    ball_qpos = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                          "ball")]

    with mujoco.Renderer(model, h, w) as rd:
        for k, s in enumerate(states):
            L = Ls[k]
            wv = np.linalg.norm(L) / I_REF
            if wv > 1e-9:
                R_orient = _rot(L, wv * CTRL_DT) @ R_orient

            c = cs[k]
            data.qpos[ball_qpos:ball_qpos + 3] = c
            data.qpos[ball_qpos + 3:ball_qpos + 7] = _R2q(R_orient)
            mujoco.mj_forward(model, data)

            rd.update_scene(data, camera=cam)
            scn = rd.scene
            # fading trail
            lo = max(0, k - _TRAIL)
            for i in range(lo, k):
                if scn.ngeom >= scn.maxgeom:
                    break
                a = 0.15 + 0.55 * (i - lo + 1) / max(k - lo, 1)
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.022, 0, 0]),
                    cs[i].astype(np.float32),
                    np.eye(3).flatten().astype(np.float32),
                    np.array([0.2, 0.5, 1.0, a], np.float32))
                scn.ngeom += 1
            # drop line: thin capsule ground -> ball
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([0.004, c[2] / 2, 0]),
                np.array([c[0], c[1], c[2] / 2], np.float32),
                np.eye(3).flatten().astype(np.float32),
                np.array([0, 0, 0, 0.45], np.float32))
            scn.ngeom += 1
            frame = rd.render()

            # caption overlay via PIL
            try:
                from PIL import Image, ImageDraw
                img = Image.fromarray(frame)
                dr = ImageDraw.Draw(img)
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/"
                        "DejaVuSans.ttf", 26)
                except Exception:
                    font = ImageFont.load_default()
                dr.text((22, _SIZE[1] - 44),
                        f"t={k * CTRL_DT:.2f}s",
                        fill=(30, 30, 30), font=font,
                        stroke_width=2, stroke_fill=(255, 255, 255))
                # height label next to the ball (projected to screen)
                px = _project(cam, model.vis.global_.fovy,
                              c + np.array([0, 0, _BALL_R + 0.04]),
                              _SIZE)
                if px:
                    dr.text((px[0], px[1] - 14),
                            f"z={c[2]:.2f}m", fill=(20, 20, 60),
                            font=font, stroke_width=2,
                            stroke_fill=(255, 255, 255))
                frame = np.asarray(img)
            except ImportError:
                pass
            ffmpeg.stdin.write(frame.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    return Path(out_mp4)
