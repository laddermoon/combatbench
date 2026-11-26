"""Kinematic-replay CoreClips on the REAL battle model -> mp4.

Writes each stored CoreState into battle_circular_v2.xml directly
(root freejoint + 21 hinge qpos/qvel — exactly what
sim.set_core_state writes), then mj_forward + a yaw-tracking front
3/4 camera (render_phc-tuned params) so feet/ground contact are
visible.  robot_b stays at its default mark.

Usage:
    clip play-env --clip <npz> --out <mp4>
    clip play-env --dir data/core --outdir data/core/videos [--only SUB]
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as _R

from .core_clip import CONTROL_FPS, CoreClip

BATTLE_XML = (Path(__file__).resolve().parents[3]
              / 'envs' / 'humanoid21' / 'battle_circular_v2.xml')
H, W = 480, 640
CAM_DIST, CAM_ELE, CAM_YAW = 3.3, -8.0, 150.0


def _mj():
    import os
    os.environ['MUJOCO_GL'] = 'osmesa'      # EGL is broken on this box
    import mujoco
    return mujoco


def render_clip(clip: CoreClip, out, robot: str = 'robot_a',
                fps: float = CONTROL_FPS) -> Path:
    mujoco = _mj()
    suffix = '_a' if robot == 'robot_a' else '_b'
    m = mujoco.MjModel.from_xml_path(str(BATTLE_XML))
    d = mujoco.MjData(m)

    root_j = m.joint(f'root{suffix}').id
    rp_adr = m.jnt_qposadr[root_j]
    rv_adr = m.jnt_dofadr[root_j]
    jids = np.array([m.joint(n + suffix).id for n in clip.meta.get(
        'dof_names', ['abdomen_z', 'abdomen_y', 'abdomen_x',
                      'hip_x_right', 'hip_z_right', 'hip_y_right',
                      'knee_right', 'ankle_y_right', 'ankle_x_right',
                      'hip_x_left', 'hip_z_left', 'hip_y_left',
                      'knee_left', 'ankle_y_left', 'ankle_x_left',
                      'shoulder1_right', 'shoulder2_right', 'elbow_right',
                      'shoulder1_left', 'shoulder2_left', 'elbow_left'])])
    jq_adr = m.jnt_qposadr[jids]
    jv_adr = m.jnt_dofadr[jids]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                            f'torso{suffix}')
    # ghost the OTHER robot so the review camera can't be occluded by it
    other = '_b' if suffix == '_a' else '_a'
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                 m.geom_bodyid[g]) or ''
        if name.endswith(other) or body.endswith(other):
            m.geom_rgba[g, 3] = 0.05

    r = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = CAM_DIST, CAM_ELE

    raw = Path('/tmp/_core_replay.mp4')
    import cv2
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (W, H))
    for k in range(clip.T):
        st = clip.state(k)
        q_wxyz = st['root_rot'].astype(np.float64)
        d.qpos[rp_adr:rp_adr + 3] = st['root_pos']
        d.qpos[rp_adr + 3:rp_adr + 7] = q_wxyz
        d.qpos[jq_adr] = st['joint_pos_norm'] * half + mid
        rot = _R.from_quat(np.roll(q_wxyz, -1))          # wxyz -> xyzw
        d.qvel[rv_adr:rv_adr + 3] = rot.apply(st['root_vel_local'])
        d.qvel[rv_adr + 3:rv_adr + 6] = st['root_angular_vel_local']
        d.qvel[jv_adr] = st['joint_vel_norm'] * half
        mujoco.mj_forward(m, d)

        fwd = rot.apply([1.0, 0, 0])
        cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0])) + CAM_YAW
        cam.lookat[:] = d.xpos[torso]
        cam.lookat[2] = max(0.45, d.xpos[torso][2] - 0.45)
        r.update_scene(d, camera=cam)
        wr.write(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    wr.release()
    r.close()

    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    str(out)], check=True)
    raw.unlink()
    return out
