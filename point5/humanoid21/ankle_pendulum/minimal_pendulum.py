"""Minimal inverted-pendulum tipping test — h-scaling verification.

Rigid body (foot plate + pole with mass concentrated at height h)
on a ground plane, horizontal force ramped at the COM height.
Theory: tipping moment about the foot edge is N*margin — constant,
so F_tip = N*margin / h must scale as 1/h.

Runs h in {0.5, 0.9, 1.3}m, renders one MP4 per height.

    PYTHONPATH=. python3 -B point5/humanoid21/ankle_pendulum/minimal_pendulum.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np

OUT = Path("/data1/dev/ankle_pendulum")
OUT.mkdir(parents=True, exist_ok=True)

M_POLE = 40.0          # kg — whole robot mass on the pole
G = 9.81
FOOT_L, FOOT_W, FOOT_T = 0.21, 0.094, 0.02   # match humanoid patch
MARGIN = FOOT_W / 2                        # lateral edge margin 0.047


def _xml(h):
    return f"""<mujoco>
  <option gravity="0 0 -9.81"/>
  <visual><quality shadowsize="1024"/></visual>
  <worldbody>
    <light directional="true" pos="1 -2 3" dir="-0.4 0.6 -1"/>
    <geom type="plane" size="4 4 .05" rgba=".8 .8 .8 1"/>
    <body name="pend" pos="0 0 0.001">
      <freejoint/>
      <geom name="plate" type="capsule" fromto="-0.07 -0.02 0.047 0.14 -0.02 0.047"
            size="0.027" mass="0.5" rgba="0.2 0.2 0.2 1"
            condim="3" friction="1.0 0.02 0.001"
            solimp="0.9 0.95 0.001" solref="0.01 1"/>
      <geom name="plate2" type="capsule" fromto="-0.07 0.02 0.047 0.14 0.02 0.047"
            size="0.027" mass="0.5" rgba="0.2 0.2 0.2 1"
            condim="3" friction="1.0 0.02 0.001"
            solimp="0.9 0.95 0.001" solref="0.01 1"/>
      <geom name="pole" type="capsule" fromto="0 0 0.07 0 0 {h}"
            size="0.05" mass="3.0" rgba="0.3 0.5 0.9 1"/>
      <geom name="bob" type="sphere" pos="0 0 {h}" size="0.10"
            mass="{M_POLE - 4.0}" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>"""


def run(h, axis=1, rate=5.0, f_max=1000.0, max_t=15.0):
    """Push horizontally (y if axis=1) at COM height h; return F_tip."""
    m = mujoco.MjModel.from_xml_string(_xml(h))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pend")
    q0 = d.xquat[bid].copy()

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, h * 0.55]
    cam.distance, cam.azimuth, cam.elevation = 2.0 + h, 90, -12
    w, hgt = 480, 360
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{hgt}", "-r", "30", "-i", "-",
         "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         str(OUT / f"pend_h{h}.mp4")], stdin=subprocess.PIPE)

    F, t, tip = 0.0, 0.0, None
    force = np.zeros(3)
    force[axis] = 1.0
    log = []
    with mujoco.Renderer(m, hgt, w) as rd:
        while t < max_t:
            F = min(rate * t, f_max)
            d.xfrc_applied[bid, :3] = force * (F * h / max(h, 1e-6))
            # apply at com: use world torque-free force at body com —
            # xfrc acts at body com by default, scale to keep M=F*h
            d.xfrc_applied[bid, 3:6] = 0.0
            mujoco.mj_step(m, d)
            # NOTE: xfrc acts at body com; to emulate a push AT height
            # h above ground we also need the lever — body com is ~h,
            # so F_eff at com already gives M = F*h about the ground
            t += m.opt.timestep
            if int(t * 30) % 3 == 0:
                rd.update_scene(d, camera=cam)
                fr = rd.render()
                try:
                    from PIL import Image, ImageDraw
                    img = Image.fromarray(fr)
                    dr = ImageDraw.Draw(img)
                    dr.text((10, hgt - 24), f"h={h}m F={F:.0f}N",
                            fill=(20, 20, 20), stroke_width=2,
                            stroke_fill=(255, 255, 255))
                    fr = np.asarray(img)
                except ImportError:
                    pass
                ff.stdin.write(fr.tobytes())
            q = d.xquat[bid]
            conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
            dq = np.empty(4)
            mujoco.mju_mulQuat(dq, q, conj)
            tilt = 2 * np.degrees(np.arccos(np.clip(abs(dq[0]), 0, 1)))
            # CoP along the push axis: force-weighted contact pos
            num = den = 0.0
            f6 = np.empty(6)
            ymin = np.inf
            for i in range(d.ncon):
                mujoco.mj_contactForce(m, d, i, f6)
                num += f6[0] * d.contact[i].pos[axis]
                den += f6[0]
                ymin = min(ymin, d.contact[i].pos[axis])
            cop = num / den if den > 1e-9 else 0.0
            if int(t * 500) % 10 == 0:
                log.append({"t": t, "F": F, "tilt": tilt,
                            "cop": cop, "ncon": d.ncon,
                            "com_z": float(d.subtree_com[1][2])})
            # real tip: all contacts clustered past 85% of margin on
            # the push side, or gross tilt
            if (den > 1e-9 and cop > 0.85 * MARGIN) or tilt > 12:
                tip = F
                break
    ff.stdin.close()
    ff.wait()
    json.dump({"h": h, "F_tip": tip, "theory_F": M_POLE * G * MARGIN / h,
               "log": log}, open(OUT / f"pend_h{h}.json", "w"))
    mom = tip * 0.94 * h if tip else None
    print(f"h={h}: F_tip={tip}  M={mom and round(mom,1)}Nm"
          f"  theory_F={M_POLE*G*MARGIN/h:.1f}N"
          f"  theory_M={M_POLE*G*MARGIN:.1f}Nm")
    return tip


def settle_test(h, F, axis=1, hold=1.5):
    """Apply constant F at com; return True if the body tips."""
    m = mujoco.MjModel.from_xml_string(_xml(h))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pend")
    q0 = d.xquat[bid].copy()
    ramp = 0.5
    t = 0.0
    while t < hold:
        f = F if t > ramp else F * t / ramp
        d.xfrc_applied[bid, :3] = 0.0
        d.xfrc_applied[bid, axis] = f
        d.xfrc_applied[bid, 3:6] = 0.0
        mujoco.mj_step(m, d)
        t += m.opt.timestep
    q = d.xquat[bid]
    conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
    dq = np.empty(4)
    mujoco.mju_mulQuat(dq, q, conj)
    tilt = 2 * np.degrees(np.arccos(np.clip(abs(dq[0]), 0, 1)))
    return tilt > 8.0, tilt


def bisect(h, lo=1.0, hi=200.0, iters=9):
    for _ in range(iters):
        mid = (lo + hi) / 2
        tipped, _ = settle_test(h, mid)
        if tipped:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    import sys
    if "--bisect" in sys.argv:
        for h in (0.5, 0.9, 1.3):
            f = bisect(h)
            print(f"h={h}: F_tip~{f:.1f}N  M={f*h:.1f}Nm"
                  f"  theory_M={M_POLE*G*MARGIN:.1f}Nm"
                  f"  theory_F={M_POLE*G*MARGIN/h:.1f}N")
    else:
        for h in (0.5, 0.9, 1.3):
            run(h)
