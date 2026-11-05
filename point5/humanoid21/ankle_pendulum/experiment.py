"""Ankle-pendulum torque-limit experiments.

Inverted-pendulum model: the body is held rigid by a root "gantry"
(external wrench servo on the root body) so the only free joint is
the ankle — the system pivots at the ankle exactly like the model.
An external force ramps on a chosen body; per step we log ankle
actuator torques (saturates at gear=20 -> 20 N m), foot tilt vs the
settled orientation, per-foot CoP + normal force, and the COM.

Two thresholds per case:
  * |ankle torque| hits 20 N m  -> ACTUATOR limit
  * support-foot tilt > 4 deg   -> SUPPORT (CoP margin) limit

Outputs (MP4 + JSON) land in /data1/dev/ankle_pendulum/.

    PYTHONPATH=. python3 -B point5/humanoid21/ankle_pendulum/experiment.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator

OUT = Path("/data1/dev/ankle_pendulum")
OUT.mkdir(parents=True, exist_ok=True)

_SUB = 25                 # physics substeps per control step
_FPS = 25
_SIZE = (640, 480)
_TAU_ACT = 20.0           # N m  (gear 20 x ctrl range 1)

_ANKLE_J = ["ankle_y_right_a", "ankle_x_right_a",
            "ankle_y_left_a", "ankle_x_left_a"]

# gantry servo gains on the root body (torso_a)
_KP_P, _KD_P = 1200.0, 200.0      # N/m, N s/m
_KP_R, _KD_R = 400.0, 80.0      # N m/rad, N m s/rad


class Rig:
    """Simulator wrapper: PD joint hold + root gantry + force ramp."""

    def __init__(self, pose: str):
        self.sim = Humanoid21Simulator(
            initial_distance=6.0, initial_pose_a=pose,
            initial_pose_b="standing")
        self.sim.reset(seed=0)
        m, d = self.sim.model, self.sim.data
        self.m, self.d = m, d
        self.a_hold = self.sim.INITIAL_POSES[pose]["action"].copy()
        self.a_b = self.sim.INITIAL_POSES["standing"]["action"].copy()
        self.ankle_dof = [
            int(m.jnt_dofadr[mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in _ANKLE_J]
        self.bid = {n: int(mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, n)) for n in
            ("torso_a", "pelvis_a", "head_a",
             "foot_right_a", "foot_left_a")}
        self.foot_geoms = {
            "right": {int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                      f"foot{i}_right_a")) for i in (1, 2)},
            "left": {int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                     f"foot{i}_left_a")) for i in (1, 2)}}
        self.foot_q0 = None
        self.ankle_xy = {}               # side -> ankle pivot xy (after settle)
        self.ankle_z = 0.05              # joint height over foot frame
        self.root_bid = self.bid["torso_a"]
        self.root_p0 = d.xpos[self.root_bid].copy()
        self.root_q0 = d.xquat[self.root_bid].copy()

    def _gantry(self):
        """Hold root body at its initial pose via external wrench."""
        d, bid = self.d, self.root_bid
        err_p = self.root_p0 - d.xpos[bid]
        q = d.xquat[bid]
        q0 = self.root_q0
        # relative rotation q0^-1 * q -> axis-angle (small-angle vec)
        conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        dq = np.empty(4)
        mujoco.mju_mulQuat(dq, conj, q)
        dq = dq / (np.linalg.norm(dq) + 1e-12)
        s = 1.0 if dq[0] >= 0 else -1.0
        aa = s * dq[1:] * 2.0
        cv = d.cvel[bid]
        F = _KP_P * err_p - _KD_P * cv[3:6]
        T = -_KP_R * aa - _KD_R * cv[0:3]
        d.xfrc_applied[bid, :3] += F
        d.xfrc_applied[bid, 3:6] += T

    def step_hold(self, force_body=None, force=None):
        self.sim.set_action({"robot_a": self.a_hold,
                             "robot_b": self.a_b})
        for _ in range(_SUB):
            if force_body is not None and force is not None:
                self.sim.apply_external_force(
                    force_body, force, robot_id="robot_a")
            self._gantry()
            self.sim.physical_step()

    def settle(self, steps=40):
        for _ in range(steps):
            self.step_hold()
        d = self.d
        self.foot_q0 = {s: d.xquat[self.bid[f"foot_{s}_a"]].copy()
                        for s in ("right", "left")}
        self.ankle_xy = {s: d.xpos[self.bid[f"foot_{s}_a"], :2].copy()
                        for s in ("right", "left")}
        self.root_p0 = d.xpos[self.root_bid].copy()
        self.root_q0 = d.xquat[self.root_bid].copy()

    def com(self):
        return np.array(self.d.subtree_com[self.root_bid])

    def foot_cop(self, side):
        gset = self.foot_geoms[side]
        num, den = np.zeros(3), 0.0
        f6 = np.empty(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 in gset or c.geom2 in gset:
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                num += f6[0] * np.asarray(c.pos)
                den += f6[0]
        return (num / den if den > 1e-9 else None), den

    def foot_tilt(self, side):
        q = self.d.xquat[self.bid[f"foot_{side}_a"]]
        q0 = self.foot_q0[side]
        conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        dq = np.empty(4)
        mujoco.mju_mulQuat(dq, q, conj)
        w = np.clip(abs(dq[0]), 0, 1)
        return 2 * np.degrees(np.arccos(w))

    def measure(self, F, app_body, direction):
        com = self.com()
        out = {"F": F, "com": com.tolist(),
               "h_app": float(self.d.xpos[app_body, 2])}
        for i, j in enumerate(_ANKLE_J):
            out[j] = float(self.d.qfrc_actuator[self.ankle_dof[i]])
        for s in ("right", "left"):
            cop, n = self.foot_cop(s)
            ank = self.d.xpos[self.bid[f"foot_{s}_a"], :2]
            # moment about the ankle axis along the push direction:
            #  F*(h_app-h_ankle) + m g (com_xy - ankle_xy) . dir
            grav = 40.844 * 9.81 * float(
                np.dot(com[:2] - ank, direction[:2]))
            out[f"M_{s}"] = (F * (out["h_app"] - self.ankle_z)
                             + grav)
            out[f"cop_off_{s}"] = (float(np.dot(cop[:2] - ank,
                                                direction[:2]))
                                   if cop is not None else None)
            out[f"N_{s}"] = n
            out[f"tilt_{s}"] = self.foot_tilt(s)
        return out


def _ffmpeg(path, w, h, fps=_FPS):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         str(path)], stdin=subprocess.PIPE)


def run_torque_case(name, pose, feet, axis, t_rate, t_max,
                    settle=40, max_t=8.0, cam_az=90):
    """Ramp an external torque (Nm) about `axis` on the foot body(ies).
    T_tip = the moment at which the foot rolls over its edge —
    the direct measure of the support limit N*margin."""
    rig = Rig(pose)
    rig.settle(settle)
    m, d = rig.m, rig.d
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)

    cam = mujoco.MjvCamera()
    com0 = rig.com()
    cam.lookat[:] = [com0[0], com0[1], 0.7]
    cam.distance, cam.azimuth, cam.elevation = 2.6, cam_az, -15

    ff = _ffmpeg(OUT / f"{name}.mp4", *_SIZE)
    log = []
    t = 0.0
    tip = None
    dt = _SUB * m.opt.timestep
    with mujoco.Renderer(m, _SIZE[1], _SIZE[0]) as rd:
        while t < max_t:
            T = min(t_rate * t, t_max)
            per = T / len(feet)
            rig.sim.set_action({"robot_a": rig.a_hold,
                                "robot_b": rig.a_b})
            for _ in range(_SUB):
                for fb in feet:
                    rig.sim.apply_external_force(
                        fb, np.zeros(3), torque=axis * per,
                        robot_id="robot_a")
                rig._gantry()
                rig.sim.physical_step()
            t += dt
            tip_dir = np.array([-axis[1], axis[0], 0.0])
            rec = rig.measure(0.0, rig.bid["pelvis_a"], tip_dir)
            rec["t"], rec["T"] = t, T
            log.append(rec)

            rd.update_scene(d, camera=cam)
            fr = rd.render()
            try:
                from PIL import Image, ImageDraw
                img = Image.fromarray(fr)
                dr = ImageDraw.Draw(img)
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/"
                        "DejaVuSans.ttf", 16)
                except Exception:
                    font = ImageFont.load_default()
                dr.text((12, _SIZE[1] - 28),
                        f"{name} t={t:.2f}s T={T:.1f}Nm"
                        f" tiltR={rec['tilt_right']:.1f}"
                        f" tiltL={rec['tilt_left']:.1f}",
                        fill=(20, 20, 20), font=font, stroke_width=2,
                        stroke_fill=(255, 255, 255))
                fr = np.asarray(img)
            except ImportError:
                pass
            ff.stdin.write(fr.tobytes())
            if rec["tilt_right"] > 7.0 or rec["tilt_left"] > 7.0:
                tip = rec
                break
    ff.stdin.close()
    ff.wait()
    res = {"name": name, "pose": pose, "feet": feet,
           "axis": axis.tolist(), "tip": tip, "log": log}
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(res, f)
    print(f"[{name}]"
          + (f" T_tip={tip['T']:.1f}Nm" if tip
             else f" no tip <= {t_max}Nm"))
    return res


def run_case(name, pose, app_body_name, direction, f_rate,
             f_max, settle=40, max_t=8.0, cam_az=90):
    rig = Rig(pose)
    rig.settle(settle)
    m, d = rig.m, rig.d
    app = rig.bid[app_body_name]
    direction = np.asarray(direction, np.float64)
    direction = direction / np.linalg.norm(direction)
    app_short = app_body_name[:-2]          # api adds the '_a' suffix

    cam = mujoco.MjvCamera()
    com0 = rig.com()
    cam.lookat[:] = [com0[0], com0[1], 0.7]
    cam.distance, cam.azimuth, cam.elevation = 2.6, cam_az, -15

    ff = _ffmpeg(OUT / f"{name}.mp4", *_SIZE)
    log = []
    t = 0.0
    events = {"act": None, "tip": None}
    dt = _SUB * m.opt.timestep
    with mujoco.Renderer(m, _SIZE[1], _SIZE[0]) as rd:
        while t < max_t:
            F = min(f_rate * t, f_max)
            rig.step_hold(app_short, direction * F)
            t += dt
            rec = rig.measure(F, app, direction)
            rec["t"] = t
            log.append(rec)

            rd.update_scene(d, camera=cam)
            fr = rd.render()
            try:
                from PIL import Image, ImageDraw
                img = Image.fromarray(fr)
                dr = ImageDraw.Draw(img)
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/"
                        "DejaVuSans.ttf", 16)
                except Exception:
                    font = ImageFont.load_default()
                dr.text((12, _SIZE[1] - 28),
                        f"{name} t={t:.2f}s F={F:.1f}N"
                        f" tiltR={rec['tilt_right']:.1f}"
                        f" tiltL={rec['tilt_left']:.1f}",
                        fill=(20, 20, 20), font=font, stroke_width=2,
                        stroke_fill=(255, 255, 255))
                fr = np.asarray(img)
            except ImportError:
                pass
            ff.stdin.write(fr.tobytes())

            tau_max = max(abs(rec[j]) for j in _ANKLE_J)
            if events["act"] is None and tau_max >= _TAU_ACT - 0.5:
                events["act"] = rec
            cop_edge = any(
                rec[f"cop_off_{s}"] is not None
                and rec[f"cop_off_{s}"] > 0.044
                for s in ("right", "left"))
            if ((rec["tilt_right"] > 7.0 or rec["tilt_left"] > 7.0
                    or cop_edge) or rec["com"][2] < 0.4):
                events["tip"] = rec
                break
    ff.stdin.close()
    ff.wait()

    res = {"name": name, "pose": pose, "app_body": app_body_name,
           "dir": direction.tolist(), "f_rate": f_rate,
           "events": events, "log": log}
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(res, f)
    a, tp = events["act"], events["tip"]
    print(f"[{name}]"
          + (f" act@F={a['F']:.1f}N" if a else " act: -")
          + (f"  tip@F={tp['F']:.1f}N h={tp['h_app']:.2f}m"
             if tp else f"  no tip <= {f_max}N"))
    return res


CASES = [
    ("single_roll", "stand_on_left_leg", "pelvis_a",
     (0, 1, 0), 8, 80, 0),
    ("single_pitch_fwd", "stand_on_left_leg", "pelvis_a",
     (1, 0, 0), 15, 120, 90),
    ("double_roll", "standing", "pelvis_a", (0, 1, 0), 15, 150, 0),
    ("double_pitch_fwd", "standing", "pelvis_a", (1, 0, 0), 25, 200, 90),
]

TORQUE_CASES = [
    # name, pose, [foot bodies], world axis, rate Nm/s, cap Nm, azimuth
    ("tq_single_roll", "stand_on_left_leg", ["foot_left"],
     (1, 0, 0), 15, 60, 0),
    ("tq_single_pitch", "stand_on_left_leg", ["foot_left"],
     (0, 1, 0), 25, 120, 90),
    ("tq_double_roll", "standing", ["foot_left", "foot_right"],
     (1, 0, 0), 25, 150, 0),
]


def main():
    for c in TORQUE_CASES:
        run_torque_case(c[0], c[1], c[2], c[3], c[4], c[5], cam_az=c[6])
    for c in CASES:
        run_case(c[0], c[1], c[2], c[3], c[4], c[5], cam_az=c[6])


if __name__ == "__main__":
    main()
