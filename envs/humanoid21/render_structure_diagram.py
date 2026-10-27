"""Render robot_structure.png — humanoid21 anatomy diagram with dimensions.

Offscreen MuJoCo render (EGL, headless-safe) of robot_a in standing
pose, front + side views, with dimension callouts.  Run:

    MUJOCO_GL=egl PYTHONPATH=. python3 envs/humanoid21/render_structure_diagram.py

Output: envs/humanoid21/robot_structure.png

All lengths are pulled live from the model (body frame origins =
joint anchors for this chain), so re-running after an XML edit keeps
the numbers honest.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

XML = Path(__file__).parent / "battle_circular_v2.xml"
OUT = Path(__file__).parent / "robot_structure.png"

# Camera params shared by both views (perspective, near-orthographic
# from ~4 m away so proportions are honest).
W, H = 900, 1080
LOOKAT = np.array([-1.0, 0.0, 0.82])
DIST = 2.4
FOVY = 45.0


def cam_frame(azimuth_deg: float):
    """MuJoCo free-camera convention → (cam_pos, forward, right, up)."""
    az = math.radians(azimuth_deg)
    fwd = np.array([math.cos(az), math.sin(az), 0.0])
    pos = LOOKAT - DIST * fwd
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up


def project(p, pos, fwd, right, up):
    """World point → pixel (x right, y down). Perspective pinhole."""
    rel = np.asarray(p, dtype=np.float64) - pos
    depth = float(rel @ fwd)
    if depth <= 0:
        return None
    f_px = (H / 2.0) / math.tan(math.radians(FOVY / 2))
    x = rel @ right
    y = rel @ up
    return np.array([W / 2 + f_px * x / depth, H / 2 - f_px * y / depth])


def render(azimuth_deg: float) -> np.ndarray:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    # Hide robot_b: sink its freejoint below the arena.
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root_b")
    adr = m.jnt_qposadr[jid]
    d.qpos[adr + 2] = -10.0
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = LOOKAT.tolist()
    cam.distance = DIST
    cam.azimuth = azimuth_deg
    cam.elevation = 0.0
    r.update_scene(d, camera=cam)
    img = r.render()
    r.close()
    return img, m, d


def _dim(ax, pa, pb, label, off=40, color="#ffd400", fs=15):
    """Engineering-style dimension line between two projected pixels,
    offset perpendicular by `off` px, with end ticks + center label."""
    pa = np.asarray(pa, float)
    pb = np.asarray(pb, float)
    v = pb - pa
    n = np.linalg.norm(v)
    if n < 1e-6:
        return
    perp = np.array([-v[1], v[0]]) / n * off
    a2, b2 = pa + perp, pb + perp
    # witness lines
    ax.plot([pa[0], a2[0]], [pa[1], a2[1]], color=color, lw=0.8, alpha=0.7)
    ax.plot([pb[0], b2[0]], [pb[1], b2[1]], color=color, lw=0.8, alpha=0.7)
    # main line with ticks
    ax.annotate("", xy=b2, xytext=a2,
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.6))
    mid = (a2 + b2) / 2
    ax.text(mid[0], mid[1], label, color="#111", fontsize=fs,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="none",
                      alpha=0.9))


def _callout(ax, pxy, label, dx=60, dy=-40, color="#ffd400", fs=14):
    ax.annotate(
        label, xy=pxy, xytext=(pxy[0] + dx, pxy[1] + dy),
        color="#111", fontsize=fs, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none",
                  alpha=0.9),
    )


def collect_points(d, m):
    """Key world positions, robot_a standing pose."""
    def bp(name):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return d.xpos[bid].copy()

    def jp(name):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return d.xanchor[jid].copy()

    return {
        "torso": bp("torso_a"),
        "head": bp("head_a"),
        "shoulder_r": bp("upper_arm_right_a"),
        "elbow_r": bp("lower_arm_right_a"),
        "hand_r": bp("hand_right_a"),
        "shoulder_l": bp("upper_arm_left_a"),
        "hand_l": bp("hand_left_a"),
        "hip_r": bp("thigh_right_a"),
        "knee_r": bp("shin_right_a"),
        "ankle_r": jp("ankle_x_right_a"),
        "foot_r": bp("foot_right_a"),
        "ground_z": 0.0,
    }


def main():
    front, m, d = render(azimuth_deg=0.0)    # looking +x at robot's face
    side, _, _ = render(azimuth_deg=90.0)    # looking +y at left profile

    P = collect_points(d, m)
    torso_z = P["torso"][2]
    head_top = P["head"][2] + 0.09
    ground_z = P["ground_z"]

    # measured segment lengths (m)
    L = {
        "height": head_top - ground_z,
        "upper_arm": np.linalg.norm(P["elbow_r"] - P["shoulder_r"]),
        "forearm": np.linalg.norm(P["hand_r"] - P["elbow_r"]),
        "thigh": np.linalg.norm(P["knee_r"] - P["hip_r"]),
        "shin": np.linalg.norm(P["ankle_r"] - P["knee_r"]),
        "ankle_h": P["ankle_r"][2] - ground_z,
        "shoulder_w": abs(P["shoulder_r"][1] - P["shoulder_l"][1]),
        "hand_fwd": P["hand_r"][0] - P["torso"][0],
        "knee_z": P["knee_r"][2] - ground_z,
    }

    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)
    for ax, img, title in (
        (axes[0], front, "Front view  (az 0°, +x forward)"),
        (axes[1], side, "Side view  (az 90°)"),
    ):
        ax.imshow(img)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8)

    # ---- projection closures per view ----
    fpos, ffwd, frt, fup = cam_frame(0.0)
    spos, sfwd, srt, sup = cam_frame(90.0)
    fp = lambda p: project(p, fpos, ffwd, frt, fup)   # noqa: E731
    sp = lambda p: project(p, spos, sfwd, srt, sup)   # noqa: E731

    # ============ FRONT view callouts ============
    ax = axes[0]
    # total height (right side of frame)
    _dim(ax, fp([-1, 0, head_top]), fp([-1, 0, ground_z]),
         f"height ≈ {L['height']:.2f} m", off=260)
    # shoulder span as a callout (dim line would collide with head/arm
    # labels in this tight crop)
    _callout(ax, fp(P["shoulder_l"]),
             f"shoulder span {L['shoulder_w']:.2f} m", dx=-310, dy=-60)
    # head
    _callout(ax, fp(P["head"] + [0.09, 0, 0.0]), "head r=0.09",
             dx=110, dy=-40)
    # torso
    _callout(ax, fp(P["torso"] + [0.0, 0.07, 0.02]),
             "torso capsule\nr=0.07, w=0.28", dx=130, dy=60)
    # arms (right side of image = robot's right arm, y<0)
    _dim(ax, fp(P["shoulder_r"]), fp(P["elbow_r"]),
         f"upper arm {L['upper_arm']:.2f}", off=150)
    _dim(ax, fp(P["elbow_r"]), fp(P["hand_r"]),
         f"forearm {L['forearm']:.2f}", off=150)
    _callout(ax, fp(P["hand_r"]), "fist sphere\nr=0.04", dx=150, dy=110)
    # legs
    _dim(ax, fp(P["hip_r"]), fp(P["knee_r"]),
         f"thigh {L['thigh']:.2f}", off=95)
    _dim(ax, fp(P["knee_r"]), fp(P["ankle_r"]),
         f"shin {L['shin']:.2f}", off=95)
    _callout(ax, fp(P["ankle_r"] + [0.06, 0, -0.02]),
             f"ankle h={L['ankle_h']:.3f}\nfoot ~0.21", dx=120, dy=50)

    # ============ SIDE view callouts ============
    ax = axes[1]
    # guard-pose fist lead
    _dim(ax, sp(P["torso"]), sp([P["hand_r"][0], 0, P["hand_r"][2]]),
         f"guard lead {L['hand_fwd']:.2f} m", off=-100)
    _callout(ax, sp(P["hand_r"]), "guard pose\nfist", dx=90, dy=-50)
    # height stack markers
    _callout(ax, sp([P["torso"][0], 0, torso_z]),
             f"torso z={torso_z:.2f}", dx=-200, dy=0)
    _callout(ax, sp(P["knee_r"]), f"knee z={L['knee_z']:.2f}",
             dx=-200, dy=30)
    # punch reach annotation (analytic, arm fully extended)
    reach = L["upper_arm"] + L["forearm"] + 0.04
    ax.text(
        30, H - 40,
        f"arm reach (shoulder→fist tip) = "
        f"{L['upper_arm']:.2f}+{L['forearm']:.2f}+0.04 ≈ {reach:.2f} m\n"
        f"hit range torso-center↔torso ≈ {reach + 0.07:.2f} m  "
        f"(D_STRIKE=0.7 in exp)",
        fontsize=13, fontweight="bold", color="#111",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffd400", ec="none",
                  alpha=0.9),
    )

    fig.suptitle(
        "humanoid21 — structure & dimensions  (battle_circular_v2.xml)",
        fontsize=20, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(OUT), dpi=100)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
