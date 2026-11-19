"""步令链平面动画：调度器采样 + 运动学落地 + 墙约束——验证采样逻辑。

场地 = battle_circular_v2 的 24 边形墙（内切半径 ~3.41m）。
机器人 = 两个脚点(L蓝/R橙) + 一条朝向线。每步 0.5s：
动脚插值落到目标槽位（声明式：以 stance 脚锚定新朝向框架），
yaw 同步转 turn·30°。撞墙预测→collision_fn 拒绝重采→踏兜底。

    PYTHONPATH=. python3 -m baseline.humanoid21.v1.gait_walk_anim \
        [--steps 80] [--seed 0] [--fps 10] [--out /tmp/gait_walk.gif]
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from baseline.humanoid21.v1.step_gait import (
    FLAT, GaitPosture, StepOrder, order_name,
    TURN_RAD, apply_order, foot_slot,
)
from baseline.humanoid21.v1.step_scheduler import (
    StepScheduler, SchedState,
)

# ---------------------------------------------------------------------------
# 场地：24 边形墙（内切半径 3.4106 取自 XML wall_00 中点）
# ---------------------------------------------------------------------------
N_SEG = 24
APOTHEM = float(np.hypot(3.3814, 0.4452))            # 内切圆半径
CIRC = APOTHEM / math.cos(math.pi / N_SEG)           # 外接半径 ≈3.44
_VERT_ANGLES = np.linspace(0, 2 * math.pi, N_SEG, endpoint=False)
WALL_VERTS = CIRC * np.stack([np.cos(_VERT_ANGLES),
                              np.sin(_VERT_ANGLES)], axis=1)
# 边中点外法线方向（顶点角 + π/24）
_EDGE = _VERT_ANGLES + math.pi / N_SEG
WALL_N = np.stack([np.cos(_EDGE), np.sin(_EDGE)], axis=1)
MARGIN = 0.30            # 脚点距墙安全边距 m


def inside_arena(p: np.ndarray, margin: float = MARGIN) -> bool:
    """24 边形凸域判定：p·n_i ≤ apothem−margin ∀i。"""
    return bool(np.all(WALL_N @ p <= APOTHEM - margin))


# ---------------------------------------------------------------------------
# 平面运动学（声明式步令的世界系执行模型）
# ---------------------------------------------------------------------------


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class GaitWalker:
    """状态：yaw + 左右脚世界坐标 + 调度上下文。"""

    def __init__(self, pos: np.ndarray, yaw: float,
                 posture: GaitPosture = FLAT):
        self.yaw = float(yaw)
        self.sched = SchedState.initial(posture)
        R = _rot(yaw)
        self.foot_l = pos + R @ foot_slot(posture, -1)
        self.foot_r = pos + R @ foot_slot(posture, +1)

    @property
    def mid(self) -> np.ndarray:
        return (self.foot_l + self.foot_r) / 2

    def target_foot_pos(self, o: StepOrder) -> np.ndarray:
        """动脚落地目标世界位（step_gait.apply_order 的动脚分量）。"""
        fl, fr, _ = apply_order(self.foot_l, self.foot_r, self.yaw, o)
        return fl if o.foot < 0 else fr

    def apply(self, o: StepOrder) -> None:
        self.foot_l, self.foot_r, self.yaw = apply_order(
            self.foot_l, self.foot_r, self.yaw, o)
        self.sched = StepScheduler.advance(self.sched, o)


# ---------------------------------------------------------------------------
# 主流程：采样 → 帧插值 → GIF
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--frames-per-step", type=int, default=5)
    ap.add_argument("--out", default="/tmp/gait_walk.gif")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 1)
    sch = StepScheduler(seed=args.seed)
    w = GaitWalker(pos=rng.uniform(-1.5, 1.5, 2),
                   yaw=float(rng.uniform(0, 2 * math.pi)))

    def collides(o: StepOrder, st: SchedState) -> bool:
        return not inside_arena(w.target_foot_pos(o))

    frames = []          # (fl,fr,yaw,swing,txt,step_i)
    mid_hist = [w.mid.copy()]

    for i in range(args.steps):
        o = sch.sample(w.sched, collision_fn=collides)
        fl0, fr0, yaw0 = w.foot_l.copy(), w.foot_r.copy(), w.yaw
        tgt = w.target_foot_pos(o)
        w.apply(o)
        txt = f"#{i:02d} {order_name(o):>8s}  →{o.target.name} " \
              f"yaw={math.degrees(w.yaw) % 360:6.1f}"
        print(txt)
        mid_hist.append(w.mid.copy())
        for k in range(args.frames_per_step):
            ph = (k + 1) / args.frames_per_step
            s_ = ph * ph * (3 - 2 * ph)            # smoothstep
            fl, fr = fl0.copy(), fr0.copy()
            if o.foot < 0:
                fl = fl0 + (tgt - fl0) * s_
            else:
                fr = fr0 + (tgt - fr0) * s_
            yaw = yaw0 + o.turn * TURN_RAD * s_
            swing = o.foot if ph < 0.8 else 0
            frames.append((fl.copy(), fr.copy(), yaw, swing, txt, i))

    _render(frames, np.vstack(mid_hist), args.out, args.fps)
    print(f"\nGIF → {args.out}  ({len(frames)} frames)")


def _render(frames, mid_hist, out, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
               "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = \
                font_manager.FontProperties(fname=fp).get_name()
            break

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    ax.set_aspect("equal")
    ax.set_xlim(-3.9, 3.9)
    ax.set_ylim(-3.9, 3.9)
    ax.axis("off")
    ax.add_patch(Polygon(WALL_VERTS, closed=True,
                         fill=False, ec="0.3", lw=1.5))
    heading, = ax.plot([], [], "-", c="0.2", lw=2, zorder=3)
    trail, = ax.plot([], [], ".", ms=3, c="0.72", zorder=1)
    pt_l, = ax.plot([], [], "o", ms=11, c="#2b6cb0", zorder=4)
    pt_r, = ax.plot([], [], "o", ms=11, c="#dd6b20", zorder=4)
    lab_l = ax.text(0, 0, "L", ha="center", va="center",
                    fontsize=7, color="w", zorder=5)
    lab_r = ax.text(0, 0, "R", ha="center", va="center",
                    fontsize=7, color="w", zorder=5)
    info = ax.text(-3.8, 3.72, "", fontsize=8.5, va="top")

    def draw(f):
        fl, fr, yaw, swing, txt, si = f
        mid = (fl + fr) / 2
        hd = mid + 0.55 * np.array([math.cos(yaw), math.sin(yaw)])
        heading.set_data([mid[0], hd[0]], [mid[1], hd[1]])
        trail.set_data(mid_hist[:si + 2, 0], mid_hist[:si + 2, 1])
        pt_l.set_data([fl[0]], [fl[1]])
        pt_r.set_data([fr[0]], [fr[1]])
        pt_l.set_alpha(0.4 if swing < 0 else 1.0)
        pt_r.set_alpha(0.4 if swing > 0 else 1.0)
        lab_l.set_position(fl)
        lab_r.set_position(fr)
        info.set_text(txt)
        return heading, pt_l, pt_r, trail, info

    ani = FuncAnimation(fig, draw, frames=frames,
                        interval=1000 // fps, blit=False)
    ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    main()
