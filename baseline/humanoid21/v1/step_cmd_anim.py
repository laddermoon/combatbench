"""V1 步令生成验证动画：step_cmd_gen 连续采样 + 运动学落地 + 墙约束。

与 gait_walk_anim 的差别：指令来自连续目标采样器（D20 残差编码的
指令源侧），并叠加显示——动脚目标落点(x 标记) + 残差箭头(踝→目标)，
信息栏直接给出 obs 将写入的本体系残差 (Δx, Δy, Δψ)。

    PYTHONPATH=. python3 -m baseline.humanoid21.v1.step_cmd_anim \
        [--steps 60] [--seed 0] [--fps 10] [--out /tmp/step_cmd.gif]
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from baseline.humanoid21.v1.arena_geom import (
    NORMALS as WALL_N, VERTS as WALL_VERTS,
    WALL_MARGIN as MARGIN, inside_arena,
)
from baseline.humanoid21.v1.step_cmd_gen import (
    StepCmd, cmd_target, residual, sample_cmd,
)
from baseline.humanoid21.v1.step_gait import WIDTH_N, foot_slot, FLAT


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class CmdWalker:
    """状态：yaw + 左右脚世界坐标 + 上一动脚（供同/异脚采样）。"""

    def __init__(self, pos: np.ndarray, yaw: float):
        self.yaw = float(yaw)
        self.last_foot = 0
        R = _rot(yaw)
        self.foot_l = pos + R @ foot_slot(FLAT, -1)
        self.foot_r = pos + R @ foot_slot(FLAT, +1)

    @property
    def mid(self) -> np.ndarray:
        return (self.foot_l + self.foot_r) / 2

    def cmd_tgt_xy(self, cmd: StepCmd) -> np.ndarray:
        tgt, _ = cmd_target(self.foot_l, self.foot_r, self.yaw, cmd)
        return tgt if tgt is not None else self.mid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--frames-per-step", type=int, default=10)
    ap.add_argument("--out", default="/tmp/step_cmd.gif")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 1)
    w = CmdWalker(pos=rng.uniform(-1.5, 1.5, 2),
                  yaw=float(rng.uniform(0, 2 * math.pi)))

    def collides(cmd: StepCmd) -> bool:
        return not inside_arena(w.cmd_tgt_xy(cmd))

    frames = []
    mid_hist = [w.mid.copy()]

    for i in range(args.steps):
        cmd = sample_cmd(rng, w.last_foot, collide_fn=collides)
        tgt, yaw_t = cmd_target(w.foot_l, w.foot_r, w.yaw, cmd)
        fl0, fr0, yaw0 = w.foot_l.copy(), w.foot_r.copy(), w.yaw

        fname = {0: "HOLD", -1: "L", +1: "R"}[cmd.foot]
        print(f"#{i:02d} {fname:>4s} sag={cmd.sag:+.2f} lat={cmd.lat:.2f} "
              f"dyaw={math.degrees(cmd.dyaw):+5.1f}°"
              + ("" if tgt is None else
                 f"  tgt=({tgt[0]:+.2f},{tgt[1]:+.2f})"))

        for k in range(args.frames_per_step):
            ph = (k + 1) / args.frames_per_step
            s_ = ph * ph * (3 - 2 * ph)
            fl, fr = fl0.copy(), fr0.copy()
            if cmd.foot < 0:
                fl = fl0 + (tgt - fl0) * s_
            elif cmd.foot > 0:
                fr = fr0 + (tgt - fr0) * s_
            yaw = yaw0 + cmd.dyaw * s_
            ankle = fl if cmd.foot < 0 else fr
            dx, dy, dpsi = (0.0, 0.0, 0.0) if tgt is None else \
                residual(ankle, yaw, tgt, yaw_t)
            txt = (f"#{i:02d} {fname:>4s}  Δx={dx:+.2f} Δy={dy:+.2f} "
                   f"Δψ={math.degrees(dpsi):+5.1f}°")
            frames.append((fl.copy(), fr.copy(), yaw,
                           cmd.foot if ph < 0.8 else 0,
                           tgt, ankle.copy(), txt, i))

        # 令末落定
        if tgt is not None:
            if cmd.foot < 0:
                w.foot_l = tgt
            else:
                w.foot_r = tgt
        w.yaw = yaw_t
        if cmd.foot != 0:
            w.last_foot = cmd.foot
        mid_hist.append(w.mid.copy())

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
    tgt_mk, = ax.plot([], [], "x", ms=12, mew=2.5, c="#c0392b", zorder=4)
    arrow, = ax.plot([], [], "-", c="#c0392b", lw=1.2,
                     alpha=0.8, zorder=3)
    pt_l, = ax.plot([], [], "o", ms=11, c="#2b6cb0", zorder=4)
    pt_r, = ax.plot([], [], "o", ms=11, c="#dd6b20", zorder=4)
    lab_l = ax.text(0, 0, "L", ha="center", va="center",
                    fontsize=7, color="w", zorder=5)
    lab_r = ax.text(0, 0, "R", ha="center", va="center",
                    fontsize=7, color="w", zorder=5)
    info = ax.text(-3.8, 3.72, "", fontsize=8.5, va="top")

    def draw(f):
        fl, fr, yaw, swing, tgt, ankle, txt, si = f
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
        if tgt is not None:
            tgt_mk.set_data([tgt[0]], [tgt[1]])
            arrow.set_data([ankle[0], tgt[0]], [ankle[1], tgt[1]])
        else:
            tgt_mk.set_data([], [])
            arrow.set_data([], [])
        info.set_text(txt)
        return heading, pt_l, pt_r, trail, tgt_mk, arrow, info

    ani = FuncAnimation(fig, draw, frames=frames,
                        interval=1000 // fps, blit=False)
    ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    main()
