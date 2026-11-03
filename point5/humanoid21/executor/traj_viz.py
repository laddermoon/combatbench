"""Tool 5: CentroidalTrajectory -> 3D MP4 animation.

A ball follows com c(t); linear momentum shows as motion + an arrow.
Three surface anchor markers make spin visible: the sphere integrates
w = L / I_REF about the world L axis, so angular momentum reads as
rotation.  A fading trail keeps the recent past; a vertical drop line
+ ground shadow show height; the whole planned path is a faint guide.

    render_traj(traj_json, out_mp4)

Caller supplies both paths.  Fixed style — nothing user-adjustable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from .centroidal import CTRL_DT, load_traj

I_REF = 0.53            # kg·m^2 nominal inertia (standing Izz)
_TRAIL = 40             # steps kept in the fading trail
_FPS = 20
_BALL_R = 0.07


def _sphere_pts(c, Rm, r=_BALL_R, n=16):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    x = r * np.outer(np.sin(u), np.cos(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.cos(u), np.ones_like(v))
    pts = np.stack([x, y, z], -1) @ Rm.T + np.asarray(c)
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _cap_pts(c, Rm, direction, r=_BALL_R * 1.005, cap_deg=28, n=8):
    """A small spherical cap patch ON the ball surface around direction."""
    d = np.asarray(Rm @ np.asarray(direction, np.float64))
    d = d / np.linalg.norm(d)
    # local basis around d
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u1 = np.cross(d, a); u1 /= np.linalg.norm(u1)
    u2 = np.cross(d, u1)
    th = np.linspace(0, np.radians(cap_deg), 3)
    ph = np.linspace(0, 2 * np.pi, n)
    TH, PH = np.meshgrid(th, ph)
    dirs = (np.sin(TH) * np.cos(PH))[..., None] * u1 + \
           (np.sin(TH) * np.sin(PH))[..., None] * u2 + \
           (np.cos(TH))[..., None] * d
    pts = np.asarray(c) + r * dirs
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _rot(axis, ang):
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _style(ax, xr, yr, zr):
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    ax.set_box_aspect((xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]))
    ax.view_init(elev=16, azim=-125)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor((0, 0, 0, 0.08))
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3, prune="both"))
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel("x fwd", fontsize=9)
    ax.set_ylabel("y left", fontsize=9)
    ax.set_zlabel("z", fontsize=9)


def render_traj(traj_json: str, out_mp4: str) -> Path:
    """Render a saved CentroidalTrajectory JSON to MP4."""
    traj = load_traj(traj_json)
    states = traj.states
    cs = np.array([s.com for s in states])
    Ls = np.array([s.ang_mom for s in states])

    # tight view box around the path (+ margins)
    m = 0.25
    xr = (min(cs[:, 0].min(), 0) - m, max(cs[:, 0].max(), 0.1) + m)
    yr = (min(cs[:, 1].min(), -0.1) - m, max(cs[:, 1].max(), 0.1) + m)
    zr = (0.0, max(1.05, cs[:, 2].max() + 0.15))

    fig = plt.figure(figsize=(9, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    marker_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    marker_colors = ["red", "green", "blue"]
    R_orient = np.eye(3)

    writer = FFMpegWriter(fps=_FPS)
    out = Path(out_mp4)
    with writer.saving(fig, str(out), dpi=120):
        for k in range(len(states)):
            L = Ls[k]
            w = np.linalg.norm(L) / I_REF
            if w > 1e-9:
                R_orient = _rot(L, w * CTRL_DT) @ R_orient

            ax.cla()
            _style(ax, xr, yr, zr)

            # full plan as faint guide
            ax.plot(cs[:, 0], cs[:, 1], cs[:, 2], ":",
                    color="gray", alpha=0.5, linewidth=1.0)
            # goal marker
            ax.scatter(*cs[-1], marker="*", s=140, color="gold",
                       edgecolor="darkgoldenrod")

            # fading trail (line + dots)
            lo = max(0, k - _TRAIL)
            seg = cs[lo:k + 1]
            for i in range(len(seg) - 1):
                a = (i + 1) / max(len(seg), 1)
                ax.plot(seg[i:i + 2, 0], seg[i:i + 2, 1],
                        seg[i:i + 2, 2], color="tab:blue",
                        alpha=a * 0.9, linewidth=2.0)
            ax.scatter(seg[:-1, 0], seg[:-1, 1], seg[:-1, 2],
                       s=10, color="tab:blue", alpha=0.35)

            # height drop line + ground shadow
            c = cs[k]
            ax.plot([c[0], c[0]], [c[1], c[1]], [0, c[2]],
                    color="black", alpha=0.35, linewidth=1.0,
                    linestyle="--")
            ax.scatter(c[0], c[1], 0.001, s=60, color="black",
                       alpha=0.35, marker="o")

            # solid ball with surface anchor dots (rotation = L)
            xs, ys, zs = _sphere_pts(c, R_orient)
            ax.plot_surface(xs, ys, zs, color="tab:blue",
                            alpha=0.85, linewidth=0,
                            antialiased=True, shade=True)
            # spots painted ON the ball surface (rotation = L) —
            # a dense disc of tiny points reads as a painted patch
            # (mpl3d sorts surfaces wrongly; scatter draws on top)
            for dvec, col in zip(marker_dirs, marker_colors):
                px, py, pz = _cap_pts(c, R_orient, dvec,
                                      r=_BALL_R * 1.02,
                                      cap_deg=30, n=14)
                ax.scatter(px.ravel(), py.ravel(), pz.ravel(),
                           s=14, color=col, depthshade=False)

            ax.text(c[0], c[1], c[2] + 0.12, f"z={c[2]:.2f}m",
                    fontsize=9, ha="center")
            ax.set_title(f"centroidal traj  t={k * CTRL_DT:.2f}s "
                         f"({k + 1}/{len(states)})", fontsize=10)
            writer.grab_frame()
    plt.close(fig)
    return out
