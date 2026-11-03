"""Tool 5: CentroidalTrajectory -> 3D MP4 animation.

A ball follows com c(t); linear momentum shows as motion.  Three
surface anchor markers make spin visible: the sphere integrates
w = L / I_REF about the world L axis, so angular momentum reads as
rotation.  A fading trail keeps the recent past; a label gives the
ball's height above the z=0 ground grid.

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa

from .centroidal import CTRL_DT, load_traj

I_REF = 0.53            # kg·m^2 nominal inertia (standing Izz)
_TRAIL = 40             # steps kept in the fading trail
_FPS = 20
_BALL_R = 0.05


def _sphere_pts(c, Rm, r=_BALL_R, n=14):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    x = r * np.outer(np.sin(u), np.cos(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.cos(u), np.ones_like(v))
    pts = np.stack([x, y, z], -1) @ Rm.T + np.asarray(c)
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _rot(axis, ang):
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def render_traj(traj_json: str, out_mp4: str) -> Path:
    """Render a saved CentroidalTrajectory JSON to MP4."""
    traj = load_traj(traj_json)
    states = traj.states
    cs = np.array([s.com for s in states])
    Ls = np.array([s.ang_mom for s in states])

    fig = plt.figure(figsize=(8, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    xr = (cs[:, 0].min() - 0.3, cs[:, 0].max() + 0.3)
    yr = (cs[:, 1].min() - 0.3, cs[:, 1].max() + 0.3)
    zr = (0.0, max(1.6, cs[:, 2].max() + 0.2))
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    ax.set_xlabel("x fwd"); ax.set_ylabel("y left"); ax.set_zlabel("z")
    # ground grid
    gx, gy = np.meshgrid(np.linspace(*xr, 5), np.linspace(*yr, 5))
    ax.plot_wireframe(gx, gy, np.zeros_like(gx), alpha=0.2,
                      color="gray", linewidths=0.5)

    marker_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    marker_colors = ["red", "green", "blue"]
    R_orient = np.eye(3)

    writer = FFMpegWriter(fps=_FPS)
    out = Path(out_mp4)
    with writer.saving(fig, str(out), dpi=110):
        for k in range(len(states)):
            # integrate spin about L
            L = Ls[k]
            w = np.linalg.norm(L) / I_REF
            if w > 1e-9:
                R_orient = _rot(L, w * CTRL_DT) @ R_orient

            for art in list(ax.collections):
                art.remove()
            for t in list(ax.texts):
                t.remove()

            # fading trail
            lo = max(0, k - _TRAIL)
            for i in range(lo, k):
                a = (i - lo + 1) / (k - lo + 1)
                ax.scatter(*cs[i], s=8, color="tab:blue", alpha=a * 0.8)

            # ball + surface anchors
            xs, ys, zs = _sphere_pts(cs[k], R_orient)
            ax.plot_wireframe(xs, ys, zs, color="tab:blue",
                              alpha=0.4, linewidths=0.4)
            for dvec, col in zip(marker_dirs, marker_colors):
                p = cs[k] + R_orient @ (dvec * _BALL_R * 1.15)
                ax.scatter(*p, s=25, color=col)

            # height label + momentum arrow
            ax.text(cs[k, 0], cs[k, 1], cs[k, 2] + 0.10,
                    f"z={cs[k, 2]:.2f}m", fontsize=8)
            v = states[k].com_vel
            if np.linalg.norm(v) > 1e-3:
                ax.quiver(*cs[k], *v, length=0.25, normalize=True,
                          color="orange", linewidth=1.5)

            ax.set_title(f"t={k * CTRL_DT:.2f}s")
            writer.grab_frame()
    plt.close(fig)
    return out
