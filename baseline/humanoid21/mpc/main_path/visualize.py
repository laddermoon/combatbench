"""Visualization helpers for generated support schedules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contracts import MainPathState, SupportInterval, SupportSchedule


FOOT_COLORS = {"left": "#1f77b4", "right": "#d62728"}
FOOT_LABELS = {"left": "L", "right": "R"}


def plot_support_schedule(
    schedule: SupportSchedule,
    state: MainPathState,
    target_pos_world: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Support schedule",
) -> Path:
    """Render foot placements and support timing to a PNG."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_map, ax_time) = plt.subplots(
        1, 2, figsize=(13, 5.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    _plot_map(ax_map, schedule, state, target_pos_world)
    _plot_timeline(ax_time, schedule, state)
    fig.suptitle(title)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plot_map(ax, schedule, state, target_pos_world) -> None:
    t0 = float(state.time)
    ax.set_title("Foot placements (top view)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    start = state.com_pos[:2]
    target = np.asarray(target_pos_world, dtype=np.float64)[:2]
    ax.scatter([start[0]], [start[1]], marker="o", s=70,
               color="black", label="COM start")
    ax.scatter([target[0]], [target[1]], marker="*", s=180,
               color="#2ca02c", label="Target")

    for foot in ("left", "right"):
        placements = _unique_placements(
            [x for x in schedule.intervals if x.foot == foot])
        if not placements:
            continue
        pts = np.asarray(placements)
        color = FOOT_COLORS[foot]
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=55,
                   label=f"{foot} supports", zorder=3)
        for i, p in enumerate(pts):
            ax.annotate(f"{FOOT_LABELS[foot]}{i}",
                        (p[0], p[1]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=color)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                ax.annotate(
                    "", xy=pts[i + 1, :2], xytext=pts[i, :2],
                    arrowprops={"arrowstyle": "->", "color": color,
                                "lw": 1.4, "alpha": 0.8})
    ax.legend(loc="best", fontsize=8)


def _plot_timeline(ax, schedule, state) -> None:
    t0 = float(state.time)
    horizon = float(schedule.horizon)
    ax.set_title("Support intervals")
    ax.set_xlabel("time [s]")
    ax.set_yticks([0, 1], labels=["left", "right"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0.0, max(1e-6, horizon - t0))
    ax.grid(True, axis="x", alpha=0.25)

    for foot, y in (("left", 0.0), ("right", 1.0)):
        for i, item in enumerate(x for x in schedule.intervals
                                 if x.foot == foot):
            start = float(item.start_time - t0)
            end = float(item.end_time - t0) if item.end_time is not None \
                else horizon - t0
            ax.barh(y, end - start, left=start, height=0.28,
                    color=FOOT_COLORS[foot], alpha=0.75,
                    edgecolor="black", linewidth=0.6)
            if item.end_time is None:
                ax.annotate("hold", ((start + end) * 0.5, y),
                            ha="center", va="center", fontsize=8,
                            color="white")
            else:
                ax.annotate(f"{FOOT_LABELS[foot]}{i}",
                            ((start + end) * 0.5, y),
                            ha="center", va="center", fontsize=8,
                            color="white")


def _unique_placements(intervals: Iterable[SupportInterval]) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    for item in intervals:
        point = np.asarray(item.point_world, dtype=np.float64)
        if not result or not np.allclose(point, result[-1]):
            result.append(point)
    return result
