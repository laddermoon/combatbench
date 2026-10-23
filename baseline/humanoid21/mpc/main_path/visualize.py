"""Visualization helpers for generated support schedules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
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


def animate_support_schedule(
    schedule: SupportSchedule,
    state: MainPathState,
    target_pos_world: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Support schedule playback",
    fps: int = 20,
) -> Path:
    """Render a generic MP4 playback for one support-sequence result."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")

    t0 = float(state.time)
    duration = float(schedule.horizon - t0)
    if duration <= 0.0:
        raise ValueError("schedule horizon must be after state.time")
    frame_count = max(2, int(np.ceil(duration * fps)) + 1)
    frame_times = np.linspace(t0, schedule.horizon, frame_count)
    curve_times = np.linspace(t0, schedule.horizon,
                              max(64, frame_count * 4))
    curves = _schedule_curves(schedule, curve_times)

    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    grid = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.15, 1.0],
        height_ratios=[1.35, 1.0],
    )
    ax_map = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])
    ax_support = fig.add_subplot(grid[1, 0])
    ax_travel = fig.add_subplot(grid[1, 1])

    def draw(frame_index: int) -> None:
        current_time = float(frame_times[frame_index])
        for ax in (ax_map, ax_time, ax_support, ax_travel):
            ax.clear()
        _plot_map(
            ax_map, schedule, state, target_pos_world,
            current_time=current_time)
        _plot_timeline(ax_time, schedule, state,
                       current_time=current_time)
        _plot_support_curve(ax_support, curves, t0, current_time)
        _plot_travel_curve(ax_travel, curves, t0, current_time)
        fig.suptitle(f"{title}  |  t={current_time - t0:5.2f} s")

    writer = animation.FFMpegWriter(
        fps=int(fps),
        metadata={"title": title},
        bitrate=1800,
    )
    render = animation.FuncAnimation(
        fig, draw, frames=frame_count, interval=1000.0 / fps,
        blit=False, repeat=False)
    render.save(output, writer=writer, dpi=130)
    plt.close(fig)
    return output


def _plot_map(
    ax,
    schedule,
    state,
    target_pos_world,
    *,
    current_time: Optional[float] = None,
) -> None:
    ax.set_title("Foot placements (top view)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    start = state.com_pos[:2]
    target = np.asarray(target_pos_world, dtype=np.float64)[:2]
    heading = np.array([
        np.cos(state.heading_yaw), np.sin(state.heading_yaw)])
    ax.scatter([start[0]], [start[1]], marker="o", s=70,
               color="black", label="COM start")
    ax.annotate(
        "", xy=start + heading * 0.16, xytext=start,
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.6})
    ax.text(
        start[0] + heading[0] * 0.18,
        start[1] + heading[1] * 0.18,
        f"yaw {np.degrees(state.heading_yaw):.0f}°",
        fontsize=8, color="black")
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
        for i in range(len(pts) - 1):
            ax.annotate(
                "", xy=pts[i + 1, :2], xytext=pts[i, :2],
                arrowprops={"arrowstyle": "->", "color": color,
                            "lw": 1.4, "alpha": 0.8})
        if current_time is not None:
            active = schedule.active_point(foot, current_time)
            if active is not None:
                ax.scatter(
                    [active[0]], [active[1]], s=190,
                    facecolors="none", edgecolors=color,
                    linewidths=2.4, zorder=4)
    if current_time is not None:
        ax.text(
            0.02, 0.03,
            "active feet: " + (", ".join(schedule.active_feet(current_time))
                              or "none"),
            transform=ax.transAxes, fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.legend(loc="best", fontsize=8)


def _plot_timeline(
    ax,
    schedule,
    state,
    *,
    current_time: Optional[float] = None,
) -> None:
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
    if current_time is not None:
        ax.axvline(
            current_time - t0, color="black", linestyle="--", linewidth=1.4)


def _schedule_curves(
    schedule: SupportSchedule,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    left = np.array([
        1.0 if "left" in schedule.active_feet(t) else 0.0
        for t in times
    ])
    right = np.array([
        1.0 if "right" in schedule.active_feet(t) else 0.0
        for t in times
    ])
    return {
        "times": np.asarray(times, dtype=np.float64),
        "left_active": left,
        "right_active": right,
        "support_count": left + right,
        "left_remaining": np.array([
            _remaining_foot_travel(schedule, "left", t) for t in times]),
        "right_remaining": np.array([
            _remaining_foot_travel(schedule, "right", t) for t in times]),
    }


def _plot_support_curve(ax, curves: dict, t0: float, current_time: float) -> None:
    times = curves["times"] - t0
    ax.set_title("Support state")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("active feet")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.2, 2.4)
    ax.set_yticks([0, 1, 2])
    ax.grid(True, alpha=0.25)
    ax.step(times, curves["support_count"], where="post",
            color="black", linewidth=1.8, label="support count")
    ax.step(times, curves["left_active"] * 0.22 - 0.11, where="post",
            color=FOOT_COLORS["left"], linewidth=1.2, label="left duty")
    ax.step(times, curves["right_active"] * 0.22 + 2.09, where="post",
            color=FOOT_COLORS["right"], linewidth=1.2, label="right duty")
    ax.axvline(current_time - t0, color="black", linestyle="--",
               linewidth=1.2)
    ax.legend(loc="upper right", fontsize=8)


def _plot_travel_curve(ax, curves: dict, t0: float, current_time: float) -> None:
    times = curves["times"] - t0
    ax.set_title("Remaining planned foot travel")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, alpha=0.25)
    ax.plot(times, curves["left_remaining"], color=FOOT_COLORS["left"],
            linewidth=1.8, label="left")
    ax.plot(times, curves["right_remaining"], color=FOOT_COLORS["right"],
            linewidth=1.8, label="right")
    ax.axvline(current_time - t0, color="black", linestyle="--",
               linewidth=1.2)
    ax.legend(loc="best", fontsize=8)


def _remaining_foot_travel(
    schedule: SupportSchedule,
    foot: str,
    time: float,
) -> float:
    own = [x for x in schedule.intervals if x.foot == foot]
    if not own:
        return 0.0
    previous = [x for x in own if x.start_time <= time]
    reference = previous[-1].point_world if previous else own[0].point_world
    future = [x for x in own if x.start_time > time]
    distance = 0.0
    for item in future:
        distance += float(np.linalg.norm(item.point_world - reference))
        reference = item.point_world
    return distance


def _unique_placements(intervals: Iterable[SupportInterval]) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    for item in intervals:
        point = np.asarray(item.point_world, dtype=np.float64)
        if not result or not np.allclose(point, result[-1]):
            result.append(point)
    return result
