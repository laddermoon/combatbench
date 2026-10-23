"""Visualization helpers for generated support schedules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
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
    """Render a real-time 2D MP4 playback for one support sequence."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")

    duration = float(schedule.horizon - state.time)
    if duration <= 0.0:
        raise ValueError("schedule horizon must be after state.time")
    frame_count = max(2, int(np.floor(duration * fps)) + 1)
    frame_times = state.time + np.arange(frame_count) / float(fps)
    frame_times = frame_times[frame_times <= schedule.horizon + 1e-9]

    fig, ax = plt.subplots(figsize=(8.0, 6.2), constrained_layout=True)

    def draw(frame_index: int) -> None:
        ax.clear()
        _plot_playback_frame(
            ax, schedule, state, target_pos_world,
            current_time=float(frame_times[frame_index]))
        fig.suptitle(
            f"{title}  |  robot t={frame_times[frame_index]:5.2f} s")

    writer = animation.FFMpegWriter(
        fps=int(fps),
        metadata={"title": title},
        bitrate=1800,
    )
    render = animation.FuncAnimation(
        fig, draw, frames=len(frame_times), interval=1000.0 / fps,
        blit=False, repeat=False)
    render.save(output, writer=writer, dpi=130)
    plt.close(fig)
    return output


def animate_main_path(
    path,
    target_pos_world: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Main path playback",
    fps: int = 20,
) -> Path:
    """Render a real-time MP4 for a validated continuous ``MainPath``."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")

    duration = float(path.end_time - path.start_time)
    if duration <= 0.0:
        raise ValueError("path end_time must be after start_time")
    frame_count = max(2, int(np.floor(duration * fps)) + 1)
    frame_times = path.start_time + np.arange(frame_count) / float(fps)
    frame_times = frame_times[frame_times <= path.end_time + 1e-9]
    if abs(frame_times[-1] - path.end_time) > 1e-6:
        frame_times = np.append(frame_times, path.end_time)
    states = path.states_at(frame_times)

    mass = path.model.mass
    gait = path.support_schedule.metadata.get("gait", {})
    leg_length = float(gait.get("leg_length", 0.8))
    speed = np.array(
        [np.linalg.norm(x.lin_mom) / mass for x in states])
    ang_scale = np.array(
        [np.linalg.norm(x.ang_mom) / (mass * leg_length)
         for x in states])
    force_curves = {
        foot: np.array([
            [np.linalg.norm(getattr(x, f"{foot}_force_world")),
             getattr(x, f"{foot}_force_world")[2]]
            for x in states
        ])
        for foot in ("left", "right")
    }

    fig, ((ax_map, ax_time), (ax_mom, ax_force)) = plt.subplots(
        2, 2, figsize=(13.2, 8.2), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 0.9]},
    )

    def draw(frame_index: int) -> None:
        for ax in (ax_map, ax_time, ax_mom, ax_force):
            ax.clear()
        current = states[frame_index]
        _plot_main_path_map(
            ax_map, path, states, frame_index, target_pos_world)
        _plot_path_timeline(ax_time, path, current.time)
        ax_mom.set_title("Momentum")
        ax_mom.set_xlabel("time [s]")
        ax_mom.set_ylabel("m/s")
        ax_mom.grid(True, alpha=0.25)
        ax_mom.plot(frame_times, speed, color="#9467bd",
                    label="|l| / m")
        ax_mom.plot(frame_times, ang_scale, color="#8c564b",
                    label="|L| / (m·leg)")
        ax_mom.set_xlim(path.start_time, path.end_time)
        ax_mom.axvline(current.time, color="black", linestyle="--", lw=1.2)
        ax_mom.legend(loc="upper right", fontsize=8)

        ax_force.set_title("Equivalent foot forces")
        ax_force.set_xlabel("time [s]")
        ax_force.set_ylabel("force [N]")
        ax_force.grid(True, alpha=0.25)
        for foot in ("left", "right"):
            color = FOOT_COLORS[foot]
            ax_force.plot(
                frame_times, force_curves[foot][:, 0], color=color,
                label=f"{foot} |f|")
            ax_force.plot(
                frame_times, force_curves[foot][:, 1], color=color,
                linestyle="--", alpha=0.75, label=f"{foot} fz")
        ax_force.set_xlim(path.start_time, path.end_time)
        ax_force.axvline(current.time, color="black", linestyle="--", lw=1.2)
        ax_force.legend(loc="upper right", fontsize=8, ncol=2)
        fig.suptitle(f"{title}  |  robot t={current.time:5.2f} s")

    writer = animation.FFMpegWriter(
        fps=int(fps), metadata={"title": title}, bitrate=2200)
    render = animation.FuncAnimation(
        fig, draw, frames=len(frame_times), interval=1000.0 / fps,
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


def _plot_playback_frame(
    ax,
    schedule: SupportSchedule,
    state: MainPathState,
    target_pos_world: np.ndarray,
    *,
    current_time: float,
) -> None:
    gait = schedule.metadata.get("gait", {})
    foot_length = float(gait.get("foot_length", 0.24))
    foot_width = float(gait.get("foot_width", 0.09))
    margin = max(0.18, foot_length * 0.75)

    points = [
        state.com_pos[:2],
        state.left_point_world[:2],
        state.right_point_world[:2],
        np.asarray(target_pos_world, dtype=np.float64)[:2],
        *[item.point_world[:2] for item in schedule.intervals],
    ]
    points = np.asarray(points, dtype=np.float64)
    x_min, y_min = points.min(axis=0) - margin
    x_max, y_max = points.max(axis=0) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.set_title("2D support playback")

    start = state.com_pos[:2]
    target = np.asarray(target_pos_world, dtype=np.float64)[:2]
    heading = np.array([
        np.cos(state.heading_yaw), np.sin(state.heading_yaw)])
    ax.plot([start[0], target[0]], [start[1], target[1]],
            color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.scatter([start[0]], [start[1]], marker="o", s=70,
               color="black", label="COM start")
    ax.annotate(
        "", xy=start + heading * 0.16, xytext=start,
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.6})
    ax.scatter([target[0]], [target[1]], marker="*", s=180,
               color="#2ca02c", label="Target")

    for foot in ("left", "right"):
        point = schedule.active_point(foot, current_time)
        if point is None:
            continue
        patch = Rectangle(
            (point[0] - foot_length * 0.5,
             point[1] - foot_width * 0.5),
            foot_length,
            foot_width,
            angle=float(np.degrees(state.heading_yaw)),
            rotation_point="center",
            facecolor=FOOT_COLORS[foot],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.82,
            zorder=4,
        )
        ax.add_patch(patch)
        ax.text(
            point[0], point[1], FOOT_LABELS[foot],
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", zorder=5)

    active = schedule.active_feet(current_time)
    ax.text(
        0.02, 0.03,
        "support: " + (", ".join(active) or "none"),
        transform=ax.transAxes, fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.scatter([], [], marker="s", s=80, color=FOOT_COLORS["left"],
               label="left on ground")
    ax.scatter([], [], marker="s", s=80, color=FOOT_COLORS["right"],
               label="right on ground")
    ax.legend(loc="upper right", fontsize=8)


def _plot_main_path_map(
    ax,
    path,
    states,
    frame_index: int,
    target_pos_world: np.ndarray,
) -> None:
    current = states[frame_index]
    schedule = path.support_schedule
    gait = schedule.metadata.get("gait", {})
    foot_length = float(gait.get("foot_length", 0.24))
    foot_width = float(gait.get("foot_width", 0.09))
    margin = max(0.2, foot_length * 0.8)
    points = np.asarray(
        [x.com_pos[:2] for x in states]
        + [x.point_world[:2] for x in schedule.intervals]
        + [np.asarray(target_pos_world, dtype=np.float64)[:2]],
        dtype=np.float64,
    )
    lower = points.min(axis=0) - margin
    upper = points.max(axis=0) + margin
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("COM, supports, and equivalent points")
    ax.grid(True, alpha=0.25)

    com_xy = np.asarray([x.com_pos[:2] for x in states])
    ax.plot(com_xy[:, 0], com_xy[:, 1], color="black", alpha=0.22,
            linewidth=1.8, label="COM path")
    ax.plot(com_xy[:frame_index + 1, 0], com_xy[:frame_index + 1, 1],
            color="black", linewidth=2.2)
    start = path.initial_state.com_pos[:2]
    target = np.asarray(target_pos_world, dtype=np.float64)[:2]
    ax.scatter([start[0]], [start[1]], marker="o", s=60,
               color="black", label="COM start")
    ax.scatter([target[0]], [target[1]], marker="*", s=170,
               color="#2ca02c", label="Target")

    heading = np.array([
        np.cos(current.heading_yaw), np.sin(current.heading_yaw)])
    ax.annotate(
        "", xy=current.com_pos[:2] + heading * 0.16,
        xytext=current.com_pos[:2],
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5})
    for foot in ("left", "right"):
        color = FOOT_COLORS[foot]
        nominal = schedule.active_point(foot, current.time)
        if nominal is not None:
            patch = Rectangle(
                (nominal[0] - foot_length * 0.5,
                 nominal[1] - foot_width * 0.5),
                foot_length,
                foot_width,
                angle=float(np.degrees(current.heading_yaw)),
                rotation_point="center",
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.78,
                zorder=4,
            )
            ax.add_patch(patch)
            ax.text(
                nominal[0], nominal[1], FOOT_LABELS[foot],
                ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=5)
        point = getattr(current, f"{foot}_point_world")
        ax.scatter(
            [point[0]], [point[1]], marker="x", s=80,
            linewidths=2.0, color=color, zorder=6,
            label=f"{foot} equivalent point")
    ax.scatter(
        [current.com_pos[0]], [current.com_pos[1]], marker="o", s=75,
        facecolors="none", edgecolors="black", linewidths=2.0,
        label="COM")
    active = schedule.active_feet(current.time)
    ax.text(
        0.02, 0.03,
        "support: " + (", ".join(active) or "none"),
        transform=ax.transAxes, fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(loc="upper left", fontsize=7, ncol=2)


def _plot_path_timeline(ax, path, current_time: float) -> None:
    ax.set_title("Support intervals")
    ax.set_xlabel("time [s]")
    ax.set_yticks([0, 1], labels=["left", "right"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(path.start_time, path.end_time)
    ax.grid(True, axis="x", alpha=0.25)
    for foot, y in (("left", 0.0), ("right", 1.0)):
        for i, item in enumerate(
                x for x in path.support_schedule.intervals
                if x.foot == foot):
            start = float(item.start_time)
            end = float(item.end_time) if item.end_time is not None \
                else path.end_time
            ax.barh(y, end - start, left=start, height=0.28,
                    color=FOOT_COLORS[foot], alpha=0.75,
                    edgecolor="black", linewidth=0.6)
            ax.annotate(
                f"{FOOT_LABELS[foot]}{i}", ((start + end) * 0.5, y),
                ha="center", va="center", fontsize=8, color="white")
    ax.axvline(current_time, color="black", linestyle="--", linewidth=1.4)


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


def _unique_placements(intervals: Iterable[SupportInterval]) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    for item in intervals:
        point = np.asarray(item.point_world, dtype=np.float64)
        if not result or not np.allclose(point, result[-1]):
            result.append(point)
    return result
