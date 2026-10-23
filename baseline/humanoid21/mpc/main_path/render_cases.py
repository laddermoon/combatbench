"""Render representative support-sequence plans to MP4 videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import MainPathRequest, MainPathState
from .support_sequence import SupportSequencePlanner
from .visualize import animate_support_schedule


CASES = {
    "hold": np.array([0.0, 0.0, 0.0]),
    "forward_step": np.array([0.4, 0.0, 0.0]),
    "diagonal_two_steps": np.array([0.9, 0.3, 0.0]),
}


def render_cases(output_dir: str | Path, *, fps: int = 20) -> list[Path]:
    planner = SupportSequencePlanner()
    half_width = planner.model.stance_width * 0.5
    state = MainPathState(
        time=0.0,
        com_pos=np.array([0.0, 0.0, planner.model.nominal_com_height]),
        lin_mom=np.zeros(3),
        ang_mom=np.zeros(3),
        heading_yaw=0.0,
        left_point_world=np.array(
            [0.0, half_width, planner.model.foot_height]),
        right_point_world=np.array(
            [0.0, -half_width, planner.model.foot_height]),
        left_force_world=np.array(
            [0.0, 0.0, planner.model.mass * planner.model.gravity * 0.5]),
        right_force_world=np.array(
            [0.0, 0.0, planner.model.mass * planner.model.gravity * 0.5]),
    )
    paths = []
    for name, target in CASES.items():
        schedule = planner.plan(MainPathRequest(state, target))
        path = animate_support_schedule(
            schedule,
            state,
            target,
            Path(output_dir) / f"{name}.mp4",
            title=f"Support sequence: {name}",
            fps=fps,
        )
        print(json.dumps({
            "case": name,
            "target": target.tolist(),
            "horizon": schedule.horizon,
            "mode": schedule.metadata.get("mode"),
            "step_counts": schedule.metadata.get("step_counts"),
            "fps": fps,
            "output": str(path),
        }, ensure_ascii=False))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "visualizations",
    )
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render_cases(args.output_dir, fps=args.fps)


if __name__ == "__main__":
    main()
