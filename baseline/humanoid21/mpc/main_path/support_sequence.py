"""Generate per-foot support intervals from a move-to-point request."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .contracts import (
    FOOT_NAMES,
    FootName,
    MainPathRequest,
    SupportInterval,
    SupportSchedule,
)
from .gait_model import GaitModel, default_gait_model


class SupportSequencePlanner:
    """First main-path stage: current state + target position -> schedule."""

    def __init__(
        self,
        gait_model: Optional[GaitModel] = None,
        *,
        first_foot: Optional[FootName] = None,
    ) -> None:
        self.model = gait_model or default_gait_model()
        if first_foot is not None and first_foot not in FOOT_NAMES:
            raise ValueError(f"unsupported first_foot {first_foot!r}")
        self.first_foot = first_foot

    def inspect_parameters(self) -> Dict[str, float]:
        """Return the measured gait envelope used by the planner."""
        data = asdict(self.model)
        data.pop("metadata", None)
        data.update(self.model.metadata)
        return data

    def plan(self, request: MainPathRequest) -> SupportSchedule:
        """Generate a deterministic support schedule for a target position."""
        state = request.current_state
        t0 = float(state.time)
        origin_xy = state.com_pos[:2]
        target_xy = request.target_pos_world[:2]
        delta = target_xy - origin_xy
        distance = float(np.linalg.norm(delta))
        heading, leftward = self._heading_frame(state.heading_yaw)
        delta_body = np.array([
            float(np.dot(delta, heading)),
            float(np.dot(delta, leftward)),
        ])
        final_left, final_right = self._stance_points(target_xy, leftward)
        current_points = {
            "left": state.left_point_world[:2],
            "right": state.right_point_world[:2],
        }

        if distance <= self.model.target_tolerance:
            intervals = (
                SupportInterval("left", t0, None,
                                self._point(current_points["left"])),
                SupportInterval("right", t0, None,
                                self._point(current_points["right"])),
            )
            return SupportSchedule(
                intervals=intervals,
                horizon=t0 + self.model.settle_duration,
                metadata={
                    "mode": "hold",
                    "heading_yaw": float(state.heading_yaw),
                    "target_delta_body": delta_body.tolist(),
                    "target_distance": distance,
                    "step_counts": {"left": 0, "right": 0},
                    "gait": self.inspect_parameters(),
                },
            )

        initial_points = dict(current_points)
        final_points = {"left": final_left, "right": final_right}
        step_counts = {
            foot: self._step_count(initial_points[foot], final_points[foot])
            for foot in FOOT_NAMES
        }
        placements = {
            foot: self._placements(
                initial_points[foot], final_points[foot], step_counts[foot])
            for foot in FOOT_NAMES
        }
        first = self._choose_first_foot(delta_body)
        events = self._event_order(step_counts, first)

        intervals: List[SupportInterval] = []
        open_start = {foot: t0 for foot in FOOT_NAMES}
        open_point = dict(initial_points)
        t = t0
        for foot, placement_index in events:
            lift_time = t + self.model.double_support_duration
            intervals.append(SupportInterval(
                foot=foot,
                start_time=open_start[foot],
                end_time=lift_time,
                point_world=self._point(open_point[foot]),
            ))
            land_time = lift_time + self.model.swing_duration
            open_start[foot] = land_time
            open_point[foot] = placements[foot][placement_index]
            t = land_time

        for foot in FOOT_NAMES:
            intervals.append(SupportInterval(
                foot=foot,
                start_time=open_start[foot],
                end_time=None,
                point_world=self._point(open_point[foot]),
            ))

        step_lengths = {
            foot: [
                float(np.linalg.norm(p - q))
                for p, q in zip(
                    [initial_points[foot], *placements[foot][:-1]],
                    placements[foot])
            ]
            for foot in FOOT_NAMES
        }
        return SupportSchedule(
            intervals=tuple(intervals),
            horizon=t + self.model.settle_duration,
            metadata={
                "mode": "walk",
                "heading_yaw": float(state.heading_yaw),
                "target_delta_body": delta_body.tolist(),
                "target_distance": distance,
                "step_counts": step_counts,
                "first_foot": first,
                "step_lengths": step_lengths,
                "last_landing_time": t,
                "gait": self.inspect_parameters(),
            },
        )

    @staticmethod
    def _heading_frame(yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        leftward = np.array([-heading[1], heading[0]], dtype=np.float64)
        return heading, leftward

    def _stance_points(
        self,
        center_xy: np.ndarray,
        lateral: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        half = self.model.stance_width * 0.5
        left_xy = center_xy + lateral * half
        right_xy = center_xy - lateral * half
        return left_xy, right_xy

    def _point(self, xy: np.ndarray) -> np.ndarray:
        return np.array([xy[0], xy[1], self.model.foot_height],
                        dtype=np.float64)

    def _step_count(self, start_xy: np.ndarray, end_xy: np.ndarray) -> int:
        distance = float(np.linalg.norm(end_xy - start_xy))
        if distance <= self.model.target_tolerance:
            return 0
        return int(np.ceil(distance / self.model.max_step_length - 1e-9))

    @staticmethod
    def _placements(
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        count: int,
    ) -> List[np.ndarray]:
        return [
            start_xy + (end_xy - start_xy) * (i + 1) / count
            for i in range(count)
        ]

    def _choose_first_foot(self, delta_body: np.ndarray) -> FootName:
        if self.first_foot is not None:
            return self.first_foot
        lateral = float(delta_body[1])
        if abs(lateral) > self.model.target_tolerance:
            return "left" if lateral > 0.0 else "right"
        return "right"

    @staticmethod
    def _event_order(
        step_counts: Dict[FootName, int],
        first_foot: FootName,
    ) -> List[Tuple[FootName, int]]:
        other = "left" if first_foot == "right" else "right"
        order: List[Tuple[FootName, int]] = []
        next_index = {foot: 0 for foot in FOOT_NAMES}
        while any(next_index[foot] < step_counts[foot] for foot in FOOT_NAMES):
            for foot in (first_foot, other):
                if next_index[foot] < step_counts[foot]:
                    order.append((foot, next_index[foot]))
                    next_index[foot] += 1
        return order
