"""Continuous MainPath generation from a support schedule."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .constraints import (
    MainPathConstraints,
    MainPathModel,
    default_main_path_setup,
)
from .contracts import (
    FOOT_NAMES,
    FootName,
    MainPathRequest,
    MainPathState,
    SupportInterval,
    SupportSchedule,
    _vec3,
)
from .diagnostics import PlanningResult, ValidationIssue, ValidationReport
from .gait_model import GaitModel
from .validation import MainPathValidator


@dataclass(frozen=True)
class MainPathTrajectoryRequest:
    """Input consumed by the second main-path planning stage."""

    current_state: MainPathState
    support_schedule: SupportSchedule
    target_pos_world: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_pos_world",
            _vec3(self.target_pos_world, "target_pos_world"))

    @classmethod
    def from_support_result(
        cls,
        request: MainPathRequest,
        schedule: SupportSchedule,
    ) -> "MainPathTrajectoryRequest":
        return cls(
            current_state=request.current_state,
            support_schedule=schedule,
            target_pos_world=request.target_pos_world,
        )


@dataclass(frozen=True)
class SwingSegment:
    """Geometric equivalent-foot transition between two support points."""

    start_time: float
    end_time: float
    start_point: np.ndarray
    end_point: np.ndarray
    lift_height: float

    @staticmethod
    def _blend(u: float) -> float:
        return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)

    @staticmethod
    def _blend_rate(u: float) -> float:
        return 30.0 * u * u * (1.0 - u) * (1.0 - u)

    def point_at(self, time: float) -> np.ndarray:
        duration = self.end_time - self.start_time
        u = np.clip((time - self.start_time) / duration, 0.0, 1.0)
        blend = self._blend(u)
        point = self.start_point + (self.end_point - self.start_point) * blend
        point[2] += self.lift_height * 4.0 * u * (1.0 - u)
        return point

    def velocity_at(self, time: float) -> np.ndarray:
        duration = self.end_time - self.start_time
        u = np.clip((time - self.start_time) / duration, 0.0, 1.0)
        rate = self._blend_rate(u) / duration
        velocity = (self.end_point - self.start_point) * rate
        velocity[2] += self.lift_height * 4.0 * (1.0 - 2.0 * u) / duration
        return velocity


@dataclass(frozen=True)
class FootPointTrack:
    """Continuous equivalent-point reference for one foot."""

    foot: FootName
    initial_point: np.ndarray
    intervals: Tuple[SupportInterval, ...]
    swing_segments: Tuple[SwingSegment, ...]

    def point_at(self, time: float) -> np.ndarray:
        for item in self.intervals:
            if item.start_time <= time and (
                    item.end_time is None or time < item.end_time):
                return item.point_world.copy()
        for item in self.swing_segments:
            if item.start_time <= time < item.end_time:
                return item.point_at(time)
        if self.intervals:
            if time < self.intervals[0].start_time:
                return self.initial_point.copy()
            return self.intervals[-1].point_world.copy()
        return self.initial_point.copy()

    def velocity_at(self, time: float) -> np.ndarray:
        for item in self.swing_segments:
            if item.start_time <= time < item.end_time:
                return item.velocity_at(time)
        return np.zeros(3, dtype=np.float64)


@dataclass(frozen=True)
class PathPiece:
    """One support-mode-constant piece with linear endpoint foot forces."""

    start_time: float
    end_time: float
    start_com: np.ndarray
    start_lin_mom: np.ndarray
    start_ang_mom: np.ndarray
    left_active: bool
    right_active: bool
    left_point_start: np.ndarray
    left_point_end: np.ndarray
    right_point_start: np.ndarray
    right_point_end: np.ndarray
    left_force_start: np.ndarray
    left_force_end: np.ndarray
    right_force_start: np.ndarray
    right_force_end: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_time):
            raise ValueError("piece start_time must be finite")
        if not np.isfinite(self.end_time) or self.end_time <= self.start_time:
            raise ValueError("piece end_time must be greater than start_time")
        for name in (
            "start_com", "start_lin_mom", "start_ang_mom",
            "left_point_start", "left_point_end",
            "right_point_start", "right_point_end",
            "left_force_start", "left_force_end",
            "right_force_start", "right_force_end",
        ):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))
        if (not self.left_active and
                (np.linalg.norm(self.left_force_start) > 1e-9 or
                 np.linalg.norm(self.left_force_end) > 1e-9)):
            raise ValueError("inactive left piece must have zero force")
        if (not self.right_active and
                (np.linalg.norm(self.right_force_start) > 1e-9 or
                 np.linalg.norm(self.right_force_end) > 1e-9)):
            raise ValueError("inactive right piece must have zero force")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


_GAUSS4_X = np.array(
    [-0.8611363115940526, -0.3399810435848563,
     0.3399810435848563, 0.8611363115940526],
    dtype=np.float64,
)
_GAUSS4_W = np.array(
    [0.3478548451374538, 0.6521451548625461,
     0.6521451548625461, 0.3478548451374538],
    dtype=np.float64,
)


class MainPath:
    """Continuous representation of `MainPathState` over a finite horizon."""

    def __init__(
        self,
        *,
        initial_state: MainPathState,
        pieces: Tuple[PathPiece, ...],
        foot_tracks: Dict[FootName, FootPointTrack],
        model: MainPathModel,
        support_schedule: SupportSchedule,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not pieces:
            raise ValueError("MainPath requires at least one piece")
        self.initial_state = initial_state
        self.pieces = tuple(pieces)
        self.foot_tracks = dict(foot_tracks)
        self.model = model
        self.support_schedule = support_schedule
        self.metadata = dict(metadata or {})
        self._starts = np.array(
            [piece.start_time for piece in self.pieces], dtype=np.float64)
        self._check_piece_chain()

    @property
    def start_time(self) -> float:
        return float(self.pieces[0].start_time)

    @property
    def end_time(self) -> float:
        return float(self.pieces[-1].end_time)

    @property
    def times(self) -> np.ndarray:
        return np.array(
            [piece.start_time for piece in self.pieces]
            + [self.pieces[-1].end_time],
            dtype=np.float64,
        )

    def _check_piece_chain(self) -> None:
        if abs(self.pieces[0].start_time - self.initial_state.time) > 1e-9:
            raise ValueError("first piece must start at initial_state.time")
        for prev, current in zip(self.pieces, self.pieces[1:]):
            if abs(prev.end_time - current.start_time) > 1e-9:
                raise ValueError("path pieces must form a continuous time chain")

    def _piece_index(self, time: float) -> int:
        if time < self.start_time - 1e-9 or time > self.end_time + 1e-9:
            raise ValueError(
                f"time {time} outside MainPath domain "
                f"[{self.start_time}, {self.end_time}]")
        if time >= self.end_time:
            return len(self.pieces) - 1
        return int(bisect_right(self._starts, time) - 1)

    def load_at(self, time: float) -> Dict[str, Any]:
        """Return support flags, equivalent points, and forces at ``time``."""
        piece = self.pieces[self._piece_index(time)]
        u = np.clip(
            (time - piece.start_time) / piece.duration, 0.0, 1.0)
        left_point = (
            piece.left_point_start
            + (piece.left_point_end - piece.left_point_start) * u
            if piece.left_active
            else self.foot_tracks["left"].point_at(time)
        )
        right_point = (
            piece.right_point_start
            + (piece.right_point_end - piece.right_point_start) * u
            if piece.right_active
            else self.foot_tracks["right"].point_at(time)
        )
        return {
            "left_active": piece.left_active,
            "right_active": piece.right_active,
            "left_point": left_point,
            "right_point": right_point,
            "left_force": piece.left_force_start
            + (piece.left_force_end - piece.left_force_start) * u,
            "right_force": piece.right_force_start
            + (piece.right_force_end - piece.right_force_start) * u,
        }

    def state_at(self, time: float) -> MainPathState:
        """Return the complete reduced state at an arbitrary time."""
        piece = self.pieces[self._piece_index(time)]
        s = np.clip(time - piece.start_time, 0.0, piece.duration)
        load = self.load_at(time)
        com, lin, ang = self._integrate_piece(piece, s)
        return MainPathState(
            time=float(time),
            com_pos=com,
            lin_mom=lin,
            ang_mom=ang,
            heading_yaw=self.initial_state.heading_yaw,
            left_point_world=load["left_point"],
            right_point_world=load["right_point"],
            left_force_world=load["left_force"],
            right_force_world=load["right_force"],
            left_support_active=bool(load["left_active"]),
            right_support_active=bool(load["right_active"]),
        )

    def states_at(self, times: Iterable[float]) -> Tuple[MainPathState, ...]:
        return tuple(self.state_at(float(t)) for t in times)

    def _integrate_piece(
        self,
        piece: PathPiece,
        s: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = piece.duration
        force0 = piece.left_force_start + piece.right_force_start
        force1 = piece.left_force_end + piece.right_force_end
        a0 = self.model.mass * self.model.gravity_world + force0
        b = (force1 - force0) / h

        lin = (
            piece.start_lin_mom
            + a0 * s
            + 0.5 * b * s * s
        )
        com = (
            piece.start_com
            + piece.start_lin_mom * (s / self.model.mass)
            + a0 * (s * s / (2.0 * self.model.mass))
            + b * (s ** 3 / (6.0 * self.model.mass))
        )
        ang = piece.start_ang_mom + self._ang_mom_delta(piece, s)
        return com, lin, ang

    def _local_com(self, piece: PathPiece, s: float) -> np.ndarray:
        h = piece.duration
        force0 = piece.left_force_start + piece.right_force_start
        force1 = piece.left_force_end + piece.right_force_end
        a0 = self.model.mass * self.model.gravity_world + force0
        b = (force1 - force0) / h
        return (
            piece.start_com
            + piece.start_lin_mom * (s / self.model.mass)
            + a0 * (s * s / (2.0 * self.model.mass))
            + b * (s ** 3 / (6.0 * self.model.mass))
        )

    def _ang_mom_delta(self, piece: PathPiece, s: float) -> np.ndarray:
        """Integrate contact torque with fourth-order Gauss quadrature."""
        if s <= 0.0:
            return np.zeros(3, dtype=np.float64)
        result = np.zeros(3, dtype=np.float64)
        h = piece.duration
        for x, weight in zip(_GAUSS4_X, _GAUSS4_W):
            u = 0.5 * s * (x + 1.0)
            fraction = u / h
            c = self._local_com(piece, u)
            torque = np.zeros(3, dtype=np.float64)
            if piece.left_active:
                p = (piece.left_point_start +
                     (piece.left_point_end - piece.left_point_start) * fraction)
                f = (piece.left_force_start +
                     (piece.left_force_end - piece.left_force_start) * fraction)
                torque += np.cross(p - c, f)
            if piece.right_active:
                p = (piece.right_point_start +
                     (piece.right_point_end - piece.right_point_start) * fraction)
                f = (piece.right_force_start +
                     (piece.right_force_end - piece.right_force_start) * fraction)
                torque += np.cross(p - c, f)
            result += weight * torque
        return result * (0.5 * s)


class MainPathTrajectoryPlanner:
    """Build a continuous reduced-order path over a support schedule."""

    def __init__(
        self,
        *,
        gait_model: Optional[GaitModel] = None,
        model: Optional[MainPathModel] = None,
        constraints: Optional[MainPathConstraints] = None,
    ) -> None:
        defaults = default_main_path_setup(gait_model)
        self.gait = gait_model or defaults[1]
        self.model = model or defaults[0]
        self.constraints = constraints or defaults[2]
        self.validator = MainPathValidator(self.constraints)

    def plan(self, request: MainPathTrajectoryRequest) -> PlanningResult:
        """Generate and independently validate one continuous MainPath."""
        issues: List[ValidationIssue] = []
        state = request.current_state
        schedule = request.support_schedule
        if schedule.horizon <= state.time:
            return self._failure(
                "INVALID_DOMAIN",
                "schedule horizon must be after current_state.time",
            )
        tracks = self._build_foot_tracks(
            schedule, state, self.constraints.swing_lift_height)
        issues.extend(self._check_initial_support(state, schedule))
        if issues:
            return self._failure_from_issues(issues)

        end_time = schedule.horizon + self.constraints.terminal_extension
        boundaries = self._time_boundaries(state.time, schedule, end_time)
        pieces: List[PathPiece] = []
        com = state.com_pos.copy()
        lin = state.lin_mom.copy()
        ang = state.ang_mom.copy()
        previous_force = state.left_force_world + state.right_force_world
        previous_foot_forces = {
            "left": state.left_force_world.copy(),
            "right": state.right_force_world.copy(),
        }
        previous_app_points = {
            "left": state.left_point_world.copy(),
            "right": state.right_point_world.copy(),
        }
        desired_com = state.com_pos.copy()

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            h = end - start
            midpoint = 0.5 * (start + end)
            active_now = schedule.active_feet(midpoint)
            active_prev = schedule.active_feet(max(state.time, start - 1e-7))
            active_next = schedule.active_feet(
                min(end_time, end + 1e-7))
            start_allowed = set(active_now) - (set(active_now) - set(active_prev))
            if start == state.time:
                start_allowed = set(active_now)
            end_allowed = set(active_now) & set(active_next)

            left_start = np.zeros(3)
            right_start = np.zeros(3)
            left_end = np.zeros(3)
            right_end = np.zeros(3)
            left_point_start = np.zeros(3)
            left_point_end = np.zeros(3)
            right_point_start = np.zeros(3)
            right_point_end = np.zeros(3)
            if active_now:
                ref = self._support_reference(
                    schedule, midpoint, fallback=desired_com)
                ref[2] = self.constraints.nominal_com_height
                desired_force = self._desired_contact_force(com, lin, ref)
                force0 = previous_force
                alpha = min(1.0, h / 0.08)
                force1 = force0 + (desired_force - force0) * alpha
                left_start, right_start = self._continue_force(
                    force0, active_now, start_allowed,
                    previous_foot_forces, issues, start)
                left_end, right_end = self._allocate_force(
                    force1, active_now, end_allowed, com, ang, tracks,
                    state.heading_yaw, issues, end)
                (left_point_start, left_point_end,
                 right_point_start, right_point_end) = self._application_points(
                    active_now, active_prev, end_allowed, previous_app_points,
                    tracks, start, end, com, ang, left_end, right_end)
            else:
                force0 = np.zeros(3)
                force1 = np.zeros(3)
                desired_com = com.copy()

            piece = PathPiece(
                start_time=start,
                end_time=end,
                start_com=com,
                start_lin_mom=lin,
                start_ang_mom=ang,
                left_active="left" in active_now,
                right_active="right" in active_now,
                left_point_start=left_point_start,
                left_point_end=left_point_end,
                right_point_start=right_point_start,
                right_point_end=right_point_end,
                left_force_start=left_start,
                left_force_end=left_end,
                right_force_start=right_start,
                right_force_end=right_end,
            )
            pieces.append(piece)
            com, lin, ang = self._piece_end_state(piece, tracks)
            desired_com = com.copy()
            previous_force = force1
            previous_foot_forces = {
                "left": left_end.copy(),
                "right": right_end.copy(),
            }
            if active_now:
                previous_app_points = {
                    "left": left_point_end.copy(),
                    "right": right_point_end.copy(),
                }
            else:
                previous_app_points = {
                    "left": tracks["left"].point_at(end),
                    "right": tracks["right"].point_at(end),
                }

        path = MainPath(
            initial_state=state,
            pieces=tuple(pieces),
            foot_tracks=tracks,
            model=self.model,
            support_schedule=schedule,
            metadata={"planner": "linear_force_tracking_v1"},
        )
        report = self.validator.validate(path)
        terminal_issues = self.validator.check_terminal(
            path, request.target_pos_world)
        report = ValidationReport(
            passed=report.passed and not issues and not terminal_issues,
            issues=tuple(issues) + report.issues + tuple(terminal_issues),
            max_com_residual=report.max_com_residual,
            max_lin_mom_residual=report.max_lin_mom_residual,
            max_ang_mom_residual=report.max_ang_mom_residual,
            method=report.method,
        )
        return PlanningResult(
            success=report.passed,
            path=path if report.passed else None,
            report=report,
            diagnostics={
                "piece_count": len(pieces),
                "constraint_source": "default_from_gait_model",
            },
        )

    def _failure(self, code: str, message: str) -> PlanningResult:
        return self._failure_from_issues([
            ValidationIssue(code=code, message=message)])

    def _failure_from_issues(
        self,
        issues: List[ValidationIssue],
    ) -> PlanningResult:
        return PlanningResult(
            success=False,
            path=None,
            report=ValidationReport(passed=False, issues=tuple(issues)),
            diagnostics={"failure_stage": "precheck"},
        )

    def _piece_end_state(
        self,
        piece: PathPiece,
        tracks: Dict[FootName, FootPointTrack],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        calc = MainPath.__new__(MainPath)
        calc.model = self.model
        calc.foot_tracks = tracks
        return MainPath._integrate_piece(calc, piece, piece.duration)

    def _time_boundaries(
        self,
        start: float,
        schedule: SupportSchedule,
        end: float,
    ) -> np.ndarray:
        values = {float(start), float(end)}
        for item in schedule.intervals:
            if start <= item.start_time <= end:
                values.add(float(item.start_time))
            if item.end_time is not None and start <= item.end_time <= end:
                values.add(float(item.end_time))
        count = int(np.ceil((end - start) / self.model.control_dt))
        for i in range(count + 1):
            t = start + i * self.model.control_dt
            if start < t < end:
                values.add(float(t))
        boundaries = np.array(sorted(values), dtype=np.float64)
        merged = [float(boundaries[0])]
        for t in boundaries[1:]:
            if t - merged[-1] > 1e-8:
                merged.append(float(t))
        return np.asarray(merged, dtype=np.float64)

    def _desired_contact_force(
        self,
        com: np.ndarray,
        lin: np.ndarray,
        ref: np.ndarray,
    ) -> np.ndarray:
        tau = self.constraints.tracking_time_constant
        velocity = lin / self.model.mass
        acceleration = (
            (ref - com) / (tau * tau)
            - 2.0 * velocity / tau
        )
        speed = float(np.linalg.norm(acceleration))
        max_accel = self.constraints.max_com_speed / tau
        if speed > max_accel:
            acceleration *= max_accel / speed
        return self.model.mass * (
            acceleration - self.model.gravity_world)

    def _support_reference(
        self,
        schedule: SupportSchedule,
        time: float,
        *,
        fallback: np.ndarray,
    ) -> np.ndarray:
        active = schedule.active_feet(time)
        if not active:
            return fallback.copy()
        if len(active) == 1:
            point = schedule.active_point(active[0], time)
            return np.array([point[0], point[1], fallback[2]])
        points = {
            foot: schedule.active_point(foot, time) for foot in active}
        ends = {}
        for foot in active:
            interval = next(
                item for item in schedule.intervals
                if item.foot == foot and item.start_time <= time
                and (item.end_time is None or time < item.end_time))
            ends[foot] = np.inf if interval.end_time is None else interval.end_time
        if ends["left"] == ends["right"]:
            xy = 0.5 * (points["left"][:2] + points["right"][:2])
        else:
            persistent = max(ends, key=ends.get)
            xy = points[persistent][:2]
        return np.array([xy[0], xy[1], fallback[2]])

    def _application_points(
        self,
        active: Tuple[FootName, ...],
        active_prev: Tuple[FootName, ...],
        active_next: set[FootName],
        previous: Dict[FootName, np.ndarray],
        tracks: Dict[FootName, FootPointTrack],
        start: float,
        end: float,
        com: np.ndarray,
        ang_mom: np.ndarray,
        left_force: np.ndarray,
        right_force: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Choose continuous per-foot equivalent force application points."""
        desired_torque = (
            -ang_mom / self.constraints.angular_tracking_time_constant)
        gravity = abs(float(self.model.gravity_world[2]))
        normal_load = max(self.model.mass * gravity, 1e-9)
        foot_forces = {"left": left_force, "right": right_force}
        total_normal = max(sum(
            force[2] for force in foot_forces.values()), 0.25 * normal_load)
        base_torque = np.zeros(3, dtype=np.float64)
        for foot in active:
            nominal_end = tracks[foot].point_at(end)
            base_torque += np.cross(
                nominal_end - com, foot_forces[foot])
        cop_x = float(np.clip(
            -(desired_torque[1] - base_torque[1]) / total_normal,
            -self.constraints.max_cop_x,
            self.constraints.max_cop_x,
        ))
        cop_y = float(np.clip(
            (desired_torque[0] - base_torque[0]) / total_normal,
            -self.constraints.max_cop_y,
            self.constraints.max_cop_y,
        ))
        offset = np.array([cop_x, cop_y, 0.0], dtype=np.float64)
        result = {}
        for foot in FOOT_NAMES:
            nominal_start = tracks[foot].point_at(start)
            nominal_end = tracks[foot].point_at(end)
            if foot not in active:
                result[f"{foot}_start"] = np.zeros(3)
                result[f"{foot}_end"] = np.zeros(3)
                continue
            if foot in active_prev:
                result[f"{foot}_start"] = previous[foot].copy()
            else:
                result[f"{foot}_start"] = nominal_start
            desired = nominal_end.copy()
            if foot in active_next:
                desired += offset
            result[f"{foot}_end"] = desired
        return (
            result["left_start"], result["left_end"],
            result["right_start"], result["right_end"],
        )

    @staticmethod
    def _continue_force(
        total: np.ndarray,
        active: Tuple[FootName, ...],
        allowed: set[FootName],
        previous: Dict[FootName, np.ndarray],
        issues: List[ValidationIssue],
        time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        forces = {
            foot: np.zeros(3, dtype=np.float64) for foot in FOOT_NAMES}
        for foot in active:
            if foot in allowed:
                forces[foot] = previous[foot].copy()
        assigned = forces["left"] + forces["right"]
        residual = total - assigned
        if np.linalg.norm(residual) > 1e-6:
            available = [foot for foot in active if foot in allowed]
            if not available:
                issues.append(ValidationIssue(
                    code="FORCE_CONTINUITY_CONFLICT",
                    message="continuing force has no allowed support foot",
                    time=time,
                    measured=float(np.linalg.norm(residual)),
                ))
            elif len(available) == 1:
                forces[available[0]] += residual
            else:
                for foot in available:
                    forces[foot] += residual * 0.5
        return forces["left"], forces["right"]

    def _allocate_force(
        self,
        total: np.ndarray,
        active: Tuple[FootName, ...],
        allowed: set[FootName],
        com: np.ndarray,
        ang_mom: np.ndarray,
        tracks: Dict[FootName, FootPointTrack],
        heading_yaw: float,
        issues: List[ValidationIssue],
        time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        forces = {"left": np.zeros(3), "right": np.zeros(3)}
        if not active:
            return forces["left"], forces["right"]
        available = [foot for foot in active if foot in allowed]
        if not available:
            if np.linalg.norm(total) > 1e-6:
                issues.append(ValidationIssue(
                    code="NO_LOAD_PATH",
                    message="nonzero force cannot be assigned to an allowed foot",
                    time=time,
                ))
            return forces["left"], forces["right"]
        if len(available) == 1:
            forces[available[0]] = total.copy()
        else:
            p_left = tracks["left"].point_at(time)[:2]
            p_right = tracks["right"].point_at(time)[:2]
            leftward = np.array([-np.sin(heading_yaw), np.cos(heading_yaw)])
            left_c = float(np.dot(p_left, leftward))
            right_c = float(np.dot(p_right, leftward))
            com_c = float(np.dot(com[:2], leftward))
            span = left_c - right_c
            share = 0.5 if abs(span) < 1e-9 else (com_c - right_c) / span
            share = float(np.clip(share, 0.0, 1.0))
            forces["left"] = total * share
            forces["right"] = total * (1.0 - share)
            p_left_3 = tracks["left"].point_at(time)
            p_right_3 = tracks["right"].point_at(time)
            base_torque = (
                np.cross(p_left_3 - com, forces["left"])
                + np.cross(p_right_3 - com, forces["right"])
            )
            desired_z = (
                -ang_mom[2] /
                self.constraints.angular_tracking_time_constant)
            span_z = p_right_3[1] - p_left_3[1]
            if abs(span_z) > 1e-9:
                correction = (desired_z - base_torque[2]) / span_z
                bounds = []
                for foot, sign in (("left", 1.0), ("right", -1.0)):
                    force = forces[foot]
                    cone = self.constraints.friction * max(force[2], 0.0)
                    margin_sq = max(cone * cone - force[1] * force[1], 0.0)
                    margin = np.sqrt(margin_sq)
                    if sign > 0.0:
                        bounds.append((-margin - force[0], margin - force[0]))
                    else:
                        bounds.append((force[0] - margin, force[0] + margin))
                lower = max(item[0] for item in bounds)
                upper = min(item[1] for item in bounds)
                if lower <= upper:
                    correction = float(np.clip(correction, lower, upper))
                    forces["left"][0] += correction
                    forces["right"][0] -= correction
        return forces["left"], forces["right"]

    @staticmethod
    def _build_foot_tracks(
        schedule: SupportSchedule,
        state: MainPathState,
        lift_height: float,
    ) -> Dict[FootName, FootPointTrack]:
        tracks = {}
        initial_points = {
            "left": state.left_point_world,
            "right": state.right_point_world,
        }
        for foot in FOOT_NAMES:
            intervals = tuple(
                item for item in schedule.intervals if item.foot == foot)
            swings: List[SwingSegment] = []
            for prev, nxt in zip(intervals, intervals[1:]):
                if prev.end_time is None:
                    continue
                gap = nxt.start_time - prev.end_time
                if gap > 1e-9:
                    swings.append(SwingSegment(
                        start_time=prev.end_time,
                        end_time=nxt.start_time,
                        start_point=prev.point_world,
                        end_point=nxt.point_world,
                        lift_height=lift_height,
                    ))
            tracks[foot] = FootPointTrack(
                foot=foot,
                initial_point=np.asarray(initial_points[foot], dtype=np.float64),
                intervals=intervals,
                swing_segments=tuple(swings),
            )
        return tracks

    @staticmethod
    def _check_initial_support(
        state: MainPathState,
        schedule: SupportSchedule,
    ) -> List[ValidationIssue]:
        issues = []
        expected = {
            "left": (state.left_support_active, state.left_point_world),
            "right": (state.right_support_active, state.right_point_world),
        }
        active = set(schedule.active_feet(state.time))
        for foot in FOOT_NAMES:
            support, point = expected[foot]
            if support != (foot in active):
                issues.append(ValidationIssue(
                    code="INPUT_SUPPORT_CONFLICT",
                    message=(
                        f"{foot} support flag does not match schedule at "
                        f"initial time"),
                    time=state.time,
                    details={"state_support": support},
                ))
                continue
            if support:
                scheduled = schedule.active_point(foot, state.time)
                if np.linalg.norm(scheduled - point) > 1e-6:
                    issues.append(ValidationIssue(
                        code="INPUT_SUPPORT_POINT_CONFLICT",
                        message=(
                            f"{foot} support point does not match schedule"),
                        time=state.time,
                        measured=float(np.linalg.norm(scheduled - point)),
                        limit=1e-6,
                    ))
        return issues
