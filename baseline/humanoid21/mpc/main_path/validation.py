"""Independent interval validation for continuous main paths."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .constraints import MainPathConstraints
from .contracts import FOOT_NAMES
from .diagnostics import ValidationIssue, ValidationReport


class MainPathValidator:
    """Check dynamics and capability constraints over arbitrary intervals."""

    def __init__(self, constraints: MainPathConstraints) -> None:
        self.constraints = constraints

    def check_interval(
        self,
        path,
        start_time: float,
        end_time: float,
    ) -> ValidationReport:
        """Independently integrate dynamics over ``[start_time, end_time]``."""
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            return self._report([ValidationIssue(
                code="INVALID_INTERVAL",
                message="interval bounds must be finite")])
        if end_time <= start_time:
            return self._report([ValidationIssue(
                code="INVALID_INTERVAL",
                message="end_time must be greater than start_time")])
        if (start_time < path.start_time - 1e-9 or
                end_time > path.end_time + 1e-9):
            return self._report([ValidationIssue(
                code="INTERVAL_OUT_OF_DOMAIN",
                message="validation interval exceeds MainPath domain",
                interval=(float(start_time), float(end_time)))])

        boundaries = [float(start_time)]
        boundaries.extend(
            float(t) for t in path.times
            if start_time < t < end_time)
        boundaries.append(float(end_time))

        max_com = 0.0
        max_lin = 0.0
        max_ang = 0.0
        issues: List[ValidationIssue] = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            integrated = self._integrate_interval(path, left, right)
            expected = path.state_at(right)
            dcom = float(np.linalg.norm(
                integrated[0] - expected.com_pos))
            dlin = float(np.linalg.norm(
                integrated[1] - expected.lin_mom))
            dang = float(np.linalg.norm(
                integrated[2] - expected.ang_mom))
            max_com = max(max_com, dcom)
            max_lin = max(max_lin, dlin)
            max_ang = max(max_ang, dang)
            if dcom > self.constraints.com_residual_tol:
                issues.append(ValidationIssue(
                    code="COM_DYNAMICS_RESIDUAL",
                    message="COM interval change does not match momentum integral",
                    interval=(left, right),
                    measured=dcom,
                    limit=self.constraints.com_residual_tol,
                ))
            if dlin > self.constraints.lin_mom_residual_tol:
                issues.append(ValidationIssue(
                    code="LIN_MOM_DYNAMICS_RESIDUAL",
                    message="linear momentum interval change does not match force integral",
                    interval=(left, right),
                    measured=dlin,
                    limit=self.constraints.lin_mom_residual_tol,
                ))
            if dang > self.constraints.ang_mom_residual_tol:
                issues.append(ValidationIssue(
                    code="ANG_MOM_DYNAMICS_RESIDUAL",
                    message="angular momentum interval change does not match torque integral",
                    interval=(left, right),
                    measured=dang,
                    limit=self.constraints.ang_mom_residual_tol,
                ))
        return ValidationReport(
            passed=not issues,
            issues=tuple(issues),
            max_com_residual=max_com,
            max_lin_mom_residual=max_lin,
            max_ang_mom_residual=max_ang,
        )

    def validate(self, path) -> ValidationReport:
        """Check dynamics, continuity, and the configured capability envelope."""
        report = self.check_interval(path, path.start_time, path.end_time)
        issues = list(report.issues)
        issues.extend(self._check_structure(path))
        issues.extend(self._check_capabilities(path))
        return ValidationReport(
            passed=not issues,
            issues=tuple(issues),
            max_com_residual=report.max_com_residual,
            max_lin_mom_residual=report.max_lin_mom_residual,
            max_ang_mom_residual=report.max_ang_mom_residual,
            method=report.method,
        )

    def check_terminal(self, path, target_pos_world: np.ndarray) -> Tuple[ValidationIssue, ...]:
        """Check final target proximity and sustainable static balance."""
        state = path.state_at(path.end_time)
        issues: List[ValidationIssue] = []
        target = np.asarray(target_pos_world, dtype=np.float64)
        pos_error = float(np.linalg.norm(
            state.com_pos[:2] - target[:2]))
        lin_error = float(np.linalg.norm(state.lin_mom))
        ang_error = float(np.linalg.norm(state.ang_mom))
        if pos_error > self.constraints.terminal_position_tol:
            issues.append(ValidationIssue(
                code="TERMINAL_POSITION_ERROR",
                message="final COM misses the requested horizontal target",
                time=path.end_time,
                measured=pos_error,
                limit=self.constraints.terminal_position_tol,
            ))
        if lin_error > self.constraints.terminal_lin_mom_tol:
            issues.append(ValidationIssue(
                code="TERMINAL_LIN_MOM_ERROR",
                message="final linear momentum is not near zero",
                time=path.end_time,
                measured=lin_error,
                limit=self.constraints.terminal_lin_mom_tol,
            ))
        if ang_error > self.constraints.terminal_ang_mom_tol:
            issues.append(ValidationIssue(
                code="TERMINAL_ANG_MOM_ERROR",
                message="final angular momentum is not near zero",
                time=path.end_time,
                measured=ang_error,
                limit=self.constraints.terminal_ang_mom_tol,
            ))
        net_force = state.left_force_world + state.right_force_world
        static_force = float(np.linalg.norm(
            net_force + path.model.mass * path.model.gravity_world))
        if static_force > self.constraints.terminal_lin_mom_tol:
            issues.append(ValidationIssue(
                code="TERMINAL_FORCE_IMBALANCE",
                message="final foot forces do not cancel gravity",
                time=path.end_time,
                measured=static_force,
                limit=self.constraints.terminal_lin_mom_tol,
            ))
        torque = np.zeros(3)
        if state.left_support_active:
            torque += np.cross(
                state.left_point_world - state.com_pos,
                state.left_force_world)
        if state.right_support_active:
            torque += np.cross(
                state.right_point_world - state.com_pos,
                state.right_force_world)
        static_torque = float(np.linalg.norm(torque))
        if static_torque > self.constraints.terminal_ang_mom_tol:
            issues.append(ValidationIssue(
                code="TERMINAL_TORQUE_IMBALANCE",
                message="final foot forces produce nonzero COM torque",
                time=path.end_time,
                measured=static_torque,
                limit=self.constraints.terminal_ang_mom_tol,
            ))
        return tuple(issues)

    def _integrate_interval(
        self,
        path,
        start_time: float,
        end_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = path.state_at(start_time)
        y = np.concatenate([
            state.com_pos, state.lin_mom, state.ang_mom])
        duration = end_time - start_time
        steps = max(1, int(np.ceil(duration / self.constraints.integration_dt)))
        dt = duration / steps
        t = start_time
        for _ in range(steps):
            y = self._rk4_step(path, t, y, dt)
            t += dt
        return y[:3], y[3:6], y[6:9]

    def _rk4_step(self, path, time: float, y: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._rhs(path, time, y)
        k2 = self._rhs(path, time + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self._rhs(path, time + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self._rhs(path, time + dt, y + dt * k3)
        return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    @staticmethod
    def _rhs(path, time: float, y: np.ndarray) -> np.ndarray:
        load = path.load_at(time)
        com = y[:3]
        lin = y[3:6]
        total_force = load["left_force"] + load["right_force"]
        dlin = path.model.mass * path.model.gravity_world + total_force
        dang = np.zeros(3)
        if load["left_active"]:
            dang += np.cross(load["left_point"] - com, load["left_force"])
        if load["right_active"]:
            dang += np.cross(load["right_point"] - com, load["right_force"])
        return np.concatenate([lin / path.model.mass, dlin, dang])

    def _check_structure(self, path) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for prev, current in zip(path.pieces, path.pieces[1:]):
            for foot in FOOT_NAMES:
                prev_force = getattr(prev, f"{foot}_force_end")
                next_force = getattr(current, f"{foot}_force_start")
                error = float(np.linalg.norm(next_force - prev_force))
                if error > 1e-6:
                    issues.append(ValidationIssue(
                        code="FORCE_DISCONTINUITY",
                        message=f"{foot} force jumps across a piece boundary",
                        time=current.start_time,
                        measured=error,
                        limit=1e-6,
                    ))
                p_prev = path.load_at(
                    current.start_time - 1e-7)[f"{foot}_point"]
                p_next = path.load_at(current.start_time)[f"{foot}_point"]
                point_error = float(np.linalg.norm(p_next - p_prev))
                if point_error > 1e-5:
                    issues.append(ValidationIssue(
                        code="FOOT_POINT_DISCONTINUITY",
                        message=f"{foot} equivalent point jumps at a boundary",
                        time=current.start_time,
                        measured=point_error,
                        limit=1e-5,
                    ))
        return issues

    def _check_capabilities(self, path) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        c = self.constraints
        for piece in path.pieces:
            for foot in FOOT_NAMES:
                active = getattr(piece, f"{foot}_active")
                f0 = getattr(piece, f"{foot}_force_start")
                f1 = getattr(piece, f"{foot}_force_end")
                rate = float(np.linalg.norm(f1 - f0) / piece.duration)
                if rate > c.max_force_rate:
                    issues.append(ValidationIssue(
                        code="FORCE_RATE_LIMIT",
                        message=f"{foot} force rate exceeds the capability envelope",
                        interval=(piece.start_time, piece.end_time),
                        measured=rate,
                        limit=c.max_force_rate,
                    ))
                if not active:
                    continue
                for label, force, time in (
                    ("start", f0, piece.start_time),
                    ("end", f1, piece.end_time),
                ):
                    issues.extend(self._check_force(
                        foot, force, time, label))

        times = self._sample_times(path.start_time, path.end_time)
        for time in times:
            state = path.state_at(time)
            height_error = abs(
                state.com_pos[2] - c.nominal_com_height)
            if height_error > c.max_com_height_deviation:
                issues.append(ValidationIssue(
                    code="COM_HEIGHT_LIMIT",
                    message="COM height leaves the configured envelope",
                    time=time,
                    measured=float(height_error),
                    limit=c.max_com_height_deviation,
                ))
            speed = float(np.linalg.norm(state.lin_mom / path.model.mass))
            if speed > c.max_com_speed:
                issues.append(ValidationIssue(
                    code="COM_SPEED_LIMIT",
                    message="COM speed exceeds the configured envelope",
                    time=time,
                    measured=speed,
                    limit=c.max_com_speed,
                ))
            ang_norm = float(np.linalg.norm(state.ang_mom))
            if ang_norm > c.max_ang_mom:
                issues.append(ValidationIssue(
                    code="ANG_MOM_LIMIT",
                    message="angular momentum exceeds the configured envelope",
                    time=time,
                    measured=ang_norm,
                    limit=c.max_ang_mom,
                ))
            for foot in FOOT_NAMES:
                active = getattr(state, f"{foot}_support_active")
                point = getattr(state, f"{foot}_point_world")
                if active:
                    nominal = path.support_schedule.active_point(foot, time)
                    cop_error = point - nominal
                    if abs(cop_error[0]) > c.max_cop_x + 1e-9:
                        issues.append(ValidationIssue(
                            code="COP_X_LIMIT",
                            message=f"{foot} equivalent point leaves the sole envelope",
                            time=time,
                            measured=float(abs(cop_error[0])),
                            limit=c.max_cop_x,
                        ))
                    if abs(cop_error[1]) > c.max_cop_y + 1e-9:
                        issues.append(ValidationIssue(
                            code="COP_Y_LIMIT",
                            message=f"{foot} equivalent point leaves the sole envelope",
                            time=time,
                            measured=float(abs(cop_error[1])),
                            limit=c.max_cop_y,
                        ))
                    distance = float(np.linalg.norm(
                        state.com_pos[:2] - point[:2]))
                    if distance > c.max_com_to_support_xy:
                        issues.append(ValidationIssue(
                            code="COM_SUPPORT_REACH_LIMIT",
                            message=f"COM is too far from the active {foot} support",
                            time=time,
                            measured=distance,
                            limit=c.max_com_to_support_xy,
                        ))
                else:
                    swing_speed = float(np.linalg.norm(
                        path.foot_tracks[foot].velocity_at(time)))
                    if swing_speed > c.max_swing_speed:
                        issues.append(ValidationIssue(
                            code="SWING_SPEED_LIMIT",
                            message=f"{foot} swing speed exceeds the envelope",
                            time=time,
                            measured=swing_speed,
                            limit=c.max_swing_speed,
                        ))
        return self._dedupe(issues)

    def _check_force(
        self,
        foot: str,
        force: np.ndarray,
        time: float,
        label: str,
    ) -> List[ValidationIssue]:
        c = self.constraints
        issues = []
        if force[2] < -1e-9:
            issues.append(ValidationIssue(
                code="NEGATIVE_NORMAL_FORCE",
                message=f"{foot} {label} force pulls on the support",
                time=time,
                measured=float(force[2]),
                limit=0.0,
            ))
        force_norm = float(np.linalg.norm(force))
        if force_norm > c.max_force:
            issues.append(ValidationIssue(
                code="FORCE_NORM_LIMIT",
                message=f"{foot} {label} force exceeds max_force",
                time=time,
                measured=force_norm,
                limit=c.max_force,
            ))
        if force[2] > c.max_normal_force:
            issues.append(ValidationIssue(
                code="NORMAL_FORCE_LIMIT",
                message=f"{foot} {label} force exceeds max_normal_force",
                time=time,
                measured=float(force[2]),
                limit=c.max_normal_force,
            ))
        tangential = float(np.linalg.norm(force[:2]))
        friction_limit = c.friction * max(force[2], 0.0)
        if tangential > friction_limit + 1e-9:
            issues.append(ValidationIssue(
                code="FRICTION_LIMIT",
                message=f"{foot} {label} tangential force exceeds friction cone",
                time=time,
                measured=tangential,
                limit=friction_limit,
            ))
        return issues

    @staticmethod
    def _sample_times(start: float, end: float) -> np.ndarray:
        count = int(np.ceil((end - start) / 0.005)) + 1
        times = np.linspace(start, end, max(2, count))
        return np.unique(times)

    @staticmethod
    def _dedupe(issues: List[ValidationIssue]) -> List[ValidationIssue]:
        seen = set()
        result = []
        for issue in issues:
            key = (issue.code, issue.time, issue.interval)
            if key not in seen:
                seen.add(key)
                result.append(issue)
        return result

    @staticmethod
    def _report(issues: List[ValidationIssue]) -> ValidationReport:
        return ValidationReport(passed=False, issues=tuple(issues))
