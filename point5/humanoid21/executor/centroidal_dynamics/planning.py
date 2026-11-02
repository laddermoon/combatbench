"""Centroidal planning context and same-budget solver comparison.

This module exposes the centroidal evidence needed by future planners, but it
does not make the reduced-order model a replacement for Reclip/full-body
control.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from mocap.humanoid21.clip.core_clip import CONTROL_DT, PHY_DT
from mocap.humanoid21.clip.objective_evaluator import ReclipObjectiveConfig
from mocap.humanoid21.clip.path_reclip import ReclipSolveResult

from .contracts import CentroidalState, DomainReport, LoadSnapshot
from .domain import CentroidalDomainChecker
from .external_loads import ExternalLoadNormalizer
from .model_spec import CentroidalModelSpec, build_model_spec
from .state import CentroidalStateExtractor


@dataclass(frozen=True)
class CentroidalPlan:
    """Evidence-backed centroidal starting plan for one motion request."""

    plan_id: str
    robot_id: str
    initial_state: CentroidalState
    initial_loads: LoadSnapshot
    domain: DomainReport
    target_summary: Dict[str, Any]
    objective_summary: Dict[str, Any]
    budget_frames: int
    model_fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def domain_valid(self) -> bool:
        return self.domain.valid


@dataclass(frozen=True)
class PlanContext:
    """Inputs shared by every solver in a same-budget comparison."""

    sim: Any
    robot_id: str
    start_state: Dict[str, np.ndarray]
    other_state: Dict[str, np.ndarray]
    path: Any
    objective: ReclipObjectiveConfig
    budget_frames: int
    plan: CentroidalPlan
    model_spec: CentroidalModelSpec


SolverMethod = Callable[[PlanContext], ReclipSolveResult]


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _other_robot(robot_id: str) -> str:
    if robot_id == "robot_a":
        return "robot_b"
    if robot_id == "robot_b":
        return "robot_a"
    raise ValueError(f"unknown robot_id {robot_id!r}")


def build_centroidal_plan(
    sim,
    robot_id: str,
    start_state: Dict[str, np.ndarray],
    path: Any,
    objective: ReclipObjectiveConfig,
    budget_frames: int,
    *,
    other_state: Optional[Dict[str, np.ndarray]] = None,
    model_spec: Optional[CentroidalModelSpec] = None,
    plan_id: Optional[str] = None,
) -> CentroidalPlan:
    """Set ``start_state`` on ``sim`` and build centroidal diagnostics."""
    if budget_frames <= 0:
        raise ValueError("budget_frames must be positive")
    model_spec = model_spec or build_model_spec(sim)
    other = _other_robot(robot_id)
    if other_state is None:
        other_state = sim.get_core_state()[other]
    sim.set_core_state({robot_id: dict(start_state),
                        other: dict(other_state)})
    normalizer = ExternalLoadNormalizer(sim, model_spec)
    extractor = CentroidalStateExtractor(sim, model_spec)
    checker = CentroidalDomainChecker(sim, model_spec)
    # Capture applied-load ledger before state refresh consumes xfrc_applied.
    loads = normalizer.snapshot(robot_id, refresh=True)
    state = extractor.state(robot_id)
    domain = checker.check(state, loads,
                           expected_fingerprint=model_spec.fingerprint)
    target_summary = {
        "path_mode": path.mode,
        "source_clip_id": path.source_clip_id,
        "frame_indices": path.frame_indices.tolist(),
        "knot_s": path.knot_s.tolist(),
        "s_end": float(path.s_end),
        "s_ext": float(path.s_ext),
        "s_max": float(path.s_max),
        "root_pos_mode": path.root_pos_mode,
    }
    objective_summary = {
        "kind": objective.kind,
        "time_budget_frames": int(objective.time_budget_frames),
        "ds_max": float(objective.ds_max),
    }
    pid = plan_id or f"{robot_id}:{model_spec.fingerprint[:12]}:{budget_frames}"
    return CentroidalPlan(
        plan_id=pid,
        robot_id=robot_id,
        initial_state=state,
        initial_loads=loads,
        domain=domain,
        target_summary=target_summary,
        objective_summary=objective_summary,
        budget_frames=int(budget_frames),
        model_fingerprint=model_spec.fingerprint,
        metadata={
            "purpose": "centroidal diagnostics for downstream planning",
            "is_full_body_executable": False,
        },
    )


def make_plan_context(
    sim,
    robot_id: str,
    start_state: Dict[str, np.ndarray],
    path: Any,
    objective: ReclipObjectiveConfig,
    budget_frames: int,
    *,
    other_state: Optional[Dict[str, np.ndarray]] = None,
    model_spec: Optional[CentroidalModelSpec] = None,
) -> PlanContext:
    model_spec = model_spec or build_model_spec(sim)
    other = _other_robot(robot_id)
    if other_state is None:
        other_state = sim.get_core_state()[other]
    plan = build_centroidal_plan(
        sim, robot_id, start_state, path, objective, budget_frames,
        other_state=other_state, model_spec=model_spec)
    return PlanContext(sim, robot_id, dict(start_state), dict(other_state),
                       path, objective, int(budget_frames), plan, model_spec)


def direct_reclip_solver(**solver_kwargs) -> SolverMethod:
    """Return a solver callable preserving the existing Reclip semantics."""
    def _solve(ctx: PlanContext) -> ReclipSolveResult:
        from mocap.humanoid21.clip.path_reclip import reclip_path_sweep
        return reclip_path_sweep(
            ctx.start_state, ctx.path, ctx.objective,
            budget_frames=ctx.budget_frames, robot=ctx.robot_id,
            sim=ctx.sim, **solver_kwargs)
    return _solve


def summarize_solver_result(method: str, ctx: PlanContext,
                            result: ReclipSolveResult,
                            wall_time_s: float) -> Dict[str, Any]:
    """Normalize a solver result into a comparable JSON-friendly row."""
    evaluation = result.evaluation
    metrics = dict(evaluation.metrics)
    actions = np.asarray(result.actions, dtype=np.float64)
    progress = np.asarray(evaluation.progress, dtype=np.float64)
    phys_per_control = int(round(CONTROL_DT / PHY_DT))
    row = {
        "method": method,
        "status": str(evaluation.status),
        "completed": evaluation.status == "completed",
        "solver_score": float(evaluation.score),
        "event_frame": evaluation.event_frame,
        "budget_frames": int(ctx.budget_frames),
        "control_frames_executed": int(len(actions)),
        "executed_physics_steps": int(len(actions) * phys_per_control),
        "wall_time_s": float(wall_time_s),
        "fall_count": int(metrics.get("fall_count", 0)),
        "path_progress_final": (None if len(progress) == 0 else
                                float(progress[-1])),
        "path_error_max": metrics.get("path_error_max"),
        "terminal_pose_error": metrics.get("terminal_pose_error"),
        "terminal_vel_error": metrics.get("terminal_vel_error"),
        "action_abs_mean": (None if len(actions) == 0 else
                            float(np.abs(actions).mean())),
        "action_saturation_frac": (None if len(actions) == 0 else
                                   float((np.abs(actions) >= 0.99).mean())),
        "replay_error": _jsonable(result.replay_metrics),
        "clip_T": int(result.clip.T),
        "centroidal_domain_valid": bool(ctx.plan.domain_valid),
        "centroidal_domain_issues": [
            {"severity": i.severity, "code": i.code, "message": i.message}
            for i in ctx.plan.domain.issues
        ],
        "centroidal_margins": _jsonable(ctx.plan.domain.margins),
    }
    return row


def run_same_budget_comparison(
    ctx: PlanContext,
    methods: Mapping[str, SolverMethod],
) -> Dict[str, Any]:
    """Run solver methods under one immutable context and budget."""
    if not methods:
        raise ValueError("methods must contain at least one solver")
    rows = []
    for name, method in methods.items():
        ctx.sim.set_core_state({ctx.robot_id: dict(ctx.start_state),
                                _other_robot(ctx.robot_id):
                                dict(ctx.other_state)})
        t0 = time.perf_counter()
        try:
            result = method(ctx)
            rows.append(summarize_solver_result(
                name, ctx, result, time.perf_counter() - t0))
        except Exception as exc:
            rows.append({
                "method": name,
                "status": "solver_exception",
                "completed": False,
                "budget_frames": int(ctx.budget_frames),
                "wall_time_s": float(time.perf_counter() - t0),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "centroidal_domain_valid": bool(ctx.plan.domain_valid),
                "centroidal_margins": _jsonable(ctx.plan.domain.margins),
            })
    return _jsonable({
        "comparison_version": 1,
        "plan": {
            "plan_id": ctx.plan.plan_id,
            "robot_id": ctx.plan.robot_id,
            "budget_frames": ctx.plan.budget_frames,
            "model_fingerprint": ctx.plan.model_fingerprint,
            "domain_valid": ctx.plan.domain_valid,
            "domain_issues": [
                {"severity": i.severity, "code": i.code,
                 "message": i.message}
                for i in ctx.plan.domain.issues
            ],
            "domain_margins": ctx.plan.domain.margins,
            "initial_centroidal_state": {
                "time": ctx.plan.initial_state.time,
                "mass": ctx.plan.initial_state.mass,
                "com_pos": ctx.plan.initial_state.com_pos,
                "lin_mom": ctx.plan.initial_state.lin_mom,
                "ang_mom": ctx.plan.initial_state.ang_mom,
            },
            "target": ctx.plan.target_summary,
            "objective": ctx.plan.objective_summary,
            "note": "CentroidalPlan is diagnostic input; it is not evidence "
                    "that a full-body action is executable.",
        },
        "runs": rows,
    })
