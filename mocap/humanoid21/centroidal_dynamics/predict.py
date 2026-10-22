"""Short-horizon discrete centroidal predictor.

This module integrates already-normalized centroidal quantities. It is a
reduced-order predictor, not a full-body action generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import (
    CentroidalState,
    LoadSnapshot,
    PredictionDomainViolation,
    SOURCE_CONTACT_GROUND,
    SOURCE_XFRC_APPLIED,
)
from .model_spec import CentroidalModelSpec


@dataclass(frozen=True)
class PredictionStep:
    index: int
    state: CentroidalState
    snapshot: LoadSnapshot
    net_force: np.ndarray
    net_moment: np.ndarray
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult:
    initial_state: CentroidalState
    steps: Tuple[PredictionStep, ...]

    @property
    def final_state(self) -> CentroidalState:
        return self.steps[-1].state if self.steps else self.initial_state


class DiscreteCentroidalPredictor:
    """Semi-implicit Euler integrator for whole-body centroidal dynamics."""

    def __init__(self, model_spec: CentroidalModelSpec,
                 *, allow_non_ground_contacts: bool = False,
                 contact_torque_tolerance: float = 1e-8,
                 time_tolerance: float = 1e-9):
        self.model_spec = model_spec
        self.dt = float(model_spec.timestep)
        self.gravity = np.asarray(model_spec.gravity, dtype=np.float64)
        self.allow_non_ground_contacts = bool(allow_non_ground_contacts)
        self.contact_torque_tolerance = float(contact_torque_tolerance)
        self.time_tolerance = float(time_tolerance)

    def predict(self, initial_state: CentroidalState,
                snapshots: Sequence[LoadSnapshot]) -> PredictionResult:
        """Integrate one step per prescribed ``LoadSnapshot``."""
        if initial_state.metadata.get("model_fingerprint") != \
                self.model_spec.fingerprint:
            raise PredictionDomainViolation(
                "initial state was extracted from a different model fingerprint")
        steps: List[PredictionStep] = []
        state = initial_state
        for i, snapshot in enumerate(snapshots):
            state, net_force, net_moment, diagnostics = self.step(
                state, snapshot, index=i)
            steps.append(PredictionStep(i, state, snapshot,
                                        net_force, net_moment, diagnostics))
        return PredictionResult(initial_state, tuple(steps))

    def step(self, state: CentroidalState, snapshot: LoadSnapshot,
             index: int = 0) -> Tuple[CentroidalState, np.ndarray,
                                      np.ndarray, Dict[str, float]]:
        """Advance one ``model_spec.timestep`` under prescribed loads."""
        self._validate_state(state)
        self._validate_snapshot(state, snapshot)
        net_force = state.mass * self.gravity + sum(
            (l.force_world for l in snapshot.spatial_loads),
            np.zeros(3, dtype=np.float64))
        net_moment = sum(
            (l.moment_about(state.com_pos)
             for l in snapshot.spatial_loads),
            np.zeros(3, dtype=np.float64))
        lin_next = state.lin_mom + net_force * self.dt
        ang_next = state.ang_mom + net_moment * self.dt
        com_next = state.com_pos + lin_next / state.mass * self.dt
        next_state = CentroidalState(
            robot_id=state.robot_id,
            time=state.time + self.dt,
            mass=state.mass,
            com_pos=com_next,
            lin_mom=lin_next,
            ang_mom=ang_next,
            source="discrete_centroidal_predictor",
            metadata={
                "model_fingerprint": self.model_spec.fingerprint,
                "prediction_step": int(index),
                "predictor": "semi_implicit_euler",
            },
        )
        diagnostics = {
            "dt": self.dt,
            "net_force_norm": float(np.linalg.norm(net_force)),
            "net_moment_norm": float(np.linalg.norm(net_moment)),
            "load_count": float(len(snapshot.spatial_loads)),
        }
        return next_state, net_force, net_moment, diagnostics

    def _validate_state(self, state: CentroidalState) -> None:
        if state.robot_id not in self.model_spec.robots:
            raise PredictionDomainViolation(
                f"unsupported robot_id {state.robot_id!r}")
        if abs(state.mass -
               self.model_spec.robots[state.robot_id].total_mass) > 1e-6:
            raise PredictionDomainViolation(
                "state mass differs from compiled robot subtree mass")

    def _validate_snapshot(self, state: CentroidalState,
                           snapshot: LoadSnapshot) -> None:
        if snapshot.robot_id != state.robot_id:
            raise PredictionDomainViolation("state/snapshot robot_id mismatch")
        if abs(snapshot.time - state.time) > self.time_tolerance:
            raise PredictionDomainViolation(
                f"snapshot time {snapshot.time} does not match state time "
                f"{state.time}")
        if not np.allclose(snapshot.gravity, self.gravity,
                           atol=1e-12, rtol=0.0):
            raise PredictionDomainViolation(
                "snapshot gravity differs from model_spec gravity")
        if snapshot.generalized_loads:
            raise PredictionDomainViolation(
                "qfrc_applied loads are not spatial centroidal inputs")
        for load in snapshot.spatial_loads:
            if load.source_type == SOURCE_XFRC_APPLIED:
                continue
            if load.source_type == SOURCE_CONTACT_GROUND:
                if np.linalg.norm(load.torque_world) > \
                        self.contact_torque_tolerance:
                    raise PredictionDomainViolation(
                        "point contact cannot carry independent torque")
                continue
            if load.source_type.startswith("contact_"):
                if self.allow_non_ground_contacts:
                    if np.linalg.norm(load.torque_world) > \
                            self.contact_torque_tolerance:
                        raise PredictionDomainViolation(
                            "point contact cannot carry independent torque")
                    continue
                raise PredictionDomainViolation(
                    f"non-ground contact {load.source_type} is outside the "
                    "first-phase predictor domain")
            raise PredictionDomainViolation(
                f"unsupported spatial load source {load.source_type!r}")
