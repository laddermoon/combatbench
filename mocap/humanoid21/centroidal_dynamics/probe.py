"""Physics-substep probes for centroidal momentum balance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .balance import momentum_balance
from .contracts import BalanceStep, CentroidalState, LoadSnapshot
from .external_loads import ExternalLoadNormalizer
from .model_spec import CentroidalModelSpec, build_model_spec
from .state import CentroidalStateExtractor


@dataclass(frozen=True)
class ProbeStepResult:
    """One physical substep with pre/post states, loads, and residuals."""

    index: int
    state0: CentroidalState
    state1: CentroidalState
    snapshot0: LoadSnapshot
    snapshot1: LoadSnapshot
    balances: Dict[str, BalanceStep]
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return self.state1.time - self.state0.time


class PhysicsStepProbe:
    """Record known external loads before the simulator clears them."""

    def __init__(self, sim, model_spec: Optional[CentroidalModelSpec] = None):
        self.sim = sim
        self.model_spec = model_spec or build_model_spec(sim)
        self.extractor = CentroidalStateExtractor(sim, self.model_spec)
        self.normalizer = ExternalLoadNormalizer(sim, self.model_spec)

    def probe_step(self, robot_id: str, index: int = 0,
                   *, strict_generalized: bool = True) -> ProbeStepResult:
        """Probe exactly one ``sim.physical_step()``."""
        # Snapshot before state refresh: mj_forward() consumes xfrc_applied in
        # MuJoCo 3.8, while snapshot() internally caches it before refreshing
        # contacts.
        snapshot0 = self.normalizer.snapshot(
            robot_id, refresh=True,
            strict_generalized=strict_generalized)
        state0 = self.extractor.state(robot_id)
        self.sim.physical_step()
        state1 = self.extractor.state(robot_id)
        snapshot1 = self.normalizer.snapshot(
            robot_id, refresh=False,
            strict_generalized=strict_generalized)
        balances = {
            mode: momentum_balance(state0, state1, snapshot0,
                                   snapshot1, mode=mode)
            for mode in ("pre", "post", "trapezoid")
        }
        return ProbeStepResult(
            index=int(index),
            state0=state0,
            state1=state1,
            snapshot0=snapshot0,
            snapshot1=snapshot1,
            balances=balances,
            diagnostics={
                "dt": state1.time - state0.time,
                "ncon0": float(snapshot0.metadata.get("ncon", 0)),
                "ncon1": float(snapshot1.metadata.get("ncon", 0)),
                "internal_contacts0": float(snapshot0.metadata.get(
                    "internal_self_contact_count", 0)),
                "internal_contacts1": float(snapshot1.metadata.get(
                    "internal_self_contact_count", 0)),
            },
        )

    def probe_steps(self, robot_id: str, n_steps: int,
                    action: Optional[Dict[str, object]] = None,
                    *, strict_generalized: bool = True
                    ) -> List[ProbeStepResult]:
        """Probe ``n_steps`` consecutive physics substeps.

        If ``action`` is provided, it is passed verbatim to
        ``sim.set_action()`` once before the first substep; the simulator's
        existing PD controller is preserved.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if action is not None:
            self.sim.set_action(action)
        return [self.probe_step(robot_id, i,
                                strict_generalized=strict_generalized)
                for i in range(int(n_steps))]
