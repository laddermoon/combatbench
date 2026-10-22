"""Normalize contacts and applied loads into centroidal ``ExternalLoad``s."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

from .contracts import (
    AFF_ENV,
    AFF_ROBOT_A,
    AFF_ROBOT_B,
    ExternalLoad,
    GeneralizedLoad,
    LoadSnapshot,
    OutOfDomainContact,
    SOURCE_CONTACT_GROUND,
    SOURCE_CONTACT_OPPONENT,
    SOURCE_CONTACT_UNKNOWN,
    SOURCE_CONTACT_WALL,
    SOURCE_XFRC_APPLIED,
    UnknownExternalSource,
    UnsupportedGeneralizedLoad,
)
from .model_spec import CentroidalModelSpec, build_model_spec


@dataclass(frozen=True)
class ContactDecision:
    """Classification and sign for one MuJoCo contact pair."""

    participates: bool
    source_type: str = ""
    counterpart_aff: Optional[int] = None
    target_side: Optional[int] = None   # 1 => target is geom1, force is -f
    internal: bool = False


class ExternalLoadNormalizer:
    """Collect known external loads acting on one robot subtree."""

    def __init__(self, sim, model_spec: Optional[CentroidalModelSpec] = None):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.model_spec = model_spec or build_model_spec(sim)
        self._body_name = {
            bid: spec.name
            for robot in self.model_spec.robots.values()
            for bid, spec in zip(robot.body_ids, robot.bodies)
        }

    def snapshot(self, robot_id: str, *, refresh: bool = True,
                 contact_torque_tolerance: float = 1e-8,
                 load_tolerance: float = 1e-9,
                 strict_generalized: bool = True) -> LoadSnapshot:
        """Return all known non-gravity loads for ``robot_id``.

        ``qfrc_applied`` cannot yet be converted to a spatial external wrench;
        a non-zero value on the target robot raises ``UnsupportedGeneralizedLoad``
        unless ``strict_generalized=False``.
        """
        if robot_id not in self.model_spec.robots:
            raise ValueError(f"unknown robot_id {robot_id!r}")
        if refresh:
            mujoco.mj_forward(self.model, self.data)

        spatial = []
        self_contacts = 0
        ignored_contacts = 0
        for contact_id in range(int(self.data.ncon)):
            decision_or_none = self._classify_contact(robot_id, contact_id)
            if decision_or_none is None:
                ignored_contacts += 1
                continue
            if decision_or_none.internal:
                self_contacts += 1
                continue
            if not decision_or_none.participates:
                ignored_contacts += 1
                continue
            spatial.append(self._contact_load(
                robot_id, contact_id, decision_or_none,
                contact_torque_tolerance))

        xfrc_loads, other_xfrc = self._xfrc_loads(robot_id, load_tolerance)
        spatial.extend(xfrc_loads)
        generalized, other_qfrc = self._generalized_loads(
            robot_id, load_tolerance, strict_generalized)

        return LoadSnapshot(
            robot_id=robot_id,
            time=float(self.data.time),
            spatial_loads=tuple(spatial),
            generalized_loads=tuple(generalized),
            gravity=np.asarray(self.model.opt.gravity, dtype=np.float64),
            metadata={
                "model_fingerprint": self.model_spec.fingerprint,
                "ncon": int(self.data.ncon),
                "internal_self_contact_count": self_contacts,
                "ignored_contact_count": ignored_contacts,
                "other_robot_xfrc_body_ids": tuple(other_xfrc),
                "other_robot_qfrc_dof_count": other_qfrc,
                "load_tolerance": float(load_tolerance),
            },
        )

    def _classify_contact(self, robot_id: str,
                          contact_id: int) -> Optional[ContactDecision]:
        contact = self.data.contact[contact_id]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        aff = self.sim._meta["geom_id_to_aff"]
        if g1 not in aff or g2 not in aff:
            raise UnknownExternalSource(
                f"contact {contact_id} has unmapped geom pair ({g1},{g2})")
        a1, a2 = int(aff[g1]), int(aff[g2])
        target_aff = self._robot_aff(robot_id)
        if a1 == target_aff and a2 == target_aff:
            return ContactDecision(participates=False, internal=True)
        if a1 == target_aff:
            counterpart = a2
            source = self._counterpart_source(counterpart, g2)
            return ContactDecision(True, source, counterpart, 1)
        if a2 == target_aff:
            counterpart = a1
            source = self._counterpart_source(counterpart, g1)
            return ContactDecision(True, source, counterpart, 2)
        return None

    def _counterpart_source(self, aff: int, geom_id: int) -> str:
        if aff == AFF_ENV:
            return (SOURCE_CONTACT_GROUND
                    if geom_id == self.model_spec.ground_geom_id
                    else SOURCE_CONTACT_WALL)
        if aff in (AFF_ROBOT_A, AFF_ROBOT_B):
            return SOURCE_CONTACT_OPPONENT
        raise UnknownExternalSource(
            f"unsupported contact counterpart affiliation {aff} "
            f"for geom {geom_id}")

    def _contact_load(self, robot_id: str, contact_id: int,
                      decision: ContactDecision,
                      torque_tolerance: float) -> ExternalLoad:
        contact = self.data.contact[contact_id]
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, contact_id, wrench)
        torque_contact = np.asarray(wrench[3:6], dtype=np.float64)
        if np.linalg.norm(torque_contact) > torque_tolerance:
            raise OutOfDomainContact(
                f"contact {contact_id} exposes unsupported torque "
                f"{torque_contact}; condim=3 point-contact model does not "
                "accept independent contact couples")
        force_on_geom2 = np.asarray(contact.frame, dtype=np.float64).reshape(
            3, 3).T @ np.asarray(wrench[:3], dtype=np.float64)
        force_on_target = (-force_on_geom2 if decision.target_side == 1
                           else force_on_geom2)
        g1, g2 = int(contact.geom1), int(contact.geom2)
        target_geom = g1 if decision.target_side == 1 else g2
        body_id = int(self.model.geom_bodyid[target_geom])
        return ExternalLoad(
            source_type=decision.source_type,
            robot_id=robot_id,
            point_world=np.asarray(contact.pos, dtype=np.float64),
            force_world=force_on_target,
            torque_world=np.zeros(3, dtype=np.float64),
            body_id=body_id,
            body_name=self._body_name.get(body_id),
            geom_pair=(g1, g2),
            counterpart_aff=decision.counterpart_aff,
            metadata={
                "contact_id": int(contact_id),
                "target_geom_id": int(target_geom),
                "normal_world": np.asarray(contact.frame,
                                           dtype=np.float64).reshape(3, 3)[0].copy(),
                "force_mag": float(np.linalg.norm(wrench[:3])),
            },
        )

    def _xfrc_loads(self, robot_id: str,
                    load_tolerance: float) -> Tuple[Tuple[ExternalLoad, ...],
                                                    Tuple[int, ...]]:
        loads = []
        other = []
        robot = self.model_spec.robots[robot_id]
        target_ids = set(int(x) for x in robot.body_ids)
        for body_id in range(int(self.model.nbody)):
            wrench = np.asarray(self.data.xfrc_applied[body_id],
                                dtype=np.float64)
            if np.linalg.norm(wrench) <= load_tolerance:
                continue
            if body_id not in target_ids:
                other.append(int(body_id))
                continue
            loads.append(ExternalLoad(
                source_type=SOURCE_XFRC_APPLIED,
                robot_id=robot_id,
                # Audited MuJoCo mapping applies the wrench at body COM.
                point_world=np.asarray(self.data.xipos[body_id],
                                       dtype=np.float64),
                force_world=wrench[:3],
                torque_world=wrench[3:6],
                body_id=int(body_id),
                body_name=self._body_name.get(body_id),
                metadata={"field": "xfrc_applied"},
            ))
        return tuple(loads), tuple(other)

    def _generalized_loads(self, robot_id: str, load_tolerance: float,
                           strict: bool) -> Tuple[Tuple[GeneralizedLoad, ...],
                                                  int]:
        robot = self.model_spec.robots[robot_id]
        root_dofs = np.arange(robot.root_dof_adr,
                              robot.root_dof_adr + 6, dtype=np.int32)
        dof_ids = np.unique(np.concatenate([
            root_dofs,
            np.asarray(robot.controlled_dof_adrs, dtype=np.int32),
        ]))
        vals = np.asarray(self.data.qfrc_applied[dof_ids],
                          dtype=np.float64)
        nz = np.abs(vals) > load_tolerance
        other_qfrc = int(np.count_nonzero(
            np.abs(np.asarray(self.data.qfrc_applied,
                              dtype=np.float64)) > load_tolerance)
            - np.count_nonzero(nz))
        loads = []
        if np.any(nz):
            load = GeneralizedLoad(
                robot_id=robot_id,
                dof_indices=dof_ids[nz],
                generalized_force=vals[nz],
                metadata={"field": "qfrc_applied"},
            )
            if strict:
                raise UnsupportedGeneralizedLoad(
                    f"{robot_id} has non-zero qfrc_applied on dofs "
                    f"{load.dof_indices.tolist()}; spatial-wrench conversion "
                    "is not part of the validated contract")
            loads.append(load)
        return tuple(loads), other_qfrc

    @staticmethod
    def _robot_aff(robot_id: str) -> int:
        if robot_id == "robot_a":
            return AFF_ROBOT_A
        if robot_id == "robot_b":
            return AFF_ROBOT_B
        raise ValueError(f"unknown robot_id {robot_id!r}")
