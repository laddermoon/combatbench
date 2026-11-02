"""Conservative applicability and contact-capability checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .contracts import (
    CentroidalState,
    DomainIssue,
    DomainReport,
    ExternalLoad,
    LoadSnapshot,
    SOURCE_CONTACT_GROUND,
    SOURCE_CONTACT_OPPONENT,
    SOURCE_CONTACT_UNKNOWN,
    SOURCE_CONTACT_WALL,
    SOURCE_XFRC_APPLIED,
)
from .model_spec import CentroidalModelSpec, build_model_spec


@dataclass(frozen=True)
class DomainLimits:
    """First-phase conservative domain limits."""

    max_tilt_rad: float = 0.5
    min_root_z: float = 0.7
    min_com_z: float = 0.4
    min_contact_normal_alignment: float = 0.5
    friction_tolerance: float = 1e-7
    normal_force_tolerance: float = -1e-6
    support_polygon_margin: float = 0.0
    foot_geom_prefixes: Tuple[str, ...] = ("foot1_", "foot2_")


@dataclass(frozen=True)
class ContactAssessment:
    contact_key: Tuple[int, int]
    geom_name: str
    body_name: Optional[str]
    point_world: np.ndarray
    normal_world_for_target: np.ndarray
    normal_force: float
    tangent_force: float
    friction_coefficient: float
    friction_margin: float
    moment_about_com_bound: float
    is_support: bool


class CentroidalDomainChecker:
    """Check whether a state/load pair stays inside the declared domain."""

    def __init__(self, sim,
                 model_spec: Optional[CentroidalModelSpec] = None,
                 limits: DomainLimits = DomainLimits()):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.model_spec = model_spec or build_model_spec(sim)
        self.limits = limits
        self._geom_by_id = {g.geom_id: g for g in self.model_spec.geoms}
        self._geom_name = {
            gid: g.name for gid, g in self._geom_by_id.items()}
        self._body_name = {
            bid: spec.name
            for robot in self.model_spec.robots.values()
            for bid, spec in zip(robot.body_ids, robot.bodies)
        }

    def check(self, state: CentroidalState,
              snapshot: LoadSnapshot,
              *, expected_fingerprint: Optional[str] = None
              ) -> DomainReport:
        """Return conservative domain validity and diagnostic margins."""
        issues = []
        margins: Dict[str, float] = {}
        if state.robot_id != snapshot.robot_id:
            issues.append(DomainIssue(
                "invalid", "robot_id_mismatch",
                "state and load snapshot refer to different robots"))
            return DomainReport(False, tuple(issues), margins)

        expected = expected_fingerprint or self.model_spec.fingerprint
        state_fp = state.metadata.get("model_fingerprint")
        snap_fp = snapshot.metadata.get("model_fingerprint")
        if state_fp != expected or snap_fp != expected:
            issues.append(DomainIssue(
                "invalid", "model_fingerprint_mismatch",
                "state/load model fingerprint does not match expected model",
                {"expected": expected, "state": state_fp,
                 "snapshot": snap_fp}))

        if snapshot.generalized_loads:
            issues.append(DomainIssue(
                "invalid", "unsupported_generalized_load",
                "non-zero qfrc_applied lacks a validated spatial mapping",
                {"count": len(snapshot.generalized_loads)}))

        margins.update(self._posture_margins(state, snapshot, issues))
        contacts = self._assess_contacts(state, snapshot, issues, margins)
        self._assess_support_polygon(state, contacts, issues, margins)
        valid = not any(i.severity == "invalid" for i in issues)
        return DomainReport(valid, tuple(issues), margins, {
            "limits": self.limits.__dict__,
            "support_contact_count": sum(c.is_support for c in contacts),
            "contact_assessment_count": len(contacts),
        })

    def _posture_margins(self, state: CentroidalState,
                         snapshot: LoadSnapshot,
                         issues) -> Dict[str, float]:
        robot = self.model_spec.robots[state.robot_id]
        root_id = robot.root_body_id
        quat = np.asarray(self.data.xquat[root_id], dtype=np.float64)
        rot = np.asarray(self.data.xmat[root_id],
                         dtype=np.float64).reshape(3, 3)
        up = rot @ np.array([0.0, 0.0, 1.0])
        tilt = float(np.arccos(np.clip(up[2] / np.linalg.norm(up),
                                      -1.0, 1.0)))
        root_z = float(self.data.xpos[root_id, 2])
        com_z = float(state.com_pos[2])
        margins = {
            "tilt_rad": tilt,
            "tilt_margin_rad": self.limits.max_tilt_rad - tilt,
            "root_z": root_z,
            "root_z_margin": root_z - self.limits.min_root_z,
            "com_z": com_z,
            "com_z_margin": com_z - self.limits.min_com_z,
            "lin_mom_norm": float(np.linalg.norm(state.lin_mom)),
            "ang_mom_norm": float(np.linalg.norm(state.ang_mom)),
        }
        if not np.isfinite(quat).all():
            issues.append(DomainIssue("invalid", "invalid_root_quat",
                                      "root quaternion is not finite"))
        if tilt > self.limits.max_tilt_rad:
            issues.append(DomainIssue(
                "invalid", "excessive_tilt",
                "root up-axis tilt exceeds conservative domain",
                {"tilt_rad": tilt, "limit": self.limits.max_tilt_rad}))
        if root_z < self.limits.min_root_z:
            issues.append(DomainIssue(
                "invalid", "low_root_height",
                "root height indicates a fallen or heavily compressed state",
                {"root_z": root_z, "limit": self.limits.min_root_z}))
        if com_z < self.limits.min_com_z:
            issues.append(DomainIssue(
                "invalid", "low_com_height",
                "COM height below first-phase domain",
                {"com_z": com_z, "limit": self.limits.min_com_z}))
        return margins

    def _assess_contacts(self, state: CentroidalState,
                         snapshot: LoadSnapshot,
                         issues,
                         margins: Dict[str, float]
                         ) -> Tuple[ContactAssessment, ...]:
        assessments = []
        unsupported = 0
        total_normal = 0.0
        moment_bound = 0.0
        friction_bound = 0.0
        for load in snapshot.spatial_loads:
            if load.source_type == SOURCE_XFRC_APPLIED:
                continue
            if load.source_type == SOURCE_CONTACT_UNKNOWN:
                unsupported += 1
                issues.append(DomainIssue(
                    "invalid", "unknown_contact", "unclassified contact",
                    dict(load.metadata)))
                continue
            if load.source_type in (SOURCE_CONTACT_OPPONENT,
                                    SOURCE_CONTACT_WALL):
                unsupported += 1
                issues.append(DomainIssue(
                    "invalid", "unsupported_contact_mode",
                    "first-phase domain only supports flat-ground contacts",
                    {"source_type": load.source_type,
                     "geom_pair": load.geom_pair}))
                continue
            if load.source_type != SOURCE_CONTACT_GROUND:
                unsupported += 1
                issues.append(DomainIssue(
                    "invalid", "unsupported_spatial_source",
                    "unsupported non-contact source",
                    {"source_type": load.source_type}))
                continue
            assessment = self._assess_ground_contact(state, load, issues)
            assessments.append(assessment)
            total_normal += max(0.0, assessment.normal_force)
            friction_bound += max(
                0.0, assessment.friction_coefficient *
                max(0.0, assessment.normal_force))
            moment_bound += assessment.moment_about_com_bound
        margins.update({
            "unsupported_contact_count": float(unsupported),
            "total_contact_normal_force": float(total_normal),
            "contact_friction_force_bound": float(friction_bound),
            "contact_moment_about_com_bound": float(moment_bound),
        })
        if not assessments:
            issues.append(DomainIssue(
                "invalid", "no_ground_support_contact",
                "no valid ground contact is available for support"))
        return tuple(assessments)

    def _assess_ground_contact(self, state: CentroidalState,
                               load: ExternalLoad,
                               issues) -> ContactAssessment:
        if load.geom_pair is None or "contact_id" not in load.metadata:
            issues.append(DomainIssue(
                "invalid", "missing_contact_metadata",
                "ground contact lacks geom pair/contact id"))
            target_geom = -1
            other_geom = -1
            contact_id = -1
        else:
            target_geom = int(load.metadata["target_geom_id"])
            other_geom = (load.geom_pair[1]
                          if load.geom_pair[0] == target_geom
                          else load.geom_pair[0])
            contact_id = int(load.metadata["contact_id"])
        geom_name = self._geom_name.get(target_geom, "")
        is_foot = geom_name.startswith(self.limits.foot_geom_prefixes)
        if not is_foot:
            issues.append(DomainIssue(
                "invalid", "non_foot_support_contact",
                "first-phase support domain only accepts foot-ground contact",
                {"geom_name": geom_name, "geom_id": target_geom}))

        condims = []
        for gid in (target_geom, other_geom):
            g = self._geom_by_id.get(gid)
            if g is None:
                issues.append(DomainIssue(
                    "invalid", "unknown_contact_geom",
                    "contact geom missing from model spec",
                    {"geom_id": gid}))
            else:
                condims.append(g.condim)
        if condims and any(c != 3 for c in condims):
            issues.append(DomainIssue(
                "invalid", "unsupported_contact_dim",
                "first-phase point-contact domain requires condim=3",
                {"condims": tuple(condims), "contact_id": contact_id}))

        normal_raw = np.asarray(load.metadata.get("normal_world"),
                                dtype=np.float64)
        if normal_raw.shape != (3,) or np.linalg.norm(normal_raw) <= 0.0:
            issues.append(DomainIssue(
                "invalid", "missing_contact_normal",
                "contact normal unavailable or invalid"))
            normal_target = np.array([0.0, 0.0, 1.0])
        else:
            normal_raw = normal_raw / np.linalg.norm(normal_raw)
            # geom1 -> geom2 normal. If target is geom1, the support normal
            # acting on the target is the opposite direction.
            target_side = 1 if load.geom_pair and \
                load.geom_pair[0] == target_geom else 2
            normal_target = -normal_raw if target_side == 1 else normal_raw
        alignment = float(normal_target[2])
        if alignment < self.limits.min_contact_normal_alignment:
            issues.append(DomainIssue(
                "invalid", "bad_support_normal",
                "ground contact normal is not sufficiently upward for target",
                {"normal_z": alignment,
                 "limit": self.limits.min_contact_normal_alignment}))

        fn = float(np.dot(load.force_world, normal_target))
        tangent = load.force_world - fn * normal_target
        ft = float(np.linalg.norm(tangent))
        mu = self._combined_friction(target_geom, other_geom)
        friction_margin = mu * fn - ft
        if fn < self.limits.normal_force_tolerance:
            issues.append(DomainIssue(
                "invalid", "negative_contact_normal_force",
                "ground contact pulls rather than pushes target body",
                {"normal_force": fn}))
        if friction_margin < -self.limits.friction_tolerance:
            issues.append(DomainIssue(
                "invalid", "friction_cone_violation",
                "contact tangential force exceeds conservative friction bound",
                {"normal_force": fn, "tangent_force": ft,
                 "mu": mu, "margin": friction_margin}))
        moment = np.cross(load.point_world - state.com_pos,
                          load.force_world)
        return ContactAssessment(
            contact_key=(target_geom, other_geom),
            geom_name=geom_name,
            body_name=load.body_name or self._body_name.get(load.body_id),
            point_world=load.point_world,
            normal_world_for_target=normal_target,
            normal_force=fn,
            tangent_force=ft,
            friction_coefficient=mu,
            friction_margin=float(friction_margin),
            moment_about_com_bound=float(np.linalg.norm(moment)),
            is_support=is_foot and fn > 0.0 and alignment > 0.0,
        )

    def _combined_friction(self, geom1: int, geom2: int) -> float:
        g1 = self._geom_by_id.get(geom1)
        g2 = self._geom_by_id.get(geom2)
        if g1 is None or g2 is None or not g1.friction or not g2.friction:
            return 0.0
        if g1.priority > g2.priority:
            return float(g1.friction[0])
        if g2.priority > g1.priority:
            return float(g2.friction[0])
        return float(max(g1.friction[0], g2.friction[0]))

    def _assess_support_polygon(self, state: CentroidalState,
                                contacts: Sequence[ContactAssessment],
                                issues,
                                margins: Dict[str, float]) -> None:
        points = np.asarray([c.point_world[:2] for c in contacts
                             if c.is_support], dtype=np.float64)
        if points.shape[0] < 3:
            margins["support_polygon_margin"] = float("-inf")
            issues.append(DomainIssue(
                "invalid", "insufficient_support_polygon",
                "fewer than three support contacts cannot define a polygon",
                {"support_contact_count": int(points.shape[0])}))
            return
        try:
            hull = ConvexHull(points)
        except QhullError:
            margins["support_polygon_margin"] = float("-inf")
            issues.append(DomainIssue(
                "invalid", "degenerate_support_polygon",
                "support contact points do not span a polygon"))
            return
        vertices = points[hull.vertices]
        margin = _point_to_convex_margin(vertices, state.com_pos[:2])
        margins["support_polygon_margin"] = float(margin)
        margins["support_polygon_area"] = float(hull.volume)
        if margin < self.limits.support_polygon_margin:
            issues.append(DomainIssue(
                "invalid", "com_outside_support_polygon",
                "COM projection lies outside conservative contact-point hull",
                {"margin": margin,
                 "required_margin": self.limits.support_polygon_margin}))


def _point_to_convex_margin(vertices_ccw: np.ndarray,
                            point: np.ndarray) -> float:
    """Signed distance to a CCW 2D convex polygon (positive inside)."""
    verts = np.asarray(vertices_ccw, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] != 2:
        raise ValueError("vertices_ccw must have shape (n>=3,2)")
    # Ensure CCW orientation.
    signed_area = 0.5 * float(np.sum(
        verts[:, 0] * np.roll(verts[:, 1], -1)
        - verts[:, 1] * np.roll(verts[:, 0], -1)))
    if signed_area < 0.0:
        verts = verts[::-1]
    edges = np.roll(verts, -1, axis=0) - verts
    rel = np.asarray(point, dtype=np.float64)[None, :] - verts
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    edge_norm = np.linalg.norm(edges, axis=1)
    if np.any(edge_norm <= 0.0):
        raise ValueError("degenerate polygon edge")
    return float(np.min(cross / edge_norm))
