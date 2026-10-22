"""Compiled-model contract and fingerprint for centroidal dynamics."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import mujoco
import numpy as np

from envs.humanoid21.meta import Humanoid21Meta

from .contracts import ROBOT_IDS


@dataclass(frozen=True)
class BodySpec:
    body_id: int
    name: str
    mass: float
    ipos: Tuple[float, float, float]
    iquat_wxyz: Tuple[float, float, float, float]
    inertia_diag: Tuple[float, float, float]
    parent_body_id: int


@dataclass(frozen=True)
class GeomSpec:
    geom_id: int
    name: str
    body_id: int
    affiliation: int
    condim: int
    friction: Tuple[float, ...]
    priority: int
    solref: Tuple[float, ...]
    solimp: Tuple[float, ...]
    margin: float
    gap: float
    contype: int
    conaffinity: int


@dataclass(frozen=True)
class JointSpec:
    joint_id: int
    name: str
    body_id: int
    joint_type: int
    joint_type_name: str
    qpos_adr: int
    dof_adr: int
    limited: bool
    range: Tuple[float, float]
    stiffness: float
    damping: Tuple[float, ...]
    armature: Tuple[float, ...]


@dataclass(frozen=True)
class ActuatorSpec:
    actuator_id: int
    name: str
    joint_id: int
    gear: float
    ctrl_range: Tuple[float, float]
    gain_type: int
    gainprm: Tuple[float, ...]
    biasprm: Tuple[float, ...]


@dataclass(frozen=True)
class RobotModelSpec:
    robot_id: str
    suffix: str
    root_body_id: int
    root_body_name: str
    root_joint_id: int
    root_joint_name: str
    root_qpos_adr: int
    root_dof_adr: int
    body_ids: Tuple[int, ...]
    body_names: Tuple[str, ...]
    bodies: Tuple[BodySpec, ...]
    joint_ids: Tuple[int, ...]
    joints: Tuple[JointSpec, ...]
    controlled_joint_ids: Tuple[int, ...]
    controlled_joint_names: Tuple[str, ...]
    controlled_qpos_adrs: Tuple[int, ...]
    controlled_dof_adrs: Tuple[int, ...]
    actuators: Tuple[ActuatorSpec, ...]
    geom_ids: Tuple[int, ...]
    keypoint_body_ids: Dict[str, int] = field(default_factory=dict)
    total_mass: float = 0.0


@dataclass(frozen=True)
class CentroidalModelSpec:
    """Dynamics-relevant compiled model contract.

    ``fingerprint`` hashes the canonical payload of this object excluding the
    fingerprint field itself. It is intended to make reports and datasets
    self-describing, not to replace MuJoCo model versioning.
    """

    model_name: str
    source_xml: str
    source_xml_sha256: str
    mujoco_version: str
    nq: int
    nv: int
    nbody: int
    njnt: int
    ngeom: int
    nactuator: int
    timestep: float
    gravity: Tuple[float, float, float]
    solver: int
    integrator: int
    cone: int
    impratio: float
    iterations: int
    ls_iterations: int
    tolerance: float
    robots: Dict[str, RobotModelSpec]
    geoms: Tuple[GeomSpec, ...]
    env_geom_ids: Tuple[int, ...]
    ground_geom_id: int
    fingerprint: str = ""

    def payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("fingerprint", None)
        return _jsonable(data)


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


def _sha256_json(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _joint_type_name(model: mujoco.MjModel, joint_id: int) -> str:
    try:
        return mujoco.mjtJoint(int(model.jnt_type[joint_id])).name
    except Exception:
        return str(int(model.jnt_type[joint_id]))


def _geom_spec(model: mujoco.MjModel, geom_id: int,
               geom_id_to_aff: Dict[int, int]) -> GeomSpec:
    return GeomSpec(
        geom_id=int(geom_id),
        name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                               geom_id) or "",
        body_id=int(model.geom_bodyid[geom_id]),
        affiliation=int(geom_id_to_aff.get(geom_id, 0)),
        condim=int(model.geom_condim[geom_id]),
        friction=tuple(float(x) for x in model.geom_friction[geom_id]),
        priority=int(model.geom_priority[geom_id]),
        solref=tuple(float(x) for x in model.geom_solref[geom_id]),
        solimp=tuple(float(x) for x in model.geom_solimp[geom_id]),
        margin=float(model.geom_margin[geom_id]),
        gap=float(model.geom_gap[geom_id]),
        contype=int(model.geom_contype[geom_id]),
        conaffinity=int(model.geom_conaffinity[geom_id]),
    )


def _joint_dofnum(model: mujoco.MjModel, joint_id: int) -> int:
    jtype = int(model.jnt_type[joint_id])
    if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    if jtype in (int(mujoco.mjtJoint.mjJNT_SLIDE),
                 int(mujoco.mjtJoint.mjJNT_HINGE)):
        return 1
    raise ValueError(f"unknown joint type {jtype} for joint {joint_id}")


def _joint_spec(model: mujoco.MjModel, joint_id: int) -> JointSpec:
    dof0 = int(model.jnt_dofadr[joint_id])
    dof1 = dof0 + _joint_dofnum(model, joint_id)
    return JointSpec(
        joint_id=int(joint_id),
        name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                               joint_id) or "",
        body_id=int(model.jnt_bodyid[joint_id]),
        joint_type=int(model.jnt_type[joint_id]),
        joint_type_name=_joint_type_name(model, joint_id),
        qpos_adr=int(model.jnt_qposadr[joint_id]),
        dof_adr=int(model.jnt_dofadr[joint_id]),
        limited=bool(model.jnt_limited[joint_id]),
        range=(float(model.jnt_range[joint_id, 0]),
               float(model.jnt_range[joint_id, 1])),
        stiffness=float(model.jnt_stiffness[joint_id]),
        damping=tuple(float(x) for x in model.dof_damping[dof0:dof1]),
        armature=tuple(float(x) for x in model.dof_armature[dof0:dof1]),
    )


def _actuator_spec(model: mujoco.MjModel, actuator_id: int) -> ActuatorSpec:
    trn_type = int(model.actuator_trntype[actuator_id])
    trn_id = int(model.actuator_trnid[actuator_id, 0])
    if trn_type != int(mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError(
            f"actuator {actuator_id} is not joint-transmission; "
            "centroidal model currently expects direct joint actuation")
    return ActuatorSpec(
        actuator_id=int(actuator_id),
        name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                               actuator_id) or "",
        joint_id=trn_id,
        gear=float(model.actuator_gear[actuator_id, 0]),
        ctrl_range=(float(model.actuator_ctrlrange[actuator_id, 0]),
                    float(model.actuator_ctrlrange[actuator_id, 1])),
        gain_type=int(model.actuator_gaintype[actuator_id]),
        gainprm=tuple(float(x) for x in model.actuator_gainprm[actuator_id]),
        biasprm=tuple(float(x) for x in model.actuator_biasprm[actuator_id]),
    )


def _robot_spec(model: mujoco.MjModel, robot_id: str,
                runtime_robot: Dict[str, Any],
                geom_id_to_aff: Dict[int, int]) -> RobotModelSpec:
    if robot_id not in ROBOT_IDS:
        raise ValueError(f"unknown robot_id {robot_id!r}")
    body_ids = tuple(int(x) for x in runtime_robot["body_ids_sorted"])
    body_set = set(body_ids)
    bodies = []
    for bid in body_ids:
        bodies.append(BodySpec(
            body_id=bid,
            name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "",
            mass=float(model.body_mass[bid]),
            ipos=tuple(float(x) for x in model.body_ipos[bid]),
            iquat_wxyz=tuple(float(x) for x in model.body_iquat[bid]),
            inertia_diag=tuple(float(x) for x in model.body_inertia[bid]),
            parent_body_id=int(model.body_parentid[bid]),
        ))
    joint_ids = tuple(jid for jid in range(model.njnt)
                      if int(model.jnt_bodyid[jid]) in body_set)
    joints = tuple(_joint_spec(model, jid) for jid in joint_ids)
    actuator_ids = tuple(int(x) for x in runtime_robot["actuator_ids"])
    actuators = tuple(_actuator_spec(model, aid) for aid in actuator_ids)
    geom_ids = tuple(gid for gid in range(model.ngeom)
                     if int(model.geom_bodyid[gid]) in body_set)
    controlled_joint_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in runtime_robot["controlled_joint_names"])
    if any(jid < 0 for jid in controlled_joint_ids):
        raise ValueError(f"missing controlled joint for {robot_id}")
    return RobotModelSpec(
        robot_id=robot_id,
        suffix=str(runtime_robot["suffix"]),
        root_body_id=int(runtime_robot["root_body_id"]),
        root_body_name=str(runtime_robot["root_body_name"]),
        root_joint_id=int(runtime_robot["root_joint_id"]),
        root_joint_name=str(runtime_robot["root_joint_name"]),
        root_qpos_adr=int(runtime_robot["root_qpos_adr"]),
        root_dof_adr=int(runtime_robot["root_qvel_adr"]),
        body_ids=body_ids,
        body_names=tuple(b.name for b in bodies),
        bodies=tuple(bodies),
        joint_ids=joint_ids,
        joints=joints,
        controlled_joint_ids=controlled_joint_ids,
        controlled_joint_names=tuple(
            runtime_robot["controlled_joint_names"]),
        controlled_qpos_adrs=tuple(
            int(model.jnt_qposadr[jid]) for jid in controlled_joint_ids),
        controlled_dof_adrs=tuple(
            int(model.jnt_dofadr[jid]) for jid in controlled_joint_ids),
        actuators=actuators,
        geom_ids=geom_ids,
        keypoint_body_ids={
            str(k): int(v)
            for k, v in runtime_robot["keypoint_body_ids"].items()},
        total_mass=float(runtime_robot["total_mass"]),
    )


def build_model_spec(sim) -> CentroidalModelSpec:
    """Build and validate the dynamics-relevant compiled-model contract."""
    model = sim.model
    errors = Humanoid21Meta.validate(model)
    if errors:
        raise ValueError("invalid Humanoid21 model:\n" + "\n".join(errors))
    runtime = sim._meta
    geom_id_to_aff = runtime["geom_id_to_aff"]
    robots = {
        rid: _robot_spec(model, rid, runtime["robots"][rid],
                         geom_id_to_aff)
        for rid in ROBOT_IDS
    }
    geoms = tuple(_geom_spec(model, gid, geom_id_to_aff)
                  for gid in range(model.ngeom))
    source_xml = Path(sim.ARENA_XML)
    xml_hash = hashlib.sha256(source_xml.read_bytes()).hexdigest()
    model_name = getattr(model, "name", "") or source_xml.stem
    spec = CentroidalModelSpec(
        model_name=str(model_name),
        source_xml=str(source_xml),
        source_xml_sha256=xml_hash,
        mujoco_version=str(mujoco.__version__),
        nq=int(model.nq), nv=int(model.nv), nbody=int(model.nbody),
        njnt=int(model.njnt), ngeom=int(model.ngeom),
        nactuator=int(model.nu),
        timestep=float(model.opt.timestep),
        gravity=tuple(float(x) for x in model.opt.gravity),
        solver=int(model.opt.solver),
        integrator=int(model.opt.integrator),
        cone=int(model.opt.cone),
        impratio=float(model.opt.impratio),
        iterations=int(model.opt.iterations),
        ls_iterations=int(model.opt.ls_iterations),
        tolerance=float(model.opt.tolerance),
        robots=robots,
        geoms=geoms,
        env_geom_ids=tuple(int(x) for x in runtime["env_geom_ids"]),
        ground_geom_id=int(runtime["ground_geom_id"]),
    )
    return CentroidalModelSpec(
        **{**spec.__dict__, "fingerprint": _sha256_json(spec.payload())})
