"""Extract ground-truth robot facts from the current arena XML.

Source of truth: the MuJoCo-compiled model (envs/humanoid21/battle_circular_v2.xml),
NOT hand-read XML — class defaults make hand-reading error-prone.

Outputs a machine-readable spec (robot_spec.json) covering everything
retargeting / prior work needs:
  - per-joint: local axis, world axis at the standing reference pose, limits,
    qpos/dof addresses, parent body, stiffness/damping/armature, actuator
    gear + ctrlrange, PD KP/KD
  - tendons (hamstring coupling), contact excludes
  - body positions / masses / geom sizes
  - control semantics (normalized action -> PD target) self-check

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.robot_spec [--out mocap/humanoid21/robot_spec.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.humanoid21.meta import Humanoid21Meta
from envs.humanoid21.simulator import Humanoid21Simulator


def _world_axes(model, data, joint_names, suffix):
    """World-frame joint axes at the standing reference pose."""
    pose = Humanoid21Meta.INITIAL_POSES["standing"]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"root{suffix}")
    qadr = model.jnt_qposadr[jid]
    data.qpos[:] = 0.0
    data.qpos[qadr:qadr + 3] = pose["root_pos"]
    data.qpos[qadr + 3:qadr + 7] = pose["root_quat"]
    for i, jn in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn + suffix)
        data.qpos[model.jnt_qposadr[jid]] = float(pose["joint_pos"][i])
    mujoco.mj_forward(model, data)
    out = {}
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn + suffix)
        bid = model.jnt_bodyid[jid]
        R = data.xmat[bid].reshape(3, 3)
        out[jn] = (R @ model.jnt_axis[jid]).tolist()
    return out


def build_spec(robot_suffix: str = "_a") -> dict:
    model = mujoco.MjSpec.from_file(Humanoid21Simulator.ARENA_XML).compile()
    data = mujoco.MjData(model)
    meta = Humanoid21Meta

    errs = meta.validate(model)
    if errs:
        raise RuntimeError(f"meta.validate failed: {errs}")

    names = meta.CONTROLLED_JOINTS
    world_axes = _world_axes(model, data, names, robot_suffix)

    joints = []
    for i, jn in enumerate(names):
        full = jn + robot_suffix
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            raise KeyError(f"joint {full} not in model")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
        joints.append({
            "index": i,
            "name": jn,
            "full_name": full,
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])),
            "qposadr": int(model.jnt_qposadr[jid]),
            "dofadr": int(model.jnt_dofadr[jid]),
            "axis_local": model.jnt_axis[jid].tolist(),
            "axis_world_at_standing": world_axes[jn],
            "range_deg": [float(np.degrees(model.jnt_range[jid][0])),
                          float(np.degrees(model.jnt_range[jid][1]))],
            "range_rad": model.jnt_range[jid].tolist(),
            "stiffness": float(np.atleast_1d(model.jnt_stiffness[jid])[0]),
            "damping": float(np.atleast_1d(model.dof_damping[model.jnt_dofadr[jid]])[0]),
            "armature": float(np.atleast_1d(model.dof_armature[model.jnt_dofadr[jid]])[0]),
            "kp": float(meta.KP[i]),
            "kd": float(meta.KD[i]),
            "actuator": None if aid < 0 else {
                "gear": float(model.actuator_gear[aid][0]),
                "ctrlrange": model.actuator_ctrlrange[aid].tolist(),
            },
        })

    tendons = []
    for tid in range(model.ntendon):
        tname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid)
        if not (tname or "").endswith(robot_suffix):
            continue
        members = []
        # fixed tendon: wrap entries are joints (tendon_adr -> tendon_joint rows)
        row0, row1 = int(model.tendon_adr[tid]), int(model.tendon_adr[tid] + model.tendon_num[tid])
        for r in range(row0, row1):
            j = int(model.wrap_objid[r])
            members.append({
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                "coef": float(model.wrap_prm[r]),
            })
        tendons.append({
            "name": tname,
            "type": "fixed",
            "range": model.tendon_range[tid].tolist(),
            "limited": bool(model.tendon_limited[tid]),
            "members": members,
        })

    bodies = []
    for bn in meta.ROBOT_BODY_NAMES:
        full = bn + robot_suffix
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full)
        bodies.append({
            "name": bn,
            "pos_in_parent": model.body_pos[bid].tolist(),
            "mass": float(model.body_mass[bid]),
        })

    # contact excludes are not exposed on the compiled model in MuJoCo 3.x;
    # read them from the XML directly.
    import xml.etree.ElementTree as ET
    excludes = [
        [el.get("body1"), el.get("body2")]
        for el in ET.parse(Humanoid21Simulator.ARENA_XML).getroot().iter("exclude")
    ]

    spec = {
        "arena_xml": Humanoid21Simulator.ARENA_XML,
        "robot_suffix": robot_suffix,
        "timestep_s": float(model.opt.timestep),
        "action_rate_hz": 1.0 / (float(model.opt.timestep) * 25),  # phy_steps_per_action=25
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "controlled_joints_order": names,
        "joints": joints,
        "tendons": tendons,
        "contact_excludes": excludes,
        "bodies": bodies,
        "keypoint_bodies": meta.KEYPOINT_BODIES,
        "initial_poses": {
            k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()}
            for k, v in meta.INITIAL_POSES.items()
        },
    }
    return spec


def selfcheck_action_mapping(spec: dict) -> dict:
    """Verify CONTROLSPEC mapping: target = action*half + mid on standing pose."""
    pose = spec["initial_poses"]["standing"]
    errs = []
    for j, act in zip(spec["joints"], pose["action"]):
        lo, hi = j["range_rad"]
        target = act * (hi - lo) / 2.0 + (hi + lo) / 2.0
        errs.append(abs(target))
    return {"max_abs_target_rad_at_zero_jointpos": max(errs),
            "note": "standing joint_pos is all-zero; action decodes to ~0 rad targets"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "robot_spec.json"))
    args = ap.parse_args()
    spec = build_spec()
    spec["selfcheck"] = selfcheck_action_mapping(spec)
    Path(args.out).write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    print(f"joints={len(spec['joints'])} tendons={len(spec['tendons'])} "
          f"excludes={len(spec['contact_excludes'])} nu={spec['nu']}")
    for t in spec["tendons"]:
        print("  tendon:", t["name"], "range", t["range"], "members", t["members"])


if __name__ == "__main__":
    main()
