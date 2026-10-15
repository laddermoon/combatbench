"""Per-action-step trajectory recorder for policy-rollout motion capture.

``TrajectoryRecorder`` is a ``PostActionRecorder`` that dumps the full
physical state of each robot on every action step.  On episode end it
writes one ``ep_<seed>_<robot>.npz`` per robot plus a shared
``ep_<seed>_meta.json`` with provenance (env/policy blueprints, seed,
termination reasons, push events).

Unlike mocap clips the recorded velocities are *true* dynamical state —
a recorded frame is a fully executable robot state, usable directly as a
Random-State-Init anchor or tracking reference.

npz layout (per robot)::

    joint_angles        (T, 21) rad      — de-normalized joint positions
    joint_velocities    (T, 21) rad/s    — true joint velocities
    joint_names         (21,)  str
    root_pos            (T, 3)  m, world
    root_quat           (T, 4)  [w,x,y,z], world
    root_linvel         (T, 3)  m/s, world (derived from local frame)
    root_angvel         (T, 3)  rad/s, world
    root_vel_local      (T, 3)  raw accessor value (body frame)
    root_angvel_local   (T, 3)  raw accessor value (body frame)
    action              (T, 21) normalized PD targets actually applied
    feet_forces         (T, 2)  raw foot force readings (contact proxy)
    body_xpos           (T, B, 3) world positions of every body
    body_xquat          (T, B, 4) world quats [w,x,y,z] of every body
    push_active         (T,)   bool — RandomPushPlugin was applying force
    push_force_vec      (T, 3) applied force vector while active
    episode_step        (T,)   int
    timestamps          (T,)   s, at action rate

Meta json (per episode, shared by both robots)::

    seed / steps / termination_reasons / fps_action / phys_hz
    provenance   {env_blueprint, policy_a, policy_b}
    body_names   ordered body names matching body_xpos axis-1
    push_events  {robot: [{start_step, duration_action_steps,
                          force_vec, force_mag, body}]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework.context import ReadOnlySimContext
from envs.framework.recorder import PostActionRecorder

_AGENTS = ("robot_a", "robot_b")


class TrajectoryRecorder(PostActionRecorder):
    """Record full robot state per action step; dump npz on episode end."""

    def __init__(
        self,
        output_dir: str,
        phy_steps_per_action: int = 25,
        robots: Sequence[str] = _AGENTS,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.phy_steps_per_action = int(phy_steps_per_action)
        self.robots = tuple(robots)
        self.provenance = dict(provenance or {})
        self._static: Optional[Dict[str, Any]] = None
        self._ep: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        if self._static is None:
            self._static = ctx.accessor.get_static_data()
        self._ep = {
            "seed": ctx.base_seed,
            "steps": [],
            "physics_steps": [],
            "push_active": {r: [] for r in self.robots},
            "push_force_vec": {r: [] for r in self.robots},
            "action": {r: [] for r in self.robots},
            "core": {r: [] for r in self.robots},
            "feet_forces": {r: [] for r in self.robots},
            "body_xpos": {r: [] for r in self.robots},
            "body_xquat": {r: [] for r in self.robots},
        }

    def on_post_action_step(
        self,
        ctx: ReadOnlySimContext,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        ep = self._ep
        if ep is None:
            return
        cs = ctx.accessor.get_core_state()
        ds = ctx.accessor.get_derived_state(list(self.robots))
        ep["steps"].append(int(ctx.episode_step))
        ep["physics_steps"].append(int(ctx.physics_step))
        for rid in self.robots:
            ep["core"][rid].append(cs[rid])
            a = action.get(rid)
            ep["action"][rid].append(
                np.asarray(a, dtype=np.float32) if a is not None
                else np.full(21, np.nan, dtype=np.float32))
            ep["feet_forces"][rid].append(
                np.asarray(ds[rid]["feet_forces"], dtype=np.float32))
            names = self._static[rid]["body_names"]
            ep["body_xpos"][rid].append(
                np.stack([ds[rid]["body_xpos"][n] for n in names]))
            ep["body_xquat"][rid].append(
                np.stack([ds[rid]["body_xquat"][n] for n in names]))
            active = bool(ctx.metrics.get(f"{rid}_push_active", False))
            ep["push_active"][rid].append(active)
            vec = ctx.metrics.get(f"{rid}_last_push_force_vec")
            ep["push_force_vec"][rid].append(
                np.asarray(vec, dtype=np.float32) if active and vec is not None
                else np.zeros(3, dtype=np.float32))

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        ep = self._ep
        if ep is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed = ep["seed"] if ep["seed"] is not None else "none"
        phys_hz = float(ctx.accessor.get_physical_frequency())
        fps = phys_hz / float(self.phy_steps_per_action)

        push_events = {}
        for rid in self.robots:
            evs = ctx.metrics.get(f"{rid}_push_events") or []
            push_events[rid] = [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in ev.items()}
                for ev in evs
            ]
        termination = {
            rid: list(ctx.agent_termination_proposals.get(rid, []))
            for rid in self.robots
        }

        for rid in self.robots:
            limits = np.asarray(self._static[rid]["joint_limits"], np.float32)
            ref = (limits[:, 0] + limits[:, 1]) / 2.0
            scale = (limits[:, 1] - limits[:, 0]) / 2.0

            core = ep["core"][rid]
            n = len(core)
            jp_norm = np.stack([c["joint_pos_norm"] for c in core])
            jv_norm = np.stack([c["joint_vel_norm"] for c in core])
            root_pos = np.stack([c["root_pos"] for c in core])
            root_quat = np.stack([c["root_rot"] for c in core])  # wxyz
            vel_local = np.stack([c["root_vel_local"] for c in core])
            angvel_local = np.stack([c["root_angular_vel_local"] for c in core])

            rot = R.from_quat(root_quat[:, [1, 2, 3, 0]])  # scipy xyzw
            linvel_w = rot.apply(vel_local).astype(np.float32)
            angvel_w = rot.apply(angvel_local).astype(np.float32)

            joint_angles = (jp_norm * scale + ref).astype(np.float32)
            joint_vel = (jv_norm * scale).astype(np.float32)

            np.savez_compressed(
                self.output_dir / f"ep_{seed}_{rid}.npz",
                joint_angles=joint_angles,
                joint_velocities=joint_vel,
                joint_names=np.asarray(self._static[rid]["dof_names"]),
                root_pos=root_pos.astype(np.float32),
                root_quat=root_quat.astype(np.float32),
                root_linvel=linvel_w,
                root_angvel=angvel_w,
                root_vel_local=vel_local.astype(np.float32),
                root_angvel_local=angvel_local.astype(np.float32),
                action=np.stack(ep["action"][rid]),
                feet_forces=np.stack(ep["feet_forces"][rid]),
                body_xpos=np.stack(ep["body_xpos"][rid]).astype(np.float32),
                body_xquat=np.stack(ep["body_xquat"][rid]).astype(np.float32),
                push_active=np.asarray(ep["push_active"][rid], dtype=bool),
                push_force_vec=np.stack(ep["push_force_vec"][rid]),
                episode_step=np.asarray(ep["steps"], dtype=np.int32),
                physics_step=np.asarray(ep["physics_steps"], dtype=np.int32),
                timestamps=np.arange(n, dtype=np.float32) / fps,
            )

        meta = {
            "seed": seed,
            "steps": len(ep["steps"]),
            "termination_reasons": termination,
            "fps_action": fps,
            "phys_hz": phys_hz,
            "phy_steps_per_action": self.phy_steps_per_action,
            "provenance": self.provenance,
            "body_names": {r: self._static[r]["body_names"] for r in self.robots},
            "push_events": push_events,
            "push_metrics": {
                r: {"count": int(ctx.metrics.get(f"{r}_push_count", 0))}
                for r in self.robots
            },
        }
        meta_path = self.output_dir / f"ep_{seed}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        self._ep = None
