"""MJX (MuJoCo XLA) 批量仿真器实现。

使用 jax.lax.scan 将 n_steps 个物理步编译为单个 XLA 计算，
中间不回 Python，实现 GPU 上全向量化仿真。

所有对外接口输入输出均为 numpy array，JAX 完全封装在内部。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import mujoco
import mujoco.mjx as mjx

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jp
from jax import tree_util

from envs.batchframework.backend import BaseBatchSimulator
from envs.humanoid21.meta import Humanoid21Meta


def _quat_to_rot_mat(quat: jp.ndarray) -> jp.ndarray:
    """[w,x,y,z] → (3,3) rotation matrix (JAX)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_rotate_inv(quat: jp.ndarray, vec: jp.ndarray) -> jp.ndarray:
    """Rotate vector by inverse of quaternion [w,x,y,z] (JAX)."""
    R = _quat_to_rot_mat(quat)
    return R.T @ vec


class MjxHumanoid21Simulator(BaseBatchSimulator):
    """基于 MJX 的 Humanoid21 批量仿真器。

    使用 jax.vmap + jax.lax.scan 实现 B 个环境的并行仿真，
    所有物理步在 GPU 上完成，仅在 get_*/set_* 时进行 host-device 传输。
    """

    DT = Humanoid21Meta.DT
    ACTION_DIM = Humanoid21Meta.ACTION_DIM
    ARENA_XML = str(Path(__file__).resolve().parent.parent / "humanoid21" / "battle_v1.xml")
    KP = Humanoid21Meta.KP
    KD = Humanoid21Meta.KD
    CONTROLLED_JOINTS = Humanoid21Meta.CONTROLLED_JOINTS
    INITIAL_POSES = Humanoid21Meta.INITIAL_POSES

    def __init__(
        self,
        batch_size: int,
        initial_distance: float = 2.0,
        initial_pose_a: str = "standing",
        initial_pose_b: str = "standing",
        device: Optional[jax.Device] = None,
    ):
        # float64 is required for MJX/MuJoCo consistency (see validation test).
        # With float32, contact solver diverges within ~10 steps.
        # With float64, per-step match is ~1e-14 (machine precision).
        self._batch_size = batch_size
        self._initial_distance = initial_distance
        self._initial_pose_a = initial_pose_a
        self._initial_pose_b = initial_pose_b
        self._device = device or jax.devices()[0]

        # --- Load MuJoCo model + MJX model ---
        self._model = mujoco.MjSpec.from_file(self.ARENA_XML).compile()
        self._model.opt.timestep = self.DT
        self._mjx_model = mjx.put_model(self._model)

        # --- Build runtime tables from meta ---
        self._meta = Humanoid21Meta.build_runtime_tables(self._model)
        self._robots = self._meta["robots"]
        self._ground_geom_id = self._meta["ground_geom_id"]

        # --- Normalization params ---
        self._norm_params = {}
        for robot_id in ["robot_a", "robot_b"]:
            jnt_ranges = self._robots[robot_id]["jnt_ranges"]  # (21, 2)
            lower = jnt_ranges[:, 0]
            upper = jnt_ranges[:, 1]
            self._norm_params[robot_id] = {
                "reference": ((lower + upper) / 2.0).astype(np.float32),
                "scale": ((upper - lower) / 2.0).astype(np.float32),
            }

        # --- PD tables ---
        self._pd_tables = {}
        for robot_id in ["robot_a", "robot_b"]:
            act_ids = self._robots[robot_id]["actuator_ids"]
            gear = np.array(self._model.actuator_gear[act_ids, 0], dtype=np.float64)
            gear[gear == 0] = 1.0
            ctrl_lo = np.array(self._model.actuator_ctrlrange[act_ids, 0], dtype=np.float64)
            ctrl_hi = np.array(self._model.actuator_ctrlrange[act_ids, 1], dtype=np.float64)
            self._pd_tables[robot_id] = {
                "actuator_ids": act_ids,
                "gear": gear,
                "ctrl_lo": ctrl_lo,
                "ctrl_hi": ctrl_hi,
            }

        # --- Precompute static JAX arrays for PD control ---
        # Per-robot: qpos_indices, qvel_indices, actuator_ids, norm ref/scale, gear, ctrl_lo/hi, KP, KD
        self._jax_statics = self._build_jax_statics()

        # --- Body name → body id mapping for external force ---
        self._body_name_to_id = {}
        for robot_id, suffix in Humanoid21Meta.ROBOT_SUFFIXES.items():
            for body_name in Humanoid21Meta.ROBOT_BODY_NAMES:
                full = f"{body_name}{suffix}"
                bid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, full
                )
                if bid >= 0:
                    self._body_name_to_id[full] = bid

        # --- Init JAX state ---
        self._mjx_data: Optional[mjx.Data] = None
        self._action_jax: Optional[Dict[str, jp.ndarray]] = None
        self._ext_force_jax: Optional[jp.ndarray] = None  # (B, nbody, 6) persistent
        self._history_buffer: Optional[mjx.Data] = None  # pytree of (B, n_steps, ...)
        self._history_n_steps: int = 0

        # --- JIT-compiled functions (built lazily after first reset) ---
        self._jit_step = None
        self._jit_step_scan = None

    def _build_jax_statics(self) -> Dict[str, Any]:
        """Precompute JAX arrays needed inside JIT-compiled step functions."""
        statics = {}
        for robot_id in ["robot_a", "robot_b"]:
            r = self._robots[robot_id]
            norm = self._norm_params[robot_id]
            pd = self._pd_tables[robot_id]
            statics[robot_id] = {
                "qpos_indices": jp.array(r["qpos_indices"], dtype=jp.int32),
                "qvel_indices": jp.array(r["qvel_indices"], dtype=jp.int32),
                "actuator_ids": jp.array(r["actuator_ids"], dtype=jp.int32),
                "norm_ref": jp.array(norm["reference"], dtype=jp.float32),
                "norm_scale": jp.array(norm["scale"], dtype=jp.float32),
                "gear": jp.array(pd["gear"], dtype=jp.float64),
                "ctrl_lo": jp.array(pd["ctrl_lo"], dtype=jp.float64),
                "ctrl_hi": jp.array(pd["ctrl_hi"], dtype=jp.float64),
                "kp": jp.array(self.KP, dtype=jp.float64),
                "kd": jp.array(self.KD, dtype=jp.float64),
                "root_qpos_adr": r["root_qpos_adr"],
                "root_qvel_adr": r["root_qvel_adr"],
            }
        return statics

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_physical_frequency(self) -> float:
        return 1.0 / self.DT

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seeds: Optional[np.ndarray] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset all envs to initial pose."""
        # Build initial qpos/qvel on host, then transfer to device
        data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, data)

        dist = self._initial_distance
        if options and "initial_distance" in options:
            dist = float(options["initial_distance"])

        pose_a_name = self._initial_pose_a
        pose_b_name = self._initial_pose_b
        if options:
            pose_a_name = options.get("initial_pose_a", pose_a_name)
            pose_b_name = options.get("initial_pose_b", pose_b_name)

        pose_a = self.INITIAL_POSES[pose_a_name]
        pose_b = self.INITIAL_POSES[pose_b_name]

        for robot_id, pose_config, x_offset in [
            ("robot_a", pose_a, -dist / 2.0),
            ("robot_b", pose_b, dist / 2.0),
        ]:
            cache = self._robots[robot_id]
            root_qpos_adr = cache["root_qpos_adr"]
            qpos_indices = cache["qpos_indices"]

            root_pos = pose_config["root_pos"].copy()
            root_pos[0] = x_offset
            data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos

            root_quat = pose_config["root_quat"].copy()
            if robot_id == "robot_b":
                # Rotate 180° around z
                from scipy.spatial.transform import Rotation as Rot

                q_scipy = np.array(
                    [root_quat[1], root_quat[2], root_quat[3], root_quat[0]]
                )
                rot_orig = Rot.from_quat(q_scipy)
                rot_z = Rot.from_euler("z", np.pi, degrees=False)
                rot_new = rot_z * rot_orig
                q_new = rot_new.as_quat()  # [x,y,z,w]
                root_quat = np.array(
                    [q_new[3], q_new[0], q_new[1], q_new[2]], dtype=np.float32
                )

            data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_quat
            data.qpos[qpos_indices] = pose_config["joint_pos"]

        data.qvel[:] = 0.0
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self._model, data)

        # Put to MJX and broadcast to batch
        single = mjx.put_data(self._model, data)
        self._mjx_data = jax.tree.map(
            lambda x: jp.broadcast_to(x, (self._batch_size,) + x.shape), single
        )

        # Initialize action: compute from actual joint positions (same as original simulator)
        # rather than using pre-defined INITIAL_POSES['action'] which is a rounded approximation
        action_a = np.zeros(21, dtype=np.float32)
        action_b = np.zeros(21, dtype=np.float32)
        for robot_id, pose_config, _ in [
            ("robot_a", pose_a, -dist / 2.0),
            ("robot_b", pose_b, dist / 2.0),
        ]:
            cache = self._robots[robot_id]
            norm = self._norm_params[robot_id]
            qpos_indices = cache["qpos_indices"]
            actual_joint_pos = data.qpos[qpos_indices]
            action = ((actual_joint_pos - norm["reference"]) / norm["scale"]).astype(np.float32)
            if robot_id == "robot_a":
                action_a = action
            else:
                action_b = action
        self._action_jax = {
            "robot_a": jp.tile(action_a, (self._batch_size, 1)),
            "robot_b": jp.tile(action_b, (self._batch_size, 1)),
        }

        # Initialize persistent external force to zeros
        self._ext_force_jax = jp.zeros(
            (self._batch_size, self._model.nbody, 6), dtype=jp.float64
        )

        # Clear history
        self._history_buffer = None
        self._history_n_steps = 0

        # Build JIT-compiled step functions
        self._build_jit_functions()

    def _build_jit_functions(self):
        """Build JIT-compiled single-step and scan-step functions."""
        mjx_model = self._mjx_model
        statics = self._jax_statics

        def _apply_pd_and_step(data, action_a, action_b, ext_force):
            """Single physics step: apply PD control, set ext force, then mjx.step."""
            # Apply persistent external forces
            data = data.replace(xfrc_applied=ext_force)

            # PD control for each robot
            ctrl = jp.zeros(data.ctrl.shape, dtype=jp.float64)
            for robot_id, action in [("robot_a", action_a), ("robot_b", action_b)]:
                s = statics[robot_id]
                target_rad = action * s["norm_scale"] + s["norm_ref"]
                current_pos = data.qpos[s["qpos_indices"]]
                current_vel = data.qvel[s["qvel_indices"]]
                torque = s["kp"] * (target_rad - current_pos) - s["kd"] * current_vel
                ctrl_val = torque / s["gear"]
                ctrl_val = jp.clip(ctrl_val, s["ctrl_lo"], s["ctrl_hi"])
                ctrl = ctrl.at[s["actuator_ids"]].set(ctrl_val)

            data = data.replace(ctrl=ctrl)
            data = mjx.step(mjx_model, data)
            return data

        self._jit_step_vmap = jax.jit(
            jax.vmap(_apply_pd_and_step, in_axes=(0, 0, 0, 0))
        )

        def _scan_step(data, _, action_a, action_b, ext_force):
            d = _apply_pd_and_step(data, action_a, action_b, ext_force)
            # mjx.step may promote some int32 fields to int64; cast back to
            # match carry input types required by jax.lax.scan.
            d = jax.tree.map(
                lambda out, inp: out.astype(inp.dtype)
                if hasattr(out, "dtype") and hasattr(inp, "dtype")
                and out.dtype != inp.dtype
                else out,
                d, data,
            )
            return d, d

        # Cache of JIT-compiled scan functions, keyed by n_steps.
        # jax.lax.scan requires length as a concrete Python int, so we
        # build a separate JIT function per n_steps value.
        self._jit_scan_cache: Dict[int, Any] = {}

        def _get_scan_fn(n_steps: int):
            if n_steps not in self._jit_scan_cache:
                def _scan_single(data, action_a, action_b, ext_force):
                    final, hist = jax.lax.scan(
                        lambda carry, _: _scan_step(
                            carry, _, action_a, action_b, ext_force
                        ),
                        data,
                        None,
                        length=n_steps,
                    )
                    return final, hist

                self._jit_scan_cache[n_steps] = jax.jit(
                    jax.vmap(_scan_single, in_axes=(0, 0, 0, 0))
                )
            return self._jit_scan_cache[n_steps]

        self._get_scan_fn = _get_scan_fn

    # ------------------------------------------------------------------
    # physical_step
    # ------------------------------------------------------------------
    def physical_step(self, n_steps: int = 1, keep_history: bool = False) -> None:
        if self._mjx_data is None:
            raise RuntimeError("Call reset() before physical_step()")

        # Clear history buffer at the start of each physical_step
        self._history_buffer = None
        self._history_n_steps = 0

        action_a = self._action_jax["robot_a"]
        action_b = self._action_jax["robot_b"]

        if n_steps == 1 and not keep_history:
            self._mjx_data = self._jit_step_vmap(
                self._mjx_data, action_a, action_b, self._ext_force_jax
            )
        else:
            scan_fn = self._get_scan_fn(n_steps)
            final, history = scan_fn(
                self._mjx_data, action_a, action_b, self._ext_force_jax
            )
            self._mjx_data = final
            if keep_history:
                self._history_buffer = history
                self._history_n_steps = n_steps

    def _single_step(self) -> None:
        """Fallback for BaseBatchSimulator default implementation."""
        self.physical_step(n_steps=1, keep_history=False)

    # ------------------------------------------------------------------
    # get_core_state
    # ------------------------------------------------------------------
    def get_core_state(self, history: bool = False) -> Dict[str, Any]:
        if history:
            if self._history_buffer is None:
                return {}
            return self._extract_core_state(self._history_buffer, is_history=True)
        return self._extract_core_state(self._mjx_data, is_history=False)

    def _extract_core_state(self, data: mjx.Data, is_history: bool) -> Dict[str, Any]:
        """Extract core state from MJX data (JAX → numpy)."""
        result: Dict[str, Any] = {}

        for robot_id in ["robot_a", "robot_b"]:
            r = self._robots[robot_id]
            norm = self._norm_params[robot_id]
            s = self._jax_statics[robot_id]

            root_qa = r["root_qpos_adr"]
            root_qva = r["root_qvel_adr"]
            qpos_idx = r["qpos_indices"]
            qvel_idx = r["qvel_indices"]

            # Root pos and rot
            root_pos = data.qpos[..., root_qa : root_qa + 3]
            root_rot = data.qpos[..., root_qa + 3 : root_qa + 7]

            # Root velocity (global → local)
            root_vel_global = data.qvel[..., root_qva : root_qva + 3]
            root_ang_vel_global = data.qvel[..., root_qva + 3 : root_qva + 6]

            # Convert to numpy first for rotation computation
            root_pos_np = np.asarray(root_pos)
            root_rot_np = np.asarray(root_rot)
            root_vel_np = np.asarray(root_vel_global)
            root_ang_np = np.asarray(root_ang_vel_global)

            # Batch rotation inverse: (B, 3, 3) from quaternion
            w, x, y, z = (
                root_rot_np[..., 0],
                root_rot_np[..., 1],
                root_rot_np[..., 2],
                root_rot_np[..., 3],
            )
            R_mat = np.stack([
                np.stack([
                    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
                ], axis=-1),
                np.stack([
                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
                ], axis=-1),
                np.stack([
                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
                ], axis=-1),
            ], axis=-2)  # (..., 3, 3)

            R_inv = np.swapaxes(R_mat, -1, -2)  # (..., 3, 3)
            root_vel_local = np.einsum("...ij,...j->...i", R_inv, root_vel_np)
            root_ang_vel_local = np.einsum("...ij,...j->...i", R_inv, root_ang_np)

            # Joint pos/vel
            joint_pos = np.asarray(data.qpos[..., qpos_idx])
            joint_vel = np.asarray(data.qvel[..., qvel_idx])

            ref = norm["reference"]
            scale = norm["scale"]
            joint_pos_norm = (joint_pos - ref) / scale
            joint_vel_norm = joint_vel / scale

            result[robot_id] = {
                "root_pos": root_pos_np.astype(np.float32),
                "root_rot": root_rot_np.astype(np.float32),
                "root_vel_local": root_vel_local.astype(np.float32),
                "root_angular_vel_local": root_ang_vel_local.astype(np.float32),
                "joint_pos_norm": joint_pos_norm.astype(np.float32),
                "joint_vel_norm": joint_vel_norm.astype(np.float32),
            }

        return result

    # ------------------------------------------------------------------
    # get_derived_state
    # ------------------------------------------------------------------
    def get_derived_state(
        self,
        fields: Optional[Sequence[str]] = None,
        history: bool = False,
    ) -> Dict[str, Any]:
        if history:
            if self._history_buffer is None:
                return {}
            return self._extract_derived_state(self._history_buffer, fields, is_history=True)
        return self._extract_derived_state(self._mjx_data, fields, is_history=False)

    def _extract_derived_state(
        self, data: mjx.Data, fields: Optional[Sequence[str]], is_history: bool
    ) -> Dict[str, Any]:
        if fields is None:
            fields = ["torso_distance", "contacts", "robot_a", "robot_b"]
        else:
            fields = list(fields)
            unknown = set(fields) - {"torso_distance", "contacts", "robot_a", "robot_b"}
            if unknown:
                raise KeyError(f"get_derived_state: unknown fields {unknown}")

        result: Dict[str, Any] = {}

        if "torso_distance" in fields:
            torso_a_id = self._robots["robot_a"]["root_body_id"]
            torso_b_id = self._robots["robot_b"]["root_body_id"]
            pos_a = np.asarray(data.xpos[..., torso_a_id, :])
            pos_b = np.asarray(data.xpos[..., torso_b_id, :])
            dist = np.linalg.norm(pos_b - pos_a, axis=-1, keepdims=True)
            result["torso_distance"] = dist.astype(np.float32)

        if "contacts" in fields:
            result["contacts"] = self._extract_contacts_batch(data)

        for rid in ("robot_a", "robot_b"):
            if rid in fields:
                opp_id = "robot_b" if rid == "robot_a" else "robot_a"
                result[rid] = self._get_robot_view_batch(data, rid, opp_id)

        return result

    def _extract_contacts_batch(self, data: mjx.Data) -> Dict[str, Any]:
        """Extract contacts from batched MJX data.

        Uses efc_force to compute proper contact forces matching
        mujoco.mj_contactForce. For pyramidal cone with condim=3:
          normal = sum(efc[0:4])
          friction1 = efc[0] - efc[1]
          friction2 = efc[2] - efc[3]
        For condim=1 (frictionless): normal = efc[0].

        force_world = frame.T @ [normal, friction1, friction2]
        """
        contact = data._impl.contact
        geom = np.asarray(contact.geom)
        dist = np.asarray(contact.dist)
        pos = np.asarray(contact.pos)
        frame = np.asarray(contact.frame)
        efc_address = np.asarray(contact.efc_address)
        contact_dim = np.asarray(contact.dim)
        efc_force = np.asarray(data._impl.efc_force)

        # Determine if batched
        if geom.ndim == 2:
            geom = geom[np.newaxis]
            dist = dist[np.newaxis]
            pos = pos[np.newaxis]
            frame = frame[np.newaxis]
            efc_force = efc_force[np.newaxis]

        # efc_address and contact_dim are NOT batched (shared model metadata)
        B, max_contacts = dist.shape
        active = dist <= 0  # (B, max_contacts)

        # Body IDs from geom IDs
        geom_bodyid = np.asarray(self._model.geom_bodyid)
        body1 = geom_bodyid[geom[..., 0].astype(np.int64)]
        body2 = geom_bodyid[geom[..., 1].astype(np.int64)]

        # Affiliation
        geom_id_to_aff = self._meta["geom_id_to_aff"]
        aff_table = np.zeros(self._model.ngeom, dtype=np.int8)
        for gid, aff in geom_id_to_aff.items():
            aff_table[gid] = aff
        aff1 = aff_table[geom[..., 0].astype(np.int64)]
        aff2 = aff_table[geom[..., 1].astype(np.int64)]

        contact_count = np.sum(active, axis=-1).astype(np.int32)

        # Compute contact forces from efc_force.
        # For pyramidal cone with dim=3: 4 constraint rows per contact.
        # For dim=1: 1 constraint row.
        # n_rows = max(1, 2*(dim-1)) for pyramidal cone.
        n_rows = np.maximum(1, 2 * (contact_dim - 1))  # (max_contacts,)

        # Gather efc_force for each contact
        # efc_address: (max_contacts,) — NOT batched (shared model)
        # efc_force: (B, max_efc) — batched
        force_mag = np.zeros((B, max_contacts), dtype=np.float64)
        force_world = np.zeros((B, max_contacts, 3), dtype=np.float64)

        for b in range(B):
            for c in range(max_contacts):
                if not active[b, c]:
                    continue
                addr = int(efc_address[c])
                nr = int(n_rows[c])
                ef = efc_force[b, addr:addr + nr]

                if nr == 1:
                    normal = ef[0]
                    f1, f2 = 0.0, 0.0
                elif nr == 4:
                    normal = ef[0] + ef[1] + ef[2] + ef[3]
                    f1 = ef[0] - ef[1]
                    f2 = ef[2] - ef[3]
                else:
                    normal = ef.sum()
                    f1 = ef[0] - ef[1] if nr >= 2 else 0.0
                    f2 = ef[2] - ef[3] if nr >= 4 else 0.0

                force_local = np.array([normal, f1, f2])
                force_mag[b, c] = np.linalg.norm(force_local)
                # frame[b,c] is (3,3) with rows [normal, tangent1, tangent2]
                # force_world = frame.T @ force_local (matches original simulator)
                force_world[b, c] = frame[b, c].T @ force_local

        return {
            "ncon": int(max_contacts),
            "contact_count": contact_count,
            "active_mask": active,
            "geom1": geom[..., 0].astype(np.int32),
            "geom2": geom[..., 1].astype(np.int32),
            "body1": body1.astype(np.int32),
            "body2": body2.astype(np.int32),
            "aff1": aff1,
            "aff2": aff2,
            "force_mag": force_mag.astype(np.float32),
            "force_world": force_world.astype(np.float32),
            "position": pos.astype(np.float32),
            "normal": frame[..., 0, :].astype(np.float32),
            "frame": frame.astype(np.float32),
        }

    def _get_robot_view_batch(
        self, data: mjx.Data, robot_id: str, opponent_id: str
    ) -> Dict[str, Any]:
        """Get per-robot observation view (batched)."""
        cache = self._robots[robot_id]
        opp_cache = self._robots[opponent_id]
        norm = self._norm_params[robot_id]

        torso_id = cache["root_body_id"]
        opp_torso_id = opp_cache["root_body_id"]

        self_pos = np.asarray(data.xpos[..., torso_id, :])  # (B, ..., 3)
        self_quat = np.asarray(data.xpos[..., torso_id, :])  # placeholder
        self_quat = np.asarray(data.xquat[..., torso_id, :])  # (B, ..., 4) [w,x,y,z]
        opp_pos = np.asarray(data.xpos[..., opp_torso_id, :])
        opp_quat = np.asarray(data.xquat[..., opp_torso_id, :])

        # Build rotation matrix from quaternion
        w, x, y, z = self_quat[..., 0], self_quat[..., 1], self_quat[..., 2], self_quat[..., 3]
        R_self = np.stack([
            np.stack([
                1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
            ], axis=-1),
            np.stack([
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
            ], axis=-1),
            np.stack([
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
            ], axis=-1),
        ], axis=-2)  # (..., 3, 3)
        R_self_inv = np.swapaxes(R_self, -1, -2)

        # Height
        height = self_pos[..., 2:3]  # (..., 1)

        # Local orientation (first 2 columns of R_self, transposed and flattened)
        local_orientation = np.swapaxes(R_self[..., :, :2], -1, -2)  # (..., 2, 3)
        local_orientation = local_orientation.reshape(*local_orientation.shape[:-2], 6)

        # Velocities (global)
        root_qva = cache["root_qvel_adr"]
        linear_vel = np.asarray(data.qvel[..., root_qva : root_qva + 3])
        angular_vel = np.asarray(data.qvel[..., root_qva + 3 : root_qva + 6])

        # Opponent basic pose
        relative_pos = opp_pos - self_pos
        relative_pos_local = np.einsum("...ij,...j->...i", R_self_inv, relative_pos)
        relative_vel_local = np.einsum("...ij,...j->...i", R_self_inv, linear_vel)

        # Opponent face vector
        ow, ox, oy, oz = opp_quat[..., 0], opp_quat[..., 1], opp_quat[..., 2], opp_quat[..., 3]
        opp_R = np.stack([
            np.stack([
                1 - 2 * (oy * oy + oz * oz), 2 * (ox * oy - oz * ow), 2 * (ox * oz + oy * ow)
            ], axis=-1),
            np.stack([
                2 * (ox * oy + oz * ow), 1 - 2 * (ox * ox + oz * oz), 2 * (oy * oz - ox * ow)
            ], axis=-1),
            np.stack([
                2 * (ox * oz - oy * ow), 2 * (oy * oz + ox * ow), 1 - 2 * (ox * ox + oy * oy)
            ], axis=-1),
        ], axis=-2)
        opp_forward = opp_R[..., :, 0]  # (..., 3) — first column = local x axis
        face_vector = np.einsum("...ij,...j->...i", R_self_inv, opp_forward)

        # Opponent keypoints
        kp = opp_cache["keypoint_body_ids"]
        kp_names = ["head", "hand_right", "hand_left", "foot_right", "foot_left"]
        kp_pos_local = {}
        kp_vel_local = {}
        for name in kp_names:
            bid = kp[name]
            kp_pos = np.asarray(data.xpos[..., bid, :])
            delta = kp_pos - self_pos
            kp_pos_local[name] = np.einsum("...ij,...j->...i", R_self_inv, delta).astype(np.float32)
            kp_vel = np.asarray(data.cvel[..., bid, 3:6])
            kp_vel_local[name] = np.einsum("...ij,...j->...i", R_self_inv, kp_vel).astype(np.float32)

        # Feet forces (simplified: from contact force magnitude)
        feet_forces = self._get_feet_forces_batch(data, robot_id)

        # Proprioception
        qpos_idx = cache["qpos_indices"]
        qvel_idx = cache["qvel_indices"]
        joint_pos = np.asarray(data.qpos[..., qpos_idx])
        joint_vel = np.asarray(data.qvel[..., qvel_idx])
        joint_pos_norm = (joint_pos - norm["reference"]) / norm["scale"]
        joint_vel_norm = joint_vel / norm["scale"]
        proprioception = np.concatenate(
            [joint_pos_norm, joint_vel_norm], axis=-1
        ).astype(np.float32)

        # Flat observation (96-dim)
        observation = np.concatenate([
            proprioception,
            local_orientation,
            height,
            linear_vel,
            angular_vel,
            feet_forces,
            relative_pos_local,
            relative_vel_local,
            face_vector,
            kp_pos_local["head"],
            kp_pos_local["hand_right"],
            kp_pos_local["hand_left"],
            kp_pos_local["foot_right"],
            kp_pos_local["foot_left"],
            kp_vel_local["head"],
            kp_vel_local["hand_right"],
            kp_vel_local["hand_left"],
            kp_vel_local["foot_right"],
            kp_vel_local["foot_left"],
        ], axis=-1).astype(np.float32)

        # Body arrays
        body_ids = cache["body_ids_sorted"]
        body_names = cache["body_names"]
        body_xpos = np.asarray(data.xpos[..., body_ids, :]).astype(np.float32)

        body_xpos_dict = {
            name: body_xpos[..., i, :] for i, name in enumerate(body_names)
        }

        return {
            "root_state": {
                "height": height.astype(np.float32),
                "local_orientation": local_orientation.astype(np.float32),
                "linear_vel": linear_vel.astype(np.float32),
                "angular_vel": angular_vel.astype(np.float32),
            },
            "feet_forces": feet_forces,
            "opponent_basic_pose": {
                "relative_pos": relative_pos_local.astype(np.float32),
                "relative_vel": relative_vel_local.astype(np.float32),
                "face_vector": face_vector.astype(np.float32),
            },
            "opponent_keypoint_pos": kp_pos_local,
            "opponent_keypoint_vel": kp_vel_local,
            "observation": observation,
            "uprightness": np.asarray(R_self[..., 2, 2:3]).astype(np.float32),
            "opponent_in_local": {
                "pos": relative_pos_local.astype(np.float32),
                "vel": relative_vel_local.astype(np.float32),
                "rot": face_vector.astype(np.float32),
            },
            "body_xpos": body_xpos_dict,
        }

    def _get_feet_forces_batch(self, data: mjx.Data, robot_id: str) -> np.ndarray:
        """Get feet contact forces, **normalized by body weight m*g** (dimensionless).

        Uses the same force_mag computed in _extract_contacts_batch (from efc_force),
        matching the original simulator's _get_feet_forces which uses mj_contactForce output.

        The division by ``body_weight`` must stay in lockstep with
        ``Humanoid21Simulator._get_feet_forces`` — the two backends are
        required to produce bit-comparable observations, so a unit change
        in one without the other would silently desync them. See that
        method's docstring for why the normalization exists.
        """
        cache = self._robots[robot_id]
        kp = cache["keypoint_body_ids"]
        foot_right_id = kp["foot_right"]
        foot_left_id = kp["foot_left"]
        ground_gid = self._ground_geom_id
        body_weight = cache["body_weight"]

        # Extract contacts to get force_mag (same as _extract_contacts_batch)
        contacts = self._extract_contacts_batch(data)

        geom1 = contacts["geom1"]  # (B, max_contacts)
        geom2 = contacts["geom2"]  # (B, max_contacts)
        body1 = contacts["body1"]  # (B, max_contacts)
        body2 = contacts["body2"]  # (B, max_contacts)
        force_mag = contacts["force_mag"]  # (B, max_contacts)
        active_mask = contacts["active_mask"]  # (B, max_contacts)

        g1_ground = geom1 == ground_gid
        g2_ground = geom2 == ground_gid
        ground_mask = (g1_ground | g2_ground) & active_mask

        other_body = np.where(g1_ground, body2, body1)

        right_force = np.sum(
            np.where(ground_mask & (other_body == foot_right_id), force_mag, 0.0),
            axis=-1,
        )
        left_force = np.sum(
            np.where(ground_mask & (other_body == foot_left_id), force_mag, 0.0),
            axis=-1,
        )

        return (
            np.stack([right_force, left_force], axis=-1) / body_weight
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # get_observation
    # ------------------------------------------------------------------
    def get_observation(self) -> Dict[str, Any]:
        result = {}
        for rid in ("robot_a", "robot_b"):
            opp_id = "robot_b" if rid == "robot_a" else "robot_a"
            view = self._get_robot_view_batch(self._mjx_data, rid, opp_id)
            result[rid] = view["observation"]
        return result

    # ------------------------------------------------------------------
    # get_sensor_data / get_action / get_static_data
    # ------------------------------------------------------------------
    def get_sensor_data(self) -> Dict[str, Any]:
        return {}

    def get_action(self) -> Dict[str, Any]:
        if self._action_jax is None:
            return {}
        return {
            "robot_a": np.asarray(self._action_jax["robot_a"]),
            "robot_b": np.asarray(self._action_jax["robot_b"]),
        }

    def get_static_data(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for robot_id in ["robot_a", "robot_b"]:
            cache = self._robots[robot_id]
            body_names = list(cache["body_names"])
            body_masses = np.asarray(cache["body_masses"], dtype=np.float32)
            result[robot_id] = {
                "dof_names": list(self.CONTROLLED_JOINTS),
                "body_names": body_names,
                "body_masses_by_name": {
                    name: float(mass) for name, mass in zip(body_names, body_masses)
                },
                "joint_names": list(cache["joint_names"]),
                "controlled_joint_names": list(cache["controlled_joint_names"]),
                "root_joint_name": cache["root_joint_name"],
                "keypoint_body_names": dict(cache["keypoint_body_names"]),
                "keypoint_joint_names": dict(cache["keypoint_joint_names"]),
                "joint_limits": cache["jnt_ranges"].copy(),
            }
        result["dt"] = float(self.DT)
        result["ground_geom_name"] = "ground"
        result["ground_geom_id"] = self._ground_geom_id
        result["geom_id_to_name"] = dict(self._meta["geom_id_to_name"])
        result["body_id_to_name"] = dict(self._meta["body_id_to_name"])
        result["body_id_to_aff"] = dict(self._meta["body_id_to_aff"])
        result["geom_id_to_aff"] = dict(self._meta["geom_id_to_aff"])
        return result

    # ------------------------------------------------------------------
    # set_action
    # ------------------------------------------------------------------
    def set_action(self, action: Dict[str, Any]) -> None:
        for robot_id in ["robot_a", "robot_b"]:
            if robot_id in action and action[robot_id] is not None:
                act = np.asarray(action[robot_id], dtype=np.float32)
                if self._action_jax is None:
                    self._action_jax = {}
                self._action_jax[robot_id] = jp.array(np.clip(act, -1.0, 1.0))

    # ------------------------------------------------------------------
    # set_core_state
    # ------------------------------------------------------------------
    def set_core_state(
        self,
        state: Dict[str, Any],
        env_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if self._mjx_data is None:
            raise RuntimeError("Call reset() before set_core_state()")

        if env_ids is None:
            env_ids = list(range(self._batch_size))

        env_ids = list(env_ids)
        n = len(env_ids)

        # Build new qpos/qvel on host for the target envs
        qpos_new = np.asarray(self._mjx_data.qpos).copy()  # (B, nq)
        qvel_new = np.asarray(self._mjx_data.qvel).copy()  # (B, nv)

        for robot_id in ["robot_a", "robot_b"]:
            if robot_id not in state:
                continue
            robot_state = state[robot_id]
            cache = self._robots[robot_id]
            norm = self._norm_params[robot_id]
            root_qa = cache["root_qpos_adr"]
            root_qva = cache["root_qvel_adr"]
            qpos_idx = cache["qpos_indices"]
            qvel_idx = cache["qvel_indices"]

            for i, eid in enumerate(env_ids):
                if "root_pos" in robot_state:
                    qpos_new[eid, root_qa : root_qa + 3] = robot_state["root_pos"][i]
                if "root_rot" in robot_state:
                    qpos_new[eid, root_qa + 3 : root_qa + 7] = robot_state["root_rot"][i]
                if "root_vel_local" in robot_state or "root_angular_vel_local" in robot_state:
                    quat = qpos_new[eid, root_qa + 3 : root_qa + 7]
                    w, x, y, z = quat
                    R_mat = np.array([
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ])
                    if "root_vel_local" in robot_state:
                        local_vel = robot_state["root_vel_local"][i]
                        qvel_new[eid, root_qva : root_qva + 3] = R_mat @ local_vel
                    if "root_angular_vel_local" in robot_state:
                        local_ang = robot_state["root_angular_vel_local"][i]
                        qvel_new[eid, root_qva + 3 : root_qva + 6] = R_mat @ local_ang
                if "joint_pos_norm" in robot_state:
                    jpn = robot_state["joint_pos_norm"][i]
                    qpos_new[eid, qpos_idx] = jpn * norm["scale"] + norm["reference"]
                if "joint_vel_norm" in robot_state:
                    jvn = robot_state["joint_vel_norm"][i]
                    qvel_new[eid, qvel_idx] = jvn * norm["scale"]

        # Transfer to device
        self._mjx_data = self._mjx_data.replace(
            qpos=jp.array(qpos_new), qvel=jp.array(qvel_new)
        )

    # ------------------------------------------------------------------
    # apply_external_force
    # ------------------------------------------------------------------
    def apply_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: Optional[np.ndarray] = None,
        robot_id: str = "robot_a",
    ) -> None:
        suffix = self._robots[robot_id]["suffix"]
        full_body_name = f"{body_name}{suffix}"
        body_id = self._body_name_to_id.get(full_body_name)
        if body_id is None:
            raise ValueError(f"Body not found: {full_body_name}")

        force = np.asarray(force, dtype=np.float32)  # (B, 3)
        ext = np.asarray(self._ext_force_jax).copy()  # (B, nbody, 6)
        ext[:, body_id, :3] = force
        if torque is not None:
            torque = np.asarray(torque, dtype=np.float32)  # (B, 3)
            ext[:, body_id, 3:6] = torque
        self._ext_force_jax = jp.array(ext)

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._mjx_data = None
        self._history_buffer = None
