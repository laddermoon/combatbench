import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import BaseSimulator

# MuJoCo/EGL cleanup quirk:
# In some environments, Python interpreter teardown can trigger EGL cleanup
# errors (e.g. EGL_NOT_INITIALIZED) from Renderer/GLContext destructors.
# These are noisy but non-fatal for completed rollouts/videos.
def _is_ignorable_egl_teardown_error(exc: Exception) -> bool:
    msg = str(exc)
    if isinstance(exc, AttributeError) and "_gl_context" in msg:
        return True
    name = exc.__class__.__name__
    if "EGLError" in name and ("EGL_NOT_INITIALIZED" in msg or "eglMakeCurrent" in msg):
        return True
    return False


if not getattr(mujoco.Renderer, "_t800_safe_del_patched", False):
    _orig_renderer_del = getattr(mujoco.Renderer, "__del__", None)

    def _safe_renderer_del(self):  # type: ignore[no-untyped-def]
        if _orig_renderer_del is None:
            return
        try:
            _orig_renderer_del(self)
        except Exception as e:
            if _is_ignorable_egl_teardown_error(e):
                return
            raise

    mujoco.Renderer.__del__ = _safe_renderer_del  # type: ignore[assignment]
    mujoco.Renderer._t800_safe_del_patched = True  # type: ignore[attr-defined]


try:
    _GLContext = mujoco.egl.GLContext  # type: ignore[attr-defined]
except Exception:
    _GLContext = None

if _GLContext is not None and not getattr(_GLContext, "_t800_safe_del_patched", False):
    _orig_glctx_del = getattr(_GLContext, "__del__", None)

    def _safe_glctx_del(self):  # type: ignore[no-untyped-def]
        if _orig_glctx_del is None:
            return
        try:
            _orig_glctx_del(self)
        except Exception as e:
            if _is_ignorable_egl_teardown_error(e):
                return
            raise

    _GLContext.__del__ = _safe_glctx_del  # type: ignore[assignment]
    _GLContext._t800_safe_del_patched = True  # type: ignore[attr-defined]


class T800Simulator(BaseSimulator):
    """T800 双机器人 MuJoCo 仿真器。"""

    DT = 0.002
    ACTION_DIM = 25
    ARENA_XML = str(Path(__file__).parent / "battle_t800_full.xml")

    CONTROLLED_JOINTS = [
        "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
        "J12_TORSO_YAW",
        "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
        "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
        "J23_HEAD_PITCH", "J24_HEAD_YAW",
    ]

    KEYPOINT_BODY_MAP = {
        "torso": "LINK_BASE",
        "pelvis": "LINK_WAIST_YAW",
        "head": "LINK_HEAD_YAW",
        "foot_left": "LINK_FOOT_L",
        "foot_right": "LINK_FOOT_R",
        "hand_left": "LINK_WRIST_END_L",
        "hand_right": "LINK_WRIST_END_R",
    }

    KP = np.array([
        220.0, 180.0, 180.0, 240.0, 120.0, 100.0,
        220.0, 180.0, 180.0, 240.0, 120.0, 100.0,
        260.0,
        140.0, 130.0, 120.0, 110.0, 80.0,
        140.0, 130.0, 120.0, 110.0, 80.0,
        60.0, 60.0,
    ], dtype=np.float32)

    KD = np.array([
        26.0, 20.0, 20.0, 28.0, 12.0, 10.0,
        26.0, 20.0, 20.0, 28.0, 12.0, 10.0,
        30.0,
        14.0, 13.0, 12.0, 11.0, 8.0,
        14.0, 13.0, 12.0, 11.0, 8.0,
        6.0, 6.0,
    ], dtype=np.float32)

    # 初始姿态配置（先提供 standing，后续可扩展 squat/prone 等）
    INITIAL_POSES = {
        "standing": {
            "root_pos": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "root_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # wxyz
            # 25-dof: 轻微屈膝站立，避免完全伸直带来的不稳定
            # order follows CONTROLLED_JOINTS
            "joint_pos": np.array([
                -0.12, 0.0, 0.0, 0.24, -0.12, 0.0,   # left leg
                -0.12, 0.0, 0.0, 0.24, -0.12, 0.0,   # right leg
                0.0,                                   # torso yaw
                0.0, 0.0, 0.0, 0.0, 0.0,              # left arm
                0.0, 0.0, 0.0, 0.0, 0.0,              # right arm
                0.0, 0.0,                              # head
            ], dtype=np.float32),
        },
    }

    def __init__(
        self,
        initial_distance: float = 3.0,
        initial_pose_a: str = "standing",
        initial_pose_b: str = "standing",
    ):
        self.dt = self.DT
        self.initial_distance = float(initial_distance)
        self.action_dim = self.ACTION_DIM
        valid_poses = list(self.INITIAL_POSES.keys())
        if initial_pose_a not in valid_poses:
            raise ValueError(f"initial_pose_a must be one of {valid_poses}, got {initial_pose_a}")
        if initial_pose_b not in valid_poses:
            raise ValueError(f"initial_pose_b must be one of {valid_poses}, got {initial_pose_b}")
        self._initial_pose_a = initial_pose_a
        self._initial_pose_b = initial_pose_b

        self.model = mujoco.MjSpec.from_file(self.ARENA_XML).compile()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.DT
        mujoco.mj_forward(self.model, self.data)

        self._cache_robot_indices()
        self._compute_normalization_params()

        self._target_pos_norm = {
            "robot_a": np.zeros(self.ACTION_DIM, dtype=np.float32),
            "robot_b": np.zeros(self.ACTION_DIM, dtype=np.float32),
        }
        self._last_action = {
            "robot_a": np.zeros(self.ACTION_DIM, dtype=np.float32),
            "robot_b": np.zeros(self.ACTION_DIM, dtype=np.float32),
        }
        self._renderer: Optional[mujoco.Renderer] = None
        # Align output resolution with humanoid21 renderer.
        self._render_width = 1280
        self._render_height = 720
        self._prev_azi: Optional[float] = None
        self._prev_ele: Optional[float] = None
        self._prev_dist: Optional[float] = None
        self._prev_lookat: Optional[np.ndarray] = None

    def _collect_subtree_body_ids(self, root_body_id: int) -> List[int]:
        body_ids: List[int] = []
        stack = [root_body_id]
        while stack:
            bid = stack.pop()
            body_ids.append(bid)
            for cid in range(self.model.nbody):
                if int(self.model.body_parentid[cid]) == bid and cid != bid:
                    stack.append(cid)
        body_ids.sort()
        return body_ids

    def _cache_robot_indices(self) -> None:
        self._robot_cache: Dict[str, Dict[str, Any]] = {}
        for robot_id, suffix in [("robot_a", "_red"), ("robot_b", "_blue")]:
            cache: Dict[str, Any] = {"suffix": suffix}
            base_name = f"LINK_BASE{suffix}"
            base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_name)
            if base_bid < 0:
                raise ValueError(f"Body {base_name} not found")
            cache["base_body_id"] = base_bid

            jnt_adr = self.model.body_jntadr[base_bid]
            jnt_num = self.model.body_jntnum[base_bid]
            if jnt_num <= 0:
                raise ValueError(f"Body {base_name} has no root joint")
            root_jid = int(jnt_adr)
            cache["root_joint_id"] = root_jid
            cache["root_qpos_adr"] = int(self.model.jnt_qposadr[root_jid])
            cache["root_qvel_adr"] = int(self.model.jnt_dofadr[root_jid])
            cache["root_joint_name"] = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, root_jid) or f"freejoint{suffix}"

            body_ids = self._collect_subtree_body_ids(base_bid)
            cache["body_ids"] = body_ids
            body_names: List[str] = []
            body_masses_by_name: Dict[str, float] = {}
            for bid in body_ids:
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if bname is not None:
                    body_names.append(bname)
                    body_masses_by_name[bname] = float(self.model.body_mass[bid])
            cache["body_names"] = body_names
            cache["body_masses_by_name"] = body_masses_by_name

            qpos_indices: List[int] = []
            qvel_indices: List[int] = []
            actuator_ids: List[int] = []
            jnt_ranges: List[np.ndarray] = []
            full_joint_names: List[str] = []
            for jn in self.CONTROLLED_JOINTS:
                full_jn = f"{jn}{suffix}"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full_jn)
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{jn}{suffix}")
                if jid < 0 or aid < 0:
                    raise ValueError(f"Joint/actuator missing: {full_jn}")
                qpos_indices.append(int(self.model.jnt_qposadr[jid]))
                qvel_indices.append(int(self.model.jnt_dofadr[jid]))
                actuator_ids.append(int(aid))
                jnt_ranges.append(self.model.jnt_range[jid].copy())
                full_joint_names.append(full_jn)

            cache["joint_qpos_indices"] = np.array(qpos_indices, dtype=np.int32)
            cache["joint_qvel_indices"] = np.array(qvel_indices, dtype=np.int32)
            cache["actuator_ids"] = np.array(actuator_ids, dtype=np.int32)
            cache["jnt_ranges"] = np.array(jnt_ranges, dtype=np.float32)
            cache["controlled_joint_names"] = full_joint_names

            joint_names: List[str] = []
            if jnt_num > 0:
                for jid in range(jnt_adr, jnt_adr + jnt_num):
                    jn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if jn is not None:
                        joint_names.append(jn)
            for jn in full_joint_names:
                if jn not in joint_names:
                    joint_names.append(jn)
            cache["joint_names"] = joint_names

            keypoint_body_names: Dict[str, str] = {}
            for k, base in self.KEYPOINT_BODY_MAP.items():
                bn = f"{base}{suffix}"
                if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bn) >= 0:
                    keypoint_body_names[k] = bn
            cache["keypoint_body_names"] = keypoint_body_names

            self._robot_cache[robot_id] = cache

    def _compute_normalization_params(self) -> None:
        self._norm: Dict[str, Dict[str, np.ndarray]] = {}
        for robot_id, cache in self._robot_cache.items():
            limits = cache["jnt_ranges"]
            lower = limits[:, 0]
            upper = limits[:, 1]
            scale = (upper - lower) / 2.0
            scale = np.where(scale < 1e-6, 1.0, scale)
            ref = (upper + lower) / 2.0
            self._norm[robot_id] = {"ref": ref.astype(np.float32), "scale": scale.astype(np.float32)}

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        distance = self.initial_distance
        if options and "initial_distance" in options:
            distance = float(options["initial_distance"])
        pose_a_name = options.get("initial_pose_a", self._initial_pose_a) if options else self._initial_pose_a
        pose_b_name = options.get("initial_pose_b", self._initial_pose_b) if options else self._initial_pose_b
        pose_a = self.INITIAL_POSES[pose_a_name]
        pose_b = self.INITIAL_POSES[pose_b_name]

        for robot_id, pose, sign in [("robot_a", pose_a, -1.0), ("robot_b", pose_b, 1.0)]:
            cache = self._robot_cache[robot_id]
            qadr = cache["root_qpos_adr"]
            root_pos = pose["root_pos"].copy()
            root_pos[0] += sign * distance / 2.0
            self.data.qpos[qadr:qadr + 3] = root_pos

            root_quat = pose["root_quat"].copy()
            if robot_id == "robot_b":
                q_scipy = np.array([root_quat[1], root_quat[2], root_quat[3], root_quat[0]], dtype=np.float32)
                rot_original = R.from_quat(q_scipy)
                rot_z = R.from_euler("z", np.pi, degrees=False)
                q_new = (rot_z * rot_original).as_quat()  # xyzw
                root_quat = np.array([q_new[3], q_new[0], q_new[1], q_new[2]], dtype=np.float32)
            self.data.qpos[qadr + 3:qadr + 7] = root_quat

            self.data.qpos[cache["joint_qpos_indices"]] = pose["joint_pos"]
            self.data.qvel[cache["root_qvel_adr"]:cache["root_qvel_adr"] + 6] = 0.0
            self.data.qvel[cache["joint_qvel_indices"]] = 0.0
            self._last_action[robot_id].fill(0.0)

        # 让 PD 控制目标与初始关节姿态一致，避免 reset 后第一步产生突兀力矩
        for robot_id in ["robot_a", "robot_b"]:
            cache = self._robot_cache[robot_id]
            norm = self._norm[robot_id]
            qpos = self.data.qpos[cache["joint_qpos_indices"]]
            self._target_pos_norm[robot_id] = ((qpos - norm["ref"]) / norm["scale"]).astype(np.float32)
        # 参照 humanoid21：reset 时重置广播镜头状态缓存
        self._prev_azi = None
        self._prev_ele = None
        self._prev_dist = None
        self._prev_lookat = None
        mujoco.mj_forward(self.model, self.data)

    def physical_step(self) -> None:
        for robot_id, cache in self._robot_cache.items():
            norm = self._norm[robot_id]
            target = self._target_pos_norm[robot_id] * norm["scale"] + norm["ref"]
            qpos = self.data.qpos[cache["joint_qpos_indices"]]
            qvel = self.data.qvel[cache["joint_qvel_indices"]]
            torque = self.KP * (target - qpos) - self.KD * qvel
            for i, aid in enumerate(cache["actuator_ids"]):
                gear = self.model.actuator_gear[aid, 0] if self.model.actuator_gear[aid, 0] != 0 else 1.0
                ctrl = torque[i] / gear
                cmin, cmax = self.model.actuator_ctrlrange[aid]
                self.data.ctrl[aid] = np.clip(ctrl, cmin, cmax)
        mujoco.mj_step(self.model, self.data)

    def get_physical_frequency(self) -> float:
        return 1.0 / self.DT

    def get_static_data(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"dt": self.DT, "ground_geom_name": "ground"}
        for robot_id, cache in self._robot_cache.items():
            out[robot_id] = {
                "dof_names": self.CONTROLLED_JOINTS.copy(),
                "body_names": cache["body_names"].copy(),
                "body_masses_by_name": dict(cache["body_masses_by_name"]),
                "joint_names": cache["joint_names"].copy(),
                "controlled_joint_names": cache["controlled_joint_names"].copy(),
                "root_joint_name": cache["root_joint_name"],
                "keypoint_body_names": dict(cache["keypoint_body_names"]),
                "joint_limits": cache["jnt_ranges"].copy(),
            }
        return out

    def get_core_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for robot_id, cache in self._robot_cache.items():
            qadr = cache["root_qpos_adr"]
            vadr = cache["root_qvel_adr"]
            qpos = self.data.qpos[cache["joint_qpos_indices"]]
            qvel = self.data.qvel[cache["joint_qvel_indices"]]
            norm = self._norm[robot_id]
            out[robot_id] = {
                "root_pos": self.data.qpos[qadr:qadr + 3].copy(),
                "root_quat_wxyz": self.data.qpos[qadr + 3:qadr + 7].copy(),
                "root_vel": self.data.qvel[vadr:vadr + 3].copy(),
                "root_ang_vel": self.data.qvel[vadr + 3:vadr + 6].copy(),
                "joint_pos": qpos.copy(),
                "joint_vel": qvel.copy(),
                "joint_pos_norm": ((qpos - norm["ref"]) / norm["scale"]).astype(np.float32),
                "joint_vel_norm": (qvel / norm["scale"]).astype(np.float32),
            }
        return out

    def _build_robot_view(self, core_state: Dict[str, Dict[str, Any]], sensor_state: Dict[str, Dict[str, Any]], robot_id: str) -> Dict[str, Any]:
        me = core_state[robot_id]
        opp_id = "robot_b" if robot_id == "robot_a" else "robot_a"
        other = core_state[opp_id]

        quat_wxyz = me["root_quat_wxyz"]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        rot = R.from_quat(quat_xyzw)
        basis6d = rot.as_matrix()[:, :2].reshape(-1)
        lin_local = rot.inv().apply(me["root_vel"])
        ang_local = rot.inv().apply(me["root_ang_vel"])

        opp_rel = other["root_pos"] - me["root_pos"]
        opp_rel_local = rot.inv().apply(opp_rel)
        opp_vel_rel_local = rot.inv().apply(other["root_vel"] - me["root_vel"])
        opp_rot = R.from_quat(np.array([other["root_quat_wxyz"][1], other["root_quat_wxyz"][2], other["root_quat_wxyz"][3], other["root_quat_wxyz"][0]], dtype=np.float32))
        opp_face_local = rot.inv().apply(opp_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32)))

        feet_forces = sensor_state[robot_id]["feet_forces"]
        opp_keypoint_pos = {k: np.zeros(3, dtype=np.float32) for k in ["head", "hand_right", "hand_left", "foot_right", "foot_left"]}
        opp_keypoint_vel = {k: np.zeros(3, dtype=np.float32) for k in ["head", "hand_right", "hand_left", "foot_right", "foot_left"]}
        opp_features = np.concatenate([
            opp_rel_local.astype(np.float32),
            opp_vel_rel_local.astype(np.float32),
            opp_face_local.astype(np.float32),
            np.concatenate([opp_keypoint_pos[k] for k in ["head", "hand_right", "hand_left", "foot_right", "foot_left"]]).astype(np.float32),
            np.concatenate([opp_keypoint_vel[k] for k in ["head", "hand_right", "hand_left", "foot_right", "foot_left"]]).astype(np.float32),
        ])

        observation = np.concatenate([
            me["joint_pos_norm"].astype(np.float32),
            me["joint_vel_norm"].astype(np.float32),
            np.array([me["root_pos"][2]], dtype=np.float32),
            basis6d.astype(np.float32),
            lin_local.astype(np.float32),
            ang_local.astype(np.float32),
            feet_forces.astype(np.float32),
            opp_features.astype(np.float32),
        ]).astype(np.float32)

        return {
            "root_state": {
                "height": np.array([me["root_pos"][2]], dtype=np.float32),
                "local_orientation": basis6d.astype(np.float32),
                "linear_vel": lin_local.astype(np.float32),
                "angular_vel": ang_local.astype(np.float32),
            },
            "feet_forces": feet_forces.astype(np.float32),
            "opponent_basic_pose": {
                "relative_pos": opp_rel_local.astype(np.float32),
                "relative_vel": opp_vel_rel_local.astype(np.float32),
                "face_vector": opp_face_local.astype(np.float32),
            },
            "opponent_keypoint_pos": opp_keypoint_pos,
            "opponent_keypoint_vel": opp_keypoint_vel,
            "observation": observation,
        }

    def _extract_contacts(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        robot_robot_contacts: List[Dict[str, Any]] = []
        robot_environment_contacts: List[Dict[str, Any]] = []
        contacts: List[Dict[str, Any]] = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            b1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            b2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""

            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            frame = c.frame.reshape(3, 3)
            f_world = frame.T @ wrench[:3]
            f_mag = float(np.linalg.norm(f_world))
            contacts.append({
                "geom_a_name": g1_name,
                "geom_b_name": g2_name,
                "body_a_name": b1_name,
                "body_b_name": b2_name,
                "position_world": c.pos.copy().astype(np.float32),
                "normal_world": frame[0].copy().astype(np.float32),
                "frame_world": frame.copy().astype(np.float32),
                "force_on_body_b_world": f_world.astype(np.float32),
                "force_magnitude": f_mag,
            })

            is_r1, is_b1 = b1_name.endswith("_red"), b1_name.endswith("_blue")
            is_r2, is_b2 = b2_name.endswith("_red"), b2_name.endswith("_blue")
            if (is_r1 and is_b2) or (is_b1 and is_r2):
                robot_robot_contacts.append({"body_a": b1_name, "body_b": b2_name, "force": f_mag})
            else:
                if (is_r1 or is_b1) and not (is_r2 or is_b2):
                    robot_environment_contacts.append({"body": b1_name, "environment_geom": g2_name, "force": f_mag})
                elif (is_r2 or is_b2) and not (is_r1 or is_b1):
                    robot_environment_contacts.append({"body": b2_name, "environment_geom": g1_name, "force": f_mag})
        return robot_robot_contacts, robot_environment_contacts, contacts

    def get_derived_state(self) -> Dict[str, Any]:
        core_state = self.get_core_state()
        sensor_state = self.get_sensor_data()
        rr_contacts, re_contacts, contacts = self._extract_contacts()

        out: Dict[str, Any] = {
            "robot_robot_contacts": rr_contacts,
            "robot_environment_contacts": re_contacts,
            "contacts": contacts,
            "torso_distance": np.array([np.linalg.norm(core_state["robot_a"]["root_pos"] - core_state["robot_b"]["root_pos"])], dtype=np.float32),
        }

        xpos, xipos, xquat, cvel, xanchor = self.data.xpos, self.data.xipos, self.data.xquat, self.data.cvel, self.data.xanchor
        for robot_id in ["robot_a", "robot_b"]:
            rv = self._build_robot_view(core_state, sensor_state, robot_id)
            cache = self._robot_cache[robot_id]
            body_names = cache["body_names"]
            body_ids = cache["body_ids"]
            rv["body_xpos"] = {n: xpos[bid].copy() for n, bid in zip(body_names, body_ids)}
            rv["body_xipos"] = {n: xipos[bid].copy() for n, bid in zip(body_names, body_ids)}
            rv["body_xquat"] = {n: xquat[bid].copy() for n, bid in zip(body_names, body_ids)}
            rv["body_angvel_world"] = {n: cvel[bid, 0:3].copy() for n, bid in zip(body_names, body_ids)}
            rv["body_linvel_world"] = {n: cvel[bid, 3:6].copy() for n, bid in zip(body_names, body_ids)}
            joint_world_anchor: Dict[str, np.ndarray] = {}
            for jn in cache["joint_names"]:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    joint_world_anchor[jn] = xanchor[jid].copy()
            rv["joint_world_anchor"] = joint_world_anchor
            out[robot_id] = rv
        return out

    def get_sensor_data(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for robot_id, suffix in [("robot_a", "_red"), ("robot_b", "_blue")]:
            f_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"force_left_foot{suffix}")
            f_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"force_right_foot{suffix}")
            imu_q = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"imu_quaternion{suffix}")
            imu_acc = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"imu_linear_acceleration{suffix}")
            imu_gyro = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"imu_angular_velocity{suffix}")

            left = np.zeros(3, dtype=np.float32)
            right = np.zeros(3, dtype=np.float32)
            q = np.zeros(4, dtype=np.float32)
            acc = np.zeros(3, dtype=np.float32)
            gyro = np.zeros(3, dtype=np.float32)
            if f_l >= 0:
                adr = self.model.sensor_adr[f_l]
                left = self.data.sensordata[adr:adr + 3].copy()
            if f_r >= 0:
                adr = self.model.sensor_adr[f_r]
                right = self.data.sensordata[adr:adr + 3].copy()
            if imu_q >= 0:
                adr = self.model.sensor_adr[imu_q]
                q = self.data.sensordata[adr:adr + 4].copy()
            if imu_acc >= 0:
                adr = self.model.sensor_adr[imu_acc]
                acc = self.data.sensordata[adr:adr + 3].copy()
            if imu_gyro >= 0:
                adr = self.model.sensor_adr[imu_gyro]
                gyro = self.data.sensordata[adr:adr + 3].copy()

            out[robot_id] = {
                "force_left_foot": left,
                "force_right_foot": right,
                "feet_forces": np.array([np.linalg.norm(left), np.linalg.norm(right)], dtype=np.float32),
                "imu_quaternion": q,
                "imu_linear_acceleration": acc,
                "imu_angular_velocity": gyro,
            }
        return out

    def get_action(self) -> Dict[str, Any]:
        return {"robot_a": self._last_action["robot_a"].copy(), "robot_b": self._last_action["robot_b"].copy()}

    def get_broadcastview_image(self) -> np.ndarray:
        """
        获取广播视角图像（动态跟踪版本，参照 humanoid21 逻辑）。
        """
        try:
            torso_a_name = self._robot_cache["robot_a"]["keypoint_body_names"].get("torso", "LINK_BASE_red")
            torso_b_name = self._robot_cache["robot_b"]["keypoint_body_names"].get("torso", "LINK_BASE_blue")
            torso_a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso_a_name)
            torso_b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso_b_name)
            if torso_a_id < 0 or torso_b_id < 0:
                torso_a_id = self._robot_cache["robot_a"]["base_body_id"]
                torso_b_id = self._robot_cache["robot_b"]["base_body_id"]

            pos_a = self.data.xpos[torso_a_id]
            pos_b = self.data.xpos[torso_b_id]
            center = (pos_a + pos_b) / 2.0

            target_lookat = center.copy()
            target_lookat[2] = 1.0

            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction)
            if dist_ab > 1e-6:
                direction = direction / dist_ab
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            target_azi = dir_angle + 90.0
            target_ele = -20.0
            target_dist = max(2.5, min(4.0, dist_ab * 1.5))

            limit = 2.55
            azi_rad = np.radians(target_azi)
            ele_rad = np.radians(target_ele)
            dx = -target_dist * np.cos(azi_rad) * np.cos(ele_rad)
            dy = -target_dist * np.sin(azi_rad) * np.cos(ele_rad)
            cam_x = target_lookat[0] + dx
            cam_y = target_lookat[1] + dy

            if abs(cam_x) > limit:
                max_dx = limit - target_lookat[0] if cam_x > 0 else -limit - target_lookat[0]
                factor = -np.cos(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dx / factor))

            if abs(cam_y) > limit:
                max_dy = limit - target_lookat[1] if cam_y > 0 else -limit - target_lookat[1]
                factor = -np.sin(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dy / factor))

            alpha_pos = 0.05
            alpha_look = 0.1
            if self._prev_azi is None:
                azi = target_azi
                ele = target_ele
                dist = target_dist
                lookat = target_lookat.copy()
            else:
                diff = (target_azi - self._prev_azi + 180) % 360 - 180
                azi = self._prev_azi + diff * alpha_pos
                ele = self._prev_ele * (1.0 - alpha_pos) + target_ele * alpha_pos
                dist = self._prev_dist * (1.0 - alpha_pos) + target_dist * alpha_pos
                lookat = self._prev_lookat * (1.0 - alpha_look) + target_lookat * alpha_look

            self._prev_azi = float(azi)
            self._prev_ele = float(ele)
            self._prev_dist = float(dist)
            self._prev_lookat = lookat.copy()

            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = float(dist)
            cam.elevation = float(ele)
            cam.azimuth = float(azi)

            # T800: 复用持久 renderer，避免每帧析构触发 EGL 清理噪声
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    height=self._render_height,
                    width=self._render_width,
                )
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render broadcast view (t800): {e}")
            return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)

    def set_core_state(self, state: Dict[str, Any]) -> None:
        for robot_id in ["robot_a", "robot_b"]:
            if robot_id not in state:
                continue
            s = state[robot_id]
            cache = self._robot_cache[robot_id]
            qadr = cache["root_qpos_adr"]
            vadr = cache["root_qvel_adr"]
            if "root_pos" in s:
                self.data.qpos[qadr:qadr + 3] = np.asarray(s["root_pos"], dtype=np.float32)
            if "root_quat_wxyz" in s:
                self.data.qpos[qadr + 3:qadr + 7] = np.asarray(s["root_quat_wxyz"], dtype=np.float32)
            if "joint_pos" in s:
                self.data.qpos[cache["joint_qpos_indices"]] = np.asarray(s["joint_pos"], dtype=np.float32)
            if "root_vel" in s:
                self.data.qvel[vadr:vadr + 3] = np.asarray(s["root_vel"], dtype=np.float32)
            if "root_ang_vel" in s:
                self.data.qvel[vadr + 3:vadr + 6] = np.asarray(s["root_ang_vel"], dtype=np.float32)
            if "joint_vel" in s:
                self.data.qvel[cache["joint_qvel_indices"]] = np.asarray(s["joint_vel"], dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)

    def set_action(self, action: Dict[str, Any]) -> None:
        for robot_id in ["robot_a", "robot_b"]:
            act = action.get(robot_id, None)
            if act is None:
                continue
            arr = np.asarray(act, dtype=np.float32)
            if arr.shape != (self.ACTION_DIM,):
                raise ValueError(f"Action for {robot_id} must have shape ({self.ACTION_DIM},), got {arr.shape}")
            arr = np.clip(arr, -1.0, 1.0)
            self._target_pos_norm[robot_id] = arr
            self._last_action[robot_id] = arr.copy()

    def apply_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: Optional[np.ndarray] = None,
        robot_id: str = "robot_a",
    ) -> None:
        suffix = "_red" if robot_id == "robot_a" else "_blue"
        full_body = body_name if body_name.endswith(suffix) else f"{body_name}{suffix}"
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_body)
        if bid < 0:
            return
        force = np.asarray(force, dtype=np.float64).reshape(3)
        torque = np.zeros(3, dtype=np.float64) if torque is None else np.asarray(torque, dtype=np.float64).reshape(3)
        self.data.xfrc_applied[bid, :3] = force
        self.data.xfrc_applied[bid, 3:] = torque

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
