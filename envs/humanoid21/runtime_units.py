from typing import Any, Dict

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseObserver, BaseRewarder, ReadOnlySimContext, TerminationReason


class Humanoid21Observer(BaseObserver):
    ACTION_DIM = 21
    OBS_DIM = 127

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output: Any = None

    def process_data(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_observation(ctx, self.agent_id)

    def get_output(self) -> Any:
        return self._output

    @classmethod
    def get_observation_space(cls) -> spaces.Dict:
        return spaces.Dict({
            "robot_a": spaces.Box(low=-np.inf, high=np.inf, shape=(cls.OBS_DIM,), dtype=np.float32),
            "robot_b": spaces.Box(low=-np.inf, high=np.inf, shape=(cls.OBS_DIM,), dtype=np.float32),
        })

    @classmethod
    def get_action_space(cls) -> spaces.Dict:
        return spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(cls.ACTION_DIM,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(cls.ACTION_DIM,), dtype=np.float32),
        })

    @classmethod
    def _build_observation(cls, ctx: ReadOnlySimContext, agent_id: str) -> np.ndarray:
        accessor = ctx.accessor
        static_data = accessor.get_static_data()
        robot_info = static_data["robot_info"]
        core_state = accessor.get_core_state()

        qpos_array = core_state["qpos"]
        qvel_array = core_state["qvel"]
        data = accessor.data
        model = accessor.model

        opponent_id = "robot_b" if agent_id == "robot_a" else "robot_a"
        obs_list = []

        pos = np.zeros(cls.ACTION_DIM)
        vel = np.zeros(cls.ACTION_DIM)
        qpos_idx_list = robot_info[agent_id]["qpos_indices"]
        qvel_idx_list = robot_info[agent_id]["qvel_indices"]
        for i in range(len(qpos_idx_list)):
            if qpos_idx_list[i] >= 0:
                pos[i] = qpos_array[qpos_idx_list[i]]
                vel[i] = qvel_array[qvel_idx_list[i]]
        obs_list.append(pos)
        obs_list.append(vel)

        body_id = robot_info[agent_id]["body_id"]
        pos_root = data.xpos[body_id].copy()
        quat_root = data.xquat[body_id].copy()
        cvel_root = data.cvel[body_id].copy()

        obs_list.append([pos_root[2]])
        rot = R.from_quat([quat_root[1], quat_root[2], quat_root[3], quat_root[0]])
        rot_mat = rot.as_matrix()
        obs_list.append(np.concatenate([rot_mat[:, 0], rot_mat[:, 1]]))
        obs_list.append(cvel_root[3:6])
        obs_list.append(cvel_root[0:3])

        left_foot_name = f"foot_left{robot_info[agent_id]['suffix']}"
        right_foot_name = f"foot_right{robot_info[agent_id]['suffix']}"
        left_contact = 0.0
        right_contact = 0.0
        for idx in range(data.ncon):
            contact = data.contact[idx]
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            is_floor = "floor" in geom1_name.lower() or "floor" in geom2_name.lower() or geom1_name == "地面" or geom2_name == "地面"
            if not is_floor:
                if model.geom_type[contact.geom1] == mujoco.mjtGeom.mjGEOM_PLANE or model.geom_type[contact.geom2] == mujoco.mjtGeom.mjGEOM_PLANE:
                    is_floor = True
            if not is_floor:
                continue
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or ""
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or ""
            if left_foot_name in body1_name or left_foot_name in body2_name:
                left_contact = 1.0
            if right_foot_name in body1_name or right_foot_name in body2_name:
                right_contact = 1.0
        obs_list.append([left_contact, right_contact])

        forces = np.zeros(6)
        for part, f_slice in [
            (f"pelvis{robot_info[agent_id]['suffix']}", slice(0, 3)),
            (f"head{robot_info[agent_id]['suffix']}", slice(0, 3)),
        ]:
            body_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
            if body_index >= 0:
                forces[f_slice] += data.cfrc_ext[body_index, :3]
        for part in [f"hand_left{robot_info[agent_id]['suffix']}", f"hand_right{robot_info[agent_id]['suffix']}"]:
            body_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
            if body_index >= 0:
                forces[3:6] += data.cfrc_ext[body_index, :3]
        obs_list.append(forces)

        opponent_body_id = robot_info[opponent_id]["body_id"]
        opponent_pos = data.xpos[opponent_body_id].copy()
        opponent_quat = data.xquat[opponent_body_id].copy()
        opponent_cvel = data.cvel[opponent_body_id].copy()

        obs_list.append(opponent_pos - pos_root)
        obs_list.append(opponent_cvel[3:6] - cvel_root[3:6])
        obs_list.append(opponent_quat)

        keypoint_parts = [
            "head", "hand_right", "hand_left", "lower_arm_right", "lower_arm_left",
            "shin_right", "shin_left", "foot_right", "foot_left",
        ]
        kp_pos = []
        kp_vel = []
        for body_part in keypoint_parts:
            body_name = f"{body_part}{robot_info[opponent_id]['suffix']}"
            body_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_index >= 0:
                body_pos = data.xpos[body_index]
                body_vel = data.cvel[body_index, 3:6]
            else:
                body_pos = opponent_pos
                body_vel = opponent_cvel[3:6]
            rel_pos = body_pos - pos_root
            kp_pos.append(rot_mat.T @ rel_pos)
            rel_vel = body_vel - cvel_root[3:6]
            kp_vel.append(rot_mat.T @ rel_vel)

        obs_list.append(np.concatenate(kp_pos))
        obs_list.append(np.concatenate(kp_vel))
        return np.concatenate([np.asarray(item).flatten() for item in obs_list]).astype(np.float32)


class Humanoid21Rewarder(BaseRewarder):
    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output = 0.0

    def process_data(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def get_output(self) -> Any:
        return self._output


def build_shared_runtime_info(ctx: ReadOnlySimContext) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "health": {
            "robot_a": float(ctx.metrics.get("health_a", 100.0)),
            "robot_b": float(ctx.metrics.get("health_b", 100.0)),
        },
        "damage_taken": {
            "robot_a": float(ctx.metrics.get("damage_taken_a", 0.0)),
            "robot_b": float(ctx.metrics.get("damage_taken_b", 0.0)),
        },
        "winner": None,
    }
    if ctx.is_terminated:
        proposals = ctx.termination_proposals
        health_a = info["health"]["robot_a"]
        health_b = info["health"]["robot_b"]
        if TerminationReason.KO in proposals:
            if health_a <= 0 and health_b > 0:
                info["winner"] = "robot_b"
            elif health_b <= 0 and health_a > 0:
                info["winner"] = "robot_a"
            else:
                info["winner"] = "draw"
        elif TerminationReason.TIMEOUT in proposals:
            if health_a > health_b:
                info["winner"] = "robot_a"
            elif health_b > health_a:
                info["winner"] = "robot_b"
            else:
                info["winner"] = "draw"
    return info
