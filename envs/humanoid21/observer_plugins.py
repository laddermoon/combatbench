"""
Humanoid21 观测插件

严格按照 DATASPEC.md 和 OBSERVATION_zh.md 实现 96 维观测空间：
- 模块一：本体感知 (42维) - joint_pos_norm, joint_vel_norm
- 模块二：全局状态 (13维) - height, local_orientation, linear_vel, angular_vel
- 模块三：触觉力反馈 (2维) - feet_forces
- 模块四：对手观测 (39维) - basic_pose, keypoint_pos, keypoint_vel
"""

from typing import Any, Dict

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseObserverPlugin, ReadOnlySimContext, TerminationReason


class Humanoid21Observer(BaseObserverPlugin):
    """Humanoid21 96维观测空间"""

    ACTION_DIM = 21
    OBS_DIM = 96  # 更新为 96 维

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output: Any = None

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_observation(ctx, self.agent_id)

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_observation(ctx, self.agent_id)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
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
        """
        构建 96 维观测空间

        按照 DATASPEC.md 规范：
        - 模块一：本体感知 (42维) - 索引 [0:42]
        - 模块二：全局状态 (13维) - 索引 [42:55]
        - 模块三：触觉力反馈 (2维) - 索引 [55:57]
        - 模块四：对手观测 (39维) - 索引 [57:96]
        """
        accessor = ctx.accessor
        derived_state = accessor.get_derived_state()

        # 直接从 derived_state 获取完整观测
        observation = derived_state[agent_id]['observation']

        return observation.astype(np.float32)


class Humanoid21BalanceAnalysisObserver(BaseObserverPlugin):
    """
    单机器人平衡状态分析插件。

    输出内容包含：
    1. 去脚质心 `center_of_mass`
    2. 左右踝关节支撑点 `left_ankle_support_point` / `right_ankle_support_point`
    3. 左右踝部支撑反力 `left_ankle_support_force` / `right_ankle_support_force`
    4. 去脚质心移动速度 `center_of_mass_velocity`
    5. 基于地面二维投影的双踝-质心几何关系与速度分解结果

    这里的“踝关节受力”做如下严格定义：
    - MuJoCo 的真实接触发生在足部 geom，而不是踝关节铰链本身。
    - 因此本插件将“对应足部与静态世界(body_id=0)的接触合力”定义为
      “通过该足部传递到对应踝部支撑点的支撑反力代理”。
    - 这是一种接触力学上的可观测量；若要得到真正的关节内力，需要额外做逆动力学/约束求解。
    """

    WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    PLANE_DISTANCE_TOLERANCE = 1e-4
    SUPPORT_SPAN_TOLERANCE = 1e-8

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output: Any = None

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_analysis(ctx)

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_analysis(ctx)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        return None

    def get_output(self) -> Any:
        return self._output

    def _build_analysis(self, ctx: ReadOnlySimContext) -> Dict[str, Any]:
        accessor = ctx.accessor
        model = getattr(accessor, "model", None)
        data = getattr(accessor, "data", None)
        robot_cache = getattr(accessor, "_robot_cache", None)
        if model is None or data is None or not isinstance(robot_cache, dict) or self.agent_id not in robot_cache:
            raise TypeError(
                "Humanoid21BalanceAnalysisObserver requires the Humanoid21 MuJoCo simulator accessor with model/data/_robot_cache"
            )

        cache = robot_cache[self.agent_id]
        center_of_mass = self._compute_center_of_mass_excluding_feet(model, data, cache)
        center_of_mass_velocity = self._compute_center_of_mass_velocity(model, data, cache)
        robot_forward_ground = self._compute_robot_forward_ground_direction(data, cache)

        left_ankle_support_point, left_ankle_anchors = self._compute_ankle_support_point(model, data, cache, side="left")
        right_ankle_support_point, right_ankle_anchors = self._compute_ankle_support_point(model, data, cache, side="right")

        left_ankle_support_force = self._compute_ankle_support_force(
            model,
            data,
            ground_geom_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "地面")),
            foot_body_id=int(cache["foot_left_body_id"]),
        )
        right_ankle_support_force = self._compute_ankle_support_force(
            model,
            data,
            ground_geom_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "地面")),
            foot_body_id=int(cache["foot_right_body_id"]),
        )

        support_geometry = self._analyze_support_geometry(
            center_of_mass=center_of_mass,
            center_of_mass_velocity=center_of_mass_velocity,
            left_support=left_ankle_support_point,
            right_support=right_ankle_support_point,
            robot_forward_ground=robot_forward_ground,
        )

        return {
            "agent_id": self.agent_id,
            "center_of_mass": center_of_mass.astype(np.float32),
            "center_of_mass_velocity": center_of_mass_velocity.astype(np.float32),
            "left_ankle_support_point": left_ankle_support_point.astype(np.float32),
            "right_ankle_support_point": right_ankle_support_point.astype(np.float32),
            "left_ankle_support_force": left_ankle_support_force.astype(np.float32),
            "right_ankle_support_force": right_ankle_support_force.astype(np.float32),
            "left_ankle_y_anchor": left_ankle_anchors["ankle_y"].astype(np.float32),
            "left_ankle_x_anchor": left_ankle_anchors["ankle_x"].astype(np.float32),
            "right_ankle_y_anchor": right_ankle_anchors["ankle_y"].astype(np.float32),
            "right_ankle_x_anchor": right_ankle_anchors["ankle_x"].astype(np.float32),
            **support_geometry,
        }

    def _compute_center_of_mass_excluding_feet(self, model: mujoco.MjModel, data: mujoco.MjData, cache: Dict[str, Any]) -> np.ndarray:
        """
        计算“去脚质心”。

        算法：
        - 机器人总 body 集合使用 `cache['body_ids']`，这是 torso 子树上的全部 body。
        - 明确剔除左右脚 body：`foot_left_body_id`、`foot_right_body_id`。
        - 对剩余 body 做质量加权平均。

        公式：
            p_com = (Σ_i m_i * p_i) / (Σ_i m_i)

        其中：
        - `m_i = model.body_mass[body_id]`
        - `p_i = data.xipos[body_id]`
        - `data.xipos` 是 MuJoCo 当前世界坐标系下 body 惯性中心位置
        """
        excluded_body_ids = {
            int(cache["foot_left_body_id"]),
            int(cache["foot_right_body_id"]),
        }
        included_body_ids = [int(body_id) for body_id in cache["body_ids"] if int(body_id) not in excluded_body_ids]
        if not included_body_ids:
            raise ValueError(f"No body remains after excluding feet for {self.agent_id}")

        masses = np.asarray(model.body_mass[included_body_ids], dtype=np.float64)
        positions = np.asarray(data.xipos[included_body_ids], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            raise ValueError(f"Invalid total mass for {self.agent_id}: {total_mass}")
        return np.sum(positions * masses[:, None], axis=0) / total_mass

    def _compute_center_of_mass_velocity(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cache: Dict[str, Any],
    ) -> np.ndarray:
        """
        计算去脚质心速度向量。

        算法：直接使用“去脚后所有 body 的瞬时线速度”做质量加权平均。

        这样不依赖相邻两帧差分，因此在 reset 后第一帧如果随机初始化插件
        已经给机器人设置了根部速度、角速度或关节速度，质心速度也能立刻正确反映出来。

        公式：
            v_com = (Σ_i m_i * v_i) / (Σ_i m_i)

        其中：
        - `m_i = model.body_mass[body_id]`
        - `v_i = data.cvel[body_id, 3:6]`

        实现约定：
        - 与去脚质心位置保持一致，显式排除 `foot_left_body_id` 和 `foot_right_body_id`
        - 本项目在其他观测实现中也使用 `data.cvel[body_id, 3:6]` 作为 body 世界系线速度
        - 若极端情况下质量求和无效，则退化为根关节线速度近似值
        
        这样得到的是“当前时刻所有刚体运动综合后的瞬时整体质心速度”。
        """
        excluded_body_ids = {
            int(cache["foot_left_body_id"]),
            int(cache["foot_right_body_id"]),
        }
        included_body_ids = [int(body_id) for body_id in cache["body_ids"] if int(body_id) not in excluded_body_ids]
        if not included_body_ids:
            root_qvel_adr = int(cache["root_qvel_adr"])
            return np.asarray(data.qvel[root_qvel_adr:root_qvel_adr + 3], dtype=np.float64).copy()

        masses = np.asarray(model.body_mass[included_body_ids], dtype=np.float64)
        linear_velocities = np.asarray(data.cvel[included_body_ids, 3:6], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            root_qvel_adr = int(cache["root_qvel_adr"])
            return np.asarray(data.qvel[root_qvel_adr:root_qvel_adr + 3], dtype=np.float64).copy()

        return np.sum(linear_velocities * masses[:, None], axis=0) / total_mass

    def _compute_robot_forward_ground_direction(self, data: mujoco.MjData, cache: Dict[str, Any]) -> np.ndarray:
        """
        计算机器人前向在地面平面上的单位方向。

        算法：
        - 取 torso 世界姿态下的局部 x 轴，作为机器人“前方”
        - 将该向量投影到地面二维平面（XY 平面）
        - 对投影结果做单位化

        公式：
            f_world = R_torso * [1, 0, 0]
            f_ground = proj_xy(f_world) / ||proj_xy(f_world)||

        若投影退化为零向量，则返回零向量，表示当前帧无法可靠定义前后符号。
        """
        torso_body_id = int(cache["torso_body_id"])
        torso_quat = np.asarray(data.xquat[torso_body_id], dtype=np.float64)
        torso_rot = R.from_quat([torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]])
        forward_world = torso_rot.apply([1.0, 0.0, 0.0])
        forward_ground = np.asarray(forward_world[:2], dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward_ground))
        if forward_norm <= self.SUPPORT_SPAN_TOLERANCE:
            return np.zeros(2, dtype=np.float64)
        return forward_ground / forward_norm

    def _compute_ankle_support_point(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cache: Dict[str, Any],
        side: str,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        计算单侧“踝关节支撑点”。

        Humanoid21 的单脚踝是 2 自由度结构，包含：
        - `ankle_y_*`
        - `ankle_x_*`

        单个脚不存在唯一一个铰链点，因此本插件将“双踝自由度锚点的几何中心”定义为脚踝支撑点。

        公式：
            p_ankle = 0.5 * (p_ankle_y + p_ankle_x)

        其中 `p_ankle_y` 和 `p_ankle_x` 使用 MuJoCo 的 `data.xanchor[joint_id]` 读取世界坐标。
        """
        suffix = str(cache["suffix"])
        ankle_y_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ankle_y_{side}{suffix}")
        ankle_x_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ankle_x_{side}{suffix}")
        if ankle_y_joint_id < 0 or ankle_x_joint_id < 0:
            raise ValueError(f"Failed to resolve ankle joints for {self.agent_id} side={side}")

        ankle_y_anchor = np.asarray(data.xanchor[ankle_y_joint_id], dtype=np.float64).copy()
        ankle_x_anchor = np.asarray(data.xanchor[ankle_x_joint_id], dtype=np.float64).copy()
        support_point = 0.5 * (ankle_y_anchor + ankle_x_anchor)
        return support_point, {
            "ankle_y": ankle_y_anchor,
            "ankle_x": ankle_x_anchor,
        }

    def _compute_ankle_support_force(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ground_geom_id: int,
        foot_body_id: int,
    ) -> np.ndarray:
        """
        计算单脚通过接触传递到踝部的支撑反力代理（世界坐标 3 维向量）。

        算法：
        - 遍历全部接触 `data.contact[i]`
        - 仅保留“足部 geom 与地面 geom(名称=`地面`)”之间的接触
        - 使用 `mujoco.mj_contactForce(model, data, i, wrench)` 读取接触坐标系下的 6D 力/矩
        - 取前三维线力 `f_c`
        - 用 `contact.frame` 将接触坐标系线力旋转到世界坐标系
        - 将所有接触对该足部的力向量求和

        公式：
            f_world = R_contact_to_world * f_contact

        其中：
        - `f_contact = wrench[:3]`
        - `R_contact_to_world = contact.frame.reshape(3, 3).T`
        - MuJoCo 的 `contact.frame` 顺序为 `[n, t1, t2]` 三个世界坐标轴向量，因此需要转置后右乘局部坐标

        符号约定：
        - `mj_contactForce` 返回的是“作用在 geom2 上，由 geom1 施加”的接触力
        - 所以当足部是 `geom2` 时直接累加
        - 当足部是 `geom1` 时，对足部受力需要取负号
        """
        support_force = np.zeros(3, dtype=np.float64)

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])

            foot_is_geom1 = body1 == foot_body_id and geom2 == ground_geom_id
            foot_is_geom2 = body2 == foot_body_id and geom1 == ground_geom_id
            if not foot_is_geom1 and not foot_is_geom2:
                continue

            contact_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, contact_wrench)

            contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            contact_force_world_on_geom2 = contact_frame.T @ np.asarray(contact_wrench[:3], dtype=np.float64)

            if foot_is_geom2:
                support_force += contact_force_world_on_geom2
            else:
                support_force -= contact_force_world_on_geom2

        return support_force

    def _analyze_support_geometry(
        self,
        center_of_mass: np.ndarray,
        center_of_mass_velocity: np.ndarray,
        left_support: np.ndarray,
        right_support: np.ndarray,
        robot_forward_ground: np.ndarray,
    ) -> Dict[str, Any]:
        """
        分析“去脚质心”与“双踝支撑点”在地面二维平面上的关系。

        首先将左右踝关节、质心位置、质心速度全部投影到地面平面：
            p_l = left_support[:2]
            p_r = right_support[:2]
            p_c = center_of_mass[:2]
            v_c = center_of_mass_velocity[:2]

        然后定义二维局部坐标系：
        - 支撑轴 `e_s`：从左踝指向右踝
        - 侧向轴 `e_n`：与 `e_s` 垂直，并选取符号使其尽量指向机器人前方

        记：
            e_s = (p_r - p_l) / ||p_r - p_l||
            e_n = sign((R90 e_s) · f_robot) * (R90 e_s)

        其中：
        - `R90 e_s = [-e_sy, e_sx]`
        - `f_robot` 是 torso 局部 x 轴投影到地面后的单位向量

        计算量包括：
        1. 双支撑点跨度：
            span = ||p_r - p_l||
        2. 质心在支撑轴上的投影坐标：
            u = (p_c - p_l) · e_s
        3. 质心相对支撑轴的有符号侧向距离：
            d = (p_c - p_l) · e_n
           约定 `d > 0` 表示位于机器人前方一侧，`d < 0` 表示位于机器人后方一侧。
        4. 质心速度在两条二维轴上的分解：
            v_parallel = v_c · e_s
            v_lateral = v_c · e_n
        """
        left_ground = np.asarray(left_support[:2], dtype=np.float64)
        right_ground = np.asarray(right_support[:2], dtype=np.float64)
        center_of_mass_ground = np.asarray(center_of_mass[:2], dtype=np.float64)
        center_of_mass_velocity_ground = np.asarray(center_of_mass_velocity[:2], dtype=np.float64)

        support_vector_ground = right_ground - left_ground
        support_span_ground = float(np.linalg.norm(support_vector_ground))

        if support_span_ground <= self.SUPPORT_SPAN_TOLERANCE:
            support_axis_unit = np.zeros(2, dtype=np.float64)
            support_lateral_unit = np.zeros(2, dtype=np.float64)
            support_projection_coordinate = np.nan
            support_projection_ratio = np.nan
            support_projection_point = np.full(2, np.nan, dtype=np.float64)
            support_lateral_signed_distance = np.nan
            center_of_mass_velocity_along_support_axis = np.nan
            center_of_mass_velocity_along_support_lateral_axis = np.nan
            projected_between_supports = False
            support_frame_defined = False
        else:
            support_axis_unit = support_vector_ground / support_span_ground
            support_lateral_unit = np.array([-support_axis_unit[1], support_axis_unit[0]], dtype=np.float64)

            if float(np.linalg.norm(robot_forward_ground)) > self.SUPPORT_SPAN_TOLERANCE:
                if float(np.dot(support_lateral_unit, robot_forward_ground)) < 0.0:
                    support_lateral_unit = -support_lateral_unit

            center_relative_ground = center_of_mass_ground - left_ground
            support_projection_coordinate = float(np.dot(center_relative_ground, support_axis_unit))
            support_projection_ratio = float(support_projection_coordinate / support_span_ground)
            support_projection_point = left_ground + support_projection_coordinate * support_axis_unit
            support_lateral_signed_distance = float(np.dot(center_relative_ground, support_lateral_unit))
            center_of_mass_velocity_along_support_axis = float(np.dot(center_of_mass_velocity_ground, support_axis_unit))
            center_of_mass_velocity_along_support_lateral_axis = float(np.dot(center_of_mass_velocity_ground, support_lateral_unit))
            projected_between_supports = (
                -self.PLANE_DISTANCE_TOLERANCE
                <= support_projection_coordinate
                <= support_span_ground + self.PLANE_DISTANCE_TOLERANCE
            )
            support_frame_defined = True

        return {
            "left_ankle_support_ground_projection": left_ground.astype(np.float32),
            "right_ankle_support_ground_projection": right_ground.astype(np.float32),
            "center_of_mass_ground_projection": center_of_mass_ground.astype(np.float32),
            "center_of_mass_velocity_ground_projection": center_of_mass_velocity_ground.astype(np.float32),
            "robot_forward_ground_direction": np.asarray(robot_forward_ground, dtype=np.float64).astype(np.float32),
            "support_span": float(support_span_ground),
            "support_span_ground": float(support_span_ground),
            "support_axis_unit_ground": support_axis_unit.astype(np.float32),
            "support_lateral_unit_ground": support_lateral_unit.astype(np.float32),
            "support_axis_projection_coordinate": float(support_projection_coordinate),
            "support_segment_parameter": float(support_projection_ratio),
            "support_axis_projection_point": support_projection_point.astype(np.float32),
            "support_lateral_signed_distance": float(support_lateral_signed_distance),
            "center_of_mass_velocity_along_support_axis": float(center_of_mass_velocity_along_support_axis),
            "center_of_mass_velocity_along_support_lateral_axis": float(center_of_mass_velocity_along_support_lateral_axis),
            "ground_support_frame_defined": bool(support_frame_defined),
            "is_projected_between_support_points": bool(projected_between_supports),
        }


class Humanoid21Rewarder(BaseObserverPlugin):
    """Humanoid21 奖励计算插件"""

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output = 0.0

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def get_output(self) -> Any:
        return self._output


def build_shared_runtime_info(ctx: ReadOnlySimContext) -> Dict[str, Any]:
    """构建共享运行时信息"""
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
