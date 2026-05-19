from typing import Any, Dict, Optional

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseObserverPlugin, ReadOnlySimContext, TerminationReason

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
    ARENA_HALF_EXTENT = 3.05

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output: Any = None
        self._last_accessor: Optional[Any] = None

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_analysis(ctx)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_analysis(ctx)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        return None

    def get_output(self) -> Any:
        return self._output

    def get_visualization_image(self) -> np.ndarray:
        if self._last_accessor is None:
            raise RuntimeError("Humanoid21BalanceAnalysisObserver has no cached accessor yet. Call runtime.reset() or runtime.step() first.")
        if not isinstance(self._output, dict):
            raise RuntimeError("Humanoid21BalanceAnalysisObserver has no analysis output yet.")
        broadcast_image = self._ensure_uint8_rgb_image(self._last_accessor.get_broadcastview_image())
        plan_image = self._render_balance_plan_view(self._output, width=int(broadcast_image.shape[1]), height=int(broadcast_image.shape[0]))
        return np.concatenate([broadcast_image, plan_image], axis=0)

    def _ensure_uint8_rgb_image(self, image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[..., None], 3, axis=2)
        elif image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)
        elif image_array.ndim == 3 and image_array.shape[2] >= 3:
            image_array = image_array[..., :3]
        else:
            raise ValueError(f"Unsupported image shape for visualization: {image_array.shape}")
        if image_array.dtype != np.uint8:
            if np.issubdtype(image_array.dtype, np.floating):
                image_array = np.clip(image_array, 0.0, 255.0)
            else:
                image_array = np.clip(image_array.astype(np.float64), 0.0, 255.0)
            image_array = image_array.astype(np.uint8)
        return np.ascontiguousarray(image_array)

    def _render_balance_plan_view(self, balance_output: Dict[str, Any], width: int, height: int) -> np.ndarray:
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        panel_size = int(min(width, height) * 0.82)
        left = int((width - panel_size) // 2)
        top = int((height - panel_size) // 2)
        right = int(left + panel_size - 1)
        bottom = int(top + panel_size - 1)
        image[top:bottom + 1, left:right + 1] = np.array([232, 236, 242], dtype=np.uint8)
        self._draw_line(image, (left, top), (right, top), (80, 80, 80), thickness=3)
        self._draw_line(image, (right, top), (right, bottom), (80, 80, 80), thickness=3)
        self._draw_line(image, (right, bottom), (left, bottom), (80, 80, 80), thickness=3)
        self._draw_line(image, (left, bottom), (left, top), (80, 80, 80), thickness=3)
        center_x = int(round((left + right) * 0.5))
        center_y = int(round((top + bottom) * 0.5))
        self._draw_line(image, (center_x, top), (center_x, bottom), (210, 210, 210), thickness=1)
        self._draw_line(image, (left, center_y), (right, center_y), (210, 210, 210), thickness=1)

        def world_to_pixel(point_xy: np.ndarray) -> tuple[int, int]:
            point = np.asarray(point_xy, dtype=np.float64)
            norm_x = (point[0] + self.ARENA_HALF_EXTENT) / (2.0 * self.ARENA_HALF_EXTENT)
            norm_y = 1.0 - (point[1] + self.ARENA_HALF_EXTENT) / (2.0 * self.ARENA_HALF_EXTENT)
            pixel_x = int(round(left + np.clip(norm_x, 0.0, 1.0) * (panel_size - 1)))
            pixel_y = int(round(top + np.clip(norm_y, 0.0, 1.0) * (panel_size - 1)))
            return pixel_x, pixel_y

        left_ankle = np.asarray(balance_output["left_ankle_support_ground_projection"], dtype=np.float64)
        right_ankle = np.asarray(balance_output["right_ankle_support_ground_projection"], dtype=np.float64)
        center_of_mass = np.asarray(balance_output["center_of_mass_ground_projection"], dtype=np.float64)
        center_of_mass_velocity = np.asarray(balance_output["center_of_mass_velocity_ground_projection"], dtype=np.float64)
        support_projection_point = np.asarray(balance_output["support_axis_projection_point"], dtype=np.float64)
        support_midpoint = 0.5 * (left_ankle + right_ankle)

        left_ankle_px = world_to_pixel(left_ankle)
        right_ankle_px = world_to_pixel(right_ankle)
        center_of_mass_px = world_to_pixel(center_of_mass)
        support_midpoint_px = world_to_pixel(support_midpoint)
        self._draw_line(image, left_ankle_px, right_ankle_px, (60, 110, 255), thickness=3)
        self._draw_circle(image, support_midpoint_px, radius=5, color=(30, 30, 30))

        if np.all(np.isfinite(support_projection_point)):
            support_projection_px = world_to_pixel(support_projection_point)
            self._draw_line(image, center_of_mass_px, support_projection_px, (120, 120, 120), thickness=2)
            self._draw_circle(image, support_projection_px, radius=6, color=(255, 170, 0))

        velocity_xy = np.asarray(center_of_mass_velocity, dtype=np.float64)
        velocity_norm = float(np.linalg.norm(velocity_xy))
        if velocity_norm > 1e-8:
            clipped_velocity = velocity_xy * min(1.0, 2.0 / velocity_norm)
            velocity_endpoint = center_of_mass + 0.45 * clipped_velocity
            velocity_endpoint_px = world_to_pixel(velocity_endpoint)
            self._draw_arrow(image, center_of_mass_px, velocity_endpoint_px, (30, 170, 30), thickness=3)

        self._draw_circle(image, left_ankle_px, radius=10, color=(235, 64, 52))
        self._draw_circle(image, right_ankle_px, radius=10, color=(52, 110, 235))
        self._draw_circle(image, center_of_mass_px, radius=10, color=(30, 170, 30))
        return image

    def _draw_line(
        self,
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, num=steps)
        ys = np.linspace(y0, y1, num=steps)
        radius = max(0, int(thickness) // 2)
        for x_value, y_value in zip(xs, ys):
            cx = int(round(float(x_value)))
            cy = int(round(float(y_value)))
            x_min = max(0, cx - radius)
            x_max = min(image.shape[1], cx + radius + 1)
            y_min = max(0, cy - radius)
            y_max = min(image.shape[0], cy + radius + 1)
            image[y_min:y_max, x_min:x_max] = np.asarray(color, dtype=np.uint8)

    def _draw_circle(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        radius = int(max(1, radius))
        x_min = max(0, cx - radius)
        x_max = min(image.shape[1], cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(image.shape[0], cy + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            return
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius
        image[y_min:y_max, x_min:x_max][mask] = np.asarray(color, dtype=np.uint8)

    def _draw_arrow(
        self,
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        self._draw_line(image, start, end, color, thickness=thickness)
        direction = np.asarray([end[0] - start[0], end[1] - start[1]], dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            return
        unit = direction / length
        head_length = min(24.0, max(10.0, length * 0.25))
        rotation_left = np.array([[0.8660254, -0.5], [0.5, 0.8660254]], dtype=np.float64)
        rotation_right = np.array([[0.8660254, 0.5], [-0.5, 0.8660254]], dtype=np.float64)
        left_head = np.asarray(end, dtype=np.float64) - head_length * (rotation_left @ unit)
        right_head = np.asarray(end, dtype=np.float64) - head_length * (rotation_right @ unit)
        self._draw_line(image, end, (int(round(left_head[0])), int(round(left_head[1]))), color, thickness=thickness)
        self._draw_line(image, end, (int(round(right_head[0])), int(round(right_head[1]))), color, thickness=thickness)

    def _build_analysis(self, ctx: ReadOnlySimContext) -> Dict[str, Any]:
        accessor = ctx.accessor
        self._last_accessor = accessor

        static_all = accessor.get_static_data()
        derived_all = accessor.get_derived_state()
        if self.agent_id not in static_all or self.agent_id not in derived_all:
            raise KeyError(
                f"Humanoid21BalanceAnalysisObserver: accessor does not provide "
                f"static/derived data for agent {self.agent_id!r}"
            )
        static_agent = static_all[self.agent_id]
        derived_agent = derived_all[self.agent_id]
        ground_geom_name = static_all.get('ground_geom_name', 'ground')

        center_of_mass = self._compute_center_of_mass_excluding_feet(static_agent, derived_agent)
        center_of_mass_velocity = self._compute_center_of_mass_velocity(static_agent, derived_agent)
        robot_forward_ground = self._compute_robot_forward_ground_direction(static_agent, derived_agent)

        left_ankle_support_point, left_ankle_anchors = self._compute_ankle_support_point(
            static_agent, derived_agent, side="left"
        )
        right_ankle_support_point, right_ankle_anchors = self._compute_ankle_support_point(
            static_agent, derived_agent, side="right"
        )

        foot_left_body_name = static_agent['keypoint_body_names']['foot_left']
        foot_right_body_name = static_agent['keypoint_body_names']['foot_right']
        contacts = derived_all.get('contacts', [])
        left_ankle_support_force = self._compute_ankle_support_force(
            contacts=contacts,
            ground_geom_name=ground_geom_name,
            foot_body_name=foot_left_body_name,
        )
        right_ankle_support_force = self._compute_ankle_support_force(
            contacts=contacts,
            ground_geom_name=ground_geom_name,
            foot_body_name=foot_right_body_name,
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

    def _compute_center_of_mass_excluding_feet(
        self,
        static_agent: Dict[str, Any],
        derived_agent: Dict[str, Any],
    ) -> np.ndarray:
        """
        计算"去脚质心"。

        使用公开接口：
        - ``static_agent['body_names']`` 子树内所有 body
        - ``static_agent['keypoint_body_names']['foot_left' / 'foot_right']``
          用于剔除双脚
        - ``static_agent['body_masses_by_name'][name]`` 质量 (kg)
        - ``derived_agent['body_xipos'][name]`` 惯性中心世界坐标 (m)

        公式： p_com = (Σ_i m_i * p_i) / (Σ_i m_i)
        """
        foot_left = static_agent['keypoint_body_names']['foot_left']
        foot_right = static_agent['keypoint_body_names']['foot_right']
        included = [n for n in static_agent['body_names'] if n not in (foot_left, foot_right)]
        if not included:
            raise ValueError(f"No body remains after excluding feet for {self.agent_id}")

        masses = np.asarray(
            [static_agent['body_masses_by_name'][n] for n in included], dtype=np.float64
        )
        positions = np.asarray(
            [derived_agent['body_xipos'][n] for n in included], dtype=np.float64
        )
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            raise ValueError(f"Invalid total mass for {self.agent_id}: {total_mass}")
        return np.sum(positions * masses[:, None], axis=0) / total_mass

    def _compute_center_of_mass_velocity(
        self,
        static_agent: Dict[str, Any],
        derived_agent: Dict[str, Any],
    ) -> np.ndarray:
        """
        计算去脚质心速度向量。

        使用公开接口：
        - ``derived_agent['body_linvel_world'][name]`` 为各 body 世界系线速度

        公式： v_com = (Σ_i m_i * v_i) / (Σ_i m_i)

        若总质量为 0（理论不可能，保留为守门条件）则退化使用躯干线速度，
        避免整个观测崩溃——这一路径在正常仿真下不会被触发。
        """
        foot_left = static_agent['keypoint_body_names']['foot_left']
        foot_right = static_agent['keypoint_body_names']['foot_right']
        included = [n for n in static_agent['body_names'] if n not in (foot_left, foot_right)]
        torso_name = static_agent['keypoint_body_names']['torso']

        if not included:
            return np.asarray(derived_agent['body_linvel_world'][torso_name], dtype=np.float64).copy()

        masses = np.asarray(
            [static_agent['body_masses_by_name'][n] for n in included], dtype=np.float64
        )
        linvels = np.asarray(
            [derived_agent['body_linvel_world'][n] for n in included], dtype=np.float64
        )
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            return np.asarray(derived_agent['body_linvel_world'][torso_name], dtype=np.float64).copy()
        return np.sum(linvels * masses[:, None], axis=0) / total_mass

    def _compute_robot_forward_ground_direction(
        self,
        static_agent: Dict[str, Any],
        derived_agent: Dict[str, Any],
    ) -> np.ndarray:
        """
        机器人前向在地面平面上的单位方向。

        使用 ``derived_agent['body_xquat'][torso]`` 旋转局部 x 轴再投影到 XY。
        """
        torso_name = static_agent['keypoint_body_names']['torso']
        torso_quat = np.asarray(derived_agent['body_xquat'][torso_name], dtype=np.float64)
        torso_rot = R.from_quat([torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]])
        forward_world = torso_rot.apply([1.0, 0.0, 0.0])
        forward_ground = np.asarray(forward_world[:2], dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward_ground))
        if forward_norm <= self.SUPPORT_SPAN_TOLERANCE:
            return np.zeros(2, dtype=np.float64)
        return forward_ground / forward_norm

    def _compute_ankle_support_point(
        self,
        static_agent: Dict[str, Any],
        derived_agent: Dict[str, Any],
        side: str,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        单侧"踝关节支撑点" = 双踝关节锚点 (ankle_x_*, ankle_y_*) 的算术平均。

        使用公开接口：
        - ``static_agent['keypoint_joint_names']['ankle_x_{side}' / 'ankle_y_{side}']``
        - ``derived_agent['joint_world_anchor'][joint_full_name]``
        """
        keypoint_joints = static_agent['keypoint_joint_names']
        joint_anchor = derived_agent['joint_world_anchor']
        try:
            ankle_y_name = keypoint_joints[f'ankle_y_{side}']
            ankle_x_name = keypoint_joints[f'ankle_x_{side}']
        except KeyError as exc:
            raise KeyError(
                f"Keypoint joint name for ankle side={side!r} missing in "
                f"static_data['keypoint_joint_names'] for {self.agent_id}"
            ) from exc
        if ankle_y_name not in joint_anchor or ankle_x_name not in joint_anchor:
            raise KeyError(
                f"Joint anchor missing for {ankle_y_name!r} or {ankle_x_name!r} "
                f"in derived_state['joint_world_anchor']"
            )

        ankle_y_anchor = np.asarray(joint_anchor[ankle_y_name], dtype=np.float64).copy()
        ankle_x_anchor = np.asarray(joint_anchor[ankle_x_name], dtype=np.float64).copy()
        support_point = 0.5 * (ankle_y_anchor + ankle_x_anchor)
        return support_point, {
            "ankle_y": ankle_y_anchor,
            "ankle_x": ankle_x_anchor,
        }

    def _compute_ankle_support_force(
        self,
        contacts: list,
        ground_geom_name: str,
        foot_body_name: str,
    ) -> np.ndarray:
        """
        单脚通过接触传递到踝部的支撑反力代理（世界坐标 3D 向量）。

        使用公开接口 ``derived_state['contacts']`` —— 其中每条接触记录
        已经把 ``force_on_body_b_world`` 事先旋转到了世界系。由此本方法
        不再需要直接访问 ``data.contact`` 或 ``mj_contactForce``。

        约定：遍历所有接触，筛选同时涉及 ``foot_body_name`` 和
        ``ground_geom_name`` 的条目；按 MuJoCo 的 "force on body B by A"
        约定，若 foot 是 body_b 则直接累加、若是 body_a 则取反后累加
        （Newton 第三定律）。
        """
        support_force = np.zeros(3, dtype=np.float64)

        for contact in contacts:
            body_a = contact.get('body_a_name', '')
            body_b = contact.get('body_b_name', '')
            geom_a = contact.get('geom_a_name', '')
            geom_b = contact.get('geom_b_name', '')
            force_on_b = np.asarray(
                contact.get('force_on_body_b_world', np.zeros(3)), dtype=np.float64
            )

            foot_is_b = (body_b == foot_body_name and geom_a == ground_geom_name)
            foot_is_a = (body_a == foot_body_name and geom_b == ground_geom_name)
            if foot_is_b:
                support_force += force_on_b
            elif foot_is_a:
                support_force -= force_on_b

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

