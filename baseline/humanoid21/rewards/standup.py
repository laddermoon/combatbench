"""Standup potential-based rewarder observer plugin for ``humanoid21``.

本模块实现了一个高可读、严谨的人形机器人（humanoid21）起身分段势能函数计算器。
设计理念基于吴恩达（Andrew Ng）提出的 Potential-Based Reward Shaping (PBRS) 理论，
旨在不借助模仿学习动捕数据的前提下，纯靠物理指标边界与光滑势能梯度引导机器人从任意跌倒姿态起立。

设计架构与判定流程
==================
每一物理控制步，系统都会执行以下计算：

1. **核心物理量提取 (Indicator Extraction)**：
   - 提取盆骨高度 $h_{\text{pelvis}}$，躯干在世界系下的直立度 $u_{\text{torso}}$ (即 `uprightness` = $\cos\theta$)。
   - 通过旋转躯干局部 X 轴（`[1, 0, 0]`）到世界坐标系，计算 $f_{\text{down}} = -x_{\text{world\_z}}$，代表“脸朝下”（俯卧）的程度。
   - 获取关节平均绝对角速度 $\bar{v}_{\text{joint}}$。

2. **接触地面的身体部位诊断 (Contact Body Part Analysis)**：
   - 双脚（`foot_left`, `foot_right`）
   - 双膝（`shin_left`, `shin_right`）
   - 双手（`hand_left`, `hand_right`）
   - 其他身体部位（`has_other_contact`：如手肘/lower_arm、大腿、臀部、躯干、头部等，即除双脚、双手、双膝外的所有部位）

3. **阶段划分优先级判定 (Stage Prioritization Routing)**：
   采用自顶向下的严格布尔流（Boolean Stream）确定唯一的当前状态阶段 $S \in \{0, 1, 2, 3, 4\}$：
   
   - **Stage 4 (完美独立平衡站立 / Perfect Stand)**:
     - 判定：双脚（单脚或双脚）着地，且双手腾空，双膝腾空，且没有任何其他部位触地。
     - 势能区间：`[0.75, 1.00]`。
     
   - **Stage 3 (双脚着地 + 手部辅助 / Feet & Hands Support)**:
     - 判定：双脚同时着地，且双手（单手或双手）触地，且双膝腾空，且没有任何其他部位触地。
     - 势能区间：`[0.60, 0.75]`。
     
   - **Stage 2 (单膝单脚半跪过渡 / Half-Kneeling Transition)**:
     - 判定：（左脚 + 右膝）或（右脚 + 左膝）同时触地。手部可以腾空，也可以只有双手（单手或双手）触地支撑，手肘等其他部位绝对不准触地。
     - 势能区间：`[0.40, 0.60]`。
     
   - **Stage 1 (双膝跪地过渡 / Double Kneeling)**:
     - 判定：双膝（左膝+右膝）同时触地。脚部接触状态不限。手部可以腾空，也可以只有双手触地支撑，手肘等其他部位绝对不准触地。
     - 势能区间：`[0.20, 0.40]`。
     
   - **Stage 0 (翻身趴地引导 / Rollover & Belly-down)**:
     - 判定：不符合上述所有更高阶段的躺平/挣扎状态。
     - 势能区间：`[0.00, 0.20]`。
     - 设计逻辑：在此状态下只追求将身体转成脸朝下（趴着），根据俯卧程度 $f_{\text{down}}$ 给予光滑过渡势能，彻底解决翻身过程的势能梯度断层。

4. **阶段内光滑势能计算 (In-stage Smooth Shaping)**：
   确定阶段后，在对应的势能区间内，根据物理指标进行光滑插值，提供无断点的策略梯度流。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class StandupPotentialRewarder(BaseObserverPlugin):
    """基于先判定阶段、后平滑计算势能的 5 阶段起身势能函数计算器"""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._stage: int = 0
        self._potential: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._stage = 0
        self._potential = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        
        # 1. 骨盆（重心）世界高度
        h_pelvis = float(core_state["root_pos"][2])
        
        # 2. 躯干直立度 cos(tilt)
        u_torso = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )
        
        # 3. 躯干朝向判定：提取四元数，计算局部 X 轴（Forward 前方）在世界系下的 Z 投影
        static_data = ctx.accessor.get_static_data()[self.agent_id]
        torso_body_name = static_data["keypoint_body_names"]["torso"]
        
        body_xquat_dict = derived_state.get("body_xquat", {})
        q_torso = body_xquat_dict.get(torso_body_name, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        x_world_z = 2.0 * (x * z - w * y)
        f_down = -x_world_z  # 当胸部（X轴正向）指向地面时为正。1.0 为完美俯卧

        # 4. 关节平均绝对角速度（用以判定站立后的稳固性）
        joint_vel = np.asarray(core_state["joint_vel_norm"], dtype=np.float32)
        mean_abs_joint_vel = float(np.mean(np.abs(joint_vel)))

        # 5. 分析接触点和接触身体部位
        contacts = self._get_detailed_contacts(ctx)
        
        # =====================================================================
        # 严格的分阶段判定流 (Stage Determination First)
        # =====================================================================
        stage = 0
        potential = 0.0

        # 获取各部位基础触地布尔值
        foot_l = contacts["foot_left"]
        foot_r = contacts["foot_right"]
        knee_l = contacts["shin_left"]
        knee_r = contacts["shin_right"]
        hand_l = contacts["hand_left"]
        hand_r = contacts["hand_right"]
        other = contacts["has_other_contact"]

        # Stage 4: 双脚着地，手、膝、其他部位完全腾空 (Perfect Stand)
        # 判定：任意一只脚或双脚着地，且没有任何非脚部接触
        if (foot_l or foot_r) and not (hand_l or hand_r) and not (knee_l or knee_r) and not other:
            stage = 4
            # 势能区间 [0.75, 1.00]
            # h_score: 骨盆高度在 0.6m - 0.9m 之间平滑过渡
            h_score = float(np.clip((h_pelvis - 0.6) / 0.3, 0.0, 1.0))
            # u_score: 直立度在 0.5 - 1.0 之间平滑过渡
            u_score = float(np.clip((u_torso - 0.5) / 0.5, 0.0, 1.0))
            # v_score: 关节晃动抑制，角速度越小越稳
            v_score = float(np.exp(-mean_abs_joint_vel))
            potential = 0.75 + 0.25 * h_score * u_score * v_score

        # Stage 3: 双脚同时着地，手部辅助撑地，膝盖及其他部位腾空 (Feet & Hands Support)
        # 判定：双脚着地，手（单手或双手）触地，膝盖及其他部分未接触
        elif foot_l and foot_r and (hand_l or hand_r) and not (knee_l or knee_r) and not other:
            stage = 3
            # 势能区间 [0.60, 0.75]
            # h_score: 骨盆高度在 0.4m - 0.8m 之间平滑过渡
            h_score = float(np.clip((h_pelvis - 0.4) / 0.4, 0.0, 1.0))
            potential = 0.60 + 0.15 * h_score

        # Stage 2: 单脚单膝半跪姿态，手部辅助允许（必须只能是手，不准手肘），其他部位腾空 (Half-Kneeling Transition)
        # 判定：（左脚 + 右膝）或（右脚 + 左膝），手接触状态不限，但绝对不准手肘等其他部位触地
        elif ((foot_l and knee_r) or (foot_r and knee_l)) and not other:
            stage = 2
            # 势能区间 [0.40, 0.60]
            # h_score: 骨盆高度在 0.35m - 0.70m 之间平滑
            h_score = float(np.clip((h_pelvis - 0.35) / 0.35, 0.0, 1.0))
            # u_score: 直立度在 0.0 - 0.80 之间平滑
            u_score = float(np.clip((u_torso - 0.0) / 0.8, 0.0, 1.0))
            potential = 0.40 + 0.20 * h_score * u_score

        # Stage 1: 双膝跪地，手部辅助允许（必须只能是手，不准手肘），膝盖以上部位不准触地 (Double Kneeling)
        # 判定：双膝着地，大腿/躯干/手肘等腾空，双脚着地状态不限
        elif knee_l and knee_r and not other:
            stage = 1
            # 势能区间 [0.20, 0.40]
            # h_score: 骨盆高度在 0.25m - 0.50m 之间平滑
            h_score = float(np.clip((h_pelvis - 0.25) / 0.25, 0.0, 1.0))
            # u_score: 直立度在 0.0 - 0.80 之间平滑
            u_score = float(np.clip((u_torso - 0.0) / 0.8, 0.0, 1.0))
            potential = 0.20 + 0.20 * h_score * u_score

        # Stage 0: 翻身趴地引导 / 任意平躺/挣扎姿势 (Rollover & Belly-down)
        # 判定：不满足以上更高阶段时，根据躯干脸朝下的对齐程度引导翻身
        else:
            stage = 0
            # 势能区间 [0.00, 0.20]
            # 对齐投影将 f_down 在 [-1, 1] 之间光滑映射到 [0, 1] 区间
            f_score = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))
            potential = 0.20 * f_score

        self._stage = stage
        self._potential = potential

    def _get_detailed_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, bool]:
        """精细化解析地面接触数据，区分双脚、双手、双膝（Shin）以及其他部位"""
        derived_state = ctx.accessor.get_derived_state()
        env_contacts = derived_state.get('robot_environment_contacts', [])
        
        static_data = ctx.accessor.get_static_data()[self.agent_id]
        keypoint_names = static_data["keypoint_body_names"]
        
        ground_geom = 'ground'
        
        contacts = {
            "foot_left": False,
            "foot_right": False,
            "hand_left": False,
            "hand_right": False,
            "shin_left": False,
            "shin_right": False,
            "has_other_contact": False,
        }
        
        foot_left_body = keypoint_names["foot_left"]
        foot_right_body = keypoint_names["foot_right"]
        hand_left_body = keypoint_names["hand_left"]
        hand_right_body = keypoint_names["hand_right"]
        
        suffix = ""
        if foot_left_body.endswith("_red"):
            suffix = "_red"
        elif foot_left_body.endswith("_blue"):
            suffix = "_blue"
            
        shin_left_body = f"shin_left{suffix}"
        shin_right_body = f"shin_right{suffix}"
        
        for c in env_contacts:
            if c.get('robot') != self.agent_id:
                continue
            if c.get('environment_geom') != ground_geom:
                continue
            if c.get('force', 0.0) < 1.0:  # 1.0 牛顿以上才认定为触地
                continue
            
            body = c.get('body', '')
            if body == foot_left_body:
                contacts["foot_left"] = True
            elif body == foot_right_body:
                contacts["foot_right"] = True
            elif body == hand_left_body:
                contacts["hand_left"] = True
            elif body == hand_right_body:
                contacts["hand_right"] = True
            elif body == shin_left_body:
                contacts["shin_left"] = True
            elif body == shin_right_body:
                contacts["shin_right"] = True
            else:
                # 凡是膝盖、脚、手之外的机器人部位触地，全标定为 other_contact
                contacts["has_other_contact"] = True
                
        return contacts

    def get_output(self) -> Dict[str, float]:
        return {
            "stage": float(self._stage),
            "potential": self._potential,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupPotentialRewarder":
        return cls(**config)
