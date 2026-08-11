"""相对角度冲量扰动插件。

与 ``ImpulsePerturbationPlugin`` 的核心逻辑相同（内部 sim + 策略生成物理一致的扰动状态），
但方向参数改为**相对机器人朝向的角度**。

方向定义（重要）：
    direction_angle 表示的是**力指向的方向**，即受力后机器人倒下的方向，
    而非力来源的方向。

    - 0°   = 向前：受力后机器人向其前方倒
    - 90°  = 向右：受力后机器人向其右方倒
    - 180° = 向后：受力后机器人向其后方倒
    - 270° = 向左：受力后机器人向其左方倒

    对两个机器人使用相同的相对角度定义，方向转换在 ``on_pre_episode`` 中
    从 ``root_rot`` 提取 heading 后完成。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BasePlugin
from envs.framework.context import SimContext


class RelativeImpulsePlugin(BasePlugin):
    """相对角度冲量扰动插件。

    方向定义：direction_angle 是**力指向的方向**（机器人倒下的方向），
    不是力来源的方向：
        0°=向前, 90°=向右, 180°=向后, 270°=向左

    工作流程：
    1. ``on_pre_episode`` 时读取真实环境的 core state，提取目标机器人 heading。
    2. 将相对角度转换为绝对方向向量（heading - angle，顺时针）。
    3. 在内部 sim 中施力 + 策略控制，持续 ``duration_action_steps`` 个 action step。
    4. 取扰动后 core state 写回真实环境。

    参数来源（优先级从高到低）：
    - **episode_options**：``ctx.episode_options["impulse_params"]`` 指定确切参数。
    - **构造器参数**：用构造器传入的固定值或随机范围。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        policy_blueprint_path: Optional[str] = None,
        force_magnitude: float = 100.0,
        duration_action_steps: int = 4,
        direction_angle: Union[float, Tuple[float, float]] = (0.0, 360.0),
        impulse_body: str = "torso",
        phy_steps_per_action: int = 25,
        random_seed: Optional[int] = None,
        weight_npz_path: Optional[str] = None,
        direction_jitter: float = 5.0,
    ):
        """
        Args:
            target_robot: 目标机器人 ID。
            policy_blueprint_path: 策略 blueprint YAML 路径（用于内部 sim）。
                None 则用零 action（PD 控制器拉回默认站姿）。
            force_magnitude: 力大小（N），固定值。weight_npz_path 提供时忽略。
            duration_action_steps: 持续时间（action step 数），固定值。weight_npz_path 提供时忽略。
            direction_angle: 相对机器人朝向的角度（度），表示**力指向的方向**
                （即受力后机器人倒下的方向，不是力来源的方向）。
                float=固定角度，(min, max)=随机采样范围。
                0°=向前, 90°=向右, 180°=向后, 270°=向左。
                weight_npz_path 提供时忽略。
            impulse_body: 施力部位，固定 torso。
            phy_steps_per_action: 每动作步的物理步数。
            random_seed: 随机种子。
            weight_npz_path: 采样分布权重文件路径（.npz）。提供后按权重采样
                (direction_angle, force, duration)，忽略 force/direction/duration 参数。
                npz 需包含: interp_angles, interp_weights, forces, durations。
            direction_jitter: 方向抖动范围（度，±），仅在 weight_npz_path 模式下生效。
        """
        self.target_robot = target_robot
        self.policy_blueprint_path = policy_blueprint_path
        self.impulse_body = impulse_body
        self.force_magnitude = float(force_magnitude)
        self.duration_action_steps = int(duration_action_steps)
        if isinstance(direction_angle, (int, float)):
            self.direction_angle_range = (float(direction_angle), float(direction_angle))
        else:
            self.direction_angle_range = (float(direction_angle[0]),
                                          float(direction_angle[1]))
        self.phy_steps_per_action = int(phy_steps_per_action)
        self._rng = np.random.RandomState(random_seed)
        self._sample_rng = np.random.RandomState((random_seed or 0) + 1)
        self._internal_sim: Optional[Any] = None
        self._policy: Optional[Any] = None
        self.direction_jitter = float(direction_jitter)
        self._weight_npz_path = weight_npz_path

        # 加载权重分布
        self._weight_interp_angles: Optional[np.ndarray] = None
        self._weight_interp_weights: Optional[np.ndarray] = None
        self._weight_forces: Optional[np.ndarray] = None
        self._weight_durations: Optional[np.ndarray] = None
        self._weight_flat_probs: Optional[np.ndarray] = None
        if weight_npz_path is not None:
            data = np.load(weight_npz_path, allow_pickle=True)
            self._weight_interp_angles = data["interp_angles"]
            self._weight_interp_weights = data["interp_weights"]
            self._weight_forces = data["forces"]
            self._weight_durations = data["durations"]
            flat = self._weight_interp_weights.flatten().astype(np.float64)
            self._weight_flat_probs = flat / flat.sum()

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))
        self._sample_rng = np.random.RandomState(int(seed) + 1)

    @property
    def name(self) -> str:
        return "relative_impulse_perturbation"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "policy_blueprint_path": self.policy_blueprint_path,
            "force_magnitude": self.force_magnitude,
            "duration_action_steps": self.duration_action_steps,
            "direction_angle": list(self.direction_angle_range),
            "impulse_body": self.impulse_body,
            "phy_steps_per_action": self.phy_steps_per_action,
            "weight_npz_path": self._weight_npz_path,
            "direction_jitter": self.direction_jitter,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RelativeImpulsePlugin":
        return cls(**config)

    def _ensure_internal_sim(self) -> Any:
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
        return self._internal_sim

    def _ensure_policy(self) -> Any:
        if self._policy is None and self.policy_blueprint_path is not None:
            from envs.framework.policy import PolicyBlueprint
            bp = PolicyBlueprint.load(Path(self.policy_blueprint_path))
            self._policy = bp.build()
        return self._policy

    @staticmethod
    def _extract_heading(root_rot: np.ndarray) -> float:
        """从 root_rot 四元数 [w,x,y,z] 提取 heading（yaw, 弧度）。

        heading = atan2(forward_y, forward_x)，其中 forward = R @ [1,0,0]
        （局部 x 轴即机器人前方，与 simulator 的 face_vector 定义一致）。

        前提假设：机器人基本直立（pitch/roll ≈ 0）。本插件在
        ``on_pre_episode`` 中施力，此时刚 reset 为 standing 姿态，满足该假设。
        若在已倾倒的状态上调用，前向轴向水平面的投影会被 pitch 污染，
        提取的 heading 不再等于真实 yaw。
        """
        # scipy 用 [x,y,z,w] 顺序
        rot = R.from_quat([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
        forward = rot.apply(np.array([1.0, 0.0, 0.0]))
        return float(np.arctan2(forward[1], forward[0]))

    def _sample_relative_angle(self) -> float:
        """采样相对角度（度）。"""
        return float(self._rng.uniform(*self.direction_angle_range))

    def _sample_from_weights(self) -> Tuple[float, float, int]:
        """从权重分布采样 (angle, force, duration)。"""
        n_interp = len(self._weight_interp_angles)
        n_forces = len(self._weight_forces)
        n_durs = len(self._weight_durations)
        idx = self._sample_rng.choice(len(self._weight_flat_probs), p=self._weight_flat_probs)
        a_idx = idx // (n_forces * n_durs)
        remainder = idx % (n_forces * n_durs)
        f_idx = remainder // n_durs
        d_idx = remainder % n_durs
        angle = float(self._weight_interp_angles[a_idx]) + self._sample_rng.uniform(-self.direction_jitter, self.direction_jitter)
        angle = angle % 360.0
        force = float(self._weight_forces[f_idx])
        duration = int(self._weight_durations[d_idx])
        return angle, force, duration

    def _resolve_params(self, ctx: SimContext) -> Dict[str, Any]:
        """解析扰动参数，支持 episode_options 覆盖和权重分布采样。"""
        params = ctx.episode_options.get("impulse_params", None)
        if params is not None:
            return {
                "body": params.get("impulse_body", self.impulse_body),
                "direction_angle": float(params["impulse_direction_angle"]),
                "force": float(params["impulse_force"]),
                "duration_action_steps": int(params["impulse_duration_steps"]),
            }
        if self._weight_flat_probs is not None:
            angle, force, duration = self._sample_from_weights()
            return {
                "body": self.impulse_body,
                "direction_angle": angle,
                "force": force,
                "duration_action_steps": duration,
            }
        return {
            "body": self.impulse_body,
            "direction_angle": self._sample_relative_angle(),
            "force": self.force_magnitude,
            "duration_action_steps": self.duration_action_steps,
        }

    def on_pre_episode(self, ctx: SimContext) -> None:
        sim = self._ensure_internal_sim()
        policy = self._ensure_policy()

        other_robot = "robot_b" if self.target_robot == "robot_a" else "robot_a"

        params = self._resolve_params(ctx)
        body = params["body"]
        rel_angle_deg = params["direction_angle"]
        force = params["force"]
        duration_action_steps = params["duration_action_steps"]
        duration_phy_steps = duration_action_steps * self.phy_steps_per_action

        # 1. 读取真实环境当前 core state，初始化内部 sim
        real_state = ctx.accessor.get_core_state()
        sim.reset()
        sim.set_core_state(real_state)

        # 2. 从目标机器人 root_rot 提取 heading，计算绝对方向
        #    direction_angle 表示力指向的方向（机器人倒下的方向）：
        #    0°=向前, 90°=向右, 180°=向后, 270°=向左
        #    MuJoCo 右手坐标系（z-up）中 +y 指向机器人左侧，
        #    因此用 heading - angle（顺时针旋转）使 90° 对应机器人右方。
        root_rot = np.asarray(real_state[self.target_robot]["root_rot"], dtype=np.float64)
        heading = self._extract_heading(root_rot)
        abs_angle = heading - np.radians(rel_angle_deg)
        direction = np.array([np.cos(abs_angle), np.sin(abs_angle), 0.0], dtype=np.float64)

        # 3. 策略 reset
        if policy is not None:
            policy.reset(seed=int(self._rng.randint(0, 2**31 - 1)))

        # 保存非目标机器人的初始状态（定期重置，防止干扰）
        non_target_state = {
            rid: {k: v.copy() for k, v in state.items()}
            for rid, state in real_state.items()
            if rid != self.target_robot
        }

        # 4. 施力 + 策略控制 + 物理步
        for i in range(duration_phy_steps):
            if policy is not None:
                obs = sim.get_observation()
                action, _ = policy.act(obs.get(self.target_robot))
                sim.set_action({
                    self.target_robot: action,
                    other_robot: np.zeros(21, dtype=np.float32),
                })
            else:
                sim.set_action({
                    rid: np.zeros(21, dtype=np.float32)
                    for rid in ("robot_a", "robot_b")
                })

            sim.apply_external_force(
                body_name=body,
                force=direction * force,
                robot_id=self.target_robot,
            )
            sim.physical_step()

            if non_target_state and (i + 1) % self.phy_steps_per_action == 0:
                sim.set_core_state(non_target_state)

        # 5. 取扰动后的 core state 写回真实环境
        perturbed_state = sim.get_core_state()
        ctx.mutator.set_core_state({
            self.target_robot: perturbed_state[self.target_robot],
        })

        # 6. 记录元数据到 metrics
        ctx.metrics[f"{self.target_robot}_impulse_body"] = body
        ctx.metrics[f"{self.target_robot}_impulse_force"] = force
        ctx.metrics[f"{self.target_robot}_impulse_duration_action_steps"] = duration_action_steps
        ctx.metrics[f"{self.target_robot}_impulse_duration_phy_steps"] = duration_phy_steps
        ctx.metrics[f"{self.target_robot}_impulse_direction"] = direction.tolist()
        ctx.metrics[f"{self.target_robot}_impulse_direction_angle"] = rel_angle_deg
        ctx.metrics[f"{self.target_robot}_impulse_heading"] = heading
