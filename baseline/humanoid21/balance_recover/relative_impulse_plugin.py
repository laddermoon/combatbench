"""相对角度冲量扰动插件。

纯执行层：从 ``episode_options["impulse_params"]`` 读取扰动参数，
在内部 sim 中用 ``EnvRuntime`` + ``ConstantForcePlugin`` 施力，
将扰动后的 core state 写回真实环境。

方向定义（重要）：
    direction_angle 表示的是**力指向的方向**，即受力后机器人倒下的方向，
    而非力来源的方向。

    - 0°   = 向前：受力后机器人向其前方倒
    - 90°  = 向右：受力后机器人向其右方倒
    - 180° = 向后：受力后机器人向其后方倒
    - 270° = 向左：受力后机器人向其左方倒

    对两个机器人使用相同的相对角度定义，方向转换在 ``on_pre_episode`` 中
    从 ``root_rot`` 提取 heading 后完成。

episode_options 格式::

    ctx.episode_options["impulse_params"] = {
        "robot_a": {
            "direction_angle": 90.0,
            "force": 200.0,
            "duration_action_steps": 4,
            "body": "torso",
        },
        "robot_b": {
            "direction_angle": 180.0,
            "force": 150.0,
            "duration_action_steps": 3,
            "body": "head",
        },
    }

    未出现在字典中的机器人不会被扰动。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from envs.framework import BasePlugin
from envs.framework.context import SimContext


class RelativeImpulsePlugin(BasePlugin):
    """相对角度冲量扰动插件（纯执行层）。

    方向定义：direction_angle 是**力指向的方向**（机器人倒下的方向），
    不是力来源的方向：
        0°=向前, 90°=向右, 180°=向后, 270°=向左

    工作流程：
    1. ``on_pre_episode`` 时从 ``episode_options["impulse_params"]`` 读取参数。
    2. 对每个待扰动机器人，在内部 sim 中施力 + 策略控制。
    3. 取扰动后 core state 写回真实环境。

    参数来源：``episode_options["impulse_params"]``，由实验类负责采样和组装。
    """

    def __init__(
        self,
        target_robots: Union[str, List[str]] = ("robot_a", "robot_b"),
        policy_blueprint_path: Optional[str] = None,
        impulse_body: str = "torso",
        phy_steps_per_action: int = 25,
    ):
        """
        Args:
            target_robots: 目标机器人 ID 或 ID 列表。默认两个机器人都扰动。
                只有同时出现在 ``target_robots`` 和 ``episode_options`` 中的
                机器人才会被实际扰动。
            policy_blueprint_path: 策略 blueprint YAML 路径（用于内部 sim）。
                None 则用零 action（PD 控制器拉回默认站姿）。
            impulse_body: 默认施力部位，可被 episode_options 中的 per-robot body 覆盖。
            phy_steps_per_action: 每动作步的物理步数。
        """
        if isinstance(target_robots, str):
            self.target_robots = [target_robots]
        else:
            self.target_robots = list(target_robots)
        self.policy_blueprint_path = policy_blueprint_path
        self.impulse_body = impulse_body
        self.phy_steps_per_action = int(phy_steps_per_action)
        self._internal_sim: Optional[Any] = None
        self._internal_runtime: Optional[Any] = None
        self._policy: Optional[Any] = None

    @property
    def name(self) -> str:
        return "relative_impulse_perturbation"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robots": self.target_robots,
            "policy_blueprint_path": self.policy_blueprint_path,
            "impulse_body": self.impulse_body,
            "phy_steps_per_action": self.phy_steps_per_action,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RelativeImpulsePlugin":
        return cls(**config)

    def _ensure_internal_sim(self) -> Any:
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
        return self._internal_sim

    def _ensure_internal_runtime(self, force_plugin) -> Any:
        """创建或复用内部 EnvRuntime。每次施力创建新的 ConstantForcePlugin。"""
        from envs.framework.env_runtime import EnvRuntime
        if self._internal_runtime is not None:
            self._internal_runtime.close()
        sim = self._ensure_internal_sim()
        self._internal_runtime = EnvRuntime(
            simulator=sim,
            plugins=[force_plugin],
            phy_steps_per_action=self.phy_steps_per_action,
        )
        return self._internal_runtime

    def _ensure_policy(self) -> Any:
        if self._policy is None and self.policy_blueprint_path is not None:
            from envs.framework.policy import PolicyBlueprint
            bp = PolicyBlueprint.load(Path(self.policy_blueprint_path))
            self._policy = bp.build()
        return self._policy

    def on_pre_episode(self, ctx: SimContext) -> None:
        from envs.humanoid21.disturbance_plugins import ConstantForcePlugin

        params_all = ctx.episode_options.get("impulse_params", {})
        robots_to_disturb = [r for r in self.target_robots if r in params_all]
        if not robots_to_disturb:
            return

        real_state = ctx.accessor.get_core_state()
        sim = self._ensure_internal_sim()
        policy = self._ensure_policy()

        for robot_id in robots_to_disturb:
            p = params_all[robot_id]
            body = p.get("body", self.impulse_body)
            rel_angle_deg = float(p["direction_angle"])
            force = float(p["force"])
            duration_action_steps = int(p["duration_action_steps"])
            duration_phy_steps = duration_action_steps * self.phy_steps_per_action

            # 1. 创建内部 EnvRuntime + ConstantForcePlugin
            force_plugin = ConstantForcePlugin(
                agent_id=robot_id,
                force=force,
                direction=rel_angle_deg,
                duration_action_steps=duration_action_steps,
                body_name=body,
            )
            runtime = self._ensure_internal_runtime(force_plugin)

            # 2. 初始化内部 sim 状态
            runtime.reset()
            sim.set_core_state(real_state)

            # 3. 策略 reset
            if policy is not None:
                policy.reset()

            # 保存非目标机器人的初始状态（每个 action step 后重置，防止干扰）
            other_robot = "robot_b" if robot_id == "robot_a" else "robot_a"
            non_target_state = {
                rid: {k: v.copy() for k, v in state.items()}
                for rid, state in real_state.items()
                if rid != robot_id
            }

            # 4. 循环 duration_action_steps 次 runtime.step()
            #    EnvRuntime 自动管理 action 步/物理步节奏：
            #    每 phy_steps_per_action 个物理步才 set_action 一次
            #    ConstantForcePlugin 在 on_pre_phy_step 中每步施力
            for _ in range(duration_action_steps):
                if policy is not None:
                    obs = sim.get_observation()
                    action, _ = policy.act(obs.get(robot_id))
                else:
                    action = np.zeros(21, dtype=np.float32)
                runtime.step(action, np.zeros(21, dtype=np.float32))

                # 每个 action step 后重置非目标机器人
                if non_target_state:
                    sim.set_core_state(non_target_state)

            # 6. 取扰动后的 core state 写回真实环境
            perturbed_state = sim.get_core_state()
            ctx.mutator.set_core_state({
                robot_id: perturbed_state[robot_id],
            })

            # 7. 记录元数据到 metrics
            ctx.metrics[f"{robot_id}_impulse_body"] = body
            ctx.metrics[f"{robot_id}_impulse_force"] = force
            ctx.metrics[f"{robot_id}_impulse_duration_action_steps"] = duration_action_steps
            ctx.metrics[f"{robot_id}_impulse_duration_phy_steps"] = duration_phy_steps
            ctx.metrics[f"{robot_id}_impulse_direction_angle"] = rel_angle_deg

            # 更新 real_state，使下一个机器人的扰动基于当前状态
            real_state = sim.get_core_state()
