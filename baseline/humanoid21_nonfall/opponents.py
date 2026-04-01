"""
对手策略模块
定义训练时使用的各种对手策略
"""
from typing import Any, Optional

import numpy as np


class BaseCombatPolicy:
    """战斗策略基类"""
    ACTION_DIM = 21

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        **kwargs,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        """根据观测返回动作"""
        raise NotImplementedError

    def reset(self) -> None:
        """重置策略内部状态"""
        pass


class StandingCombatPolicy(BaseCombatPolicy):
    """站立不动策略"""

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        return np.zeros(self.ACTION_DIM, dtype=np.float32)


class RandomCombatPolicy(BaseCombatPolicy):
    """随机动作策略"""

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        scale: float = 0.1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        return self.rng.uniform(-self.scale, self.scale, self.ACTION_DIM).astype(np.float32)


class ScriptedActiveCombatPolicy(BaseCombatPolicy):
    """
    脚本化主动攻击策略
    使用周期性的动作模式模拟攻击行为
    """

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        arm_swing_scale: float = 0.7,
        torso_scale: float = 0.2,
        leg_scale: float = 0.25,
        step_period: int = 12,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)
        self.arm_swing_scale = float(arm_swing_scale)
        self.torso_scale = float(torso_scale)
        self.leg_scale = float(leg_scale)
        self.step_period = max(1, int(step_period))
        self.rng = np.random.default_rng(seed)
        self._step = 0

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        phase_index = self._step // self.step_period
        phase = 1.0 if phase_index % 2 == 0 else -1.0
        jitter = self.rng.uniform(-0.05, 0.05, self.ACTION_DIM).astype(np.float32)
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        # 躯干动作
        action[0] = 0.25 * phase * self.torso_scale
        action[1] = -0.15 * phase * self.torso_scale
        action[2] = 0.20 * phase * self.torso_scale

        # 腿部动作
        action[3] = -phase * self.leg_scale
        action[4] = 0.5 * phase * self.leg_scale
        action[9] = phase * self.leg_scale
        action[10] = -0.5 * phase * self.leg_scale

        # 手臂动作
        action[15] = phase * self.arm_swing_scale
        action[16] = -0.7 * phase * self.arm_swing_scale
        action[17] = 0.4 * phase * self.arm_swing_scale
        action[18] = -phase * self.arm_swing_scale
        action[19] = 0.7 * phase * self.arm_swing_scale
        action[20] = -0.4 * phase * self.arm_swing_scale

        self._step += 1
        return np.clip(action + jitter, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        self._step = 0


def make_opponent_policy(
    opponent: Any = "standing",
    *,
    seed: Optional[int] = None,
    random_scale: float = 0.1,
) -> BaseCombatPolicy:
    """
    创建对手策略的工厂函数

    Args:
        opponent: 对手策略规格，可以是：
            - BaseCombatPolicy 实例
            - 返回 BaseCombatPolicy 的可调用对象
            - "standing": 站立不动
            - "random": 随机动作
            - "active"/"scripted"/"scripted_active": 脚本化主动攻击
        seed: 随机种子
        random_scale: 随机策略的动作幅度

    Returns:
        BaseCombatPolicy 实例
    """
    if isinstance(opponent, BaseCombatPolicy):
        return opponent

    if callable(opponent) and not isinstance(opponent, str):
        candidate = opponent()
        if not isinstance(candidate, BaseCombatPolicy):
            raise TypeError("Opponent factory must return a BaseCombatPolicy instance")
        return candidate

    if opponent is None:
        return StandingCombatPolicy()

    if not isinstance(opponent, str):
        raise TypeError("Opponent must be a policy instance, factory, or string spec")

    spec = opponent.strip().lower()
    if spec == "standing":
        return StandingCombatPolicy()
    if spec == "random":
        return RandomCombatPolicy(scale=random_scale, seed=seed)
    if spec in {"active", "scripted", "scripted_active"}:
        return ScriptedActiveCombatPolicy(seed=seed)

    raise ValueError(f"Unsupported opponent spec: {opponent}")
