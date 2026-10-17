"""Warmup-stepping episode-init plugin (experiment-side).

Episode 直接从"步态中间态"开始，而不是在 episode 内部先跑一段
零指令 lead-in。

参照 ``ImpulsePerturbationPlugin`` 的内部 sim 模式：
1. ``on_pre_episode`` 时读取真实环境的 core state，写入内部 sim。
2. 在内部 sim 中用一个策略跑 ``action_steps`` 个动作步的原地踏步
   （观测 = 本体 57 维 + 全零指令，即 v2 契约下的 in-place stepping）。
3. 将结束时的 core state 写回真实环境作为 episode 初始状态。

引导策略按 episode 注入：``episode_options["warmup_policy_cls"]``
传入一个 PolicyBlueprint 的 cls 字符串（如
``file:<export>/policy.py:ExportedXxx``）。训练实验把**当前**
post-update 导出的策略写进来——热身状态始终跟随训练中的步态，
且每次 episode 因采样动作而不同。缺省时退化零 action（PD 站姿）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework.context import SimContext
from envs.framework.plugin import BasePlugin


class WarmupSteppingPlugin(BasePlugin):
    """热身踏步初始化插件（详见模块 docstring）。"""

    def __init__(
        self,
        target_robots: str = "both",
        action_steps: int = 10,
        phy_steps_per_action: int = 25,
        body_dim: int = 57,
        cmd_dim: int = 9,
        sample_actions: bool = True,
        random_seed: Optional[int] = None,
    ):
        if target_robots == "both":
            self._target_set = {"robot_a", "robot_b"}
        else:
            self._target_set = {target_robots}
        self.action_steps = int(action_steps)
        self.phy_steps_per_action = int(phy_steps_per_action)
        self.body_dim = int(body_dim)
        self.cmd_dim = int(cmd_dim)
        self.sample_actions = bool(sample_actions)
        self._rng = np.random.RandomState(random_seed)
        self._internal_sim: Optional[Any] = None
        self._policies: Dict[str, Any] = {}

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return "warmup_stepping"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robots": "both" if len(self._target_set) > 1
            else list(self._target_set)[0],
            "action_steps": self.action_steps,
            "phy_steps_per_action": self.phy_steps_per_action,
            "body_dim": self.body_dim,
            "cmd_dim": self.cmd_dim,
            "sample_actions": self.sample_actions,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "WarmupSteppingPlugin":
        return cls(**config)

    def _ensure_internal_sim(self) -> Any:
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
        return self._internal_sim

    def on_detach(self) -> None:
        if self._internal_sim is not None:
            self._internal_sim.close()
            self._internal_sim = None

    def _ensure_policy(self, cls_str: str) -> Any:
        if cls_str not in self._policies:
            from envs.framework.policy import PolicyBlueprint
            self._policies[cls_str] = PolicyBlueprint(
                cls=cls_str, config={}).build()
            # Bound the cache — a new export arrives every update.
            while len(self._policies) > 8:
                self._policies.pop(next(iter(self._policies)))
        return self._policies[cls_str]

    def on_pre_episode(self, ctx: SimContext) -> None:
        sim = self._ensure_internal_sim()
        cls_str = (ctx.episode_options or {}).get("warmup_policy_cls")
        policy = self._ensure_policy(cls_str) if cls_str else None

        # 1. 真实环境当前 core state → 内部 sim
        real_state = ctx.accessor.get_core_state()
        sim.reset()
        sim.set_core_state(real_state)

        if policy is not None:
            policy.reset(seed=int(self._rng.randint(0, 2**31 - 1)))

        zero_cmd = np.zeros(self.cmd_dim, dtype=np.float32)
        zero_action = np.zeros(21, dtype=np.float32)

        # 2. 内部 sim 跑原地踏步（全零指令 = in-place stepping）
        for _ in range(self.action_steps):
            obs = sim.get_observation()
            actions: Dict[str, np.ndarray] = {}
            for rid in ("robot_a", "robot_b"):
                if rid not in self._target_set or rid not in obs:
                    actions[rid] = zero_action
                    continue
                o = np.concatenate([
                    np.asarray(obs[rid], dtype=np.float32)[:self.body_dim],
                    zero_cmd,
                ])
                if policy is not None:
                    if self.sample_actions and hasattr(policy, "sample"):
                        a, _ = policy.sample(o)
                    else:
                        a, _ = policy.act(o)
                    actions[rid] = np.asarray(a, dtype=np.float32)
                else:
                    actions[rid] = zero_action
            sim.set_action(actions)
            for _ in range(self.phy_steps_per_action):
                sim.physical_step()

        # 3. 结束状态写回真实环境
        final_state = sim.get_core_state()
        ctx.mutator.set_core_state({
            rid: final_state[rid]
            for rid in self._target_set if rid in final_state
        })

        ctx.metrics["warmup_policy_cls"] = cls_str
