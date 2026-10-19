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

技能衔接热身（v2）：``episode_options["warmup"]`` 可按 robot 提供
``{"skill": str, "steps": int, "schedule": rows}`` ——schedule 与
``move_schedule`` 同格式（含打击 eff 字段）。此时内部 sim 切换为
``CommandSimulator(strike_dyn_target=True)``，热身动作的目标指令
序列由 schedule 驱动：任意技能（含打击中段/后段、移动中、转身中）
都可作为 episode 初始分布，覆盖技能衔接瞬态。未提供 spec 的
robot 走原有零指令踏步路径。
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
        # Two internal sims keyed by warmup mode: legacy zero-cmd stepping
        # uses the plain simulator; skill-schedule warmup needs
        # CommandSimulator (move_schedule + strike dynamic targets).
        self._internal_sims: Dict[str, Any] = {}
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

    def _ensure_internal_sim(self, mode: str) -> Any:
        if mode not in self._internal_sims:
            if mode == "skills":
                # CommandSimulator drives the warmup cmd sequence itself
                # via reset(options['move_schedule']) — reuses strike
                # dynamic-target anchoring; obs comes back 66-dim already.
                from baseline.humanoid21.hierarchy.command_simulator import (
                    CommandSimulator)
                self._internal_sims[mode] = CommandSimulator(
                    cmd_version="v2", strike_dyn_target=True)
            else:
                from envs.humanoid21.simulator import Humanoid21Simulator
                self._internal_sims[mode] = Humanoid21Simulator()
        return self._internal_sims[mode]

    def on_detach(self) -> None:
        for sim in self._internal_sims.values():
            sim.close()
        self._internal_sims = {}

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
        cls_str = (ctx.episode_options or {}).get("warmup_policy_cls")
        policy = self._ensure_policy(cls_str) if cls_str else None
        specs = (ctx.episode_options or {}).get("warmup") or {}
        skill_specs = {
            rid: s for rid, s in specs.items()
            if isinstance(s, dict) and s.get("schedule")
        }
        if skill_specs:
            self._run_skill_warmup(ctx, policy, specs)
        else:
            self._run_stepping_warmup(ctx, policy)
        ctx.metrics["warmup_policy_cls"] = cls_str
        if specs:
            ctx.metrics["warmup_skills"] = {
                rid: s.get("skill") for rid, s in specs.items()
                if isinstance(s, dict)
            }

    def _act(self, policy, o):
        if policy is None:
            return np.zeros(21, dtype=np.float32)
        if self.sample_actions and hasattr(policy, "sample"):
            a, _ = policy.sample(o)
        else:
            a, _ = policy.act(o)
        return np.asarray(a, dtype=np.float32)

    def _run_stepping_warmup(self, ctx: SimContext, policy) -> None:
        """Legacy path: zero-cmd in-place stepping (unchanged)."""
        sim = self._ensure_internal_sim("stepping")
        real_state = ctx.accessor.get_core_state()
        sim.reset()
        sim.set_core_state(real_state)
        if policy is not None:
            policy.reset(seed=int(self._rng.randint(0, 2**31 - 1)))

        zero_cmd = np.zeros(self.cmd_dim, dtype=np.float32)
        zero_action = np.zeros(21, dtype=np.float32)
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
                actions[rid] = self._act(policy, o)
            sim.set_action(actions)
            for _ in range(self.phy_steps_per_action):
                sim.physical_step()

        final_state = sim.get_core_state()
        ctx.mutator.set_core_state({
            rid: final_state[rid]
            for rid in self._target_set if rid in final_state
        })

    def _run_skill_warmup(self, ctx: SimContext, policy, specs) -> None:
        """Skill-schedule warmup: drive each target robot through a
        truncated skill ``move_schedule`` (incl. strike dynamic targets),
        then hand the mid-skill state to the real episode.

        ``specs[rid] = {"skill": str, "steps": int, "schedule": rows}``;
        robots without a schedule spec warm up with zero-cmd stepping
        for ``action_steps`` steps.  Each robot's warmup length is
        independent: its core-state snapshot is captured when its own
        step budget expires (the shared sim keeps running for the
        longer warmups, so the two snapshots may come from different
        sim times — each robot still starts n_i steps into its own
        warmup, which is the intended semantics).  A robot whose
        spec says ``steps: 0`` keeps its spawn state untouched."""
        sim = self._ensure_internal_sim("skills")
        real_state = ctx.accessor.get_core_state()
        schedules: Dict[str, Any] = {}
        steps_map: Dict[str, int] = {}
        for rid in ("robot_a", "robot_b"):
            spec = specs.get(rid) or {}
            n = int(spec.get("steps", self.action_steps))
            steps_map[rid] = max(0, n)
            rows = spec.get("schedule")
            schedules[rid] = rows if rows else [
                [steps_map[rid], 0.0, 0.0, 0.0]]
        sim.reset(options={"move_schedule": schedules})
        sim.set_core_state(real_state)
        if policy is not None:
            policy.reset(seed=int(self._rng.randint(0, 2**31 - 1)))

        zero_action = np.zeros(21, dtype=np.float32)
        n_total = max(steps_map.values(), default=0)
        final_states: Dict[str, Any] = {}
        for i in range(n_total):
            obs = sim.get_observation()   # {rid: 66-dim obs incl. cmd}
            actions: Dict[str, np.ndarray] = {}
            for rid in ("robot_a", "robot_b"):
                if rid not in self._target_set or rid not in obs:
                    actions[rid] = zero_action
                    continue
                actions[rid] = self._act(policy, np.asarray(
                    obs[rid], dtype=np.float32))
            sim.set_action(actions)
            for _ in range(self.phy_steps_per_action):
                sim.physical_step()
            done = [rid for rid, n in steps_map.items() if n == i + 1]
            if done:
                state_now = sim.get_core_state()
                for rid in done:
                    final_states[rid] = state_now[rid]

        ctx.mutator.set_core_state({
            rid: s for rid, s in final_states.items()
            if rid in self._target_set
        })
