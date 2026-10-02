"""binding_registry — blueprint simulator cls → 设备绑定解析。

E3-W1：collector 不再硬编码具体后端与任务 schema。绑定声明：

- ``make_sim(batch_size, device)``  → 满足 BatchRuntime 契约的后端
  对象（build_sim_namespace/attach_state/dev_*/physical_step/
  capture/reset/task_tables/device_obs_builder）；
- ``io_schema(sim)``                → agent_ids + 每 agent
  obs_dim/action_dim——RecordStore 预分配与 Episode 导出据此取形，
  不再写死 96/21/"robot_a"/"robot_b"；
- ``episode_options_keys``          → reset options 的白名单键
  （按任务绑定声明，不在 collector 内联）。

未注册的 simulator cls 显式拒绝（与 capability_registry 同风格：
不支持即报错，不静默降级）。

新任务接入 = 实现一个绑定类 + ``register_binding(sim_cls, binding)``。
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import torch


class IoSchema(NamedTuple):
    """per-agent IO 形状契约（由绑定声明，collector/store/exporter 消费）。"""

    agent_ids: Tuple[str, ...]
    obs_dim: int        # 每 agent 观测维度
    action_dim: int     # 每 agent 动作维度


class DeviceBinding:
    """设备绑定基类（任务×后端对）。"""

    #: 展示/诊断名
    name: str = "unnamed-binding"

    #: reset options 白名单键（per-env episode_options 可广播的键）
    episode_options_keys: Tuple[str, ...] = ()

    def make_sim(self, batch_size: int, device: str):
        """构造后端 sim 对象（见模块 docstring 的契约清单）。"""
        raise NotImplementedError

    def io_schema(self, sim) -> IoSchema:
        """从已构造的 sim 推导 IO schema。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
_BINDINGS: Dict[str, DeviceBinding] = {}


def register_binding(sim_cls: str, binding: DeviceBinding) -> None:
    if sim_cls in _BINDINGS:
        raise ValueError(f"binding already registered for {sim_cls!r}")
    _BINDINGS[sim_cls] = binding


def resolve_binding(sim_cls: str) -> DeviceBinding:
    b = _BINDINGS.get(sim_cls)
    if b is None:
        raise ValueError(
            f"no device binding registered for simulator {sim_cls!r}; "
            f"registered: {sorted(_BINDINGS)}")
    return b


def registered_sim_clses() -> Tuple[str, ...]:
    return tuple(sorted(_BINDINGS))


# ---------------------------------------------------------------------------
# 内置：humanoid21 → warp 后端
# ---------------------------------------------------------------------------
class _Humanoid21WarpBinding(DeviceBinding):
    """envs.humanoid21.simulator:Humanoid21Simulator 的 warp 实现绑定。"""

    name = "humanoid21-warp"
    SIM_CLS = "envs.humanoid21.simulator:Humanoid21Simulator"
    episode_options_keys = ("initial_distance", "initial_pose_a",
                            "initial_pose_b")

    def make_sim(self, batch_size: int, device: str):
        from .warp_simulator import WarpHumanoid21Simulator
        return WarpHumanoid21Simulator(batch_size=batch_size, device=device)

    def io_schema(self, sim) -> IoSchema:
        obs_builder = sim.device_obs_builder()
        return IoSchema(agent_ids=("robot_a", "robot_b"),
                        obs_dim=int(obs_builder.obs_dim()),
                        action_dim=int(sim.ACTION_DIM))


register_binding(_Humanoid21WarpBinding.SIM_CLS, _Humanoid21WarpBinding())
