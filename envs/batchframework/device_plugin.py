"""BaseDevicePlugin + DeviceCtx — 设备端原生插件契约（M3 W2）。

与 ``batch_plugin.py``（numpy 契约原型）的关系：hook 序列与权限语义
相同，但 ctx 暴露的是 **设备张量视图**（torch.Tensor, CUDA）。DEVICE
插件在整个 action step 内不得触发任何 host↔device 传输（``Tensor.cpu()``
/ ``.item()`` / ``print(tensor)`` 均违规）——由约定 + 性能审计强制，
不在解释层拦截。

终止编码（episode.term_reason i8）::

    -1 = 无, 0 = TIMEOUT, 1 = KO, 2 = FOUL, 3 = OUT_OF_BOUNDS, 4 = CUSTOM
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .device_state import DeviceBatchState, ExecutionPlane

# 终止原因 → i8 code（与 batch_context.TerminationReason 字符串对应）
TERM_CODES: Dict[str, int] = {
    "timeout": 0, "ko": 1, "foul": 2, "out_of_bounds": 3, "custom": 4,
}
TERM_NAMES: Dict[int, str] = {v: k for k, v in TERM_CODES.items()}


# ---------------------------------------------------------------------------
# Mutator — 可写 hook 中授予的写接口
# ---------------------------------------------------------------------------
class DeviceMutator:
    """对后端 device 写方法的窄封装。只暴露契约内操作。"""

    __slots__ = ("_sim",)

    def __init__(self, sim):
        object.__setattr__(self, "_sim", sim)

    def __setattr__(self, name, value):
        raise AttributeError("DeviceMutator is immutable")

    def set_action(self, action_a: torch.Tensor, action_b: torch.Tensor) -> None:
        """(B,21) 动作张量 → PD target。on_pre_action_step 内可调用。"""
        self._sim.dev_set_action(action_a, action_b)

    def add_ext_force(self, body_id: int, force: torch.Tensor,
                      torque: Optional[torch.Tensor] = None) -> None:
        """累加挂起外力 (B,3)[,(B,3)]；作用于下一物理块的首个子步。"""
        self._sim.dev_add_ext_force(body_id, force, torque)

    def upload_force_schedule(self, sched: torch.Tensor) -> None:
        """(B, n_steps, nbody, 6) 逐子步外力表；下一 physical_step 消费。"""
        self._sim.dev_upload_force_schedule(sched)

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        """部分 reset：恢复初始姿态并重建 derived（forward）。"""
        self._sim.dev_reset_rows(env_ids)

    def set_integration_rows(self, env_ids: torch.Tensor,
                             qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """写原始 qpos/qvel 行 + forward 刷新（跨后端搬运/回放用）。"""
        self._sim.dev_set_integration_rows(env_ids, qpos, qvel)


# ---------------------------------------------------------------------------
# DeviceCtx — hook 收到的上下文
# ---------------------------------------------------------------------------
class DeviceCtx:
    """设备端插件上下文。

    - ``state``    DeviceBatchState（sim 视图只读；episode/io 见权限）
    - ``mutator``  可写 hook 中由 runtime 授予，否则为 None
    - ``pstate``   本插件在 plugin_pool 中的张量字典
    - ``reset_env_ids`` / ``terminated_env_ids``：本事件的 env 行索引
      (device long tensor; 空长 tensor = 无/全量的语义按 hook 文档)
    """

    __slots__ = ("state", "mutator", "plugin_name", "reset_env_ids",
                 "terminated_env_ids", "phy_substeps", "_mutator_impl")

    def __init__(self, state: DeviceBatchState, plugin_name: str = ""):
        self.state = state
        self.plugin_name = plugin_name
        self.mutator: Optional[DeviceMutator] = None
        self.reset_env_ids: Optional[torch.Tensor] = None
        self.terminated_env_ids: Optional[torch.Tensor] = None
        self.phy_substeps: int = 0   # 本物理块子步数（pre/post_batch_step 有效）
        self._mutator_impl: Optional[DeviceMutator] = None

    @property
    def sim(self):
        return self.state.sim

    @property
    def episode(self):
        return self.state.episode

    @property
    def io(self):
        return self.state.io

    @property
    def pstate(self) -> Dict[str, torch.Tensor]:
        return self.state.plugin.setdefault(self.plugin_name, {})

    @property
    def batch_size(self) -> int:
        return self.state.batch_size

    def request_termination(
        self,
        env_ids: torch.Tensor,
        reason: str,
        agents: Optional[Sequence[int]] = None,
    ) -> None:
        """标记终止。

        - ``agents=None``：env 级终止（整 episode 结束，等价旧框架
          ``agent_id=None`` 的全员终止）——同时写 agent_terminated 全部位
          并置 env 级 terminated_flag。
        - ``agents=[0]``：仅终止指定 agent（0=robot_a, 1=robot_b）；
          env 在该 env 全部 agent 终止后才由 runtime 消费为 env 终止
          （对齐旧框架 ``all_agents_terminated`` 语义）。
        """
        code = TERM_CODES.get(reason, TERM_CODES["custom"])
        ep = self.state.episode
        if agents is None:
            ep.agent_terminated[env_ids] = True
            ep.agent_term_reason[env_ids] = code
            ep.terminated_flag[env_ids] = True
            ep.term_reason[env_ids] = code
            ep.active_mask[env_ids] = False
        else:
            for a in agents:
                ep.agent_terminated[env_ids, a] = True
                ep.agent_term_reason[env_ids, a] = code

    # runtime 内部使用
    def _grant_mutator(self) -> None:
        self.mutator = self._mutator_impl

    def _revoke_mutator(self) -> None:
        self.mutator = None


# ---------------------------------------------------------------------------
# BaseDevicePlugin
# ---------------------------------------------------------------------------
class BaseDevicePlugin:
    """设备端原生插件基类。

    声明式属性：

    - ``plane``           ExecutionPlane（DEVICE 插件由 runtime 计 sync=0）
    - ``priority``        越大越先执行（observer dispatcher = 1_000_000）
    - ``require_mutator`` 最小权限：False 时即使可写 hook mutator 也为 None

    生命周期：

    - ``declare_state(state)``  attach 时调用一次，分配持久张量
      （``state.declare_state(name, key, shape, dtype)``）
    - hook 签名均为 ``(ctx: DeviceCtx) -> None``
    - ``on_envs_reset(ctx)``    本插件状态池行已清零后的回调
      （env_ids 在 ctx.reset_env_ids；默认 no-op——约定优于隐式发现）

    hook 表（与 batch_plugin.py 同序）::

        on_pre_episode      env 重置后、第一步前           read-write
        on_pre_action_step  action 入队后、物理块前      read-write(io.action)
        on_pre_batch_step   physical_step(n) 前          read-write(外力)
        on_post_batch_step  physical_step(n) 后          read-write(受限投影)
        on_post_action_step 每 action step               read-only(+term/reward)
        on_post_episode     env 终止后                   read-only
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def plane(self) -> ExecutionPlane:
        return ExecutionPlane.DEVICE

    @property
    def priority(self) -> int:
        return 0

    @property
    def require_mutator(self) -> bool:
        return False

    # --- attach 期 ---
    def declare_state(self, state: DeviceBatchState) -> None:
        """分配插件持久张量。默认 no-op。"""
        return None

    def on_attach(self) -> None:
        return None

    def on_detach(self) -> None:
        return None

    def set_episode_seeds(self, seeds: torch.Tensor) -> None:
        """(B,) i64 per-env 种子；reset 前调用。默认 no-op。"""
        return None

    # --- 生命周期 hooks（默认全 no-op） ---
    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        return None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        """部分 reset 后：本插件 pool 行已清零，可在此重新初始化。"""
        return None

    def on_pre_action_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_pre_batch_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_batch_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        return None


# ---------------------------------------------------------------------------
# Observer：只读单元（reward/指标），由 dispatcher 统一刷新
# ---------------------------------------------------------------------------
class BaseDeviceObserver:
    """设备端 observer——只读，输出入 get_output()。

    约定：reward unit 返回 (B,) tensor；debug/metric unit 返回任意值。
    """

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        return None

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        return None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        return None

    def get_output(self) -> Any:
        raise NotImplementedError


OBSERVER_DISPATCHER_PRIORITY = 1_000_000


class DeviceObserverDispatcher(BaseDevicePlugin):
    """observer 调度器——抢占用户插件之前刷新（priority 最高）。

    保证同一步内下游 reward/终止插件读到的 observer 输出对应当前状态。
    """

    def __init__(self):
        self.observers: Dict[str, BaseDeviceObserver] = {}

    @property
    def name(self) -> str:
        return "device_observer_dispatcher"

    @property
    def priority(self) -> int:
        return OBSERVER_DISPATCHER_PRIORITY

    def set_observer(self, name: str, unit: Optional[BaseDeviceObserver]) -> None:
        if unit is None:
            self.observers.pop(name, None)
        else:
            self.observers[name] = unit

    def get_output(self, name: str) -> Any:
        u = self.observers.get(name)
        return u.get_output() if u is not None else None

    def on_pre_episode(self, ctx):
        for u in self.observers.values():
            u.on_pre_episode(ctx)

    def on_envs_reset(self, ctx):
        for u in self.observers.values():
            u.on_envs_reset(ctx)

    def on_post_action_step(self, ctx):
        for u in self.observers.values():
            u.on_post_action_step(ctx)

    def on_post_episode(self, ctx):
        for u in self.observers.values():
            u.on_post_episode(ctx)
