"""原生插件最小模板（M3 W4）——四种典型形态的可复用样例。

1. ``HeightObserver``     — 无状态 observer（只读 sim 视图 → get_output）
2. ``EpLenCounterPlugin`` — 有状态插件（declare_state 持久张量 + 行清零）
3. ``PulseForcePlugin``   — 逐子步施力（force schedule 上传）
4. ``JitterResetPlugin``  — reset 时重初始化本插件的 pool 行

每个模板都是可直接 attach 到 BatchRuntime 的完整实现，
测试文件直接复用它们验证生命周期语义。
"""
from __future__ import annotations

import torch

from .device_plugin import BaseDeviceObserver, BaseDevicePlugin, DeviceCtx
from .device_state import ExecutionPlane


class HeightObserver(BaseDeviceObserver):
    """模板1：无状态 observer——每步读 torso 高度。"""

    def __init__(self, torso_body: int = 0):
        self._bid = int(torso_body)
        self._out = None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        self._out = ctx.sim.xpos[:, self._bid, 2].clone()

    def get_output(self):
        return self._out


class EpLenCounterPlugin(BaseDevicePlugin):
    """模板2：有状态插件——per-env 计数器，partial reset 自动清零行。"""

    @property
    def name(self) -> str:
        return "ep_len_counter"

    def declare_state(self, state) -> None:
        state.declare_state(self.name, "count", (), torch.float32)

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        ctx.pstate["count"] += 1.0


class PulseForcePlugin(BaseDevicePlugin):
    """模板3：逐子步施力——pre_batch_step 上传 force schedule。

    在前 ``pulse_steps`` 个子步对 ``body_id`` 施加恒力，
    其余子步为 0（设备端逐子步消费，对应旧框架"每物理步施力"
    语义的正确等价物，而不是块边界的一次性冲量）。
    """

    def __init__(self, body_id: int, force=(0.0, 0.0, 1.0),
                 pulse_steps: int = 5):
        self._bid = int(body_id)
        self._force = tuple(float(f) for f in force)
        self._pulse = int(pulse_steps)

    @property
    def name(self) -> str:
        return "pulse_force"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_batch_step(self, ctx: DeviceCtx) -> None:
        n = ctx.sim.xfrc_applied.shape[1]  # nbody
        sched = torch.zeros(ctx.batch_size, ctx.phy_substeps, n, 6,
                            device=ctx.sim.qpos.device)
        sched[:, : self._pulse, self._bid, :3] = torch.as_tensor(
            self._force, device=sched.device)
        ctx.mutator.upload_force_schedule(sched)


class JitterResetPlugin(BaseDevicePlugin):
    """模板4：reset 感知插件——episode 开始时给本插件 pool 行写初值。

    展示 on_pre_episode / on_envs_reset 分工：全量 reset 与部分 reset
    走不同 hook；pool 行已由 runtime 清零，插件在此写非零初值。
    """

    def __init__(self, init_value: float = 1.0):
        self._v = float(init_value)

    @property
    def name(self) -> str:
        return "jitter_reset"

    @property
    def require_mutator(self) -> bool:
        return True

    def declare_state(self, state) -> None:
        state.declare_state(self.name, "phase", (2,), torch.float32)

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        # reset_env_ids=None → 全量；否则只写本批 reset 的行
        if ctx.reset_env_ids is None:
            ctx.pstate["phase"][:] = self._v
        else:
            ctx.pstate["phase"][ctx.reset_env_ids] = self._v

    on_envs_reset = on_pre_episode  # 同一初始化逻辑；分开覆写亦可


class SubstepProbePlugin(BaseDevicePlugin):
    """模板5：空子步 hook——E7-W0 计量探针。

    覆写 ``on_post_phy_step``（空体）使 runtime 进入子步驱动路径：
    每子步 pre/post 回调 + 终止屏障全走，但回调本身零负载——
    测得的增量即"子步路径固定开销"（屏障/调度），不含插件业务。
    """

    @property
    def name(self) -> str:
        return "e7_substep_probe"

    def on_post_phy_step(self, ctx: DeviceCtx) -> None:
        pass
