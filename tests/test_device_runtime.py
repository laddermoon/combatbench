"""M3 生命周期测试——FakeBatchBackend 上的契约验证（不依赖 warp 编译）。

覆盖 ROADMAP M3 放行条件：
- hook 顺序与 mutator 授予/回收
- per-agent 终止 ≠ env 终止（all_agents_terminated 语义）
- 部分 reset 的 env 行隔离（物理行/插件池/episode 簿记）
- force schedule 逐子步生效（正确子步，非事后 history）
- HOST_SLOW 准入控制、能力注册表未注册类拒绝启动
"""
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent

from envs.batchframework.device_plugin import (  # noqa: E402
    BaseDevicePlugin, DeviceCtx,
)
from envs.batchframework.device_runtime import (  # noqa: E402
    BatchRuntime, DeviceTimeoutPlugin,
)
from envs.batchframework.device_state import ExecutionPlane  # noqa: E402
from envs.batchframework.fake_backend import FakeBatchBackend  # noqa: E402


# ---------------------------------------------------------------------------
# 测试插件
# ---------------------------------------------------------------------------
class RecorderPlugin(BaseDevicePlugin):
    """记录 hook 调用顺序与 mutator 授予情况。"""

    def __init__(self, name="recorder", mutator=False):
        self._name = name
        self._mut = mutator
        self.log = []

    @property
    def name(self):
        return self._name

    @property
    def require_mutator(self):
        return self._mut

    def _rec(self, ctx, hook):
        self.log.append((hook, ctx.mutator is not None,
                         None if ctx.reset_env_ids is None
                         else ctx.reset_env_ids.tolist()))

    def on_pre_episode(self, ctx):        self._rec(ctx, "pre_episode")
    def on_envs_reset(self, ctx):         self._rec(ctx, "envs_reset")
    def on_pre_action_step(self, ctx):    self._rec(ctx, "pre_action")
    def on_pre_batch_step(self, ctx):     self._rec(ctx, "pre_batch")
    def on_post_batch_step(self, ctx):    self._rec(ctx, "post_batch")
    def on_post_action_step(self, ctx):   self._rec(ctx, "post_action")
    def on_post_episode(self, ctx):       self._rec(ctx, "post_episode")


class AgentTerminator(BaseDevicePlugin):
    """第 kill_step 步终止 env kill_env 的指定 agent。"""

    def __init__(self, kill_step=1, kill_env=1, agent=0):
        self._ks, self._ke, self._ka = kill_step, kill_env, agent

    @property
    def name(self):
        return "agent_terminator"

    def on_post_action_step(self, ctx: DeviceCtx):
        if int(ctx.state.rng.step_counter.item()) == self._ks:
            ctx.request_termination(
                torch.tensor([self._ke],
                             device=ctx.episode.terminated_flag.device),
                "ko", agents=[self._ka])


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def _rt(B=4):
    sim = FakeBatchBackend(batch_size=B)
    sim.reset()
    return sim, BatchRuntime(sim, phy_substeps=5)


def test_hook_order_and_mutator_grant():
    """hook 顺序 + mutator 只在可写 hook、且插件声明 require_mutator 时授予。"""
    sim, rt = _rt()
    rec = RecorderPlugin(mutator=True)
    rt.attach(rec)
    rt.reset()
    rec.log.clear()
    rt.step()
    hooks = [h for h, _, _ in rec.log]
    assert hooks == ["pre_action", "pre_batch", "post_batch", "post_action"]
    muts = dict((h, m) for h, m, _ in rec.log)
    assert muts["pre_action"] and muts["pre_batch"] and muts["post_batch"]
    assert muts["post_action"] is False     # read-only hook 不授予

    rec2 = RecorderPlugin(name="ro", mutator=False)
    rt.attach(rec2)
    rt.step()
    # require_mutator=False → 即使可写 hook 也是 None（最小权限）
    assert all(m is False for _, m, _ in rec2.log)


def test_per_agent_termination_not_env_end():
    """单个 agent 终止不能误停整个 env——两个 agent 都终止才复位。"""
    sim, rt = _rt()
    rt.attach(AgentTerminator(kill_step=1, kill_env=1, agent=0))
    rt.step()
    ep = rt.state.episode
    assert ep.agent_terminated[1, 0].item() is True
    assert ep.episode_steps.tolist() == [1, 1, 1, 1]   # env 未复位
    assert ep.active_mask.tolist() == [True] * 4

    rt2_sim = FakeBatchBackend(batch_size=4)
    rt2_sim.reset()
    rt2 = BatchRuntime(rt2_sim, phy_substeps=5)
    rt2.attach(AgentTerminator(kill_step=1, kill_env=1, agent=0))
    rt2.attach(AgentTerminator(kill_step=1, kill_env=1, agent=1))
    rt2.step()
    ep2 = rt2.state.episode
    # 两 agent 均终止 → env 1 复位：episode_steps 清 0，其余 env 不动
    assert ep2.episode_steps.tolist() == [1, 0, 1, 1]
    assert ep2.agent_terminated[1].tolist() == [False, False]


def test_partial_reset_isolation():
    """plugin pool 行与 episode 行只清被 reset 的 env。"""
    sim, rt = _rt()
    from envs.batchframework.device_examples import (
        EpLenCounterPlugin, JitterResetPlugin)
    cnt, jit = EpLenCounterPlugin(), JitterResetPlugin(init_value=7.0)
    rt.attach(cnt)
    rt.attach(jit)
    rt.reset()
    rt.step(); rt.step()
    rt.attach(AgentTerminator(kill_step=3, kill_env=2, agent=0))
    rt.attach(AgentTerminator(kill_step=3, kill_env=2, agent=1))
    rt.step()
    st = rt.state
    # env2 复位：计数器行清零→on_envs_reset 不写 count；jitter 行重置为 7
    assert st.plugin["ep_len_counter"]["count"].tolist() == [3, 3, 0, 3]
    assert st.plugin["jitter_reset"]["phase"][:, 0].tolist() == [7, 7, 7, 7]
    assert st.episode.episode_steps.tolist() == [3, 3, 0, 3]


def test_device_timeout_plugin():
    sim, rt = _rt()
    rt.attach(DeviceTimeoutPlugin(max_steps=2))
    rt.reset()
    rt.step()
    assert rt.state.episode.episode_steps.tolist() == [1] * 4
    rt.step()
    assert rt.state.episode.episode_steps.tolist() == [0] * 4


def test_force_schedule_per_substep():
    """schedule 必须逐子步生效——xfrc_log 第 i 行等于 sched[:,i]。"""
    sim, rt = _rt()
    from envs.batchframework.device_examples import PulseForcePlugin
    rt.attach(PulseForcePlugin(body_id=0, force=(0, 0, 9.0), pulse_steps=3))
    rt.reset()
    rt.step()   # phy_substeps=5
    log = sim.xfrc_log
    assert len(log) == 5
    for i, snap in enumerate(log):
        expect = 9.0 if i < 3 else 0.0
        assert torch.allclose(
            snap[:, 0, 2], torch.full((4,), expect)), f"substep {i}"


def test_host_slow_rejected_and_registry():
    """HOST_SLOW 插件默认拒绝；注册表未登记类启动即失败。"""
    sim, rt = _rt()

    class Slow(BaseDevicePlugin):
        @property
        def name(self): return "slowp"
        @property
        def plane(self): return ExecutionPlane.HOST_SLOW

    with pytest.raises(ValueError):
        rt.attach(Slow())
    rt2 = BatchRuntime(sim, phy_substeps=5, allow_host_slow=True)
    rt2.attach(Slow())   # 显式允许后可接

    from envs.batchframework import capability_registry as reg
    with pytest.raises(ValueError, match="pending|unsupported|unregistered"):
        reg.resolve_plugin("some.module:UnregisteredPlugin", {}, sim)
    # NATIVE 工厂可实例化
    p = reg.resolve_plugin(
        "envs.batchframework.device_runtime:DeviceTimeoutPlugin",
        {"max_steps": 9}, sim)
    assert isinstance(p, DeviceTimeoutPlugin) and p._max == 9
    # UNSUPPORTED 条目拒绝（RandomFallenStatePlugin 已于 M4 转 NATIVE）
    with pytest.raises(ValueError):
        reg.resolve_plugin(
            "baseline.humanoid21.plugins.standup_termination"
            ":StandupTerminationPlugin",
            {}, sim)


def test_observer_dispatcher_ordering():
    """observer dispatcher 先于普通插件刷新同一步的输出。"""
    sim, rt = _rt()
    from envs.batchframework.device_examples import HeightObserver
    rt.set_observer("height", HeightObserver(torso_body=0))
    out_holder = {}

    class Reader(BaseDevicePlugin):
        @property
        def name(self): return "reader"
        def on_post_action_step(self, ctx):
            out_holder["v"] = rt.get_observer_output("height").clone()

    rt.attach(Reader())
    rt.reset(); rt.step()
    # dispatcher priority 最高 → reader 读到的是本步刷新后的值
    assert out_holder["v"].shape == (4,)
    sim.build_device_state().sim.xpos[:, 0, 2] = 5.0
    rt.step()
    assert torch.allclose(out_holder["v"], torch.full((4,), 5.0))
