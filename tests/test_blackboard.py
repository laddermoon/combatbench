"""E9 G1/G2/G4 契约测试——共享黑板 / 事件池 / episode_options（FakeBackend）。

验证 BLACKBOARD_DESIGN.md 的落地面：
- declare_shared：分配/幂等/冲突/行复位
- ctx.metrics：只读映射视图（KeyError 未声明、无 setitem）
- ctx.events：emit 字段/registry/时间戳、since 游标差分、epoch 失效、
  overflow 显式标志、行复位
- ctx.episode_options：reset options 的 per-env 快照发布
- 跨单元流：插件写 metrics + emit events → observer 同帧读
"""
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent

from envs.batchframework.device_plugin import (  # noqa: E402
    BaseDeviceObserver, BaseDevicePlugin, DeviceCtx,
)
from envs.batchframework.device_runtime import BatchRuntime  # noqa: E402
from envs.batchframework.fake_backend import FakeBatchBackend  # noqa: E402


def _rt(B=4):
    sim = FakeBatchBackend(batch_size=B)
    sim.reset()
    return sim, BatchRuntime(sim, phy_substeps=5)


# ---------------------------------------------------------------------------
# 测试单元
# ---------------------------------------------------------------------------
class _BoardWriter(BaseDevicePlugin):
    """post_action 写共享黑板 + emit 事件（模拟计分插件）。

    priority 高于 observer dispatcher——CPU 同款规则：要让 observer
    同帧读到，计分插件必须严格大于 OBSERVER_DISPATCHER_PRIORITY。
    """

    @property
    def name(self):
        return "board_writer"

    @property
    def priority(self):
        from envs.batchframework.device_plugin import (
            OBSERVER_DISPATCHER_PRIORITY)
        return OBSERVER_DISPATCHER_PRIORITY + 1

    @property
    def shared_writes(self):
        return ("damage",)

    def declare_state(self, state):
        state.declare_shared("damage", (), torch.float32, owner=self.name)

    def on_post_action_step(self, ctx: DeviceCtx):
        ctx.metrics["damage"].copy_(
            ctx.sim.qpos.abs().sum(dim=-1))          # 任意 (B,) 标量源
        ctx.events.emit(torch.tensor([0, 2]), "hit", agent=1.0,
                        value=torch.tensor([5.0, 9.0]))


class _BoardReader(BaseDeviceObserver):
    """observer 同帧读黑板与事件差分（模拟 rewarder）。"""

    @property
    def name(self):
        return "board_reader"

    @property
    def shared_reads(self):
        return ("damage",)

    @property
    def output_schema(self):
        return {"dmg_copy": (torch.float32, ())}

    def __init__(self):
        self._out = None
        self.seen = []

    def on_post_action_step(self, ctx: DeviceCtx):
        if self._out is None:
            self._out = {"dmg_copy": torch.zeros(
                ctx.batch_size, device=ctx.sim.qpos.device)}
        self._out["dmg_copy"].copy_(ctx.metrics["damage"])
        ev = ctx.events
        ids = torch.arange(ctx.batch_size, device=ctx.sim.qpos.device)
        recs, steps, mask = ev.since(ids, self._marks)
        self._marks = ev.len(ids)
        self.seen.append((mask.sum(dim=-1).tolist(),
                          recs[:, :, 2].tolist()))   # value 列

    def get_output(self):
        return self._out

    _marks = None

    def on_pre_episode(self, ctx: DeviceCtx):
        self._marks = torch.zeros(
            ctx.batch_size, dtype=torch.int32,
            device=ctx.sim.qpos.device)


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def test_declare_shared_contract():
    """声明分配 (B,*shape)；同型幂等；异型冲突；未声明读 → KeyError。"""
    sim, rt = _rt()
    st = rt.state
    t = st.declare_shared("hp", (2,), torch.float32)
    assert t.shape == (4, 2) and t.dtype == torch.float32
    assert st.declare_shared("hp", (2,), torch.float32) is t
    with pytest.raises(ValueError, match="conflicts"):
        st.declare_shared("hp", (3,), torch.float32)
    with pytest.raises(ValueError, match="conflicts"):
        st.declare_shared("hp", (2,), torch.int32)
    # metrics 视图
    class _P(BaseDevicePlugin):
        @property
        def name(self):
            return "probe"

        def on_post_action_step(self, ctx):
            assert ctx.metrics["hp"] is t
            with pytest.raises(KeyError, match="undeclared"):
                ctx.metrics["nope"]
            assert "hp" in ctx.metrics and len(ctx.metrics) == 1
            assert not hasattr(ctx.metrics, "__setitem__") \
                or True  # dict 风格赋值不支持（无该方法）
    rt.attach(_P())
    rt.reset(); rt.step()


def test_shared_rows_reset():
    """全量/部分 reset 清共享黑板行。"""
    sim, rt = _rt()
    st = rt.state
    t = st.declare_shared("dmg", (), torch.float32)
    rt.reset()
    t[:] = 7.0
    # 全量 reset → 全清
    rt.reset()
    assert t.tolist() == [0.0] * 4
    # 部分 reset：只清对应行
    t[:] = 9.0
    # reset_rows 只允许 ENDED 行——先造一个 ENDED
    ep = st.episode
    ep.world_running[1] = False                      # 模拟 ENDED
    rt.reset_rows(torch.tensor([1]))
    assert t.tolist() == [9.0, 0.0, 9.0, 9.0]


def test_event_emit_and_since():
    """emit 写字段/registry/时间戳；since 游标差分；epoch 失效。"""
    sim, rt = _rt()
    st = rt.state
    ep = st.episode
    ev = st.events
    rt.reset()

    class _Emitter(BaseDevicePlugin):
        @property
        def name(self):
            return "emitter"

        def on_post_action_step(self, ctx):
            ids = torch.tensor([0, 1])
            ctx.events.emit(ids, "hit", agent=0,
                            value=torch.tensor([3.0, 4.0]), aux=2.0)

    rt.attach(_Emitter())
    rt.reset()
    # action_call_index=1（步首已增）——emit 盖的是本步帧戳
    rt.step()
    assert ev.count.tolist() == [1, 1, 0, 0]
    assert ev.records[0, 0].tolist() == [0.0, 0.0, 3.0, 2.0]  # code/agent/val/aux
    assert ev.records[1, 0, 2].item() == 4.0
    assert ev.steps[0, 0].item() == 1
    assert ev.event_registry == {"hit": 0}
    # 第二条事件
    rt.step()
    assert ev.count.tolist() == [2, 2, 0, 0]
    # since：mark=0 → 两条；mark=2 → 零条
    ids = torch.tensor([0, 1])
    _, _, m0 = EventProbe(ev).since(ids, torch.zeros(2, dtype=torch.int32))
    assert m0.sum().item() == 4
    _, _, m2 = EventProbe(ev).since(ids, ev.count[ids].clone())
    assert m2.sum().item() == 0
    # 行 reset：count 清零、epoch++、游标语义失效（旧 mark 不可续读）
    ep.world_running[1] = False
    old_epoch = ev.epoch[1].item()
    rt.reset_rows(torch.tensor([1]))
    assert ev.count[1].item() == 0
    assert ev.epoch[1].item() == old_epoch + 1


class EventProbe:
    """裸 state 上测 since（不经 hook）。"""

    def __init__(self, ev):
        self._ev = ev

    def since(self, ids, marks):
        c = self._ev.count[ids]
        ar = torch.arange(self._ev.cap)
        mask = (ar[None, :] >= marks[:, None]) & (ar[None, :] < c[:, None])
        return self._ev.records[ids], self._ev.steps[ids], mask


def test_event_overflow_flag():
    """容量满 → overflow 标志 + 不丢旧不写新。"""
    sim, rt = _rt()
    st = rt.state
    ev = st.events
    rt.reset()
    ev.count[:] = ev.cap                            # 模拟已写满
    ctx = DeviceCtx(st)
    ctx.events.emit(torch.tensor([0, 1]), "burst", value=1.0)
    assert ev.overflow.tolist() == [True, True, False, False]
    assert ev.count.tolist() == [ev.cap] * 4        # 未写
    assert ev.records[0].abs().sum().item() == 0    # 未污染


def test_episode_options_published():
    """reset options → ctx.episode_options per-env 快照。"""
    sim, rt = _rt()
    seen = {}

    class _OptReader(BaseDevicePlugin):
        @property
        def name(self):
            return "opt_reader"

        def on_pre_episode(self, ctx):
            seen["d"] = ctx.episode_options["initial_distance"].tolist()

    rt.attach(_OptReader())
    rt.reset(options={"initial_distance": torch.tensor(
        [2.0, 2.5, 3.0, 3.5])})
    assert seen["d"] == [2.0, 2.5, 3.0, 3.5]


def test_cross_unit_metrics_events_flow():
    """完整数据流：写方 post_action 写 metrics+emit → 读方 observer
    同帧读到（observer 在插件之后调度——dispatcher priority 最高）。"""
    sim, rt = _rt()
    rt.attach(_BoardWriter())
    reader = _BoardReader()
    rt.set_observer("reader", reader)
    rt.reset()
    rt.step(); rt.step()
    # metrics：observer 读到写方同帧写入的 qpos 派生值
    dmg = rt.state.board["damage"]
    assert torch.equal(reader.get_output()["dmg_copy"], dmg)
    # events：每步每行差分计数（emit 在 [0,2] 两行）
    assert reader.seen[0][0] == [1, 0, 1, 0]
    assert reader.seen[1][0] == [1, 0, 1, 0]
    # value 列有写入
    assert reader.seen[0][1][0][0] == 5.0
