"""ef_program — 声明式 explore_factor 程序的端到端契约。

覆盖：
- obs_threshold evaluator：numpy 行 / torch 批量双后端、全部比较 op；
- 未知 kind / 畸形 spec 显式拒绝；
- dump 源码发射（ef_program_to_source）与 eval 等价；
- CPU 路径：SamplingPolicy dict spec → 逐帧 ef == 等价 callable；
- device 路径：_EfSchedule 行级混合（program+scalar 共存）、
  run_wave 逐帧求值 + store.ef 记录与观测阈值一致。
"""
import tempfile

import numpy as np
import pytest
import torch

from baseline.framework.rollout.ef_programs import (
    ef_program_to_source, eval_ef_program)
from baseline.framework.rollout.job import SamplingSpec
from baseline.framework.rollout.exploratory_policy import SamplingPolicy

PROG = {"kind": "obs_threshold", "index": 0, "threshold": 0.5,
        "op": "ge", "then": 2.0, "else": -1.0}


# ---------------------------------------------------------------------------
# evaluator 单元
# ---------------------------------------------------------------------------
def test_obs_threshold_numpy_row():
    lo, hi = np.zeros(8), np.zeros(8)
    lo[0], hi[0] = 0.1, 0.9
    assert eval_ef_program(PROG, lo) == pytest.approx(-1.0)
    assert eval_ef_program(PROG, hi) == pytest.approx(2.0)


def test_obs_threshold_torch_batch():
    obs = torch.tensor([[0.1], [0.5], [0.9], [0.0]])
    out = eval_ef_program(PROG, obs)
    assert out.shape == (4,)
    assert out.tolist() == pytest.approx([-1.0, 2.0, 2.0, -1.0])


@pytest.mark.parametrize("op,col,expect", [
    ("ge", 0.5, 2.0), ("gt", 0.5, -1.0), ("lt", 0.5, -1.0),
    ("le", 0.5, 2.0), ("eq", 0.5, 2.0), ("ne", 0.5, -1.0),
])
def test_obs_threshold_ops(op, col, expect):
    spec = dict(PROG, op=op)
    assert eval_ef_program(spec, np.array([col])) == pytest.approx(
        expect)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown ef program kind"):
        eval_ef_program({"kind": "magic"}, np.zeros(4))


def test_malformed_spec_rejected():
    with pytest.raises(ValueError, match="malformed"):
        eval_ef_program({"kind": "obs_threshold", "index": 0},
                        np.zeros(4))


def test_source_emitter_equivalent():
    src = ef_program_to_source(PROG)
    ns = {}
    exec(src, ns)
    fn = ns["_explore_factor"]
    obs = np.zeros(8)
    assert fn(np.array([0.1]), 0) == pytest.approx(-1.0)
    assert fn(np.array([0.9]), 0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# CPU 路径：SamplingPolicy dict spec ≈ 等价 callable
# ---------------------------------------------------------------------------
class _StubPolicy:
    """记录每次 sample 收到的 ctx.explore_factor。"""

    def __init__(self):
        self.seen_ef = []

    def sample(self, observation, *, ctx=None, want_extra=False):
        self.seen_ef.append(float(ctx.explore_factor))
        return np.zeros(4), {"ok": 1.0}

    def act(self, observation, *, want_extra=False):
        return np.zeros(4), None


def test_cpu_dict_ef_matches_callable():
    callable_ef = lambda obs, step: 2.0 if float(obs[0]) >= 0.5 else -1.0
    seq = [np.array([0.1]), np.array([0.9]), np.array([0.5])]
    inner_a, inner_b = _StubPolicy(), _StubPolicy()
    pa = SamplingPolicy(inner_a, SamplingSpec(explore_factor=callable_ef))
    pb = SamplingPolicy(inner_b, SamplingSpec(explore_factor=dict(PROG)))
    for obs in seq:
        pa.act(obs)
        pb.act(obs)
    assert inner_b.seen_ef == inner_a.seen_ef == [-1.0, 2.0, 2.0]


# ---------------------------------------------------------------------------
# device 路径：_EfSchedule + run_wave 逐帧求值 + 记录
# ---------------------------------------------------------------------------
def test_ef_schedule_mixed_rows():
    from envs.batchframework.device_rollouter import _EfSchedule
    # 行 0/2：程序；行 1/3：标量 7.0 / 标量 0.0
    sched = _EfSchedule([dict(PROG), 7.0, dict(PROG), 0.0],
                        batch_size=4, device=torch.device("cpu"))
    obs = torch.tensor([[0.9], [0.9], [0.1], [0.9]])
    out = sched.eval(obs)
    assert out.tolist() == pytest.approx([2.0, 7.0, -1.0, 0.0])


def _make_mini_wave():
    """FakeBackend 上的最小 wave：obs[:,0] = qpos[:,0]，Bump 插件
    每步把它置为步号，形成跨阈值序列。"""
    from envs.batchframework.device_runtime import (
        BatchRuntime, DeviceTimeoutPlugin)
    from envs.batchframework.binding_registry import IoSchema
    from envs.batchframework.fake_backend import FakeBatchBackend
    from envs.batchframework.record_store import RecordStore
    from envs.batchframework.device_plugin import BaseDevicePlugin
    from envs.batchframework.device_rollouter import _RecorderAdapter

    B_, T_, OD, AD = 4, 10, 8, 4
    AGENTS = ("robot_a", "robot_b")

    class Obs:
        def obs_dim(self):
            return OD

        def build(self, st):
            q = st.sim.qpos
            st.io.obs_a.copy_(torch.cat([q, q[:, :1].abs()], -1))
            st.io.obs_b.copy_(torch.cat([q, q[:, :1].abs()], -1))

    class Bump(BaseDevicePlugin):
        name = "bump"

        def on_post_action_step(self, ctx):
            ctx.state.sim.qpos[:, 0] = \
                ctx.state.episode.episode_steps.to(torch.float32)

    io = IoSchema(agent_ids=AGENTS, obs_dim=OD, action_dim=AD)
    sim = FakeBatchBackend(batch_size=B_, device="cpu")
    rt = BatchRuntime(sim, obs_builder=Obs(), phy_substeps=3)
    rt.attach(Bump())
    rt.attach(DeviceTimeoutPlugin(T_))
    store = RecordStore(io, T_, B_, torch.device("cpu"))
    rec = _RecorderAdapter([], store)
    rec.set_runtime(rt)
    rt.attach(rec)
    return rt, store, rec, B_, T_


def test_wave_per_frame_ef_recorded():
    from envs.batchframework.device_rollouter import _EfSchedule
    from envs.batchframework.wave_runner import run_wave
    rt, store, rec, B_, T_ = _make_mini_wave()
    rt.reset()
    rt.obs_builder.build(rt.state)
    rec.begin_wave()
    sched = _EfSchedule([dict(PROG)] * B_, B_, torch.device("cpu"))
    from envs.batchframework.policy_executor import (
        PolicyCapabilities, PolicyExecutor)

    class ConstExec(PolicyExecutor):
        capabilities = PolicyCapabilities(
            stochastic=True, deterministic=True,
            ctx_fields=frozenset({"explore_factor"}), stateful=False)
        version = "t"

        def act(self, obs_a, obs_b, *, stochastic, ctx_a, ctx_b, ctx_ab,
                shared, u_a=None, u_b=None, u_ab=None):
            a = torch.zeros(obs_a.shape[0], 4)
            b = torch.zeros(obs_b.shape[0], 4)
            if not stochastic:
                return a, b, None, None
            return a, b, -obs_a.sum(-1), -obs_b.sum(-1)

    run_wave(rt, ConstExec(), ConstExec(), store, stochastic=True,
             ef_sched_a=sched, ef_sched_b=sched, T=T_)
    # store.ef 逐帧 == 该帧记录 obs[:,0] 的阈值判定
    obs_col = store.obs["robot_a"][:, :, 0]
    expect = (obs_col >= 0.5) * 3.0 - 1.0
    torch.testing.assert_close(store.ef["robot_a"], expect)
    # 波内确实发生过切换（排除常量假阳性）
    assert bool((store.ef["robot_a"] < 0).any()) and \
           bool((store.ef["robot_a"] > 0).any())
