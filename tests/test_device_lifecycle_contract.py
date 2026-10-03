"""E2-W6 生命周期契约矩阵——FakeBatchBackend 上的显式语义验证。

对照 E2_PLAN.md W6 清单与 LIFECYCLE_TRACE.md 的 CPU 语义：

- 终止历史：多 reason 保留/去重/保序/已终止后新 reason 仍记/自定义
  reason 字符串经 reason_registry 往返
- sealed-ENDED：ENDED 行物理冻结（qpos 不漂移）、不自动 reset、
  reset_rows 只对 ENDED/FAILED 行开放、RUNNING 行拒绝
- 四 mask 分离：slot_valid(padding) / world_running / agent_done /
  policy_eval_mask（policy vs hold 模式）
- 子步屏障：子步 hook 内提出终止 → 当子步屏障封存，episode_step 不
  自增（CPU：子步内终止该 action step 不计）
- abandon / FAILED 显式通道
- RNG：unit_seed 绑定 seed_offsets 行（分片重排不变形）、counter
  递增换序列、同 salt 冲突拒绝、不同 unit 序列独立
- 空 mask / 空 ids 操作无退化
"""
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent

from envs.batchframework.device_plugin import (  # noqa: E402
    BaseDeviceObserver,
    BaseDevicePlugin,
    DeviceCtx,
)
from envs.batchframework.device_runtime import (  # noqa: E402
    BatchRuntime,
    DeviceTimeoutPlugin,
)
from envs.batchframework.device_state import (  # noqa: E402
    ExecutionPlane,
    RngView,
    TERM_CODES,
)
from envs.batchframework.fake_backend import FakeBatchBackend  # noqa: E402
from envs.batchframework.physics import ContractError  # noqa: E402


def _rt(B=4, **kw):
    sim = FakeBatchBackend(batch_size=B)
    sim.reset()
    return sim, BatchRuntime(sim, phy_substeps=5, **kw)


class _Terminator(BaseDevicePlugin):
    """第 kill_step 步对 kill_env 提出终止（可指定 agent 子集与 reason）。"""

    def __init__(self, kill_step=1, kill_env=0, agent=None,
                 reason="ko", name="term"):
        self._ks, self._ke = kill_step, kill_env
        self._ka, self._reason, self._name = agent, reason, name

    @property
    def name(self):
        return self._name

    def on_post_action_step(self, ctx: DeviceCtx):
        if int(ctx.state.rng.step_counter.item()) == self._ks:
            agents = None if self._ka is None else [self._ka]
            ctx.request_termination(
                torch.tensor([self._ke], device=ctx.state.sim.qpos.device),
                self._reason, agents=agents)


# ---------------------------------------------------------------------------
# 终止历史
# ---------------------------------------------------------------------------
def test_multiple_reasons_preserved_in_order():
    """同 env 同 agent 不同 reason 全部保留，按提出顺序归档。"""
    sim, rt = _rt()

    class TwoReasons(BaseDevicePlugin):
        @property
        def name(self):
            return "two_reasons"

        def on_post_action_step(self, ctx):
            if int(ctx.state.rng.step_counter.item()) == 1:
                ids = torch.tensor([0], device=ctx.state.sim.qpos.device)
                ctx.request_termination(ids, "ko", agents=[0])
                ctx.request_termination(ids, "foul", agents=[0])
                ctx.request_termination(ids, "ko", agents=[0])   # 重复去重

    rt.attach(TwoReasons())
    rt.attach(_Terminator(kill_step=1, kill_env=0, agent=1, name="t1"))
    rt.reset()
    rt.step()
    ep = rt.state.episode
    hist = ep.term_history[0, 0]
    assert ep.term_history_len[0, 0].item() == 2
    assert hist[0, 0].item() == TERM_CODES["ko"]
    assert hist[1, 0].item() == TERM_CODES["foul"]
    assert hist[0, 1].item() == hist[1, 1].item() == 1  # 同一步提议


def test_reason_after_done_still_recorded():
    """agent 已终止后再提出的新 reason 仍记入历史（CPU 帧扫描语义）。"""
    sim, rt = _rt()
    rt.attach(_Terminator(kill_step=1, kill_env=0, agent=0,
                          reason="ko", name="t0"))
    rt.attach(_Terminator(kill_step=2, kill_env=0, agent=0,
                          reason="out_of_bounds", name="t1"))
    rt.attach(_Terminator(kill_step=2, kill_env=0, agent=1,
                          reason="timeout", name="t2"))
    rt.reset()
    rt.step(); rt.step()
    ep = rt.state.episode
    hist = ep.term_history[0, 0]
    assert ep.term_history_len[0, 0].item() == 2
    assert hist[0, 0].item() == TERM_CODES["ko"]
    assert hist[0, 1].item() == 1                        # 首 reason 在 step 1
    assert hist[1, 0].item() == TERM_CODES["out_of_bounds"]
    assert hist[1, 1].item() == 2                        # 次 reason 在 step 2


def test_custom_reason_string_roundtrip():
    """自定义 reason（如 imbalance_robot_a）原始字符串经 registry 往返。"""
    sim, rt = _rt()
    rt.attach(_Terminator(kill_step=1, kill_env=0, agent=0,
                          reason="imbalance_robot_a", name="tc"))
    rt.attach(_Terminator(kill_step=1, kill_env=0, agent=1,
                          reason="imbalance_robot_b", name="td"))
    rt.reset()
    rt.step()
    ep = rt.state.episode
    names = {v: k for k, v in ep.reason_registry.items()}
    c0 = ep.term_history[0, 0, 0, 0].item()
    c1 = ep.term_history[0, 1, 0, 0].item()
    assert names[c0] == "imbalance_robot_a"
    assert names[c1] == "imbalance_robot_b"
    assert c0 >= 6 and c1 >= 6       # 动态 code 不占内置段


# ---------------------------------------------------------------------------
# sealed-ENDED
# ---------------------------------------------------------------------------
def test_ended_row_frozen_no_drift():
    """ENDED 行每步写回封存态：qpos 不再漂移（FakeBackend 每步推进 qpos）。"""
    sim, rt = _rt()
    rt.attach(_Terminator(kill_step=1, kill_env=1, name="tk"))  # 全员终止
    rt.reset()
    rt.step()
    ep = rt.state.episode
    assert ep.world_running.tolist() == [True, False, True, True]
    qpos_end = sim.views()["qpos"][1].clone()
    rt.step(); rt.step()
    assert torch.equal(sim.views()["qpos"][1], qpos_end)  # 冻结不漂移
    assert ep.episode_steps[1].item() == 1                # 停在终止步
    assert ep.episode_steps[0].item() == 3                # 正常行推进


def test_reset_rows_rejects_running():
    """RUNNING 行 reset_rows = 契约错误；ENDED 行可 reset。"""
    sim, rt = _rt()
    rt.attach(_Terminator(kill_step=1, kill_env=1, name="tk"))
    rt.reset(); rt.step()
    with pytest.raises(ContractError):
        rt.reset_rows(torch.tensor([0]))        # RUNNING
    rt.reset_rows(torch.tensor([1]))            # ENDED → OK
    assert rt.state.episode.world_running[1].item() is True
    assert rt.state.episode.episode_steps[1].item() == 0


def test_abandon_then_reuse():
    """abandon 记 'abandoned' 终止并封存；reset_rows 后行可复用。"""
    sim, rt = _rt()
    rt.reset(); rt.step()
    rt.abandon(torch.tensor([2]))
    ep = rt.state.episode
    assert ep.world_running.tolist() == [True, True, False, True]
    assert ep.term_history[2, 0, 0, 0].item() == TERM_CODES["abandoned"]
    assert ep.term_history[2, 1, 0, 0].item() == TERM_CODES["abandoned"]
    rt.reset_rows(torch.tensor([2]))
    assert ep.world_running[2].item() is True
    assert ep.term_history_len[2].tolist() == [0, 0]


def test_mark_failed_is_explicit():
    """FAILED ≠ ENDED：fail_reason 记入，term_history 不受影响。"""
    sim, rt = _rt()
    rt.reset(); rt.step()
    rt.mark_failed(torch.tensor([1]), reason="capacity")
    ep = rt.state.episode
    assert ep.world_failed[1].item() is True
    assert ep.world_running[1].item() is False
    assert ep.fail_reason[1].item() == 0        # FAIL_CODES["capacity"]
    assert ep.term_history_len[1].tolist() == [0, 0]
    assert bool(rt.failed_mask.any()) is True
    # FAILED 行也封存可复用
    rt.reset_rows(torch.tensor([1]))
    assert ep.world_failed[1].item() is False


# ---------------------------------------------------------------------------
# 四 mask 分离 / padding / 空 mask
# ---------------------------------------------------------------------------
def test_slot_valid_padding_rows():
    """padding 行（slot_valid=False）不推进、不结束、不计数。"""
    sim, rt = _rt()
    ep = rt.state.episode
    ep.slot_valid[3] = False
    rt.reset()
    assert ep.world_running[3].item() is False   # reset 不复活 padding
    rt.attach(_Terminator(kill_step=1, kill_env=3, name="tk"))
    rt.step()
    # padding 行即使被提议终止也不进入 ENDED 输出（world_running 本就 0）
    assert ep.terminated_flag[3].item() is False
    assert ep.episode_steps.tolist() == [1, 1, 1, 0]
    # 运行行不受影响
    rt.step()
    assert ep.episode_steps.tolist() == [2, 2, 2, 0]


def test_empty_mask_ops_noop():
    """空 env_ids 的 reset/abandon/mark_failed 全部无操作不报错。"""
    sim, rt = _rt()
    rt.reset()
    empty = torch.empty(0, dtype=torch.long,
                        device=rt.state.sim.qpos.device)
    rt.reset_rows(empty)
    rt.abandon(empty)
    rt.mark_failed(empty)
    assert rt.state.episode.world_running.tolist() == [True] * 4


def test_policy_eval_mask_modes():
    """policy 模式：已终止 agent 仍被采样（CPU 默认）；hold 模式：不采样。"""
    sim, rt = _rt()
    rt.attach(_Terminator(kill_step=1, kill_env=0, agent=0, name="t0"))
    rt.reset()
    rt.step()     # step 内先按全 live 采样，屏障后才 done
    rt.step()     # agent_done 已置位：policy 模式仍采样
    ep = rt.state.episode
    assert ep.policy_eval_mask[0].tolist() == [True, True]
    assert ep.world_running[0].item() is True      # agent1 未终止 → RUNNING

    sim2, rt2 = _rt(post_termination_action="hold")
    rt2.attach(_Terminator(kill_step=1, kill_env=0, agent=0, name="t0"))
    rt2.reset(); rt2.step(); rt2.step()
    ep2 = rt2.state.episode
    assert ep2.policy_eval_mask[0].tolist() == [False, True]


# ---------------------------------------------------------------------------
# 子步屏障（子步 hook 模式）
# ---------------------------------------------------------------------------
def test_substep_termination_barrier():
    """子步 hook 内提议终止 → 当子步屏障封存；episode_steps 无条件 +1
    （含端点语义：终止帧仍是一次进入的 step），physics_steps 只记实际
    执行的 3 个子步。"""
    sim, rt = _rt()

    class MidBlockTerm(BaseDevicePlugin):
        @property
        def name(self):
            return "mid_block"

        def on_post_phy_step(self, ctx):
            if int(ctx.state.rng.step_counter.item()) == 0 \
                    and int(ctx.substep_index[0].item()) == 2:
                ctx.request_termination(
                    torch.tensor([0], device=ctx.state.sim.qpos.device),
                    "ko")

    rt.attach(MidBlockTerm())
    rt.reset()
    rt.step()
    ep = rt.state.episode
    # env0 在第 3 子步后 ENDED——episode_steps 对进入的 step 无条件 +1；
    # physics_steps 记实际执行的 3 个子步（帧物理增量 = 3 > 0 → 该
    # 终止帧是有效 transition）
    assert ep.world_running[0].item() is False
    assert ep.episode_steps.tolist() == [1, 1, 1, 1]
    assert ep.physics_steps[0].item() == 3
    # 提议归档值 = 本步 1-based 帧序号（CPU 记录帧扫描的 episode_step）
    assert ep.term_history[0, 0, 0, 1].item() == 1


# ---------------------------------------------------------------------------
# RNG 服务
# ---------------------------------------------------------------------------
def test_rng_row_shuffle_invariance():
    """unit_seed 绑定 seed_offsets 行内容：重排 offsets → 输出同步重排。"""
    sim, rt = _rt(B=4)
    ep_state = rt.state
    v = RngView(ep_state, salt=0x1234)
    base = ep_state.rng.seed_offsets.clone()
    seeds0 = v.unit_seed()
    # 重排行（模拟分片换槽）
    perm = torch.tensor([2, 0, 3, 1], device=base.device)
    ep_state.rng.seed_offsets.copy_(base[perm])
    seeds1 = v.unit_seed()
    assert torch.equal(seeds1, seeds0[perm])


def test_rng_counter_changes_stream():
    """同一 unit 不同 counter → 不同种子；同 counter → 可复现。"""
    sim, rt = _rt(B=2)
    v = RngView(rt.state, salt=0x99)
    c0 = torch.zeros(2, dtype=torch.int64, device="cpu").to(
        rt.state.rng.seed_offsets.device)
    s_a = v.unit_seed(counter=c0)
    s_b = v.unit_seed(counter=c0 + 1)
    assert not torch.equal(s_a, s_b)
    assert torch.equal(v.unit_seed(counter=c0), s_a)


def test_rng_salt_clash_rejected():
    """两个插件声明同一 salt → 装配拒绝。"""
    sim, rt = _rt()

    class R1(BaseDevicePlugin):
        @property
        def name(self): return "r1"
        @property
        def rng_salt(self): return 0x55

    class R2(BaseDevicePlugin):
        @property
        def name(self): return "r2"
        @property
        def rng_salt(self): return 0x55

    rt.attach(R1())
    with pytest.raises(ValueError, match="salt"):
        rt.attach(R2())


# ---------------------------------------------------------------------------
# 装配校验补充
# ---------------------------------------------------------------------------
def test_duplicate_plugin_name_rejected():
    sim, rt = _rt()
    rt.attach(DeviceTimeoutPlugin(max_steps=3))
    with pytest.raises(ValueError, match="duplicate"):
        rt.attach(DeviceTimeoutPlugin(max_steps=9))


def test_metric_export_collision_rejected():
    sim, rt = _rt()

    class M(BaseDevicePlugin):
        def __init__(self, name):
            self._n = name

        @property
        def name(self): return self._n

        def export_episode_metrics(self, state):
            return {"m": torch.zeros(4)}

    rt.attach(M("m1")); rt.attach(M("m2"))
    with pytest.raises(ValueError, match="m1.*m2|both"):
        rt.export_episode_metrics()


def test_observer_gets_no_mutator():
    """observer 隔离：hook ctx 中 mutator 必须为 None。"""
    sim, rt = _rt()
    seen = {}

    class Obs(BaseDeviceObserver):
        def on_post_action_step(self, ctx):
            seen["mut"] = ctx.mutator

        def get_output(self):
            return {"v": torch.zeros(rt.state.batch_size)}

    rt.set_observer("obs", Obs())
    rt.reset(); rt.step()
    assert seen["mut"] is None
