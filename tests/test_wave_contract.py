"""E3-W3/W5 波契约测试——FakeBackend 驱动，无 GPU 依赖。

覆盖（E3 放行条件）：
- 混长波：env 在不同 step 终止 → 导出不同 num_frames；
- 早退行封存：ENDED 行物理冻结、不漂移、不串台（行隔离）；
- padding/不满波与全 ENDED 早退；
- frame_valid / env_term_step / term_records 语义；
- 向量化 exporter 与 ``Episode.from_buffer_frames`` golden 路径
  逐字段等价。
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from envs.batchframework.device_plugin import (
    BaseDeviceObserver, BaseDevicePlugin)
from envs.batchframework.device_runtime import (
    BatchRuntime, DeviceTimeoutPlugin)
from envs.batchframework.episode_exporter import export_episode
from envs.batchframework.fake_backend import FakeBatchBackend
from envs.batchframework.binding_registry import IoSchema
from envs.batchframework.policy_executor import (
    PolicyCapabilities, PolicyExecutor)
from envs.batchframework.record_store import ObserverLeaf, RecordStore
from envs.batchframework.wave_runner import run_wave
from baseline.framework.rollout.episode import Episode

B, T, OBS_DIM, ACT_DIM = 4, 10, 8, 4
AGENTS = ("robot_a", "robot_b")
IO = IoSchema(agent_ids=AGENTS, obs_dim=OBS_DIM, action_dim=ACT_DIM)


class _ObsBuilder:
    """obs_a = [qpos, |qpos|0]；obs_b = 2×obs_a——确定性可核对。"""

    def obs_dim(self):
        return OBS_DIM

    def build(self, st):
        q = st.sim.qpos
        st.io.obs_a.copy_(torch.cat([q, q[:, :1].abs()], dim=-1))
        st.io.obs_b.copy_(torch.cat([q * 2.0, q[:, :1].abs()], dim=-1))


class _ConstExec(PolicyExecutor):
    """常量动作 + lp = -obs.sum（可核对的伪 log_prob）。"""

    capabilities = PolicyCapabilities(
        stochastic=True, deterministic=True,
        ctx_fields=frozenset({"explore_factor", "delta_factor"}),
        stateful=False)
    version = "test-exec"

    def act(self, obs_a, obs_b, *, stochastic, ctx_a, ctx_b, ctx_ab,
            shared, u_a=None, u_b=None, u_ab=None):
        n = obs_a.shape[0]
        a = torch.full((n, ACT_DIM), 0.25, device=obs_a.device)
        b = torch.full((n, ACT_DIM), -0.10, device=obs_b.device)
        if not stochastic:
            return a, b, None, None
        return a, b, -obs_a.sum(-1), -obs_b.sum(-1)


class _KoAt(BaseDevicePlugin):
    """{row: episode_step} 到点对全 agent 提 reason 终止。"""

    def __init__(self, table, reason="ko"):
        self._table = dict(table)
        self._reason = reason

    @property
    def name(self):
        return "ko_at"

    def on_post_action_step(self, ctx):
        ep = ctx.episode
        hit = torch.tensor(
            [r for r, k in self._table.items()
             if int(ep.episode_steps[r]) == k
             and bool(ep.world_running[r])],
            dtype=torch.long, device=ep.episode_steps.device)
        if hit.numel():
            ctx.request_termination(hit, self._reason)


class _SumObserver(BaseDeviceObserver):
    """声明 schema 的 (B,) 标量叶 observer。"""

    def __init__(self):
        self._out = None

    @property
    def output_schema(self):
        return {"qsum": (torch.float32, ())}

    def on_post_action_step(self, ctx):
        self._out = ctx.state.sim.qpos.sum(-1)

    def on_pre_episode(self, ctx):
        self._out = ctx.state.sim.qpos.sum(-1)

    def get_output(self):
        return {"qsum": self._out}


def _make_wave(ko_table=None, ko_reason="ko"):
    from envs.batchframework.device_rollouter import _RecorderAdapter
    sim = FakeBatchBackend(batch_size=B, device="cpu")
    rt = BatchRuntime(sim, obs_builder=_ObsBuilder(), phy_substeps=3)
    if ko_table:
        rt.attach(_KoAt(ko_table, ko_reason))
    rt.attach(DeviceTimeoutPlugin(T))
    rt.set_observer("sum", _SumObserver())
    store = RecordStore(IO, T, B, torch.device("cpu"),
                        {"sum": {"qsum": ObserverLeaf(torch.float32, ())}})
    rec = _RecorderAdapter(["sum"], store)
    rec.set_runtime(rt)
    rt.attach(rec)
    return sim, rt, store, rec


def _run(rt, store, rec, T_=T, stochastic=True):
    from baseline.framework.ppo.sampling_context import SamplingContext
    ex = _ConstExec()
    rt.reset()
    rt.obs_builder.build(rt.state)
    rec.begin_wave()
    ctx = SamplingContext(explore_factor=torch.zeros(B))
    run_wave(rt, ex, ex, store, stochastic=stochastic,
             ctx_a=ctx, ctx_b=ctx, ctx_ab=ctx, T=T_)
    return ex


def _np_bufs(store):
    return dict(
        obs={r: store.obs[r].cpu().numpy() for r in AGENTS},
        act={r: store.act[r].cpu().numpy() for r in AGENTS},
        log_prob={r: store.log_prob[r].cpu().numpy() for r in AGENTS},
        final_obs={r: store.final_obs[r].cpu().numpy() for r in AGENTS},
        obs_out={n: {k: v.cpu().numpy() for k, v in fs.items()}
                 for n, fs in store.obs_out.items()},
        env_term=store.env_term_step.cpu().numpy())


def _export(store, row, ep_index=0, stochastic=True):
    """要求调用方先 ``store.finalize`` 一次（重复 finalize 会重复 append
    term_records——与 rollouter 每波一次的使用约定一致）。"""
    return export_episode(
        _np_bufs(store),
        job=SimpleNamespace(seed=100 + row, episode_options={}),
        ep_index=ep_index, row=row, agent_ids=AGENTS,
        ef_pair=((0.0, 0.0), (0.0, 0.0)), stochastic=stochastic,
        env_hash="h", term_records=store.term_records[row],
        metrics_np={}, T=T)


# ---------------------------------------------------------------------------
def test_mixed_length_wave():
    """不同行不同步终止 → 不同 num_frames + 正确 term records。"""
    global rt
    sim, rt, store, rec = _make_wave(ko_table={0: 3, 1: 7})
    _run(rt, store, rec)
    store.finalize(rt.state.episode, rt.state.io)

    assert store.env_term_step.tolist() == [3, 7, T, T]
    for row, t_use in ((0, 3), (1, 7)):
        ep = _export(store, row)
        assert ep.num_frames == t_use
        assert ep.observations["robot_a"].shape == (t_use, OBS_DIM)
        assert ep.actions["robot_b"].shape == (t_use, ACT_DIM)
        assert ep.agent_termination_proposal_records["robot_a"] == (
            ("ko", t_use),)
    ep = _export(store, 2)
    assert ep.num_frames == T
    assert ep.agent_termination_proposal_records["robot_a"] == (
        ("timeout", T),)


def test_ended_row_frozen_and_isolated():
    """ENDED 行物理冻结（sealed write-back），其他行继续推进。"""
    global rt
    sim, rt, store, rec = _make_wave(ko_table={0: 2})
    rt.reset()
    rt.obs_builder.build(rt.state)
    store.begin_wave()
    ex = _ConstExec()
    from baseline.framework.ppo.sampling_context import SamplingContext
    ctx = SamplingContext(explore_factor=torch.zeros(B))
    # 跑到 row0 终止（step 计数到 2 后 ENDED），取其 qpos 快照
    for _ in range(2):
        store.write_inputs(0, rt.state,
                           rt.state.episode.world_running)
        a, b, lp_a, lp_b = ex.act(rt.state.io.obs_a, rt.state.io.obs_b,
                                  stochastic=True, ctx_a=ctx, ctx_b=ctx,
                                  ctx_ab=ctx, shared=True)
        rt.step((a, b))
    assert not bool(rt.state.episode.world_running[0])
    frozen = sim.build_sim_namespace().qpos[0].clone()
    for _ in range(3):
        rt.step((torch.zeros(B, ACT_DIM), torch.zeros(B, ACT_DIM)))
    assert torch.equal(sim.build_sim_namespace().qpos[0], frozen)
    # 其他行生命周期继续推进，ENDED 行计数器冻结
    ep = rt.state.episode
    assert int(ep.episode_steps[0]) == 2
    assert int(ep.episode_steps[1]) == 5


def test_frame_valid_and_custom_reason():
    """frame_valid 标 RUNNING 帧；自定义 reason 串保留。"""
    global rt
    sim, rt, store, rec = _make_wave(ko_table={1: 4},
                                ko_reason="imbalance_robot_a")
    _run(rt, store, rec)
    fv = store.frame_valid.cpu().numpy()
    assert fv.shape == (T, B)
    assert fv[:, 0].all() and fv[:, 2].all() and fv[:, 3].all()
    # row1 在 episode_step=4 终止 → 前 4 步 RUNNING，其后 ENDED
    assert fv[:4, 1].all() and not fv[4:, 1].any()
    store.finalize(rt.state.episode, rt.state.io)
    ep = _export(store, 1)
    assert ep.agent_termination_proposal_records["robot_a"] == (
        ("imbalance_robot_a", 4),)
    assert ep.agent_termination_proposal_records["robot_b"] == (
        ("imbalance_robot_a", 4),)


def test_early_exit_all_ended():
    """全 ENDED 早退不报错；未填帧不参与导出。"""
    global rt
    sim, rt, store, rec = _make_wave(ko_table={0: 1, 1: 1, 2: 1, 3: 1})
    _run(rt, store, rec, T_=T)
    assert not rt.any_running()
    store.finalize(rt.state.episode, rt.state.io)
    for row in range(B):
        ep = _export(store, row)
        assert ep.num_frames == 1
        assert ep.agent_termination_proposal_records["robot_a"] == (
            ("ko", 1),)


def test_final_observation_is_term_obs():
    """final_obs = 终止时刻观测（非波末态）。"""
    global rt
    sim, rt, store, rec = _make_wave(ko_table={2: 5})
    _run(rt, store, rec)
    store.finalize(rt.state.episode, rt.state.io)
    ep = _export(store, 2)
    obs_np = store.obs["robot_a"].cpu().numpy()
    # 末帧观测是 t_use 帧之后构建的 obs —— 与 CPU 语义相同：obs_{t_use}
    # 记录在 obs 缓冲的下一行（波内该行 ENDED 后 obs 不再更新，
    # 故 final_obs 独立捕获）。
    assert np.allclose(ep.final_observation["robot_a"],
                       store.final_obs["robot_a"][row := 2].cpu().numpy())


def test_deterministic_wave_no_extras():
    """stochastic=False：无 log_prob/extras（CPU eval 波语义）。"""
    global rt
    sim, rt, store, rec = _make_wave()
    _run(rt, store, rec, stochastic=False)
    store.finalize(rt.state.episode, rt.state.io)
    ep = _export(store, 0, stochastic=False)
    assert ep.action_extras == {} and ep.explore_factors == {}
    assert ep.num_frames == T


def test_exporter_matches_golden_path():
    """向量化导出 vs Episode.from_buffer_frames 逐字段等价。"""
    global rt
    sim, rt, store, rec = _make_wave(ko_table={0: 6})
    _run(rt, store, rec)
    store.finalize(rt.state.episode, rt.state.io)
    np_bufs = _np_bufs(store)

    job = SimpleNamespace(seed=42, episode_options={"initial_distance": 2.0})
    got = export_episode(
        np_bufs, job=job, ep_index=7, row=0, agent_ids=AGENTS,
        ef_pair=((0.3, 0.0), (0.5, 0.0)), stochastic=True,
        env_hash="hh", term_records=store.term_records[0],
        metrics_np={"m": np.arange(B, dtype=np.float32)}, T=T)

    # golden：用同一批 np 数据走逐帧 dict + from_buffer_frames
    row = 0
    t_use = int(np_bufs["env_term"][row])
    frames = []
    for t in range(t_use):
        frames.append({
            "observation": {r: np_bufs["obs"][r][t, row] for r in AGENTS},
            "action": {r: np_bufs["act"][r][t, row] for r in AGENTS},
            "observer_outputs": {
                n: {k: v[t, row].item() for k, v in fs.items()}
                for n, fs in np_bufs["obs_out"].items()},
            "action_extras": {
                "robot_a": {
                    "log_prob": float(np_bufs["log_prob"]["robot_a"][t, row]),
                    "explore_factor": np.float32(0.3),
                    "sctx__delta_factor": np.float32(0.0)},
                "robot_b": {
                    "log_prob": float(np_bufs["log_prob"]["robot_b"][t, row]),
                    "explore_factor": np.float32(0.5),
                    "sctx__delta_factor": np.float32(0.0)}},
        })
    golden = Episode.from_buffer_frames(
        frames=frames,
        final_observation={r: np_bufs["final_obs"][r][row] for r in AGENTS},
        base_seed=42, episode_index=7, blueprint_hash="hh",
        agent_termination_proposal_records={
            r: tuple(store.term_records[row][r]) for r in AGENTS},
        episode_options=dict(job.episode_options),
        episode_metrics={"backend": "warp-fp32", "m": float(
            np.arange(B, dtype=np.float32)[row])})

    assert got.num_frames == golden.num_frames == t_use
    assert got.base_seed == golden.base_seed
    assert got.episode_options == golden.episode_options
    assert (got.agent_termination_proposal_records
            == golden.agent_termination_proposal_records)
    for r in AGENTS:
        assert np.array_equal(got.observations[r], golden.observations[r])
        assert np.array_equal(got.actions[r], golden.actions[r])
        assert np.array_equal(got.final_observation[r],
                              golden.final_observation[r])
        for k in got.action_extras[r]:
            assert np.allclose(got.action_extras[r][k],
                               golden.action_extras[r][k])
        assert np.array_equal(got.explore_factors[r],
                              golden.explore_factors[r])
        # sampling_contexts 派生视图等价
        for k in golden.sampling_contexts[r]:
            assert np.allclose(got.sampling_contexts[r][k],
                               golden.sampling_contexts[r][k])
    for name in got.observer_outputs:
        for k, leaf in got.observer_outputs[name].items():
            gold_leaf = golden.observer_outputs[name][k]
            assert np.allclose(np.asarray(leaf), np.asarray(gold_leaf))
            # 标量叶 CPU 语义为 list[T]
            assert isinstance(leaf, type(gold_leaf))
    assert got.episode_metrics == golden.episode_metrics


class _UNoiseExec(PolicyExecutor):
    """动作 = 注入 u 映射到 [-1,1]——验证 E6-W1 job-keyed 噪声接线。"""

    capabilities = PolicyCapabilities(
        stochastic=True, deterministic=True,
        ctx_fields=frozenset(), stateful=False)
    version = "u-exec"

    def act(self, obs_a, obs_b, *, stochastic, ctx_a, ctx_b, ctx_ab,
            shared, u_a=None, u_b=None, u_ab=None):
        n = obs_a.shape[0]
        if not stochastic:
            z = torch.zeros(n, ACT_DIM, device=obs_a.device)
            return z, z, None, None
        assert u_a is not None and u_b is not None
        return (u_a * 2 - 1, u_b * 2 - 1,
                torch.zeros(n, device=obs_a.device),
                torch.zeros(n, device=obs_b.device))


def _run_u(rt, rec, store, seeds):
    rt.reset(seeds=torch.as_tensor(seeds, dtype=torch.int64))
    rt.obs_builder.build(rt.state)
    rec.begin_wave()
    run_wave(rt, _UNoiseExec(), _UNoiseExec(), store,
             stochastic=True, T=T)


def test_job_keyed_action_noise():
    """E6-W1：动作噪声 = f(job_seed, episode_step)——同 seed 重跑逐位
    一致、改 seed 变、行位（行号/pad）不影响。"""
    seeds = [10, 11, 12, 13]
    sim, rt, store, rec = _make_wave()
    _run_u(rt, rec, store, seeds)
    act1 = {r: store.act[r].cpu().clone() for r in AGENTS}

    sim2, rt2, store2, rec2 = _make_wave()
    _run_u(rt2, rec2, store2, seeds)
    for r in AGENTS:                                # 同 seed 重跑逐位一致
        torch.testing.assert_close(store2.act[r], act1[r])

    sim3, rt3, store3, rec3 = _make_wave()
    _run_u(rt3, rec3, store3, [s + 100 for s in seeds])
    assert not torch.allclose(store3.act["robot_a"],
                              act1["robot_a"])      # 改 seed → 不同流

    # 行位无关：把 seed 11 放到行 3，其噪声流应与原先行 1 相同
    sim4, rt4, store4, rec4 = _make_wave()
    _run_u(rt4, rec4, store4, [20, 21, 22, 11])
    torch.testing.assert_close(store4.act["robot_a"][:, 3],
                               act1["robot_a"][:, 1])


def test_debug_capture(tmp_path):
    """E6-W2：命中 (job,frame) → npz 快照+帧切片 + manifest provenance。"""
    from envs.batchframework.debug_capture import (
        CaptureRequest, WaveDebugCapture)
    sim, rt, store, rec = _make_wave()
    jobs = [SimpleNamespace(seed=100 + i, episode_options={})
            for i in range(B)]
    cap = WaveDebugCapture(
        CaptureRequest(job_refs=(1, 3), frames=(2,),
                       out_dir=str(tmp_path)), jobs)
    cap.set_wave({r: r for r in range(B)}, store, {"binding": "fake"})
    rt.reset(seeds=torch.arange(B, dtype=torch.int64))
    rt.obs_builder.build(rt.state)
    rec.begin_wave()
    run_wave(rt, _ConstExec(), _ConstExec(), store, stochastic=True,
             ctx_a=None, ctx_b=None, ctx_ab=None, T=T, step_hook=cap)

    assert len(cap.entries) == 2
    for e in cap.entries:
        assert e["wave_step"] == 2 and e["episode_step"] >= 0
        npz = np.load(tmp_path / e["file"])
        assert f"obs/{AGENTS[0]}" in npz and f"act/{AGENTS[1]}" in npz
        assert "obs_out/sum/qsum" in npz
        assert npz["state/qpos"].shape[-1] == store.obs[AGENTS[0]].shape[-1] - 1
    mpath = cap.write_manifest(collect_id=1)
    doc = json.loads(mpath.read_text())
    assert doc["kind"] == "device_capture"
    assert doc["collect_id"] == 1
    assert len(doc["captures"]) == 2
    assert "git_commit" in doc and "versions" in doc
    assert doc["provenance"]["binding"] == "fake"
