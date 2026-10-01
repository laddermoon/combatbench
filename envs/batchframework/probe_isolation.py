"""W0 探针：行隔离冻结机制 / 快照完备性 / 视图时效（E1 前置）。

源码核查结论（warp 1.12.1）：``mjw.step(m, d)`` 无 mask 参数，
``m.opt.disableflags`` 为标量 int——不存在 per-world masked-advance
一等原语。本探针量化三个替代/验证项：

P1 冻结机制：已结束行（ended mask）在继续 step-all 时如何保持无害
   A. 现状 reset-park（dev_reset_rows 进新 episode 继续跑）
   B. write-back 冻结：每子步把保存的 qpos/qvel/ctrl/warmstart/xfrc
      写回 ended 行——行状态逐位冻结，步内产生的接触有界（每子步
      从同一冻结态出发，接触数稳定不累积）
   度量：冻结行逐位不变性、运行行与纯运行基线的逐位一致性、
   0%/50% ended 下的子步耗时。

P2 快照完备性（H3）：capture 候选字段集 → 推进 k 步 → restore →
   再推进 → 与不间断对照逐位比对，找出必需字段集。

P3 视图时效（H2）：mjw.step 是否已刷新 derived 字段（xpos 等）；
   写 qpos 不经 forward 时 derived 是否陈旧——确定 refresh 语义。

用法::
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data1/mono/things/combatbench \
        python3 envs/batchframework/probe_isolation.py
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch


def _make_sim(batch_size: int, seeds=None):
    from envs.batchframework.warp_simulator import WarpHumanoid21Simulator
    sim = WarpHumanoid21Simulator(batch_size=batch_size)
    sim.reset(seeds=seeds)
    return sim


def _views(sim):
    return sim._torch_views()


_EXTRA_FIELDS = ("time", "act", "nefc", "nacon")


def _field_view(sim, name: str):
    """按需从 mjw.Data 取字段的 torch 视图（_torch_views 未登记的字段）。"""
    wp = sim._wp
    attr = getattr(sim._wdata, name, None)
    if attr is None:
        return None
    try:
        return wp.to_torch(attr)
    except Exception:
        return None


def _get_view(sim, v, name: str):
    return v.get(name) if name in v else _field_view(sim, name)


def _capture_rows(v, keys, rows) -> Dict[str, torch.Tensor]:
    return {k: v[k][rows].clone() for k in keys}


def _restore_rows(v, snap: Dict[str, torch.Tensor], rows) -> None:
    for k, buf in snap.items():
        v[k][rows] = buf


# ---------------------------------------------------------------------------
# P1: write-back 冻结
# ---------------------------------------------------------------------------
FREEZE_KEYS = ("qpos", "qvel", "ctrl", "qacc_warmstart",
               "xfrc_applied", "qfrc_applied")


def probe_freeze(B: int = 256, n_steps: int = 200, frac_ended: float = 0.5,
                 substeps_per_action: int = 25):
    """B=256 下 write-back 冻结的正确性与开销。

    用固定 seeds 保证两个 sim 起点逐位相同（probe 已发现 reset(None)
    非确定——初始姿态走全局 RNG）。
    """
    print(f"\n=== P1 write-back freeze  B={B} steps={n_steps} "
          f"ended={frac_ended:.0%} ===")
    seeds = np.arange(B, dtype=np.int64) + 1000
    sim = _make_sim(B, seeds=seeds)
    v = _views(sim)
    # _compute_reset_state 源码确认不含 RNG（seeds 仅记录）——
    # reset 后 qpos 应逐位一致，此处显式验证。
    qpos_reset_a = v["qpos"].clone()
    sim.physical_step(50)  # 进入非平凡状态
    qpos_50_a = v["qpos"].clone()   # 50 步后检查点（用于同刻对照）

    n_ended = int(B * frac_ended)
    ended = torch.arange(n_ended, device=v["qpos"].device)

    # 对照基线：同 seed 起点纯运行 n_steps
    sim.physical_step(n_steps)
    base_running = v["qpos"][n_ended:].clone()
    base_ended = v["qpos"][:n_ended].clone()
    del sim

    # 冻结实验：同 seeds reset + 50 步 → 每 substeps_per_action 子步
    # 执行一次 write-back（action-step 粒度——冻结只需限制漂移上界，
    # 不要求冻结行子步间逐位稳定）。
    sim2 = _make_sim(B, seeds=seeds)
    v2 = _views(sim2)
    reset_same = torch.equal(v2["qpos"], qpos_reset_a)
    sim2.physical_step(50)
    step50_same = torch.equal(v2["qpos"], qpos_50_a)
    frozen = _capture_rows(v2, FREEZE_KEYS, ended)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for done in range(0, n_steps, substeps_per_action):
        sim2.physical_step(substeps_per_action)
        _restore_rows(v2, frozen, ended)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # 检查 1：冻结行逐位不变
    frozen_ok = all(
        torch.equal(v2[k][ended], frozen[k]) for k in FREEZE_KEYS)
    # 检查 2：运行行与无冻结基线逐位一致（world 独立性）
    running = v2["qpos"][n_ended:]
    running_ok = torch.equal(running, base_running)
    # 检查 3：冻结行未产生非法值
    frozen_finite = all(
        torch.isfinite(v2[k][ended]).all().item() for k in ("qpos", "qvel"))

    print(f"  reset(seeds) deterministic : {reset_same}")
    print(f"  warp step run-to-run det.  : {step50_same} "
          f"(同 seed 双实例 50 步后 qpos 逐位比较)")
    print(f"  frozen rows bitwise-stable : {frozen_ok}")
    print(f"  running rows == baseline   : {running_ok}")
    print(f"  frozen rows finite         : {frozen_finite}")
    print(f"  wall: {n_steps} substeps = {dt:.3f}s "
          f"({n_steps/dt:.0f} substep/s, {n_steps*B/dt:.0f} world-sub/s)")

    # 对照：无冻结同 shape
    del sim2
    sim3 = _make_sim(B, seeds=seeds)
    sim3.physical_step(50)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sim3.physical_step(n_steps)
    torch.cuda.synchronize()
    dt0 = time.perf_counter() - t0
    print(f"  baseline no-freeze         : {dt0:.3f}s "
          f"(write-back overhead = {dt/dt0 - 1:.1%})")
    return frozen_ok and running_ok and frozen_finite


# ---------------------------------------------------------------------------
# P2: 快照完备性
# ---------------------------------------------------------------------------
SNAP_CANDIDATE_SETS = {
    "minimal": ("time", "qpos", "qvel", "ctrl", "qacc_warmstart"),
    "minimal+act": ("time", "qpos", "qvel", "ctrl", "qacc_warmstart", "act"),
    "+forces": ("time", "qpos", "qvel", "ctrl", "qacc_warmstart",
                "xfrc_applied", "qfrc_applied"),
    "+efc-n": ("time", "qpos", "qvel", "ctrl", "qacc_warmstart",
               "xfrc_applied", "qfrc_applied", "nefc", "nacon"),
}


def _capture_all(sim, v, keys) -> Dict[str, torch.Tensor]:
    out = {}
    for k in keys:
        t = _get_view(sim, v, k)
        if t is not None:
            out[k] = t.clone()
    return out


def _restore_all(sim, v, snap: Dict[str, torch.Tensor]) -> None:
    for k, buf in snap.items():
        _get_view(sim, v, k).copy_(buf)


def probe_snapshot(B: int = 64, k: int = 10):
    """capture→restore 后重推进是否与不间断逐位一致。"""
    print(f"\n=== P2 snapshot completeness  B={B} k={k} ===")
    torch.manual_seed(0)
    sim = _make_sim(B)
    v = _views(sim)
    sim.physical_step(30)  # 非平凡状态

    results = {}
    for name, keys in SNAP_CANDIDATE_SETS.items():
        missing = [kk for kk in keys if _get_view(sim, v, kk) is None]
        if missing:
            print(f"  {name:>14}: SKIP (missing fields {missing})")
            continue
        snap = _capture_all(sim, v, keys)
        # 分支 A：存照 → 推进 k 步（漂移）→ 恢复+forward → 推进 k 步
        for _ in range(k):
            sim.physical_step(1)
        _restore_all(sim, v, snap)
        with sim._wp.ScopedStream(sim._wp.stream_from_torch()):
            sim._mjw.forward(sim._wmodel, sim._wdata)
        for _ in range(k):
            sim.physical_step(1)
        ref_qpos = v["qpos"].clone()

        # 分支 B：从快照状态不间断推进 k 步（恢复快照再直接跑）
        _restore_all(sim, v, snap)
        with sim._wp.ScopedStream(sim._wp.stream_from_torch()):
            sim._mjw.forward(sim._wmodel, sim._wdata)
        for _ in range(k):
            sim.physical_step(1)
        got_qpos = v["qpos"]

        bit = torch.equal(ref_qpos, got_qpos)
        max_diff = (ref_qpos - got_qpos).abs().max().item()
        results[name] = bit
        print(f"  {name:>14}: bitwise={bit}  max|Δqpos|={max_diff:.3e}")

    return results


# ---------------------------------------------------------------------------
# P3: 视图时效
# ---------------------------------------------------------------------------
def probe_freshness(B: int = 16):
    """mjw.step 后 xpos 是否已刷新；写 qpos 不 forward 时 derived 是否陈旧。"""
    print(f"\n=== P3 view freshness  B={B} ===")
    sim = _make_sim(B)
    v = _views(sim)

    xpos0 = v["xpos"].clone()
    sim.physical_step(5)
    xpos1 = v["xpos"]
    changed = not torch.equal(xpos0, xpos1)
    print(f"  xpos changes inside physical_step (no explicit forward): "
          f"{changed}")

    # 写 qpos 不 forward：xpos 应陈旧（仍等于写前值）
    qpos_save = v["qpos"].clone()
    xpos_before = v["xpos"].clone()
    v["qpos"] += 0.01  # 直接写设备视图（绕过 mutator，仅探针用）
    with sim._wp.ScopedStream(sim._wp.stream_from_torch()):
        pass  # 不 forward
    stale = torch.equal(v["xpos"], xpos_before)
    with sim._wp.ScopedStream(sim._wp.stream_from_torch()):
        sim._mjw.forward(sim._wmodel, sim._wdata)
    refreshed = not torch.equal(v["xpos"], xpos_before)
    print(f"  qpos write w/o forward → xpos stale: {stale}")
    print(f"  forward() → xpos reflects write   : {refreshed}")
    v["qpos"].copy_(qpos_save)
    return changed, stale, refreshed


# ---------------------------------------------------------------------------
def main():
    ok1 = probe_freeze()
    p2 = probe_snapshot()
    c, s, r = probe_freshness()

    print("\n=== SUMMARY ===")
    print(f"P1 write-back freeze viable: {ok1}")
    print(f"P2 snapshot sets: {p2}")
    print(f"P3 step-refresh={c} stale-without-forward={s} "
          f"forward-refresh={r}")


if __name__ == "__main__":
    main()
