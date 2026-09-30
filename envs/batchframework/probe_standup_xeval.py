"""M4 T4 — standup_floor04 固定策略交叉评估（CPU vs Warp）。

协议（对齐 M4_PLAN T4）：

1. **冻结初态集**：CPU env reset(N 个种子) → fallen 插件跑完 → 取
   core_state 作为冻结初态。两侧都从相同初态起步 ⇒ 物理/策略/奖励
   差异与摔倒分布差异解耦（摔倒分布已在 T3 验收 B 单独验证）。
2. **CPU 侧**：blueprint EnvRuntime，reset 后注入冻结态，确定性策略
   跑满 200 步。
3. **Warp 侧**：BatchRuntime + WarpObsBuilder + DeviceFallenResetPlugin
   + DeviceStandup4StageRewarder×2；同样 reset→注入→200 步。
4. **指标**（与 exp_standup.on_eval 一致）：success = max_pot≥0.9；
   max_pot/final_pot/max_stage/max_h 均值；B ∈ {16,64} 敏感性。

用法::

    PYTHONPATH=. python3 envs/batchframework/probe_standup_xeval.py \
        [--n-frozen 64] [--ckpt RUN:UPDATE]...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RUN_DIR = ROOT / "baseline/runs/train_standup_floor04_ppo_20260920_164819"
CKPTS = {
    "u0001": RUN_DIR / "checkpoints/checkpoint_u00001.pt",
    "u0300": RUN_DIR / "checkpoints/checkpoint_u00300.pt",
    "u1500": RUN_DIR / "checkpoints/checkpoint_u01500.pt",
}
MAX_STEPS = 200
SUCCESS_TH = 0.9


def load_policy(ckpt_path, device="cpu"):
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol = TruncatedNormalPolicy(obs_dim=96, action_dim=21, hidden_dim=256,
                                device=device)
    pol.load_state_dict(ck["actor_state_dict"])
    pol.eval()
    return pol


def build_cpu_runtime():
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    pb = ParameterizedEnvBlueprint.load(
        ROOT / "baseline/humanoid21/blueprints"
        / "standup_4stage_dense_v2_env.yaml")
    bp = pb.materialize(max_steps=MAX_STEPS)
    return bp.build()


def gen_frozen_states(n: int):
    """CPU env 产出的冻结摔倒初态集。"""
    rt = build_cpu_runtime()
    states = []
    for i in range(n):
        rt.reset(seed=10_000 + i)
        states.append(rt.simulator.get_core_state())
    return states


def _empty_metrics():
    return dict(max_pot=[], final_pot=[], max_stage=[], max_h=[],
                success=0)


def _accumulate(m, pots, stages, hs):
    """pots/stages/hs: (T,) arrays per agent-episode。"""
    mx = float(np.max(pots)) if len(pots) else 0.0
    m["max_pot"].append(mx)
    m["final_pot"].append(float(pots[-1]) if len(pots) else 0.0)
    m["max_stage"].append(float(np.max(stages)) if len(stages) else 0.0)
    m["max_h"].append(float(np.max(hs)) if len(hs) else 0.0)
    m["success"] += int(mx >= SUCCESS_TH)


def eval_cpu_collect(ckpt_path, frozen_states, policy_device="cpu"):
    """逐 episode：reset → 注入冻结态 → 确定性策略 200 步。"""
    rt = build_cpu_runtime()
    pol = load_policy(ckpt_path, policy_device)
    pots = {"a": [], "b": []}
    stages = {"a": [], "b": []}
    hs = {"a": [], "b": []}
    for st in frozen_states:
        pa, pb, sa, sb, ha_, hb_ = [], [], [], [], [], []
        rt.reset(seed=0)
        rt.simulator.set_core_state(st)
        for _ in range(MAX_STEPS):
            obs_a, obs_b = rt.get_observation()
            with torch.no_grad():
                a_a = pol.deterministic_action(torch.as_tensor(
                    obs_a, dtype=torch.float32,
                    device=policy_device)).cpu().numpy()
                a_b = pol.deterministic_action(torch.as_tensor(
                    obs_b, dtype=torch.float32,
                    device=policy_device)).cpu().numpy()
            rt.step(a_a, a_b)
            oa = rt.get_observer_output("standing_balance_a")
            ob = rt.get_observer_output("standing_balance_b")
            pa.append(oa["potential"]); pb.append(ob["potential"])
            sa.append(oa["stage"]); sb.append(ob["stage"])
            ha_.append(oa["h_torso"]); hb_.append(ob["h_torso"])
        pots["a"].append(pa); pots["b"].append(pb)
        stages["a"].append(sa); stages["b"].append(sb)
        hs["a"].append(ha_); hs["b"].append(hb_)
    m = _empty_metrics()
    for aid in ("a", "b"):
        for i in range(len(frozen_states)):
            _accumulate(m, pots[aid][i], stages[aid][i], hs[aid][i])
    n = 2 * len(frozen_states)
    return dict(
        success=m["success"] / n,
        max_pot=float(np.mean(m["max_pot"])),
        final_pot=float(np.mean(m["final_pot"])),
        max_stage=float(np.mean(m["max_stage"])),
        max_h=float(np.mean(m["max_h"])),
        _series=dict(pots=pots, stages=stages, hs=hs),
    )


class _MetricRecorder:
    """on_post_action_step 时快照 observer 输出（在 reset 消费之前）。

    优先级低于 dispatcher(-1 < 1e6)，保证读到的是本步新输出；
    终止 env 在本步的输出仍计入该 episode（与 CPU 侧逐步收集一致）。
    """

    KEYS = ("potential", "stage", "h_torso")

    def __init__(self, observers: dict):
        self.obs = observers            # {"a": unit, "b": unit}
        self.buffers = {a: {k: [] for k in self.KEYS} for a in observers}
        self.rows = None                # 当前 chunk 有效行数

    @property
    def name(self):
        return "metric_recorder"

    @property
    def priority(self):
        return -1                       # dispatcher(1e6) 之后

    def declare_state(self, state):
        pass

    def on_attach(self):
        pass

    def on_detach(self):
        pass

    def set_episode_seeds(self, seeds):
        pass

    def reset_buffers(self):
        """chunk 级清空——由采集端显式调用。

        注意不能挂在 on_pre_episode：步内终止触发的部分 reset 也会走
        该 hook，会把已捕获的本步输出一并清掉。
        """
        for bufs in self.buffers.values():
            for b in bufs.values():
                b.clear()

    def on_post_action_step(self, ctx):
        rows = self.rows if self.rows is not None else ctx.batch_size
        for aid, unit in self.obs.items():
            out = unit.get_output()
            for k in self.KEYS:
                self.buffers[aid][k].append(out[k][:rows].cpu())


def _stack(buffers):
    """buffers[aid][key] list-of-(rows,) → {aid: {key: (rows,T)}}"""
    return {a: {k: torch.stack(v, dim=-1).numpy() for k, v in bufs.items()}
            for a, bufs in buffers.items()}


def eval_warp(ckpt_path, frozen_states, B: int):
    """batch 版：reset → 注入冻结行 → 200 步；B 个 episode 并行。"""
    from envs.batchframework.warp_simulator import WarpHumanoid21Simulator
    from envs.batchframework.device_runtime import (
        BatchRuntime, DeviceTimeoutPlugin)
    from envs.batchframework.device_plugin import BaseDevicePlugin
    from envs.batchframework.device_standup import (
        DeviceFallenResetPlugin, DeviceStandup4StageRewarder)

    sim = WarpHumanoid21Simulator(batch_size=B)
    sim.reset(seeds=np.arange(B, dtype=np.int64))
    rt = BatchRuntime(sim, obs_builder=sim.device_obs_builder(),
                      phy_substeps=25)
    fall = DeviceFallenResetPlugin(
        sim_factory=lambda b: WarpHumanoid21Simulator(batch_size=b),
        target_robots=["robot_a", "robot_b"], max_phy_steps=1000,
        height_threshold=0.3, reset_interval=5).bind_shared_sim(sim)
    rt.attach(fall)
    rt.attach(DeviceTimeoutPlugin(MAX_STEPS))
    obs_a = DeviceStandup4StageRewarder.from_sim(sim, 0)
    obs_b = DeviceStandup4StageRewarder.from_sim(sim, 1)
    rt.set_observer("standing_balance_a", obs_a)
    rt.set_observer("standing_balance_b", obs_b)
    pol = load_policy(ckpt_path, "cuda")

    class _Rec(_MetricRecorder, BaseDevicePlugin):
        def __init__(self):
            _MetricRecorder.__init__(self, {"a": obs_a, "b": obs_b})
    rec = _Rec()
    rt.attach(rec)

    # 冻结态打包：(B,) env 一行的 batched core-state dict
    def inject(states_slice):
        batched = {}
        for rid in ("robot_a", "robot_b"):
            batched[rid] = {k: np.stack([s[rid][k] for s in states_slice])
                            for k in states_slice[0][rid]}
        sim.set_core_state(batched)

    pots = {"a": [], "b": []}
    stages = {"a": [], "b": []}
    hs = {"a": [], "b": []}
    n_chunks = (len(frozen_states) + B - 1) // B
    for c in range(n_chunks):
        chunk = frozen_states[c * B:(c + 1) * B]
        n_rows = len(chunk)
        rt.reset(seeds=torch.arange(n_rows, dtype=torch.int64)
                 + c * 10_000)
        if n_rows < B:
            # 尾部不满批：多余行注入最后一行（指标只统计前 n_rows）
            inject(chunk + [chunk[-1]] * (B - n_rows))
        else:
            inject(chunk)
        rt.obs_builder.build(rt.state)   # obs(t=0) 对应注入态
        rec.rows = n_rows
        rec.reset_buffers()              # 清空上一 chunk 缓冲
        for _ in range(MAX_STEPS):
            oa, ob = rt.state.io.obs_a, rt.state.io.obs_b
            with torch.no_grad():
                a_a = pol.deterministic_action(oa[:n_rows])
                a_b = pol.deterministic_action(ob[:n_rows])
            fa = torch.zeros(B, 21, device="cuda"); fb = fa.clone()
            fa[:n_rows] = a_a; fb[:n_rows] = a_b
            rt.step((fa, fb))
        stacked = _stack(rec.buffers)
        for aid in ("a", "b"):
            pots[aid].extend(stacked[aid]["potential"].tolist())
            stages[aid].extend(stacked[aid]["stage"].tolist())
            hs[aid].extend(stacked[aid]["h_torso"].tolist())

    m = _empty_metrics()
    for aid in ("a", "b"):
        for p, s, h in zip(pots[aid], stages[aid], hs[aid]):
            _accumulate(m, np.asarray(p), np.asarray(s), np.asarray(h))
    n = len(pots["a"]) + len(pots["b"])
    return dict(
        success=m["success"] / n,
        max_pot=float(np.mean(m["max_pot"])),
        final_pot=float(np.mean(m["final_pot"])),
        max_stage=float(np.mean(m["max_stage"])),
        max_h=float(np.mean(m["max_h"])),
        _series=dict(pots=pots, stages=stages, hs=hs),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frozen", type=int, default=64)
    ap.add_argument("--sizes", type=str, default="16,64")
    ap.add_argument("--ckpts", type=str, default="u0001,u0300,u1500")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--warp-only", action="store_true")
    ap.add_argument("--out", type=str,
                    default="envs/batchframework/m4_t4_results.json")
    args = ap.parse_args()

    ckpts = {k: CKPTS[k] for k in args.ckpts.split(",")}
    sizes = [int(x) for x in args.sizes.split(",")]

    t0 = time.time()
    frozen = gen_frozen_states(args.n_frozen)
    print(f"[gen] {len(frozen)} frozen states in {time.time()-t0:.0f}s")

    results = {}
    for tag, path in ckpts.items():
        if not path.exists():
            print(f"[skip] {path} missing"); continue
        results[tag] = {}
        if not args.warp_only:
            t = time.time()
            r = eval_cpu_collect(path, frozen)
            r.pop("_series")
            results[tag]["cpu"] = r
            print(f"[cpu ] {tag}: {r} ({time.time()-t:.0f}s)")
        if not args.cpu_only:
            for B in sizes:
                t = time.time()
                r = eval_warp(path, frozen, B)
                r.pop("_series")
                results[tag][f"warp_B{B}"] = r
                print(f"[warp] {tag} B={B}: {r} ({time.time()-t:.0f}s)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
