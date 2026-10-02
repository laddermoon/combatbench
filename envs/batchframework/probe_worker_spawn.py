"""E4-W0 探针：spawn worker + per-device CUDA init + Episode 传输实测。

回答的问题（discuss.md H12 / E4 计划 J1/J2）：

1. ``mp.spawn`` 子进程内 ``torch.cuda.set_device(k)`` + warp init 是否
   可靠（不设 CUDA_VISIBLE_DEVICES 遮罩，物理卡直选）；
2. Job/EnvBlueprint/PolicyBlueprint 是否可 pickle 跨进程传递；
3. Episode pickle 尺寸/序列化与队列传输耗时（决策：首版 pickle
   直传是否够快，要不要共享内存/npz 通道）；
4. worker 异常退出时 coordinator 侧检测延迟。

用法：

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python3 \
        envs/batchframework/probe_worker_spawn.py [--devices 0,1] \
        [--jobs 8] [--steps 32] [--kill-test]

只探测，不改框架代码。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _make_inputs(max_steps: int):
    """在调用进程内构造 env_bp/policy_bp/jobs（可 pickle 性一并验证）。"""
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    pb = ParameterizedEnvBlueprint.load(
        ROOT / "baseline/humanoid21/blueprints"
        / "standup_4stage_dense_v2_env.yaml")
    env_bp = pb.materialize(max_steps=max_steps)
    tmp = tempfile.mkdtemp(prefix="e4probe_policy_")
    pol = TruncatedNormalPolicy(obs_dim=96, action_dim=21, hidden_dim=256,
                                device="cpu")
    policy_bp = pol.to_blueprint(str(Path(tmp) / "export"))
    return env_bp, policy_bp


def _make_jobs(env_bp, policy_bp, n):
    from baseline.framework.rollout.job import Job, SamplingSpec
    return [
        Job(policy_a_bp=policy_bp, policy_b_bp=policy_bp, env_bp=env_bp,
            seed=1000 + i,
            episode_options={"initial_distance": 2.0},
            sampling_a=SamplingSpec(0.0), sampling_b=SamplingSpec(0.0),
            stochastic=True)
        for i in range(n)]


def _worker_main(device_index: int, in_q, out_q, env_bp, policy_bp,
                 n_jobs: int):
    """worker：先定设备再初始化 CUDA/warp；握手 → collect → 上行结果。"""
    try:
        import torch
        torch.cuda.set_device(device_index)
        from envs.batchframework.device_rollouter import DeviceRollouter
        jobs = _make_jobs(env_bp, policy_bp, n_jobs)
        with DeviceRollouter(batch_size=max(4, n_jobs),
                             device=f"cuda:{device_index}") as dr:
            t0 = time.perf_counter()
            eps = dr.collect(jobs)
            collect_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            payload = pickle.dumps(eps)
            pickle_s = time.perf_counter() - t0
            out_q.put({"kind": "result", "device": device_index,
                       "phys": torch.cuda.get_device_name(device_index),
                       "collect_s": collect_s, "pickle_s": pickle_s,
                       "n_eps": len(eps), "payload_bytes": len(payload),
                       "episodes": payload})
    except Exception:
        out_q.put({"kind": "error", "device": device_index,
                   "tb": traceback.format_exc()})


def _sleep_worker(in_q):
    """故障注入用：挂起等待被 kill。"""
    try:
        in_q.get(timeout=120)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--kill-test", action="store_true")
    args = ap.parse_args()

    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    ctx = mp.get_context("spawn")

    print(f"[probe] devices={devices} jobs/dev={args.jobs} "
          f"steps={args.steps}", flush=True)

    # --- P1: bp/job picklability（不经过 CUDA）---
    env_bp, policy_bp = _make_inputs(args.steps)
    jobs = _make_jobs(env_bp, policy_bp, args.jobs)
    t0 = time.perf_counter()
    blob = pickle.dumps((env_bp, policy_bp, jobs))
    env_bp2, policy_bp2, jobs2 = pickle.loads(blob)
    assert jobs2[0].seed == jobs[0].seed
    assert env_bp2.to_dict() == env_bp.to_dict()
    print(f"[P1] bp/jobs pickle ok: {len(blob)/1e6:.2f} MB, "
          f"{time.perf_counter()-t0:.3f}s roundtrip", flush=True)

    # --- P2: spawn worker + collect + Episode 传输 ---
    procs, ins, outs = [], [], []
    t_start = time.perf_counter()
    for k in devices:
        iq, oq = ctx.Queue(), ctx.Queue()
        p = ctx.Process(target=_worker_main,
                        args=(k, iq, oq, env_bp, policy_bp, args.jobs))
        p.start()
        procs.append(p); ins.append(iq); outs.append(oq)
    results = []
    t_deadline = time.perf_counter() + 600
    for k, oq, p in zip(devices, outs, procs):
        try:
            r = oq.get(timeout=max(1, t_deadline - time.perf_counter()))
        except Exception as e:
            print(f"[P2] device {k}: queue timeout/err {e} "
                  f"(alive={p.is_alive()})", flush=True)
            r = None
        results.append((k, r))
    for p in procs:
        p.join(timeout=30)
    wall = time.perf_counter() - t_start

    ok = 0
    for k, r in results:
        if r is None or r.get("kind") != "result":
            print(f"[P2] device {k}: FAIL {r and r.get('tb', r)}",
                  flush=True)
            continue
        eps = pickle.loads(r["episodes"])
        e0 = eps[0]
        print(f"[P2] cuda:{k} ({r['phys']}): {r['n_eps']} eps, "
              f"collect={r['collect_s']:.2f}s "
              f"pickle={r['pickle_s']*1e3:.1f}ms "
              f"payload={r['payload_bytes']/1e6:.2f}MB "
              f"({r['payload_bytes']/max(1,r['n_eps'])/1e6:.3f}MB/ep); "
              f"ep0 T={e0.num_frames} obs={e0.observations['robot_a'].shape}",
              flush=True)
        ok += 1
    print(f"[P2] {ok}/{len(devices)} workers ok, wall={wall:.1f}s",
          flush=True)

    # --- P3: kill 检测延迟 ---
    if args.kill_test:
        iq, oq = ctx.Queue(), ctx.Queue()
        p = ctx.Process(target=_sleep_worker, args=(iq,))
        p.start()
        time.sleep(1.0)
        p.kill()
        t0 = time.perf_counter()
        p.join(timeout=10)
        print(f"[P3] kill→join 检测 {time.perf_counter()-t0:.3f}s, "
              f"exitcode={p.exitcode}", flush=True)

    return 0 if ok == len(devices) else 1


if __name__ == "__main__":
    sys.exit(main())
