"""E7-W0 计量基线探针——DeviceRollouter/MultiDeviceRollouter 分项吞吐。

矩阵：``--batch`` ∈ {64,512,2048} × ``--substep-hooks`` ∈ {off,on}
× ``--devices`` ∈ {1,2,8} 卡。固定模型（humanoid21 双机）+ 固定
任务语义（standup_4stage_dense_v2 蓝图）——吞吐只在同语义下可比。

每格输出（``--out`` 写 JSON；终端打摘要）：
  collect_wall    每次 collect 的端到端 host 秒（含 sync 等待）
  timing          reset/policy/step/finalize/d2h/export 分解
                  （host 提交时间视角）
  step_segments   physics_wall / obs_build
  hook_timing     每 hook 累计 host 秒（observer/记录器在
                  on_post_action_step 内，不再细分——其内部成本用
                  无 observer 变体或单独 profile 估）
  barrier_time    _consume_terminations 内部 host 秒
  sync_stats      各 host sync 点调用计数（bool(.any())/D2H）
  env_substeps    Σ各行累计物理子步（world physics，不算 agent 维）
  transitions     Σ各 agent 轨迹帧数（agent transitions，与
                  env_substeps 分列，不混算）

"子步 hook" 格在蓝图上追加 ``SubstepProbePlugin``（device_examples
模板5，空 on_post_phy_step）——增量即子步驱动路径固定开销。

用法::

    CUDA_VISIBLE_DEVICES=0 python3 -B -u \
        envs/batchframework/probe_e7_baseline.py --batch 512
    python3 -B -u envs/batchframework/probe_e7_baseline.py \
        --batch 512 --devices 0,1 --substep-hooks --out e7.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ENV_YAML = (REPO / "baseline" / "humanoid21" / "blueprints"
                    / "standup_4stage_dense_v2_env.yaml")
SUBSTEP_PROBE_CLS = ("envs.batchframework.device_examples"
                     ":SubstepProbePlugin")


def _env_bp(env_yaml: Path, max_steps: int, substep_hooks: bool):
    from envs.framework.blueprint import ClassSpec
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    bp = ParameterizedEnvBlueprint.load(env_yaml).materialize(
        max_steps=max_steps)
    if substep_hooks:
        bp = dataclasses.replace(
            bp, plugins=bp.plugins + (ClassSpec(
                cls=SUBSTEP_PROBE_CLS, config={}),))
    return bp


def _policy_bp(tmpdir: Path):
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    pol = TruncatedNormalPolicy(obs_dim=96, action_dim=21,
                                hidden_dim=256, device="cpu")
    return pol.to_blueprint(str(tmpdir / "policy_export"))


def _jobs(env_bp, policy_bp, n: int):
    from baseline.framework.rollout.job import Job, SamplingSpec
    return [
        Job(policy_a_bp=policy_bp, policy_b_bp=policy_bp, env_bp=env_bp,
            seed=1000 + i,
            episode_options={"initial_distance": 2.0},
            sampling_a=SamplingSpec(0.0), sampling_b=SamplingSpec(0.0),
            stochastic=True)
        for i in range(n)
    ]


def _episode_counts(episodes):
    """(env_substeps, transitions)——两个口径分列，互不混算。"""
    substeps, transitions = 0, 0
    for ep in episodes:
        if ep.physics_steps is not None and len(ep.physics_steps):
            substeps += int(ep.physics_steps[-1])
        else:
            substeps += int(ep.num_frames) * int(
                getattr(ep, "phy_steps_per_action", 25))
        try:
            transitions += int(sum(ep.agent_frame_boundary.values()))
        except Exception:
            transitions += int(ep.num_frames) * 2
    return substeps, transitions


def _reset_counters(dr) -> None:
    """单次 collect 前归零累计器（与 ppo/loop 同模式）。"""
    for k in dr.timing:
        dr.timing[k] = 0.0 if isinstance(dr.timing[k], float) else 0
    for k in dr.sync_stats:
        dr.sync_stats[k] = 0
    rt = getattr(dr, "_rt", None)
    if rt is not None:
        for k in rt.sync_stats:
            rt.sync_stats[k] = 0
        rt.hook_timing.clear()
        rt.hook_plugin_timing.clear()
        rt.dispatcher.observer_timing.clear()
        rt.seg_timing.update({k: 0.0 for k in rt.seg_timing})
        rt.barrier_time = 0.0


def run_cell(args) -> dict:
    env_bp = _env_bp(Path(args.env), args.max_steps, args.substep_hooks)
    devices = [int(d) for d in args.devices.split(",")]
    n_dev = len(devices)
    n_jobs = args.batch
    tmp = Path(tempfile.mkdtemp(prefix="e7_probe_"))
    policy_bp = _policy_bp(tmp)
    jobs = _jobs(env_bp, policy_bp, n_jobs)

    result = {"batch": n_jobs, "devices": devices,
              "substep_hooks": args.substep_hooks,
              "max_steps": args.max_steps,
              "env_yaml": str(Path(args.env).name),
              "collects": []}

    if n_dev == 1:
        from envs.batchframework.device_rollouter import DeviceRollouter
        with DeviceRollouter(batch_size=args.batch,
                             device=f"cuda:{devices[0]}") as dr:
            for rep in range(args.repeats):
                _reset_counters(dr)
                t0 = time.perf_counter()
                eps = dr.collect(jobs)
                wall = time.perf_counter() - t0
                sub, tr = _episode_counts(eps)
                result["collects"].append({
                    "wall": wall, "timing": dict(dr.timing),
                    "report": dict(dr.last_collect_report),
                    "env_substeps": sub, "transitions": tr})
    else:
        from envs.batchframework.multi_rollouter import (
            MultiDeviceRollouter)
        per = max(1, n_jobs // n_dev)
        with MultiDeviceRollouter(
                devices=devices,
                batch_size_per_worker=per) as mr:
            for rep in range(args.repeats):
                t0 = time.perf_counter()
                eps = mr.collect(jobs)
                wall = time.perf_counter() - t0
                sub, tr = _episode_counts(eps)
                # worker 侧计数跨 collect 累计——报告按 collect 数
                # 平均（同格各次 job 相同，均值即单次水平）
                reps = mr.last_collect_report.get(
                    "per_worker_reports", [])
                result["collects"].append({
                    "wall": wall, "timing": dict(mr.timing),
                    "per_worker_reports": reps,
                    "n_collects_so_far": rep + 1,
                    "env_substeps": sub, "transitions": tr})

    # 汇总：预热后中位数
    walls = [c["wall"] for c in result["collects"]]
    timed = walls[1:] if len(walls) > 1 else walls
    result["summary"] = {
        "wall_median": statistics.median(timed),
        "wall_first": walls[0],
        "env_substeps_per_s": (result["collects"][-1]["env_substeps"]
                               / statistics.median(timed)),
        "transitions_per_s": (result["collects"][-1]["transitions"]
                              / statistics.median(timed)),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--devices", type=str, default="0",
                    help="逗号分隔 GPU 索引，如 0 / 0,1 / 0,1,2,3,4,5,6,7")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3,
                    help="含首次预热；>1 时汇总取第 2 次起中位数")
    ap.add_argument("--substep-hooks", action="store_true")
    ap.add_argument("--env", type=str, default=str(DEFAULT_ENV_YAML))
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    res = run_cell(args)
    s = res["summary"]
    tag = f"B={res['batch']} dev={res['devices']} " \
          f"substep_hooks={res['substep_hooks']}"
    print(f"[{tag}] wall_first={s['wall_first']:.2f}s "
          f"median={s['wall_median']:.2f}s  "
          f"env_substeps/s={s['env_substeps_per_s']:.0f}  "
          f"transitions/s={s['transitions_per_s']:.0f}")
    last = res["collects"][-1]
    if "timing" in last:
        print("  timing:", {k: round(v, 3) for k, v in
                           last["timing"].items()})
        rep = last.get("report") or {}
        print("  step_segments:", rep.get("step_segments"))
        print("  barrier_time:", round(rep.get("barrier_time", 0), 3))
        print("  hook_timing:", {k: round(v, 3) for k, v in
                               (rep.get("hook_timing") or {}).items()})
        pt = rep.get("hook_plugin_timing") or {}
        print("  hook_plugin_timing:",
              {k: round(v, 3) for k, v in
               sorted(pt.items(), key=lambda kv: -kv[1])[:8]})
        ot = rep.get("observer_timing") or {}
        print("  observer_timing:",
              {k: round(v, 3) for k, v in
               sorted(ot.items(), key=lambda kv: -kv[1])[:10]})
        print("  sync_stats:", rep.get("sync_stats"))
    else:
        print("  multi-dev wall only; per-worker report 见 --out JSON")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
