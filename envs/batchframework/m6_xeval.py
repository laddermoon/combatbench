"""M6 — pilot 终版策略的后端交叉评估。

在 M4 T4 的冻结初态协议上，把三个 pilot run 的 checkpoint 分别放到
CPU EnvRuntime 与 warp BatchRuntime 里各评估一遍：

- dev 策略 → CPU env：主验收。设备训出的策略必须在参考物理上同样站立，
  不能依赖 warp 特有的数值行为。
- cpu 策略 → warp env：反向检查。CPU 参考策略在 warp 物理上保持能力，
  排除「该策略只在原生后端成立」。
- 同策略 → 同后端的两个数字应与该 run 自身 eval 序列末段同量级。

用法::

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 -B envs/batchframework/m6_xeval.py \
        [--n-frozen 64] [--out envs/batchframework/m6_xeval_results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from envs.batchframework.probe_standup_xeval import (  # noqa: E402
    eval_cpu_collect, eval_warp, gen_frozen_states)

RUNS = ROOT / "baseline/runs"
CKPTS = {
    "cpu_s42@500": RUNS / "m6_pilot_cpu_s42/checkpoints/checkpoint_u00500.pt",
    "cpu_s43@500": RUNS / "m6_pilot_cpu_s43/checkpoints/checkpoint_u00500.pt",
    "dev_s42@475": RUNS / "m6_pilot_dev_s42b/checkpoints/checkpoint_u00475.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frozen", type=int, default=64)
    ap.add_argument("--warp-b", type=int, default=64)
    ap.add_argument("--out", type=str,
                    default="envs/batchframework/m6_xeval_results.json")
    args = ap.parse_args()

    t0 = time.time()
    frozen = gen_frozen_states(args.n_frozen)
    print(f"[gen] {len(frozen)} frozen states in {time.time()-t0:.0f}s",
          flush=True)

    results = {}
    for tag, path in CKPTS.items():
        if not path.exists():
            print(f"[skip] {path} missing", flush=True)
            continue
        results[tag] = {}
        t = time.time()
        r = eval_cpu_collect(path, frozen)
        r.pop("_series")
        results[tag]["cpu_eval"] = r
        print(f"[cpu ] {tag}: {r} ({time.time()-t:.0f}s)", flush=True)
        t = time.time()
        r = eval_warp(path, frozen, args.warp_b)
        r.pop("_series")
        results[tag][f"warp_B{args.warp_b}"] = r
        print(f"[warp] {tag} B={args.warp_b}: {r} ({time.time()-t:.0f}s)",
              flush=True)

    out = ROOT / args.out
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
