"""M6 pilot 对照工具：从各 run 的 train.log 抽取 per-update 指标对齐比较。

用法：
    PYTHONPATH=. python3 envs/batchframework/m6_compare.py \
        m6_pilot_cpu_s42 m6_pilot_cpu_s43 m6_pilot_dev_s42

输出两段：
  1. eval 曲线（[eval N] 行：max_pot/final_pot/max_stage/success）
  2. per-update 关键 stats（reward_mean / kl_mean / ev / uncertainty /
     final_potential_mean），按 update 对齐输出成表
"""
import json
import re
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "baseline" / "runs"

EVAL_RE = re.compile(
    r"\[eval\s+(\d+)\] max_pot=([\d.]+) final_pot=([\d.]+) "
    r"max_stage=([\d.]+) max_h=([\d.]+) success=([\d.]+)")


def parse_run(name):
    log = RUNS / name / "train.log"
    evals, stats = {}, {}
    for line in log.read_text().splitlines():
        m = EVAL_RE.search(line)
        if m:
            u = int(m.group(1))
            evals[u] = dict(max_pot=float(m.group(2)), final_pot=float(m.group(3)),
                            max_stage=float(m.group(4)), max_h=float(m.group(5)),
                            success=float(m.group(6)))
            continue
        if "__RAW_STATS__" in line:
            d = json.loads(line.split("__RAW_STATS__", 1)[1])
            u = d["update"]
            ch = d["buffer_stats"]["per_channel"]["r_potential"]
            stats[u] = dict(
                reward_mean=ch["reward_mean"], reward_std=ch["reward_std"],
                kl=d["stats"]["kl_mean"],
                ev=d["stats"].get("ev_r_potential"),
                uncertainty=d["stats"]["uncertainty"],
                std=d["stats"]["std_mean"],
                final_pot=d["experiment"].get("final_potential_mean"),
                rollout_s=d.get("timing", {}).get("rollout"),
            )
    return evals, stats


def main(names):
    runs = {n: parse_run(n) for n in names}
    print("=== eval 曲线（update: max_pot / final_pot / stage / success）===")
    all_eval_u = sorted({u for e, _ in runs.values() for u in e})
    for u in all_eval_u:
        row = [f"u{u:<4}"]
        for n in names:
            e = runs[n][0].get(u)
            row.append("%-40s" % ("%.3f/%.3f/s%.1f/suc%.3f" % (
                e["max_pot"], e["final_pot"], e["max_stage"], e["success"])
                if e else "-"))
        print("  ".join(row))
    print("\n=== per-update stats（update: reward_mean / kl / ev / final_pot_mean）===")
    all_u = sorted({u for _, s in runs.values() for u in s})
    for u in all_u:
        row = [f"u{u:<4}"]
        for n in names:
            s = runs[n][1].get(u)
            row.append("%-44s" % ("%.5f/%.4f/%s/%.4f" % (
                s["reward_mean"], s["kl"],
                "%.2f" % s["ev"] if s["ev"] is not None else "  -",
                s["final_pot"]) if s else "-"))
        print("  ".join(row))


if __name__ == "__main__":
    main(sys.argv[1:] or ["m6_pilot_cpu_s42", "m6_pilot_cpu_s43", "m6_pilot_dev_s42"])
