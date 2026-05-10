#!/usr/bin/env python3
"""Lightweight log-tail monitor for ``curriculum.py`` runs.

Parses the training log produced by ``baseline/humanoid21/curriculum.py``
(one structured line per update) and prints a health summary:

  * Stage progression (1 -> 2 -> 3) and dwell counters.
  * Rolling mean over the last N updates of:
      - mean episode length / max_steps
      - imbalance termination rate
      - in-range fraction
      - per-component reward sums (r1 / r2 / r3)
      - policy KL (PPO health)
  * Health verdicts (PASS / WARN / FAIL) for:
      - "is the trainer alive?"           (recent line in last 5 min)
      - "is balance progressing?"         (term_rate trending down)
      - "is approach progressing?"        (in_range trending up once stage>=2)
      - "is PPO stable?"                  (KL not blowing up)
  * Best eval (length, reward) seen so far.

Usage:
    python3 baseline/humanoid21/curriculum_monitor.py [LOG_PATH] [-w WINDOW]

If LOG_PATH is omitted, the monitor picks the newest ``curriculum_*.log``
file under ``baseline/humanoid21/logs/``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional


HUMANOID_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = HUMANOID_DIR / "logs"


# Capture update lines like:
# update=  17 target=robot_a stage=1 weights=(1.0, 0.0, 0.0) reward=-0.7 len= 88.50 term=0.625 in_range=0.412 final_in_zone=0.000 r1=-0.7 r2=-12.5 r3=+0.0 term_pen=-0.6 policy_loss=+0.012 value_loss=+0.04 kl=0.083 gate_reason='...' | eval_target=robot_a eval_reward=-0.6 eval_length=110.0 eval_in_range=0.0 eval_final_in_zone=0.0  [new_best]
_UPDATE_RE = re.compile(
    r"update=\s*(?P<update>\d+)\s+"
    r"(?:target=(?P<target>\w+)\s+)?"
    r"stage=(?P<stage>\d+)\s+"
    r"weights=\((?P<w1>[-+0-9.]+),\s*(?P<w2>[-+0-9.]+),\s*(?P<w3>[-+0-9.]+)\)\s+"
    r"reward=(?P<reward>[-+0-9.eE]+)\s+"
    r"len=\s*(?P<length>[-+0-9.eE]+)\s+"
    r"term=(?P<term>[-+0-9.eE]+)\s+"
    r"in_range=(?P<in_range>[-+0-9.eE]+)\s+"
    r"(?:final_in_zone=(?P<final_in_zone>[-+0-9.eE]+)\s+)?"
    r"r1=(?P<r1>[-+0-9.eE]+)\s+"
    r"r2=(?P<r2>[-+0-9.eE]+)\s+"
    r"r3=(?P<r3>[-+0-9.eE]+)\s+"
    r"(?:term_pen=(?P<term_pen>[-+0-9.eE]+)\s+)?"
    r"policy_loss=(?P<policy_loss>[-+0-9.eE]+)\s+"
    r"value_loss=(?P<value_loss>[-+0-9.eE]+)\s+"
    r"kl=(?P<kl>[-+0-9.eE]+)\s+"
    r"gate_reason=(?P<gate_reason>'[^']*'|\"[^\"]*\")"
    r"(?:.*?eval_reward=(?P<eval_reward>[-+0-9.eE]+)\s+eval_length=\s*(?P<eval_length>[-+0-9.eE]+))?"
    r"(?:.*?eval_final_in_zone=(?P<eval_final_in_zone>[-+0-9.eE]+))?"
    r"(?P<new_best>\s*\[new_best\])?"
)


@dataclass
class UpdateRecord:
    update: int
    stage: int
    weights: tuple
    reward: float
    length: float
    term: float
    in_range: float
    final_in_zone: float
    r1: float
    r2: float
    r3: float
    policy_loss: float
    value_loss: float
    kl: float
    gate_reason: str
    target: Optional[str] = None
    eval_reward: Optional[float] = None
    eval_length: Optional[float] = None
    eval_final_in_zone: Optional[float] = None
    new_best: bool = False


@dataclass
class RunSummary:
    log_path: Path
    last_modified_sec_ago: float
    n_records: int = 0
    records: List[UpdateRecord] = field(default_factory=list)
    resume_line: Optional[str] = None
    last_lines: List[str] = field(default_factory=list)


def parse_log(log_path: Path, *, tail_lines: int = 5) -> RunSummary:
    summary = RunSummary(
        log_path=log_path,
        last_modified_sec_ago=time.time() - log_path.stat().st_mtime,
    )
    last_lines: Deque[str] = deque(maxlen=tail_lines)
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            last_lines.append(line)
            if line.startswith("[resume]") and "loaded actor" in line:
                summary.resume_line = line
            m = _UPDATE_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            summary.records.append(UpdateRecord(
                update=int(d["update"]),
                stage=int(d["stage"]),
                weights=(float(d["w1"]), float(d["w2"]), float(d["w3"])),
                reward=float(d["reward"]),
                length=float(d["length"]),
                term=float(d["term"]),
                in_range=float(d["in_range"]),
                final_in_zone=float(d["final_in_zone"]) if d.get("final_in_zone") else 0.0,
                r1=float(d["r1"]), r2=float(d["r2"]), r3=float(d["r3"]),
                policy_loss=float(d["policy_loss"]),
                value_loss=float(d["value_loss"]),
                kl=float(d["kl"]),
                gate_reason=d["gate_reason"].strip("\"'"),
                target=d.get("target"),
                eval_reward=float(d["eval_reward"]) if d.get("eval_reward") else None,
                eval_length=float(d["eval_length"]) if d.get("eval_length") else None,
                eval_final_in_zone=(
                    float(d["eval_final_in_zone"]) if d.get("eval_final_in_zone") else None
                ),
                new_best=bool(d.get("new_best")),
            ))
    summary.n_records = len(summary.records)
    summary.last_lines = list(last_lines)
    return summary


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _trend(xs: List[float]) -> float:
    """Crude linear trend: (last quartile mean) - (first quartile mean)."""
    if len(xs) < 4:
        return 0.0
    q = max(1, len(xs) // 4)
    return _mean(xs[-q:]) - _mean(xs[:q])


def render_report(summary: RunSummary, *, window: int) -> str:
    out: List[str] = []
    out.append(f"# Curriculum monitor — {summary.log_path}")
    out.append(
        f"  log mtime: {summary.last_modified_sec_ago:.0f}s ago "
        f"({'ALIVE' if summary.last_modified_sec_ago < 300 else 'STALE>5min'})"
    )
    if summary.resume_line:
        out.append(f"  {summary.resume_line}")
    out.append(f"  parsed update records: {summary.n_records}")
    if summary.n_records == 0:
        out.append("  (no update lines parsed yet — trainer is still warming up "
                   "or the log format changed)")
        out.append("")
        out.append("Last raw log lines:")
        for ln in summary.last_lines:
            out.append(f"  | {ln}")
        return "\n".join(out)

    recent = summary.records[-window:]
    rec_lengths = [r.length for r in recent]
    rec_terms = [r.term for r in recent]
    rec_in_range = [r.in_range for r in recent]
    rec_r1 = [r.r1 for r in recent]
    rec_r2 = [r.r2 for r in recent]
    rec_r3 = [r.r3 for r in recent]
    rec_kls = [r.kl for r in recent]

    last = summary.records[-1]
    first = summary.records[0]

    out.append("")
    out.append(f"## Last update: {last.update}")
    out.append(
        f"  stage={last.stage} weights={last.weights}  "
        f"reward={last.reward:+.4f}  len={last.length:.1f}  "
        f"term={last.term:.3f}  in_range={last.in_range:.3f}  "
        f"kl={last.kl:.4f}  gate_reason={last.gate_reason!r}"
    )

    out.append("")
    out.append(f"## Rolling means over last {len(recent)} updates")
    rec_final_in_zone = [r.final_in_zone for r in recent]
    out.append(
        f"  mean_length={_mean(rec_lengths):.2f}  "
        f"term_rate={_mean(rec_terms):.3f}  "
        f"in_range={_mean(rec_in_range):.3f}  "
        f"final_in_zone={_mean(rec_final_in_zone):.3f}"
    )
    out.append(
        f"  r1={_mean(rec_r1):+.3f}  r2={_mean(rec_r2):+.3f}  "
        f"r3={_mean(rec_r3):+.3f}  kl={_mean(rec_kls):.4f}"
    )

    out.append("")
    out.append("## Trends (last quartile - first quartile of recent window)")
    out.append(
        f"  Δlength={_trend(rec_lengths):+.2f}  "
        f"Δterm={_trend(rec_terms):+.3f}  "
        f"Δin_range={_trend(rec_in_range):+.3f}  "
        f"Δr1={_trend(rec_r1):+.3f}  Δr2={_trend(rec_r2):+.3f}  "
        f"Δr3={_trend(rec_r3):+.3f}"
    )

    # Stage history
    stage_changes = []
    prev = None
    for r in summary.records:
        if r.stage != prev:
            stage_changes.append((r.update, r.stage))
            prev = r.stage
    out.append("")
    out.append(f"## Stage history: {' -> '.join(f'@{u}:s{s}' for u, s in stage_changes)}")

    # Best eval
    eval_records = [r for r in summary.records if r.eval_length is not None]
    if eval_records:
        best_by_len = max(eval_records, key=lambda r: r.eval_length)
        out.append("")
        out.append(
            f"## Best eval so far: update={best_by_len.update} "
            f"stage={best_by_len.stage} eval_length={best_by_len.eval_length:.1f} "
            f"eval_reward={best_by_len.eval_reward:+.4f}"
        )
        n_best = sum(1 for r in summary.records if r.new_best)
        out.append(f"  total [new_best] saves: {n_best}")

    # Health verdicts
    out.append("")
    out.append("## Health verdicts")
    verdicts: List[tuple] = []

    # Alive
    alive = summary.last_modified_sec_ago < 300
    verdicts.append((
        "alive",
        "PASS" if alive else "FAIL",
        f"log mtime {summary.last_modified_sec_ago:.0f}s ago",
    ))

    # Balance learning trend (mean_length up OR term down)
    if len(summary.records) >= 8:
        len_trend = _trend([r.length for r in summary.records])
        term_trend = _trend([r.term for r in summary.records])
        improving = len_trend > 5.0 or term_trend < -0.05
        verdicts.append((
            "balance_progress",
            "PASS" if improving else "WARN",
            f"Δlength_overall={len_trend:+.1f}  Δterm_overall={term_trend:+.3f}",
        ))
    else:
        verdicts.append(("balance_progress", "INIT", f"only {len(summary.records)} updates so far"))

    # PPO stability
    kl_max = max(rec_kls) if rec_kls else 0.0
    kl_status = "PASS" if kl_max < 0.5 else ("WARN" if kl_max < 1.0 else "FAIL")
    verdicts.append((
        "ppo_stable",
        kl_status,
        f"max KL in last window={kl_max:.3f} (target_kl=0.05; >0.5 unhealthy)",
    ))

    # Stage advancement
    has_advanced = any(r.stage > 1 for r in summary.records)
    if has_advanced:
        verdicts.append(("stage_advance", "PASS", "stage>=2 reached at least once"))
    elif len(summary.records) > 100:
        verdicts.append(("stage_advance", "WARN",
                         f"still stage 1 after {len(summary.records)} updates"))
    else:
        verdicts.append(("stage_advance", "INIT",
                         f"stage 1 (only {len(summary.records)} updates so far)"))

    for name, status, detail in verdicts:
        out.append(f"  [{status:<4}] {name:<20} — {detail}")

    out.append("")
    out.append("Last raw log lines:")
    for ln in summary.last_lines:
        out.append(f"  | {ln}")
    return "\n".join(out)


def auto_pick_log() -> Path:
    candidates = sorted(
        DEFAULT_LOG_DIR.glob("curriculum_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No curriculum_*.log files found under {DEFAULT_LOG_DIR}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_path", nargs="?", default=None,
                        help="Path to the curriculum training log. Defaults to "
                             "the newest one under baseline/humanoid21/logs/")
    parser.add_argument("-w", "--window", type=int, default=20,
                        help="Rolling-mean window size (in update records)")
    args = parser.parse_args()

    log_path = Path(args.log_path) if args.log_path else auto_pick_log()
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(2)
    summary = parse_log(log_path)
    print(render_report(summary, window=args.window))


if __name__ == "__main__":
    main()
