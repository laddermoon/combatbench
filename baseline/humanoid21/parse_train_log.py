#!/usr/bin/env python3
"""
Parse training log and extract key metrics per update.

Usage:
    python3 parse_train_log.py <logfile> [output_file]

Output columns (TSV):
    update  stage  train_len  eval_len  final_in_zone  r_relation  vloss_r_relation  kl  new_best
"""

import re
import sys

# Regex for the main update line (train stats)
# Matches: update=N stage=S ... len=X.XX ... final_in_zone=X.XXX ... r_relation=+/-X.XXX ...
# vloss_r_relation=X.XXXX  kl=X.XXXX
RE_UPDATE = re.compile(
    r"update=(\d+)"
    r"\s+stage=(\d+)"
    r".*?\blen=(\d+\.\d+)"
    r".*?\bfinal_in_zone=(\d+\.\d+)"
    r".*?\br_relation=([+-]?\d+\.\d+)"
    r".*?\bvloss_r_relation=(\d+\.\d+)"
    r".*?\bkl=(\d+\.\d+)"
)

# eval block appears after kl, separated by |: len=X.XX term=X.XXX ... [new_best]?
# We look for "| len=X.XX" within the same line after kl
RE_EVAL_INLINE = re.compile(r"\|\s*len=(\d+\.\d+).*?(\[new_best\])?")


def parse_log(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            m = RE_UPDATE.search(line)
            if not m:
                continue

            update        = int(m.group(1))
            stage         = int(m.group(2))
            train_len     = float(m.group(3))
            final_in_zone = float(m.group(4))
            r_relation    = float(m.group(5))
            vloss         = float(m.group(6))
            kl            = float(m.group(7))

            # Check for inline eval block after the kl match
            rest = line[m.end():]
            eval_m = RE_EVAL_INLINE.search(rest)
            eval_len  = float(eval_m.group(1)) if eval_m else None
            new_best  = bool(eval_m and eval_m.group(2))

            records.append(dict(
                update=update,
                stage=stage,
                train_len=train_len,
                eval_len=eval_len,
                final_in_zone=final_in_zone,
                r_relation=r_relation,
                vloss_r_relation=vloss,
                kl=kl,
                new_best=new_best,
            ))
    return records


def format_table(records: list[dict]) -> str:
    header = (
        f"{'update':>7}  {'stage':>5}  {'train_len':>9}  {'eval_len':>8}  "
        f"{'in_zone':>7}  {'r_rel':>7}  {'vloss_rel':>9}  {'kl':>7}  {'best':>4}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    prev_stage = None
    for r in records:
        if prev_stage is not None and r["stage"] != prev_stage:
            lines.append(sep)  # blank separator between stage transitions
        prev_stage = r["stage"]

        eval_str = f"{r['eval_len']:8.2f}" if r["eval_len"] is not None else "        "
        best_str = "***" if r["new_best"] else ""
        lines.append(
            f"{r['update']:7d}  {r['stage']:5d}  {r['train_len']:9.2f}  {eval_str}  "
            f"{r['final_in_zone']:7.3f}  {r['r_relation']:+7.4f}  "
            f"{r['vloss_r_relation']:9.4f}  {r['kl']:7.4f}  {best_str}"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    records = parse_log(log_path)
    if not records:
        print("No update lines found.", file=sys.stderr)
        sys.exit(1)

    table = format_table(records)

    if out_path:
        with open(out_path, "w") as f:
            f.write(table + "\n")
        print(f"Written {len(records)} updates → {out_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()
