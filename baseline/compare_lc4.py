"""Strict bit-identical comparison: lifecycle-refactor run vs ctx3 run.

Compares __RAW_STATS__ update records (excluding timing) between
verify_lc4_standup_floor04_s42 (new version-stream loop) and
verify_ctx3_standup_floor04_s42 (verified equal to the ef05 baseline).

Also compares exported model weights under the version-shifted naming:
new u{K} (post-update-K)  ==  old u{K+1} (pre-update-(K+1))  for K>=1
new u00000 (init weights) ==  old u00001 (pre-update-1 = same init)
new u00005 (post-update-5)==  old u00005_eval (eval export = same weights)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

RUNS = Path("baseline/runs")
NEW = RUNS / "verify_lc4_standup_floor04_s42"
OLD = RUNS / "verify_ctx3_standup_floor04_s42"


def raw_stats(log_path):
    out = {}
    for line in open(log_path):
        if line.startswith("__RAW_STATS__"):
            rec = json.loads(line[len("__RAW_STATS__"):])
            out[int(rec["update"])] = rec
    return out


def strip_timing(rec):
    rec = dict(rec)
    rec.pop("timing", None)
    s = dict(rec.get("stats", {}))
    rec["stats"] = {k: v for k, v in s.items() if not k.endswith("_time_s")}
    return rec


def deep_diff(a, b, path=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys differ {sorted(set(a) ^ set(b))[:8]}")
            return diffs
        for k in a:
            diffs += deep_diff(a[k], b[k], f"{path}.{k}")
        return diffs
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            diffs += deep_diff(x, y, f"{path}[{i}]")
        return diffs
    if a != b and not (
        isinstance(a, float) and isinstance(b, float)
        and np.isnan(a) and np.isnan(b)
    ):
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def main():
    new = raw_stats(NEW / "train.log")
    old = raw_stats(OLD / "train.log")
    assert set(new) == set(old) == {1, 2, 3, 4, 5}, (sorted(new), sorted(old))

    total = 0
    for u in sorted(new):
        d = deep_diff(strip_timing(new[u]), strip_timing(old[u]), f"u{u}")
        status = "IDENTICAL" if not d else f"{len(d)} DIFFS"
        print(f"update {u}: {status}")
        for line in d[:10]:
            print("   ", line)
        total += len(d)
    print(f"records: {'ALL IDENTICAL' if total == 0 else f'{total} diffs'}")

    # --- weight parity across the version shift ---
    def _model_equal(pa, pb):
        if isinstance(pa, torch.Tensor) and isinstance(pb, torch.Tensor):
            return bool(torch.equal(pa, pb))
        if isinstance(pa, dict) and isinstance(pb, dict):
            return set(pa) == set(pb) and all(
                _model_equal(pa[k], pb[k]) for k in pa
            )
        try:
            return bool(pa == pb)
        except Exception:
            return False

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    ok = True
    for k_new, k_old in pairs:
        pa = torch.load(NEW / "policy_exports" / f"u{k_new:05d}" / "model.pt",
                        map_location="cpu", weights_only=False)
        pb = torch.load(OLD / "policy_exports" / f"u{k_old:05d}" / "model.pt",
                        map_location="cpu", weights_only=False)
        same = _model_equal(pa, pb)
        ok &= same
        print(f"weights u{k_new:05d}(new) vs u{k_old:05d}(old): {'EQUAL' if same else 'DIFF'}")
    # post-update-5 == old eval export (det export of same weights)
    pa = torch.load(NEW / "policy_exports" / "u00005" / "model.pt",
                    map_location="cpu", weights_only=False)
    pb = torch.load(OLD / "policy_exports" / "u00005_eval" / "model.pt",
                    map_location="cpu", weights_only=False)
    same = _model_equal(pa, pb)
    ok &= same
    print(f"weights u00005(new) vs u00005_eval(old): {'EQUAL' if same else 'DIFF'}")

    sys.exit(0 if (total == 0 and ok) else 1)


if __name__ == "__main__":
    main()
