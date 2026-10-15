"""Scan every AMC clip: FK the source skeleton, extract content features.

Goals (M0 W3):
  - per-clip content features -> provisional action labels for the minimal set
  - flag anomalies (below-ground feet, teleports, nonfinite)

FK: mocap.humanoid21.ref_fk (official-CMU-viewer convention; legacy fk.py was
found incorrect — see ref_fk.py / M0_AUDIT.md).

Unit handling: ref_fk output is in "CMU units" (raw AMC/ASF x 0.45). Absolute
meter scale is uncalibrated in M0; provisional calibration hip_height~0.95 m
gives METERS_PER_UNIT ~= 0.11 (revisit at M1 retargeting).

Output: content_scan.json next to this file.

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.content_scan
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mocap.humanoid21.parsers.asf_parser import parse_asf
from mocap.humanoid21.parsers.amc_parser import parse_amc
from mocap.humanoid21.ref_fk import RefSkeleton, fk_batch

RAW_DIRS = [
    Path("/data1/dev/combatbench_mocap/raw"),
]
OUT = Path(__file__).parent / "content_scan.json"

FPS = 120.0
METERS_PER_UNIT = 0.11  # provisional: standing hip ~0.95 m / ~8.5 CMU units

HAND_BONES = ["lhand", "rhand"]          # use distal endpoints
FOOT_BONES = ["lfoot", "rfoot", "ltoes", "rtoes"]


def _find_raw(name: str) -> Path:
    for d in RAW_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def clip_features(skel: RefSkeleton, frames) -> dict:
    T = len(frames)
    J, D, names = fk_batch(skel, frames)
    idx = {n: i for i, n in enumerate(names)}

    def track(name):
        return D[:, idx[name]] * METERS_PER_UNIT if name in idx else None

    root_pos = J[:, idx["root"]] * METERS_PER_UNIT  # (T,3); y is up
    head_pos = track("head")
    feet = {n: track(n) for n in FOOT_BONES}
    hands = {n: track(n) for n in HAND_BONES}
    hands = {k: v for k, v in hands.items() if v is not None}
    feet = {k: v for k, v in feet.items() if v is not None}

    dt = 1.0 / FPS

    # --- root ---
    horiz = root_pos[:, [0, 2]]
    horiz_speed = np.linalg.norm(np.diff(horiz, axis=0), axis=1) / dt
    root_h = root_pos[:, 1]

    # --- hands ---
    hand_stats = {}
    for side, pos in hands.items():
        vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
        hand_stats[side] = {
            "peak_speed_mps": float(vel.max()),
            "peak_t_s": round(float(np.argmax(vel) * dt), 2),
            "p99_speed_mps": float(np.quantile(vel, 0.99)),
            "mean_speed_mps": float(vel.mean()),
        }

    # --- feet: height + contact estimate ---
    all_foot_h = np.stack([v[:, 1] for v in feet.values()], axis=0).min(axis=0)
    foot_speed = np.stack(
        [np.linalg.norm(np.diff(v, axis=0), axis=1) / dt for v in feet.values()], axis=0
    ).min(axis=0)
    contact_est = (all_foot_h[1:] < 0.08) & (foot_speed < 0.5)

    # --- simple labels ---
    labels = []
    if horiz_speed.mean() > 0.15 or np.ptp(horiz, axis=0).max() > 0.8:
        labels.append("locomotion")
    else:
        labels.append("in_place")
    peak_hand = max(v["peak_speed_mps"] for v in hand_stats.values())
    if peak_hand > 3.0:
        labels.append("fast_hand_motion")
    if root_h.mean() > 0.9 * (head_pos[:, 1].mean() - 0.55):  # upright-ish
        labels.append("upright")
    else:
        labels.append("bent_or_low")
    contact_frac = float(contact_est.mean())
    if contact_frac < 0.3:
        labels.append("feet_often_off_ground")

    return {
        "frames": T,
        "duration_s": round(T / FPS, 2),
        "root_height_m": {"min": round(float(root_h.min()), 2),
                          "mean": round(float(root_h.mean()), 2),
                          "max": round(float(root_h.max()), 2)},
        "root_horiz_displacement_m": round(float(np.ptp(horiz, axis=0).max()), 2),
        "root_speed_mean_mps": round(float(horiz_speed.mean()), 2),
        "root_speed_max_mps": round(float(horiz_speed.max()), 2),
        "hands": hand_stats,
        "min_foot_height_m": round(float(all_foot_h.min()), 2),
        "foot_contact_frac": round(contact_frac, 2),
        "provisional_labels": labels,
    }


def main():
    # collect all (asf, amc) pairs across raw dirs
    amcs = {}
    for d in RAW_DIRS:
        for p in sorted(d.glob("*.amc")):
            amcs.setdefault(p.stem, p)
    results = {}
    for clip_name in sorted(amcs):
        subj = clip_name.split("_")[0]
        skel = RefSkeleton.from_asf(parse_asf(str(_find_raw(f"{subj}.asf"))))
        try:
            frames = parse_amc(str(amcs[clip_name]))
            results[clip_name] = clip_features(skel, frames)
            r = results[clip_name]
            print(f"{clip_name:<7} {r['duration_s']:>6.1f}s "
                  f"disp={r['root_horiz_displacement_m']:>5.2f}m "
                  f"v={r['root_speed_mean_mps']:>4.1f}/{r['root_speed_max_mps']:>4.1f} "
                  f"hand={r['hands']['lhand']['peak_speed_mps']:>4.1f}/"
                  f"{r['hands']['rhand']['peak_speed_mps']:>4.1f} "
                  f"minFoot={r['min_foot_height_m']:>5.2f} "
                  f"{','.join(r['provisional_labels'])}")
        except Exception as e:
            results[clip_name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{clip_name:<7} ERROR {e}")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
