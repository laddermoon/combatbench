"""Scan every AMC clip: FK the source skeleton, extract content features.

Goals (M0 W3):
  - verify parser/FK correctness numerically (skeleton size, feet near ground)
  - per-clip content features -> provisional action labels for the minimal set
  - flag anomalies (below-ground feet, teleports, nonfinite)

Unit handling: raw AMC/ASF values share one consistent unit system
(fk.py scales both root and bones by units_length=0.45 — self-consistent).
Real-world scale empirically: meters = raw_unit / 0.45 * 0.0254 (raw*0.05644),
which yields ~1.7 m human height. Recorded here as `SCALE_M` with method note.

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

from baseline.humanoid21.mocap.asf_parser import parse_asf
from baseline.humanoid21.mocap.amc_parser import parse_amc
from baseline.humanoid21.mocap.fk import compute_fk_batch

RAW = PROJECT_ROOT / "baseline/humanoid21/mocap/data/raw"
OUT = Path(__file__).parent / "content_scan.json"

FPS = 120.0
# raw AMC/ASF unit -> meters (see module docstring for derivation)
SCALE_M = 0.0254 / 0.45
# fk.py already multiplies positions by units_length=0.45, so fk output is in
# "0.45-scaled units"; meters = fk_value / 0.45 * 0.0254 -> fk_value * 0.05644
FK_TO_M = 0.0254 / 0.45

# bones used as end-effectors / contacts
HAND_BONES = ["lhand", "rhand"]      # distal hand bone position
FOOT_BONES = ["lfoot", "rfoot", "ltoes", "rtoes"]


def clip_features(skel, frames) -> dict:
    T = len(frames)
    transforms = compute_fk_batch(skel, frames)

    def track(name):
        return np.array([tr[name][1] for tr in transforms]) * FK_TO_M if name in transforms[0] else None

    root_pos = track("root")          # (T,3), meters; y is up in CMU FK convention
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
    foot_h = np.minimum(feet["lfoot"][:, 1], feet["rfoot"][:, 1]) if "lfoot" in feet else None
    all_foot_h = np.stack([v[:, 1] for v in feet.values()], axis=0).min(axis=0)
    foot_speed = np.stack(
        [np.linalg.norm(np.diff(v, axis=0), axis=1) / dt for v in feet.values()], axis=0
    ).min(axis=0)
    contact_est = (all_foot_h[1:] < 0.08) & (foot_speed < 0.5)
    anomalies = {}
    anomalies["min_foot_height_m"] = float(all_foot_h.min())
    anomalies["max_root_speed_mps"] = float(horiz_speed.max())

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
        "min_foot_height_m": anomalies["min_foot_height_m"],
        "foot_contact_frac": round(contact_frac, 2),
        "provisional_labels": labels,
    }


def main():
    index_path = Path(__file__).parent / "assets_index.json"
    assets = json.loads(index_path.read_text())
    results = {}
    for asf_name in sorted(assets["asf"]):
        subj = asf_name.split(".")[0]
        asf_path = RAW / asf_name
        skel = parse_asf(str(asf_path))
        for clip_name in sorted(assets["amc"]):
            if not clip_name.startswith(f"{subj}_"):
                continue
            amc_path = RAW / f"{clip_name}.amc"
            try:
                frames = parse_amc(str(amc_path))
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
