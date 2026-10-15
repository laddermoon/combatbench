"""Audit existing mocap assets under baseline/humanoid21/mocap/data/.

For every raw file: sha256, size, parse success, frame count, bone signature.
For every retargeted npy: shape, dtype, finiteness, limit violations against the
*current* model (robot_spec.json), clip fraction, frozen-frame ratio.

Output: assets_index.json next to this file (git add -f; *.json is gitignored).

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.audit_assets
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mocap.humanoid21.parsers.amc_parser import parse_amc
from mocap.humanoid21.parsers.asf_parser import parse_asf

DATA_DIR = Path("/data1/dev/combatbench_mocap")
RAW = DATA_DIR / "raw"
RET = DATA_DIR / "retargeted"
# supplementary downloads (walk/punch clips to fill the minimal action set);
# on the GPU server, data lives under /data1/dev per repo convention
RAW_SUPPLEMENT = Path("/data1/dev/combatbench_mocap/raw")
OUT = Path(__file__).parent / "assets_index.json"

# CMU AMC sampling rate — standard for this dataset; marked as assumed until
# cross-verified against the official subject notes (W3/W4).
CMU_FPS_ASSUMED = 120.0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_asf(path: Path) -> dict:
    skel = parse_asf(str(path))
    n_dof = sum(len(b.dof) for b in skel.bones.values())
    return {
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "bones": len(skel.bones),
        "total_dof": n_dof,
        "units_length": skel.units_length,
        "units_angle": skel.units_angle,
        "root_order": skel.root_order,
    }


def audit_amc(path: Path) -> dict:
    frames = parse_amc(str(path))
    rec = {
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "frames": len(frames),
        "duration_s_assumed_120hz": round(len(frames) / CMU_FPS_ASSUMED, 2),
    }
    if frames:
        keys = sorted(frames[0].keys())
        rec["bone_keys_frame0"] = keys
        # bones present in every frame (missing-bone check)
        all_keys = set(keys)
        missing = sum(1 for f in frames if set(f.keys()) != all_keys)
        rec["frames_with_missing_bones"] = missing
        root = np.array([f["root"] for f in frames if "root" in f])
        if len(root):
            rec["root_pos_range_m"] = (
                np.ptp(root[:, :3], axis=0) * 0.0
            ).tolist()  # placeholder; units applied in W3 via skeleton
            rec["root_tz_min"] = float(root[:, 2].min())
            rec["root_tz_max"] = float(root[:, 2].max())
        bad = sum(1 for f in frames for v in f.values() if not np.all(np.isfinite(v)))
        rec["nonfinite_values"] = bad
    return rec


def audit_npy(path: Path, limits_deg: np.ndarray, joint_names: list) -> dict:
    a = np.load(path)
    rec = {
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite": bool(np.all(np.isfinite(a))),
    }
    if a.ndim == 2 and a.shape[1] == len(joint_names):
        lo, hi = limits_deg[:, 0], limits_deg[:, 1]
        tol = 1e-4
        below = a < lo[None, :] - tol
        above = a > hi[None, :] + tol
        rec["limit_violations"] = int((below | above).sum())
        rec["max_overshoot_deg"] = float(
            max((lo[None, :] - a).max(), (a - hi[None, :]).max(), 0.0))
        # how many values sit exactly at a limit (clipping artifact)
        at_lim = (np.abs(a - lo[None, :]) < 1e-6) | (np.abs(a - hi[None, :]) < 1e-6)
        rec["at_limit_frac"] = float(at_lim.mean())
        # frozen frames: consecutive rows identical
        rec["frozen_frame_frac"] = float(np.mean(np.all(np.diff(a, axis=0) == 0, axis=1))) if len(a) > 1 else 0.0
        per_joint = []
        for i, jn in enumerate(joint_names):
            per_joint.append({
                "joint": jn,
                "min": float(a[:, i].min()),
                "max": float(a[:, i].max()),
                "at_limit_frac": float(at_lim[:, i].mean()),
            })
        rec["per_joint"] = per_joint
    else:
        rec["note"] = "unexpected shape"
    return rec


def main() -> None:
    spec = json.loads((Path(__file__).parent / "robot_spec.json").read_text())
    limits_deg = np.array([j["range_deg"] for j in spec["joints"]])
    joint_names = spec["controlled_joints_order"]

    index = {
        "source_dataset": "CMU Graphics Lab Motion Capture Database",
        "source_url": "http://mocap.cs.cmu.edu/",
        "fps_assumed": CMU_FPS_ASSUMED,
        "asf": {}, "amc": {}, "retargeted": {},
    }

    for root, tag in ((RAW, "raw"), (RAW_SUPPLEMENT, "raw_supplement")):
        if not root.exists():
            continue
        for p in sorted(root.glob("*.asf")):
            key = p.name if tag == "raw" else f"{tag}:{p.name}"
            try:
                rec = audit_asf(p)
                rec["path"] = str(p)
                index["asf"][key] = rec
            except Exception as e:
                index["asf"][key] = {"error": f"{type(e).__name__}: {e}"}

    for root, tag in ((RAW, "raw"), (RAW_SUPPLEMENT, "raw_supplement")):
        if not root.exists():
            continue
        for p in sorted(root.glob("*.amc")):
            key = p.stem if tag == "raw" else f"{tag}:{p.stem}"
            try:
                rec = audit_amc(p)
                rec["path"] = str(p)
                index["amc"][key] = rec
            except Exception as e:
                index["amc"][key] = {"error": f"{type(e).__name__}: {e}"}

    for p in sorted(RET.glob("*.npy")):
        try:
            index["retargeted"][p.stem] = audit_npy(p, limits_deg, joint_names)
        except Exception as e:
            index["retargeted"][p.stem] = {"error": f"{type(e).__name__}: {e}"}

    OUT.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    print(f"asf={len(index['asf'])} amc={len(index['amc'])} npy={len(index['retargeted'])}")

    # summary table
    print(f"\n{'clip':<8}{'frames':>7}{'dur_s':>7}{'missing':>8}{'nonfinite':>10}")
    for k, v in sorted(index["amc"].items()):
        if "error" in v:
            print(f"{k:<8} ERROR {v['error']}")
            continue
        print(f"{k:<8}{v['frames']:>7}{v['duration_s_assumed_120hz']:>7}"
              f"{v.get('frames_with_missing_bones', 0):>8}{v.get('nonfinite_values', 0):>10}")

    print(f"\n{'npy':<16}{'shape':<14}{'viol':>6}{'at_lim%':>8}{'frozen%':>8}")
    for k, v in sorted(index["retargeted"].items()):
        if "error" in v:
            print(f"{k:<16} ERROR {v['error']}")
            continue
        if "limit_violations" in v:
            print(f"{k:<16}{str(v['shape']):<14}{v['limit_violations']:>6}"
                  f"{v['at_limit_frac']*100:>7.1f}%{v['frozen_frame_frac']*100:>7.1f}%")
        else:
            print(f"{k:<16}{str(v['shape']):<14} {v.get('note','')}")


if __name__ == "__main__":
    main()
