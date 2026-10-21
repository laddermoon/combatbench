"""Repair ground contact in an existing humanoid21_v2 clip npz.

Uses the same contact-aware grounding pass as the retarget pipelines
(mocap.humanoid21.grounding): true capsule-sole measurement, per-frame
anchor lift, IK plant for the other contact foot.  foot_contacts in
the npz are already robot-frame (mirrored clips included).

Derived fields (root_linvel, joint_velocities, keypoints) are
recomputed so the file stays self-consistent.

Usage:
  PYTHONPATH=. python3 -m mocap.humanoid21.repair_grounding CLIP.npz
      [--out OUT.npz] [--check] [--margin 0.005]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .grounding import repair_npz


def _report(rep: dict) -> str:
    lift = rep["lift"]
    lines = [
        f"  lift: max={lift.max() * 100:.1f}cm "
        f"mean={lift.mean() * 100:.1f}cm",
        f"  ik repairs: {int(rep['ik_fixed'].sum())} frames fixed, "
        f"residual>tol: {int((rep['ik_resid'] > 0.008).sum())}",
        f"  final soles: left {rep['sole_left'].min() * 100:+.1f}.."
        f"{rep['sole_left'].max() * 100:+.1f}cm   "
        f"right {rep['sole_right'].min() * 100:+.1f}.."
        f"{rep['sole_right'].max() * 100:+.1f}cm",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Contact-aware ground repair for clip npz")
    ap.add_argument("clip")
    ap.add_argument("--out", default=None,
                    help="default: overwrite in place")
    ap.add_argument("--check", action="store_true",
                    help="report only, do not write")
    ap.add_argument("--margin", type=float, default=0.005)
    args = ap.parse_args()

    z = np.load(args.clip, allow_pickle=True)
    npz = {k: z[k] for k in z.files}
    out = repair_npz(npz, margin=args.margin)
    rep = out.pop("_ground_report")
    print(f"{args.clip}:")
    print(_report(rep))
    if not args.check:
        out_path = args.out or args.clip
        np.savez(out_path, **{k: v for k, v in out.items()})
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
