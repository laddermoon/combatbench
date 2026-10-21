"""Automatic repair: ground penetration via root-height lift.

Method: for every frame, measure the lowest foot point and lift
``root_pos.z`` so it sits at ``margin`` (default ~0).  Joint angles,
velocities and the other fields are untouched — this is the minimal
kinematic patch for retarget clips whose ground reference is wrong.

LIMITS of this repair (documented in the output meta):
  * It only fixes sinking below z=0.  If the source pose has feet at
    inconsistent heights (one leg effectively longer), the other foot
    may still float — that needs IK-based leg adjustment (not this tool).
  * root_vel_local.z is NOT modified; kinetic consistency with the
    lifted root trajectory is not guaranteed.
  * Nothing here proves dynamic executability — the fixed clip is
    still only a kinematic reference.

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.clip.fix \
        --clip x.npz [--out x_fixed.npz] [--mode lowest|constant]
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from .check import ground_report
from .core_clip import CORE_FIELDS, CoreClip


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average, edge-padded. w<=1 returns x."""
    if w <= 1 or len(x) < 3:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")[:len(x)]


def fix_ground_penetration(clip: CoreClip, robot: str = "robot_a",
                           mode: str = "lowest", margin: float = 0.002,
                           smooth: int = 3, sim=None) -> CoreClip:
    """Return a new CoreClip lifted so no foot point is below ``margin``.

    mode "lowest":   per-frame lift = penetration + margin (exact touch)
    mode "constant": one lift for all frames = max penetration + margin
    """
    rep = ground_report(clip, robot=robot, sim=sim)
    if rep.max_penetration <= margin:
        return clip.slice(0, clip.T,
                          clip_id=clip.clip_id)  # nothing to fix
    if mode == "lowest":
        offsets = _smooth(rep.penetration, smooth) + margin
    elif mode == "constant":
        offsets = np.full(clip.T, rep.max_penetration + margin)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    states = {f: clip.states[f].copy() for f in CORE_FIELDS}
    states["root_pos"][:, 2] += offsets
    meta = dict(clip.meta)
    meta["clip_id"] = f"{clip.clip_id}_groundfix"
    meta["source"] = {
        "type": "fix_ground_penetration",
        "parent": clip.clip_id,
        "method": f"root_lift_{mode}",
        "margin_m": margin,
        "max_offset_m": float(offsets.max()),
        "mean_offset_m": float(offsets.mean()),
        "note": "kinematic patch: lifts root_pos.z only; feet may "
                "still float if source legs are inconsistent; "
                "root_vel_local.z unchanged",
    }
    return CoreClip(states, meta,
                    None if clip.ref_action is None
                    else clip.ref_action.copy())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Lift clip root so feet do not sink below ground")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--robot", default="robot_a")
    ap.add_argument("--mode", default="lowest",
                    choices=("lowest", "constant"))
    ap.add_argument("--margin", type=float, default=0.002,
                    help="target lowest-foot height above ground (m)")
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    clip = CoreClip.load(args.clip)
    fixed = fix_ground_penetration(clip, robot=args.robot,
                                   mode=args.mode, margin=args.margin,
                                   smooth=args.smooth)
    rep = ground_report(fixed, robot=args.robot)
    print(rep.summary())
    out = args.out or str(args.clip).replace(".npz", "_groundfix.npz")
    fixed.save(out)
    print(f"[fix] saved {out}")


if __name__ == "__main__":
    main()
