"""reclip — synthesize an EXECUTABLE clip between two poses.

Pose interpolation between start/end provides a soft guide rail;
each frame's ref_action is CEM-solved to bring the simulated robot
closest to the next guide state, and the ACTUAL simulated landing
state becomes clip[k+1].  Therefore

    sim(clip[k], ref_action[k]) == clip[k+1]

by construction — the clip is executable by definition (deterministic
sim), with real measured velocities and contacts.  The guide rail is
only a target schedule; the stored trajectory is physics.

Intended scope (documented limitation): SHORT transitions — punch /
retract / recovery moves of ~1-2 s where balance failure is unlikely.
Do not use for long locomotion or unstable-pose sequences (greedy
per-frame solving does not reason about fall recovery).

Usage (unified CLI):
    PYTHONPATH=. python3 -m mocap.humanoid21.clip reclip \
        --start s.npz --end e.npz --out r.npz [--T 40] \
        [--via guide.npz] [--settle 20] [--tol-final 0.1]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .core_clip import (CONTROL_DT, CORE_FIELDS, CoreClip, _slerp,
                        SPEC_VERSION, record_lineage)
from .rederive_action import (DEFAULT_WEIGHTS, frame_distance,
                              solve_frame_action)


def lerp_state(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
               t: float) -> Dict[str, np.ndarray]:
    """Interpolate two CoreState dicts (slerp root_rot, lerp rest)."""
    out = {}
    for f in CORE_FIELDS:
        if f == "root_rot":
            out[f] = _slerp(a[f], b[f], t)
        else:
            out[f] = ((1.0 - t) * a[f] + t * b[f]).astype(np.float32)
    return out


def interp_guide(start: Dict[str, np.ndarray],
                 end: Dict[str, np.ndarray], T: int
                 ) -> List[Dict[str, np.ndarray]]:
    return [lerp_state(start, end, k / max(T - 1, 1))
            for k in range(T)]


def guide_from_clip(via: CoreClip) -> List[Dict[str, np.ndarray]]:
    return [via.state(k) for k in range(via.T)]


def reclip(start: Dict[str, np.ndarray],
           guide: List[Dict[str, np.ndarray]],
           robot: str = "robot_a",
           weights: Optional[Dict[str, float]] = None,
           settle: int = 0, tol_final: float = 0.1,
           n_samples: int = 64, n_iters: int = 4,
           seed: int = 0, sim=None,
           clip_id: str = "reclip") -> CoreClip:
    """Roll an executable clip: CEM-solve each step toward the guide.

    ``guide`` is a list of target CoreStates (its length is the
    transit-frame count).  After the guide is exhausted, ``settle``
    extra frames keep targeting its last state until
    D_final < ``tol_final`` or the budget runs out.
    """
    own = sim is None
    if own:
        from .rederive_action import _mk_sim
        sim = _mk_sim()
    other = "robot_b" if robot == "robot_a" else "robot_a"
    other_state = sim.get_core_state()[other]

    T_guide = len(guide)
    states: List[Dict[str, np.ndarray]] = [dict(start)]
    actions: List[np.ndarray] = []
    per_D: List[float] = []

    def step_toward(cur, tgt, k):
        sol = solve_frame_action(
            sim, robot, cur, tgt, other_state,
            weights=weights, n_samples=n_samples, n_iters=n_iters,
            seed=seed + k)
        a = sol["a"].astype(np.float32)
        sim.set_core_state({robot: cur, other: other_state})
        sim.set_action({robot: a, other: np.zeros(21, np.float32)})
        for _ in range(int(CONTROL_DT / 0.002)):
            sim.physical_step()
        nxt = sim.get_core_state()[robot]
        actions.append(a)
        states.append(nxt)
        per_D.append(sol["D"])
        return nxt

    cur = dict(start)
    for k in range(T_guide - 1):
        cur = step_toward(cur, guide[k + 1], k)
    end = guide[-1]
    final_D = frame_distance(cur, end, weights=weights)["total"]
    n_settle = 0
    while final_D > tol_final and n_settle < settle:
        cur = step_toward(cur, end, T_guide - 1 + n_settle)
        n_settle += 1
        final_D = frame_distance(cur, end, weights=weights)["total"]

    T = len(states)
    ra = np.zeros((T, 21), np.float32)
    ra[:len(actions)] = np.stack(actions)
    ra[T - 1] = np.clip(states[T - 1]["joint_pos_norm"], -1, 1)

    if own:
        sim.close()

    meta: Dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "clip_id": clip_id,
        "ref_action_semantics": "executed_pd_target",
        "source": {"type": "reclip"},
        "reclip": {
            "T_guide": T_guide, "settle_used": n_settle,
            "final_distance": float(final_D),
            "per_frame_target_D": [float(d) for d in per_D],
            "weights": dict(DEFAULT_WEIGHTS if weights is None
                            else weights),
            "n_samples": n_samples, "n_iters": n_iters, "seed": seed,
            "note": "states are simulated landings; guide rail is a "
                    "soft target schedule — executable by "
                    "construction under deterministic physics",
        },
    }
    meta = record_lineage(meta, "reclip",
                          clip_id=clip_id, T=T, settle=n_settle,
                          final_D=float(final_D))
    stacked = {f: np.stack([s[f] for s in states]) for f in CORE_FIELDS}
    return CoreClip(stacked, meta, ra)
