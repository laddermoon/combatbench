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
                              solve_action_window, solve_frame_action)


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


RECLIP_WEIGHTS = dict(DEFAULT_WEIGHTS, fall=2.0)


def _roll_from(sim, robot, other, state_k, other_state, aseq,
               targets, weights, k0):
    """Roll aseq over the sim from state_k; return
    (suffix_score, landed_states).  suffix score = Σ D vs targets."""
    sim.set_core_state({robot: state_k, other: other_state})
    tot, landed = 0.0, []
    for t, a in enumerate(aseq):
        sim.set_action({robot: a.astype(np.float32),
                        other: np.zeros(21, np.float32)})
        for _ in range(int(CONTROL_DT / 0.002)):
            sim.physical_step()
        st = sim.get_core_state()[robot]
        landed.append(st)
        tgt = targets[min(k0 + t + 1, len(targets) - 1)]
        tot += frame_distance(st, tgt,
                              action=a, weights=weights)["total"]
    return tot, landed


def reclip_sweep(start, guide,
                 init_actions: np.ndarray,
                 robot: str = "robot_a",
                 weights: Optional[Dict[str, float]] = None,
                 sweeps: int = 3,
                 n_samples: int = 32, n_iters: int = 2,
                 sigma0: float = 0.15,
                 seed: int = 0, sim=None,
                 clip_id: str = "reclip") -> CoreClip:
    """Iterated-shooting trajectory optimization.

    Starts from an initial action sequence (e.g. the guide clip's own
    ref_action, or per-frame PD targets) and improves it in sweeps:
    for each frame k, candidate actions are scored by the TOTAL
    landing error of the whole remaining sequence (suffix score),
    not by one-step distance — so actions that preserve the future
    trajectory win over actions that only look good now.

    The landed states become the clip: executable by construction.
    """
    own = sim is None
    if own:
        from .rederive_action import _mk_sim
        sim = _mk_sim()
    rng = np.random.default_rng(seed)
    other = "robot_b" if robot == "robot_a" else "robot_a"
    other_state = sim.get_core_state()[other]
    if weights is None:
        weights = RECLIP_WEIGHTS

    T_guide = len(guide)
    Na = T_guide - 1
    seq = np.clip(np.asarray(init_actions[:Na], np.float64), -1, 1)
    if len(seq) < Na:
        seq = np.vstack([seq, np.zeros((Na - len(seq), 21))])

    states = [dict(start)]
    total, landed = _roll_from(sim, robot, other, start, other_state,
                               seq, guide, weights, 0)
    states += landed
    per_D = [frame_distance(states[k + 1], guide[k + 1],
                            weights=weights)["total"]
             for k in range(Na)]

    def roll_from(k):
        return _roll_from(sim, robot, other, states[k], other_state,
                          seq[k:], guide, weights, k)

    for sw in range(sweeps):
        for k in range(Na):
            base = states[k]
            best_suffix, tail = roll_from(k)
            best_a, best_tail = seq[k], tail
            mu = seq[k]
            sigma = np.full(21, sigma0)
            for it in range(n_iters):
                cand = np.clip(rng.normal(mu, sigma,
                                          (n_samples, 21)), -1, 1)
                if it == 0:
                    cand[0] = seq[k]
                for c in cand:
                    test = seq[k:].copy()
                    test[0] = c
                    sc, tl = _roll_from(sim, robot, other, base,
                                        other_state, test, guide,
                                        weights, k)
                    if sc < best_suffix:
                        best_suffix, best_a, best_tail = sc, c, tl
                elite = np.asarray([best_a])
                mu = 0.7 * mu + 0.3 * elite.mean(axis=0)
                sigma = np.maximum(0.5 * sigma, 0.03)
            seq[k] = best_a
            states[k + 1:] = best_tail

    T = len(states)
    ra = np.zeros((T, 21), np.float32)
    ra[:Na] = seq.astype(np.float32)
    ra[T - 1] = np.clip(states[T - 1]["joint_pos_norm"], -1, 1)
    final_D = frame_distance(states[-1], guide[-1],
                             weights=weights)["total"]
    per_D = [frame_distance(states[k + 1], guide[k + 1],
                            weights=weights)["total"]
             for k in range(Na)]
    if own:
        sim.close()

    meta: Dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "clip_id": clip_id,
        "ref_action_semantics": "executed_pd_target",
        "source": {"type": "reclip"},
        "reclip": {
            "method": "sweep",
            "T_guide": T_guide,
            "final_distance": float(final_D),
            "per_frame_target_D": [float(d) for d in per_D],
            "weights": dict(weights),
            "sweeps": sweeps, "n_samples": n_samples,
            "n_iters": n_iters, "sigma0": sigma0, "seed": seed,
            "note": "iterated shooting: per-frame actions scored by "
                    "full-suffix landing error; states are simulated "
                    "landings — executable by construction",
        },
    }
    meta = record_lineage(meta, "reclip", clip_id=clip_id, T=T,
                          final_D=float(final_D), method="sweep")
    stacked = {f: np.stack([s[f] for s in states])
               for f in CORE_FIELDS}
    return CoreClip(stacked, meta, ra)


def reclip(start: Dict[str, np.ndarray],
           guide: List[Dict[str, np.ndarray]],
           robot: str = "robot_a",
           weights: Optional[Dict[str, float]] = None,
           settle: int = 0, tol_final: float = 0.1,
           n_samples: int = 64, n_iters: int = 4,
           horizon: int = 1, disc: float = 0.9,
           guide_actions: Optional[np.ndarray] = None,
           seed: int = 0, sim=None,
           clip_id: str = "reclip") -> CoreClip:
    """Roll an executable clip: CEM-solve each step toward the guide.

    ``guide`` is a list of target CoreStates (its length is the
    transit-frame count).  After the guide is exhausted, ``settle``
    extra frames keep targeting its last state until
    D_final < ``tol_final`` or the budget runs out.

    ``horizon > 1`` switches to receding-horizon MPC: each step solves
    a discounted action sequence toward the next ``horizon`` guide
    targets, applies only the first action, and warm-starts the next
    solve with the shifted previous sequence.  This lets the solver
    commit to multi-frame dynamics (weight transfer, wind-up) that
    greedy single-frame solving cannot discover.
    """
    own = sim is None
    if own:
        from .rederive_action import _mk_sim
        sim = _mk_sim()
    other = "robot_b" if robot == "robot_a" else "robot_a"
    other_state = sim.get_core_state()[other]
    if weights is None:
        weights = RECLIP_WEIGHTS

    T_guide = len(guide)

    def hint(k):
        """Guide-clip action prior for step k -> k+1 (if provided)."""
        if guide_actions is None:
            return None
        return np.asarray(
            guide_actions[min(k + 1, len(guide_actions) - 1)],
            np.float64)

    states: List[Dict[str, np.ndarray]] = [dict(start)]
    actions: List[np.ndarray] = []
    per_D: List[float] = []
    prev_seq = None

    def step_toward(cur, tgt, k):
        nonlocal prev_seq
        if horizon > 1:
            win = [guide[min(k + 1 + t, T_guide - 1)]
                   for t in range(horizon)]
            if prev_seq is None and guide_actions is not None:
                prev_seq = np.clip(np.stack(
                    [guide_actions[min(k + 1 + t,
                                       len(guide_actions) - 1)]
                     for t in range(horizon)]), -1, 1)
            sol = solve_action_window(
                sim, robot, cur, win, other_state,
                weights=weights, init_seq=prev_seq,
                n_samples=n_samples, n_iters=n_iters,
                disc=disc, seed=seed + k)
            a = sol["seq"][0].astype(np.float32)
            # warm start next solve: shifted best seq + last action
            prev_seq = np.vstack([sol["seq"][1:], sol["seq"][-1:]])
            D_here = sol["detail"]["total"]
        else:
            sol = solve_frame_action(
                sim, robot, cur, tgt, other_state,
                weights=weights, init=hint(k),
                n_samples=n_samples,
                n_iters=n_iters, seed=seed + k)
            a = sol["a"].astype(np.float32)
            D_here = sol["D"]
        sim.set_core_state({robot: cur, other: other_state})
        sim.set_action({robot: a, other: np.zeros(21, np.float32)})
        for _ in range(int(CONTROL_DT / 0.002)):
            sim.physical_step()
        nxt = sim.get_core_state()[robot]
        actions.append(a)
        states.append(nxt)
        per_D.append(D_here)
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
            "weights": dict(weights),
            "n_samples": n_samples, "n_iters": n_iters, "seed": seed,
            "horizon": horizon, "disc": disc,
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
