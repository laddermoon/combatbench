"""Best-effort ref_action regeneration.

ref_action is a REFERENCE only — no guarantee that applying it at
state k reproduces clip[k+1].  This tool solves, per frame, the
nearest action under a weighted distance:

    a* = argmin_a  D( sim(state_k, a), state_{k+1} )

where D is a weighted sum of state-component errors (pose primary).
Frames where even the best action leaves D above tolerance are marked
``unsolved`` — that is a property of the clip's transition under this
dynamics model, not a solver failure.

Method: per-frame CEM shooting (teleport semantics — each frame is
solved from the EXACT clip state k, not from the previous frame's
landing state, so results do not compound).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .core_clip import CoreClip, record_lineage

DEFAULT_WEIGHTS: Dict[str, float] = {
    "jp": 1.0,    # mean |joint_pos_norm diff|        — primary
    "jv": 0.05,   # mean |joint_vel_norm diff| (norm/s)
    "rp": 1.0,    # 5 * |root_pos diff| (m)
    "rq": 0.5,    # geodesic root_rot diff / pi
    "rv": 0.2,    # |root_vel_local diff| / 2 (m/s)
    "ra": 0.1,    # |root_angvel_local diff| / 5 (rad/s)
    "reg": 0.02,  # mean(a^2) — prefer low-effort among near-ties
    "fall": 0.0,  # penalty max(0, 1.0 - root_z): discourage sinking
                  # (opt-in; reclip enables it)
}
TOL_SOLVED = 0.05
TOL_APPROX = 0.2


def frame_distance(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
                   action: Optional[np.ndarray] = None,
                   weights: Optional[Dict[str, float]] = None
                   ) -> Dict[str, Any]:
    """Weighted distance between two CoreState dicts.

    Components are pre-scaled to ~[0,1] magnitude so that D ~ 0.05 is
    'close'.  Returns {"total": D, "components": {...}}.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    q1 = np.asarray(a["root_rot"], np.float64)
    q2 = np.asarray(b["root_rot"], np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    geo = 2.0 * np.arccos(np.clip(abs(float(np.dot(q1, q2))), 0, 1))
    comp = {
        "jp": float(np.abs(a["joint_pos_norm"]
                           - b["joint_pos_norm"]).mean()),
        "jv": float(np.abs(a["joint_vel_norm"]
                           - b["joint_vel_norm"]).mean()),
        "rp": float(5.0 * np.linalg.norm(a["root_pos"] - b["root_pos"])),
        "rq": float(geo / np.pi),
        "rv": float(np.linalg.norm(a["root_vel_local"]
                                   - b["root_vel_local"]) / 2.0),
        "ra": float(np.linalg.norm(a["root_angular_vel_local"]
                                   - b["root_angular_vel_local"]) / 5.0),
        "reg": float(np.mean(np.square(action)))
        if action is not None else 0.0,
        "fall": float(max(0.0, 1.0 - a["root_pos"][2])),
    }
    total = sum(w.get(k, 0.0) * comp[k] for k in comp)
    return {"total": float(total), "components": comp}


def _mk_sim():
    from envs.humanoid21.simulator import Humanoid21Simulator
    return Humanoid21Simulator(initial_distance=6.0,
                               initial_pose_b="standing")


def solve_frame_action(sim, robot: str,
                       state_k: Dict[str, np.ndarray],
                       target: Dict[str, np.ndarray],
                       other_state: Optional[Dict[str, np.ndarray]],
                       weights: Optional[Dict[str, float]] = None,
                       init: Optional[np.ndarray] = None,
                       n_samples: int = 64, n_elite: int = 16,
                       n_iters: int = 4, sigma0: float = 0.3,
                       seed: int = 0) -> Dict[str, Any]:
    """CEM per-frame: best action taking state_k closest to target."""
    rng = np.random.default_rng(seed)
    other = ("robot_b" if robot == "robot_a" else "robot_a")
    zero_other = np.zeros(21, np.float32)

    mu = (init.astype(np.float64) if init is not None
          else np.asarray(target["joint_pos_norm"], np.float64))
    mu = np.clip(mu, -1.0, 1.0)
    sigma = np.full(21, sigma0)
    best = {"a": mu.copy(), "D": np.inf, "detail": None}

    def eval_a(a):
        st = {robot: state_k}
        if other_state is not None:
            st[other] = other_state
        sim.set_core_state(st)
        sim.set_action({robot: a.astype(np.float32), other: zero_other})
        for _ in range(25):
            sim.physical_step()
        got = sim.get_core_state()[robot]
        d = frame_distance(got, target, action=a, weights=weights)
        return d

    for it in range(n_iters):
        cand = rng.normal(mu, sigma, (n_samples, 21))
        cand = np.clip(cand, -1.0, 1.0)
        if it == 0 and init is not None:
            cand[0] = np.clip(init, -1, 1)
        scores = np.array([eval_a(c)["total"] for c in cand])
        elite = cand[np.argsort(scores)[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        bi = int(np.argmin(scores))
        if scores[bi] < best["D"]:
            best = {"a": cand[bi].copy(), "D": float(scores[bi])}

    best["detail"] = eval_a(best["a"])
    best["D"] = best["detail"]["total"]
    return best


def solve_action_window(sim, robot: str,
                        state_k: Dict[str, np.ndarray],
                        targets: List[Dict[str, np.ndarray]],
                        other_state: Optional[Dict[str, np.ndarray]],
                        weights: Optional[Dict[str, float]] = None,
                        init_seq: Optional[np.ndarray] = None,
                        n_samples: int = 64, n_elite: int = 16,
                        n_iters: int = 4, sigma0: float = 0.3,
                        seed: int = 0,
                        disc: float = 0.9) -> Dict[str, Any]:
    """Receding-horizon CEM: optimize an action SEQUENCE
    ``(horizon, 21)`` over h steps, minimizing discounted
    ``Σ disc^t · D(state_t, target_t)``.  Returns best sequence and
    the first-step detail.  Used by reclip's MPC mode.
    """
    rng = np.random.default_rng(seed)
    other = "robot_b" if robot == "robot_a" else "robot_a"
    zero_other = np.zeros(21, np.float32)
    h = len(targets)
    disc_w = np.power(disc, np.arange(h))

    if init_seq is not None:
        mu = np.asarray(init_seq, np.float64).copy()
    else:
        mu = np.stack([np.asarray(t["joint_pos_norm"], np.float64)
                       for t in targets])
    mu = np.clip(mu, -1.0, 1.0)
    sigma = np.full((h, 21), sigma0)
    best = {"seq": mu.copy(), "score": np.inf}

    def eval_seq(seq):
        st = {robot: state_k}
        if other_state is not None:
            st[other] = other_state
        sim.set_core_state(st)
        tot = 0.0
        for t in range(h):
            sim.set_action({robot: seq[t].astype(np.float32),
                            other: zero_other})
            for _ in range(25):
                sim.physical_step()
            got = sim.get_core_state()[robot]
            tot += disc_w[t] * frame_distance(
                got, targets[t], action=seq[t], weights=weights
            )["total"]
        return tot

    for it in range(n_iters):
        cand = np.clip(rng.normal(mu, sigma, (n_samples, h, 21)),
                       -1.0, 1.0)
        if it == 0 and init_seq is not None:
            cand[0] = np.clip(init_seq, -1, 1)
        scores = np.array([eval_seq(c) for c in cand])
        elite = cand[np.argsort(scores)[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        bi = int(np.argmin(scores))
        if scores[bi] < best["score"]:
            best = {"seq": cand[bi].copy(),
                    "score": float(scores[bi])}

    # detail of the first action's landing
    st = {robot: state_k}
    if other_state is not None:
        st[other] = other_state
    sim.set_core_state(st)
    sim.set_action({robot: best["seq"][0].astype(np.float32),
                    other: zero_other})
    for _ in range(25):
        sim.physical_step()
    got = sim.get_core_state()[robot]
    best["detail"] = frame_distance(got, targets[0],
                                    action=best["seq"][0],
                                    weights=weights)
    return best


def grade(D: float, tol_solved: float = TOL_SOLVED,
          tol_approx: float = TOL_APPROX) -> str:
    if D < tol_solved:
        return "solved"
    if D < tol_approx:
        return "approx"
    return "unsolved"


def rederive_actions(clip: CoreClip, robot: str = "robot_a",
                     weights: Optional[Dict[str, float]] = None,
                     tol_solved: float = TOL_SOLVED,
                     tol_approx: float = TOL_APPROX,
                     n_samples: int = 64, n_iters: int = 4,
                     seed: int = 0, sim=None) -> CoreClip:
    """Solve a best-effort action per frame (k -> k+1)."""
    T = clip.T
    own = sim is None
    if own:
        sim = _mk_sim()
    other = "robot_b" if robot == "robot_a" else "robot_a"
    other_state = sim.get_core_state()[other]

    actions = np.zeros((T, 21), np.float32)
    per_D = np.zeros(T)
    per_grade = []
    inits = (clip.ref_action if clip.ref_action is not None
             else clip.states["joint_pos_norm"])
    for k in range(T - 1):
        init = np.asarray(inits[min(k + 1, T - 1)], np.float64)
        sol = solve_frame_action(
            sim, robot, clip.state(k), clip.state(k + 1),
            other_state, weights=weights, init=init,
            n_samples=n_samples, n_iters=n_iters, seed=seed + k)
        actions[k] = sol["a"]
        per_D[k] = sol["D"]
    # last frame: hold its own pose as PD target
    actions[T - 1] = np.clip(clip.states["joint_pos_norm"][T - 1],
                             -1, 1)
    got = None
    sim.set_core_state({robot: clip.state(T - 1),
                        other: other_state})
    sim.set_action({robot: actions[T - 1], other: np.zeros(21)})
    for _ in range(25):
        sim.physical_step()
    got = sim.get_core_state()[robot]
    per_D[T - 1] = frame_distance(got, clip.state(T - 1),
                                  action=actions[T - 1],
                                  weights=weights)["total"]
    per_grade = [grade(d, tol_solved, tol_approx) for d in per_D]
    unsolved = [int(k) for k in range(T) if per_grade[k] == "unsolved"]

    if own:
        sim.close()

    meta = dict(clip.meta)
    meta["clip_id"] = f"{clip.clip_id}_act"
    meta["ref_action_semantics"] = "best_effort_pd_target"
    meta["ref_action_solver"] = {
        "method": "cem_per_frame",
        "weights": dict(DEFAULT_WEIGHTS if weights is None
                        else weights),
        "n_samples": n_samples, "n_iters": n_iters, "seed": seed,
        "tol_solved": tol_solved, "tol_approx": tol_approx,
        "per_frame_D": [float(d) for d in per_D],
        "per_frame_grade": per_grade,
        "unsolved_frames": unsolved,
        "note": "per-frame teleport solve; sequence-level replay is "
                "verified separately by `clip check` ref_action",
    }
    meta["source"] = {"type": "rederive_action", "parent": clip.clip_id}
    meta = record_lineage(meta, "rederive_action",
                          parent=clip.clip_id, method="cem_per_frame")
    return CoreClip({f: v.copy() for f, v in clip.states.items()},
                    meta, actions)
