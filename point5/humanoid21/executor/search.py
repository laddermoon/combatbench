"""E5: finite-horizon CEM search over normalized PD target sequences."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .contracts import ExecutorConfig, RefFrame
from .state import capture_current, restore
from .scoring import score_landed


def _meas_dict(cur) -> Dict[str, Any]:
    yaw_deg = cur.frame_pose[1]
    return {"p5_world": cur.p5_world, "com_pos": cur.com_pos,
            "com_vel": cur.com_vel,
            "yaw": np.deg2rad(yaw_deg),
            "torso_tilt": cur.torso_tilt, "contacts": cur.contacts}


def rollout_score(wsim, snap, seq, refs: List[RefFrame],
                  prev_action: np.ndarray, robot_id: str,
                  other_id: str, cfg: ExecutorConfig) -> Dict[str, Any]:
    """Score an action sequence by simulating landings per control step."""
    restore(wsim, snap)
    H = len(refs)
    total, details = 0.0, []
    prev_a = np.asarray(prev_action, np.float64)
    prev_pw = None
    for t in range(H):
        a = np.clip(seq[t], -cfg.action_clip, cfg.action_clip)
        wsim.set_action({robot_id: a.astype(np.float32),
                         other_id: np.zeros(21, np.float32)})
        for _ in range(cfg.phy_steps):
            wsim.physical_step()
        cur = capture_current(wsim, robot_id, with_snapshot=False)
        pv = np.zeros((5, 3)) if prev_pw is None else \
            (cur.p5_world - prev_pw) / cfg.dt
        prev_pw = cur.p5_world
        sc = score_landed(_meas_dict(cur), refs[t], pv, a, prev_a, cfg)
        total += (cfg.disc ** t) * sc["total"]
        details.append(sc)
        prev_a = a
        if sc["fallen"]:
            break
    return {"score": float(total), "details": details,
            "landed": prev_pw}


def search_window(wsim, snap, refs: List[RefFrame],
                  init_seq: np.ndarray, prev_action: np.ndarray,
                  standing_action: np.ndarray,
                  robot_id: str, other_id: str,
                  cfg: ExecutorConfig, seed: int = 0) -> Dict[str, Any]:
    """CEM over (H,21); always includes warm-start baselines."""
    rng = np.random.default_rng(seed)
    H = len(refs)
    mu = np.asarray(init_seq, np.float64)[:H]
    if len(mu) < H:
        pad = mu[-1] if len(mu) else np.asarray(prev_action, np.float64)
        mu = np.vstack([mu, np.tile(pad, (H - len(mu), 1))])
    mu = np.clip(mu, -1, 1)
    sigma = np.full((H, 21), cfg.sigma0)
    best = {"seq": mu.copy(), "score": np.inf, "details": None}
    baselines = [mu.copy(),
                 np.tile(np.asarray(prev_action, np.float64), (H, 1)),
                 np.tile(np.asarray(standing_action, np.float64), (H, 1))]
    for cand in baselines:
        r = rollout_score(wsim, snap, cand, refs, prev_action,
                          robot_id, other_id, cfg)
        if r["score"] < best["score"]:
            best.update(seq=cand, score=r["score"],
                        details=r["details"])
    for it in range(cfg.n_iters):
        cand = np.clip(rng.normal(mu, sigma,
                                  (cfg.n_samples, H, 21)), -1, 1)
        scores = []
        for c in cand:
            r = rollout_score(wsim, snap, c, refs, prev_action,
                              robot_id, other_id, cfg)
            scores.append(r["score"])
            if r["score"] < best["score"]:
                best.update(seq=c.copy(), score=r["score"],
                            details=r["details"])
        elite = cand[np.argsort(scores)[:cfg.n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
    return {"action": best["seq"][0].astype(np.float32),
            "seq": best["seq"], "score": float(best["score"]),
            "details": best["details"]}
