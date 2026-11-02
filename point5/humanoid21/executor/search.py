"""Executor scoring + finite-horizon simulation search (big-logic MPC)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from point5.contracts import POINT_INDEX
from .contracts import ExecutorConfig, RefFrame, PHY_STEPS
from .state import foot_contacts, torso_tilt, restore

FOOT_IDX = {"right": POINT_INDEX["foot_right"],
            "left": POINT_INDEX["foot_left"]}
HAND_IDX = (POINT_INDEX["hand_right"], POINT_INDEX["hand_left"])


def _world_p5(p5) -> np.ndarray:
    from baseline.humanoid21.hierarchy.p5_world import (
        f1_from_state, p5_to_world)
    return p5_to_world(p5.pos, f1_from_state(p5))


def score_state(world_p5: np.ndarray, com_pos: np.ndarray,
                com_vel: np.ndarray, yaw_act: float, tilt: float,
                head_z: float, contacts: Dict[str, float],
                ref: RefFrame, action: np.ndarray,
                prev_action: np.ndarray, cfg: ExecutorConfig) -> Dict:
    """Score one landed state against a RefFrame (lower is better)."""
    c = {}
    perr = np.linalg.norm(world_p5 - ref.p5_world, axis=1)
    over = np.maximum(0.0, perr - ref.tol_point)
    c["point"] = float(over.sum() + 0.25 * perr.sum())
    c["com"] = float(max(0.0, np.linalg.norm(com_pos - ref.com_pos)
                         - cfg.tol_com))
    c["com_vel"] = float(np.linalg.norm(com_vel - ref.com_vel))
    dy = (yaw_act - ref.yaw + np.pi) % (2 * np.pi) - np.pi
    c["yaw"] = float(max(0.0, abs(dy) - cfg.tol_yaw))
    slip = 0.0
    for side, planted in (("right", ref.support[0]),
                          ("left", ref.support[1])):
        if planted:
            i = FOOT_IDX[side]
            v = float(np.linalg.norm(
                world_p5[i] - ref.p5_world[i]) / cfg.dt)
            slip += max(0.0, v - cfg.slip_v)
    c["slip"] = slip
    c["contact"] = float(contacts.get("extra_contacts", 0))
    c["action"] = float(np.mean(action ** 2))
    c["delta"] = float(np.mean((action - prev_action) ** 2))
    total = (cfg.w_point * c["point"] + cfg.w_com * c["com"]
             + cfg.w_com_vel * c["com_vel"] * 0.1 + cfg.w_yaw * c["yaw"]
             + cfg.w_support_slip * c["slip"]
             + cfg.w_extra_contact * c["contact"]
             + cfg.w_action * c["action"] + cfg.w_action_delta * c["delta"])
    fallen = tilt > cfg.tilt_fail or head_z < cfg.head_min_z
    if fallen:
        total += 50.0
    return {"total": float(total), "components": c, "fallen": fallen}


def _measure(sim, robot_id: str, p5_prev_w, dt, cfg):
    """Measure actual world p5, com, yaw, tilt, contacts after a step."""
    from point5.humanoid21.adapter import point5_from_view
    from baseline.humanoid21.hierarchy.p5_world import (
        f1_from_state, p5_to_world)
    from .state import com_state
    ds = sim.get_derived_state([robot_id])[robot_id]
    p5 = point5_from_view(ds, robot_id, with_vel=False)
    f1 = f1_from_state(p5)
    pw = p5_to_world(p5.pos, f1)
    core = sim.get_core_state()[robot_id]
    yaw = np.deg2rad(float(p5.metadata["frame_yaw_deg"]))
    cst = com_state(sim, robot_id)
    con = foot_contacts(sim, robot_id)
    head_z = float(pw[POINT_INDEX["head"], 2])
    tilt = torso_tilt(core)
    return {"p5w": pw, "com": np.asarray(cst.com_pos),
            "com_vel": np.asarray(cst.lin_mom) / cst.mass,
            "yaw": yaw, "tilt": tilt, "head_z": head_z,
            "contacts": con, "f1": f1, "p5": p5}


def search_window(wsim, snap, robot_id: str, other_id: str,
                  refs: List[RefFrame], init_seq: np.ndarray,
                  prev_action: np.ndarray, cfg: ExecutorConfig,
                  seed: int = 0) -> Dict[str, Any]:
    """CEM over an H-step action sequence; returns best first action.

    Candidates are scored by discounted per-step costs measured on
    real simulated landings.  Always includes init_seq + previous
    seq + hold-at-current as baseline candidates.
    """
    rng = np.random.default_rng(seed)
    H = min(len(refs), cfg.horizon)
    other = other_id
    mu = np.asarray(init_seq[:H], dtype=np.float64)
    if len(mu) < H:
        mu = np.vstack([mu, np.tile(mu[-1], (H - len(mu), 1))])
    mu = np.clip(mu, -cfg.action_clip, cfg.action_clip)
    sigma = np.full((H, 21), cfg.sigma0)
    disc = np.power(cfg.disc, np.arange(H))

    def roll(seq):
        restore(wsim, snap)
        tot, details = 0.0, []
        prev_a = prev_action
        for t in range(H):
            wsim.set_action({robot_id: seq[t].astype(np.float32),
                             other: np.zeros(21, np.float32)})
            for _ in range(PHY_STEPS):
                wsim.physical_step()
            m = _measure(wsim, robot_id, None, cfg.dt, cfg)
            sc = score_state(m["p5w"], m["com"], m["com_vel"], m["yaw"],
                             m["tilt"], m["head_z"], m["contacts"],
                             refs[t], seq[t], prev_a, cfg)
            tot += disc[t] * sc["total"]
            details.append(sc)
            prev_a = seq[t]
            if sc["fallen"]:
                break
        return tot, details

    best = {"seq": mu.copy(), "score": np.inf, "details": None}
    cands = [mu.copy(), np.tile(prev_action, (H, 1)),
             np.zeros((H, 21))]
    for c in cands:
        sc, dt_ = roll(np.clip(c, -1, 1))
        if sc < best["score"]:
            best.update(seq=c.copy(), score=sc, details=dt_)
    for it in range(cfg.n_iters):
        cand = np.clip(rng.normal(mu, sigma, (cfg.n_samples, H, 21)),
                       -1, 1)
        for c in cand:
            sc, dt_ = roll(c)
            if sc < best["score"]:
                best.update(seq=c.copy(), score=sc, details=dt_)
        elite = np.array([best["seq"]])
        mu = 0.7 * mu + 0.3 * elite[0]
        sigma = np.maximum(0.5 * sigma, 0.05)
    return {"action": best["seq"][0].astype(np.float32),
            "seq": best["seq"], "score": float(best["score"]),
            "details": best["details"]}
