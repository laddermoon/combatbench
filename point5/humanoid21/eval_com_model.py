"""Evaluate P5 -> COM proxy models against real (MuJoCo) body-mass COM.

Ground truth: sample joint configs / planned punch keyframes, run the
mechanics Skeleton FK, place per-body masses at their inertial offsets
(extracted from battle_circular_v2.xml, robot _a) and take the mass
centroid.  The P5 (5 landmarks) of the same pose feeds the candidate
proxy models; report per-model errors.

    PYTHONPATH=. python3 point5/humanoid21/eval_com_model.py [--n 4000]
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import nnls

from point5.contracts import POINT_INDEX, P5Clip, P5, is_degenerate
from point5.humanoid21.mechanics import (
    JOINT_NAMES, LANDMARK_ORDER, Skeleton, norm_to_rad, solve_equilibrium)
from point5.humanoid21.stand_punch import (
    StandPunchConfig, plan_stand_punch)

# --- real inertial data: (mass, local com offset) per mechanics body name ---
_BODY_MASS: Dict[str, Tuple[float, np.ndarray]] = {
    "torso":           (5.8538, np.array([-0.0039, 0.0, -0.0464])),
    "head":            (3.0536, np.zeros(3)),
    "waist_lower":     (2.2619, np.zeros(3)),
    "pelvis":          (6.6162, np.array([-0.02, 0.0, 0.0])),
    "thigh_right":     (4.7518, np.array([0.0, 0.005, -0.17])),
    "shin_right":      (2.7557, np.array([0.0, 0.0, -0.15])),
    "foot_right":      (1.1311, np.array([0.035, 0.0, 0.0])),
    "thigh_left":      (4.7518, np.array([0.0, -0.005, -0.17])),
    "shin_left":       (2.7557, np.array([0.0, 0.0, -0.15])),
    "foot_left":       (1.1311, np.array([0.035, 0.0, 0.0])),
    "upper_arm_right": (1.6611, np.array([0.08, -0.08, -0.08])),
    "lower_arm_right": (0.9615, np.array([0.09, 0.09, 0.09])),
    "hand_right":      (0.2681, np.zeros(3)),
    "upper_arm_left":  (1.6611, np.array([0.08, 0.08, -0.08])),
    "lower_arm_left":  (0.9615, np.array([0.09, -0.09, 0.09])),
    "hand_left":       (0.2681, np.zeros(3)),
}
_TOTAL_MASS = sum(v[0] for v in _BODY_MASS.values())

_JIDX = {n: i for i, n in enumerate(JOINT_NAMES)}
_ARM_J = ["shoulder1_right", "shoulder2_right", "elbow_right",
          "shoulder1_left", "shoulder2_left", "elbow_left"]
_BODY_J = [n for n in JOINT_NAMES if n not in _ARM_J]


def real_com(sk: Skeleton, root_pos, root_quat, q_rad) -> np.ndarray:
    xf = sk.body_transforms(root_pos, root_quat, q_rad)
    c = np.zeros(3)
    for name, (p, R) in xf.items():
        m, ipos = _BODY_MASS[name]
        c += m * (p + R @ ipos)
    return c / _TOTAL_MASS


def _pose_tuple(sk, root_pos, root_quat, q_norm):
    q_rad = norm_to_rad(np.asarray(q_norm, dtype=np.float64))
    lm = sk.landmarks(root_pos, root_quat, q_rad)
    com = real_com(sk, root_pos, root_quat, q_rad)
    return lm, com, q_rad


# ---------------------------------------------------------------------------
# Dataset: sampled joint configs + solved punch keyframes
# ---------------------------------------------------------------------------

def _natural_ok(lm, xf) -> bool:
    fr = lm[POINT_INDEX["foot_right"]]
    fl = lm[POINT_INDEX["foot_left"]]
    h = lm[POINT_INDEX["head"]]
    mid = 0.5 * (fr + fl)
    if is_degenerate(lm):
        return False
    if not (0.0 <= min(fr[2], fl[2]) <= 0.35):
        return False
    if h[2] < mid[2] + 0.55 or h[2] < 0.55:
        return False
    pel = xf["pelvis"][0]
    if np.linalg.norm(pel[:2] - mid[:2]) > 0.45:
        return False
    torso_up = xf["torso"][1][2, 2]
    if torso_up < 0.3:                      # torso too tilted
        return False
    return True


def sample_dataset(rng, n_want: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(landmarks, real_com) pairs from sampled joint configs."""
    sk = Skeleton()
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    # per-group sampling std in normalized units
    std = {"abdomen": 0.45, "hip": 0.45, "knee": 0.45, "ankle": 0.30,
           "shoulder1": 0.55, "shoulder2": 0.55, "elbow": 0.55}
    tries = 0
    while len(out) < n_want and tries < n_want * 60:
        tries += 1
        q = np.zeros(len(JOINT_NAMES))
        for j, name in enumerate(JOINT_NAMES):
            grp = next(g for g in std if name.startswith(g))
            q[j] = np.clip(rng.normal(0.0, std[grp]), -1.0, 1.0)
        # keep feet near the ground: place root so the lowest ankle
        # sits at its standing height (~0.087)
        q_rad = norm_to_rad(q)
        lm0 = sk.landmarks(np.zeros(3), np.array([1, 0, 0, 0]), q_rad)
        root_pos = np.array([0.0, 0.0, 0.087 - lm0[3:, 2].min()])
        if not (0.55 < root_pos[2] < 1.75):
            continue
        root_quat = np.array([1.0, 0, 0, 0])
        lm, com, _ = _pose_tuple(sk, root_pos, root_quat, q)
        xf = sk.body_transforms(root_pos, root_quat, q_rad)
        if _natural_ok(lm, xf):
            out.append((lm, com))
    return out


def punch_dataset() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Solved stand-punch keyframes -> (landmarks, real_com)."""
    sk = Skeleton()
    out = []
    cfg = StandPunchConfig()
    for hand in ("left", "right"):
        for tx in np.linspace(0.95, 1.30, 4):
            for ty in np.linspace(-0.30, 0.30, 3):
                for tz in np.linspace(0.75, 1.35, 3):
                    plan = plan_stand_punch(
                        np.array([tx, ty, tz]), hand, cfg)
                    if not plan.reachable or plan.task is None:
                        continue
                    for kf in plan.task.keyframes:
                        res = solve_equilibrium(kf.pos, skeleton=sk)
                        lm = np.asarray(res["landmarks"])
                        com = real_com(sk, res["root_pos"],
                                       res["root_quat_wxyz"], res["q_rad"])
                        out.append((lm, com))
    return out


def natural_dataset(rng, n_want: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Natural poses: perturbed standing P5 targets -> solve_equilibrium.

    Only satisfied solves count; the achieved landmarks (not the target)
    pair with the real COM, matching the eval criterion.
    """
    from point5.humanoid21.stand_punch import stand_p5
    sk = Skeleton()
    base = stand_p5()
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    tries = 0
    while len(out) < n_want and tries < n_want * 40:
        tries += 1
        tgt = base.copy()
        tgt[POINT_INDEX["head"]] += np.array(
            [rng.uniform(-0.05, 0.15), rng.uniform(-0.08, 0.08),
             rng.uniform(-0.22, 0.02)])
        for hn in ("hand_right", "hand_left"):
            tgt[POINT_INDEX[hn]] += np.array(
                [rng.uniform(-0.15, 0.25), rng.uniform(-0.10, 0.10),
                 rng.uniform(-0.20, 0.20)])
        for fn in ("foot_right", "foot_left"):
            tgt[POINT_INDEX[fn]] += np.array(
                [rng.uniform(-0.10, 0.15), rng.uniform(-0.06, 0.06),
                 rng.uniform(-0.02, 0.03)])
        if is_degenerate(tgt):
            continue
        res = solve_equilibrium(tgt, skeleton=sk, max_iter=300)
        if not res["satisfied"]:
            continue
        lm = np.asarray(res["landmarks"])
        com = real_com(sk, res["root_pos"], res["root_quat_wxyz"],
                       res["q_rad"])
        out.append((lm, com))
    return out


# ---------------------------------------------------------------------------
# Candidate COM proxy families (model-free: P5 in, com out)
# ---------------------------------------------------------------------------

def _features_v0(p5: np.ndarray) -> np.ndarray:
    """Basis points for the convex-mixture family: the 5 points themselves."""
    return p5                                     # rows = 5 point positions


def _features_v1(p5: np.ndarray, alpha: float, beta: float):
    """Torso-axis segment model.

    Anchors on the ankle-mid -> head axis:  P = F + a(H-F), S = F + b(H-F).
    Segment mass centres (uniform-line assumption):
        trunk (P,S): (P+S)/2          head: H
        armL (S,AL): (S+AL)/2         armR (S,AR): (S+AR)/2
        legL (P,FL): (P+FL)/2         legR (P,FR): (P+FR)/2
    """
    h = p5[POINT_INDEX["head"]]
    ar = p5[POINT_INDEX["hand_right"]]
    al = p5[POINT_INDEX["hand_left"]]
    fr = p5[POINT_INDEX["foot_right"]]
    fl = p5[POINT_INDEX["foot_left"]]
    F = 0.5 * (fr + fl)
    P = F + alpha * (h - F)
    S = F + beta * (h - F)
    return np.stack([0.5 * (P + S), h, 0.5 * (S + al), 0.5 * (S + ar),
                     0.5 * (P + fl), 0.5 * (P + fr)])


def _fit_weights(basis: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1: min ||sum w_i b_i - gt||.

    Solved by augmenting NNLS with a sum-to-one equality row scaled up.
    """
    A = basis.copy()                              # (N, K, 3)
    K = A.shape[1]
    # stack: each coord is an independent target; weights shared
    X = np.concatenate([A[:, :, 0], A[:, :, 1], A[:, :, 2]], axis=0)
    y = np.concatenate([gt[:, 0], gt[:, 1], gt[:, 2]])
    # encode sum(w)=1 as a big-weight row
    lam = 10.0 * np.sqrt(np.sum(y ** 2) / len(y) + 1.0)
    X = np.vstack([X, lam * np.ones(K)])
    y = np.append(y, lam)
    w, _ = nnls(X, y)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _eval_weights(basis: np.ndarray, gt: np.ndarray, w: np.ndarray):
    est = np.einsum("k,nkc->nc", w, basis)
    err = np.linalg.norm(est - gt, axis=1)
    return err


# ---------------------------------------------------------------------------
# Family v2: virtual-body reconstruction (closed-form, model-free)
#
#   anchors:  P = pelvis anchor on/near the head-foot axis
#             S = shoulder anchor near the head
#   knees/elbows recovered by two-link closure with a fixed bend prior:
#       mid = A + along*x + e*h   (e = bend prior ⊥ along)
#   segment mass centres are midpoints of the reconstructed bones.
# ---------------------------------------------------------------------------

# generic humanoid bone lengths (m) — defaults, fitted offsets applied on top
_LEG_UPPER = 0.42          # hip -> knee
_LEG_LOWER = 0.37          # knee -> ankle
_ARM_UPPER = 0.31          # shoulder -> elbow
_ARM_LOWER = 0.31          # elbow -> hand


def _perp_unit(v: np.ndarray, along: np.ndarray,
               fallback: np.ndarray) -> np.ndarray:
    e = np.asarray(v, dtype=np.float64)
    e = e - float(e @ along) * along
    n = float(np.linalg.norm(e))
    if n < 1e-9:
        e = np.asarray(fallback, dtype=np.float64)
        e = e - float(e @ along) * along
        n = float(np.linalg.norm(e))
        if n < 1e-9:
            return np.zeros(3)
    return e / n


def _two_link_mid(A: np.ndarray, T: np.ndarray, l1: float, l2: float,
                  bend_dir: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Closed-form elbow/knee position between anchor A and target T."""
    d = T - A
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return A.copy()
    along = d / dist
    dc = min(dist, l1 + l2 - 1e-6)
    x = (l1 * l1 - l2 * l2 + dc * dc) / (2.0 * dc)
    h = float(np.sqrt(max(l1 * l1 - x * x, 0.0)))
    e = _perp_unit(bend_dir, along, fallback)
    return A + along * x + e * h


def _features_v2(p5: np.ndarray, l_torso: float, lean: float,
                 elbow_mode: str):
    """Reconstructed segment midpoints.

    Anchors (column model):
      pelvis  P = over the ankle mid, L_torso below the head
      torso   direction n = (H - P)/|H-P|
      shoulder S = P + 0.78 (H - P)
    knees/elbows closed by two-link IK with fixed bend priors.

    Returns 15 segment mass centres:
    pelvis/abdomen/chest (trunk split x3), head,
    upper_arm l/r, lower_arm l/r,
    thigh l/r, shin l/r, foot l/r.
    """
    h = p5[POINT_INDEX["head"]]
    ar = p5[POINT_INDEX["hand_right"]]
    al = p5[POINT_INDEX["hand_left"]]
    fr = p5[POINT_INDEX["foot_right"]]
    fl = p5[POINT_INDEX["foot_left"]]
    F = 0.5 * (fr + fl)
    up = np.array([0.0, 0.0, 1.0])

    P = np.array([F[0] + lean * (h[0] - F[0]),
                  F[1] + lean * (h[1] - F[1]),
                  h[2] - l_torso])
    d = h - P
    u = float(np.linalg.norm(d))
    n = d / u if u > 1e-9 else up
    left = _perp_unit(np.array([0.0, 1.0, 0.0]), n,
                      np.array([1.0, 0.0, 0.0]))
    fwd = _perp_unit(np.array([1.0, 0.0, 0.0]), n, up)

    S = h - 0.13 * n                             # shoulder line below head

    hip_l = P + left * 0.10
    hip_r = P - left * 0.10
    sho_l = S + left * 0.17
    sho_r = S - left * 0.17

    knee_l = _two_link_mid(hip_l, fl, _LEG_UPPER, _LEG_LOWER, fwd, up)
    knee_r = _two_link_mid(hip_r, fr, _LEG_UPPER, _LEG_LOWER, fwd, up)
    if elbow_mode == "down":
        e_l = e_r = -up
    elif elbow_mode == "down_back":
        e_l = e_r = -up - 0.5 * fwd
    elif elbow_mode == "out":
        e_l, e_r = left, -left
    else:                                        # "down_out"
        e_l, e_r = -up + 0.7 * left, -up - 0.7 * left
    if elbow_mode == "none":                     # straight arm approximation
        elb_l = 0.5 * (sho_l + al)
        elb_r = 0.5 * (sho_r + ar)
    else:
        elb_l = _two_link_mid(sho_l, al, _ARM_UPPER, _ARM_LOWER, e_l, fwd)
        elb_r = _two_link_mid(sho_r, ar, _ARM_UPPER, _ARM_LOWER, e_r, fwd)

    toes_l = fl + fwd * 0.07                     # foot extends forward
    toes_r = fr + fwd * 0.07
    # trunk anchors at real mass heights along the pelvis->head axis:
    # pelvis block at P, waist at P+0.17n, chest com at P+0.38n
    return np.stack([
        P, P + 0.17 * n, P + 0.38 * n,
        h,
        0.5 * (sho_l + elb_l), 0.5 * (elb_l + al),
        0.5 * (sho_r + elb_r), 0.5 * (elb_r + ar),
        0.5 * (hip_l + knee_l), 0.5 * (knee_l + fl),
        0.5 * (hip_r + knee_r), 0.5 * (knee_r + fr),
        0.5 * (fl + toes_l), 0.5 * (fr + toes_r),
    ])


def fit_family(basis_fn, p5_all: np.ndarray, gt_all: np.ndarray,
               grid: List[Tuple[float, float]]):
    best = None
    for a, b in grid:
        basis = np.stack([basis_fn(p, a, b) for p in p5_all])
        w = _fit_weights(basis, gt_all)
        err = _eval_weights(basis, gt_all, w)
        score = float(np.sqrt(np.mean(err ** 2)))
        if best is None or score < best[0]:
            best = (score, a, b, w)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print(f"[data] sampling {a.n} joint poses ...")
    samp = sample_dataset(rng, a.n)
    print(f"[data] got {len(samp)} sampled poses")
    print("[data] solving natural targets (equilibrium) ...")
    nat = natural_dataset(rng, max(300, a.n // 3))
    print(f"[data] got {len(nat)} natural solved poses")
    punch = punch_dataset()
    print(f"[data] got {len(punch)} punch keyframe poses "
          f"(solved, incl. unsatisfied best-effort)")

    def pack(ds):
        return (np.stack([d[0] for d in ds]),
                np.stack([d[1] for d in ds]))

    p5_s, gt_s = pack(samp)
    p5_n, gt_n = pack(nat) if nat else (None, None)
    p5_p, gt_p = pack(punch) if punch else (None, None)

    SETS = (("sampled", (p5_s, gt_s)), ("natural", (p5_n, gt_n)),
            ("punch", (p5_p, gt_p)))

    # ---- family v0: convex mixture of the 5 points ----------------------
    basis0 = np.stack([_features_v0(p) for p in p5_s])
    w0 = _fit_weights(basis0, gt_s)
    for name, (p5, gt) in SETS:
        if p5 is None:
            continue
        basis = np.stack([_features_v0(p) for p in p5])
        err = _eval_weights(basis, gt, w0)
        de = np.einsum("k,nkc->nc", w0, basis) - gt
        print(f"[v0/{name}] w={np.round(w0,3)} "
              f"rmse={np.sqrt(np.mean(err**2))*100:.2f}cm "
              f"mean={err.mean()*100:.2f}cm p90={np.percentile(err,90)*100:.2f}cm "
              f"max={err.max()*100:.2f}cm bias={np.round(de.mean(0)*100,1)}cm")

    # ---- family v1: torso-axis segments --------------------------------
    grid = [(a, b) for a in np.arange(0.0, 0.61, 0.05)
            for b in np.arange(0.55, 1.01, 0.05) if b > a]
    best = fit_family(_features_v1, p5_s, gt_s, grid)
    score, a_best, b_best, w1 = best
    print(f"[v1/fit] alpha={a_best:.2f} beta={b_best:.2f} "
          f"w={np.round(w1,3)} train_rmse={score*100:.2f}cm")
    for name, (p5, gt) in SETS:
        if p5 is None:
            continue
        basis = np.stack([_features_v1(p, a_best, b_best) for p in p5])
        est = np.einsum("k,nkc->nc", w1, basis)
        err = np.linalg.norm(est - gt, axis=1)
        de = est - gt
        print(f"[v1/{name}] rmse={np.sqrt(np.mean(err**2))*100:.2f}cm "
              f"mean={err.mean()*100:.2f}cm p90={np.percentile(err,90)*100:.2f}cm "
              f"max={err.max()*100:.2f}cm bias={np.round(de.mean(0)*100,1)}cm")

    # ---- family v2: virtual-body reconstruction -------------------------
    grid2 = [(lt, ln, em)
             for lt in np.arange(0.55, 0.66, 0.02)
             for ln in (0.0, 0.15, 0.30)
             for em in ("down", "down_back", "down_out")]
    best2 = None
    for lt, ln, em in grid2:
        basis = np.stack([_features_v2(p, lt, ln, em) for p in p5_s])
        w2 = _fit_weights(basis, gt_s)
        err = _eval_weights(basis, gt_s, w2)
        score2 = float(np.sqrt(np.mean(err ** 2)))
        if best2 is None or score2 < best2[0]:
            best2 = (score2, lt, ln, em, w2)
    score2, lt, ln, em, w2 = best2
    seg_names = ["pelvis", "abdomen", "chest", "head",
                 "uarm_l", "larm_l", "uarm_r", "larm_r",
                 "thigh_l", "shin_l", "thigh_r", "shin_r",
                 "foot_l", "foot_r"]
    print(f"[v2/fit] L_torso={lt:.2f} lean={ln:.2f} elbow={em} "
          f"train_rmse={score2*100:.2f}cm")
    print("         masses: " + ", ".join(
        f"{n}={w:.3f}" for n, w in zip(seg_names, w2) if w > 0.005))
    for name, (p5, gt) in SETS:
        if p5 is None:
            continue
        basis = np.stack([_features_v2(p, lt, ln, em) for p in p5])
        est = np.einsum("k,nkc->nc", w2, basis)
        err = np.linalg.norm(est - gt, axis=1)
        de = est - gt
        print(f"[v2/{name}] rmse={np.sqrt(np.mean(err**2))*100:.2f}cm "
              f"mean={err.mean()*100:.2f}cm p90={np.percentile(err,90)*100:.2f}cm "
              f"max={err.max()*100:.2f}cm bias={np.round(de.mean(0)*100,1)}cm")

    # ---- family v3: shipped proxy (point5.mass_model defaults) ----------
    from point5.mass_model import mass_proxy
    for name, (p5, gt) in SETS:
        if p5 is None:
            continue
        est = np.stack([mass_proxy(p).com for p in p5])
        de = est - gt
        err = np.linalg.norm(de, axis=1)
        print(f"[v3/{name}] rmse={np.sqrt(np.mean(err**2))*100:.2f}cm "
              f"mean={err.mean()*100:.2f}cm p90={np.percentile(err,90)*100:.2f}cm "
              f"max={err.max()*100:.2f}cm bias={np.round(de.mean(0)*100,1)}cm "
              f"axis_rmse={np.round(np.sqrt((de**2).mean(0))*100,1)}cm")


if __name__ == "__main__":
    main()
