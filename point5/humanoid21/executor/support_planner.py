"""Support planning (algorithm-1 branch).

Deepened requirements (user-driven, v1):

1. A Footstep is a LANDING only — mid-air swing waypoints
   (lifted z in the clip) are not support points and emit nothing.
2. The physical gate is the LIP capture point xi = c + v/w against
   the CURRENT support hull — no velocity thresholds.
3. After a corrective step the virtual state is RE-ANCHORED:
   the bracing foot moves, momentum is considered braked, and the
   same segment is rechecked before any next step is emitted.
4. Corrective foot choice prefers the foot that was going to move
   anyway (near-tie) — least damage to clip intent.
5. Momentum that agrees with the intent needs no corrective: if
   the mover's own landing produces a hull containing xi, the move
   itself is the capture step.
6. A move step's validity assumes the clip's support geometry —
   if a corrective displaced a clip-required planted foot, a
   recover step returns it BEFORE the mover lifts.
7. The plan's terminal stance must satisfy the clip's last
   keyframe; residual displacement gets a recover step.

Two layers:

1. ``support_transitions`` — which segments need weight transfer
   (foot displacement between consecutive keyframes in a shared
   frame).  Detection, not planning.

2. ``plan_footsteps`` — the support-planning sub-algorithm per
   bootstrip: from the start LandMark + clip emit an ordered
   Footstep list.  Physics rule: if COM momentum points away from
   where the next support must go, insert a corrective step
   (垫步) on the support side to brake/catch the momentum before
   the intended mover lifts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from point5.contracts import compose_frame, to_world

from .cop_solver import FLAT_Z as LIFT_Z

_FIDX = {"right": 3, "left": 4}
_STEP_MAX = 0.35          # prior: nominal max single-step length (m)
_G = 9.81                 # LIP: capture point xi = c + v/w, w=sqrt(g/z0)


# ---------- detection -------------------------------------------------------
@dataclass
class SupportTransition:
    """One segment that needs support planning."""

    seg: int
    mover: str
    support: str
    mover_dxy: float
    mover_dz: float


def support_transitions(clip, xy_eps: float = 0.05,
                        lift_eps: float = 0.06
                        ) -> Optional[List[SupportTransition]]:
    """Segments requiring support planning, or None."""
    out = []
    kfs = clip.keyframes
    for k in range(1, len(kfs)):
        prev, cur = kfs[k - 1], kfs[k]
        tr = cur.transfer
        cur_in_prev = to_world(cur.pos, [tr.dx, tr.dy], tr.dyaw)
        movers = []
        for side, fi in (("right", 3), ("left", 4)):
            d = cur_in_prev[fi] - prev.pos[fi]
            if (np.linalg.norm(d[:2]) > xy_eps or d[2] > lift_eps):
                movers.append((side, float(np.linalg.norm(d[:2])),
                               float(d[2])))
        if movers:
            movers.sort(key=lambda t: -t[1])
            m = movers[0][0]
            out.append(SupportTransition(
                seg=k, mover=m,
                support="left" if m == "right" else "right",
                mover_dxy=movers[0][1], mover_dz=movers[0][2]))
    return out or None


# ---------- footstep sub-algorithm (bootstrip) ------------------------------
@dataclass
class Footstep:
    """One planned step: which foot goes to which xy point."""

    foot: str
    target: np.ndarray                  # (2,) local frame xy
    kind: str = "move"                  # 'move' | 'corrective'
    reason: str = ""


def _foot_patches(feet) -> List[np.ndarray]:
    """Contact patches (4,2 each) of loaded feet."""
    from .capability_solver import _PATCH
    out = []
    for f in feet.values():
        if f.force_ratio > 0.05 and f.foot_pos is not None:
            out.append(np.asarray(f.foot_pos) + np.asarray(_PATCH))
    return out


def _inside(pt, poly) -> bool:
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        e = b - a
        if e[0] * (pt - a)[1] - e[1] * (pt - a)[0] < -1e-9:
            return False
    return True


def _outside_polygon(xi, feet) -> bool:
    # support polygon = CONVEX HULL of all contact patches
    pts = np.vstack(_foot_patches(feet)) if _foot_patches(feet) \
        else np.empty((0, 2))
    if len(pts) == 0:
        return False
    from .capability_solver import _hull2d
    return not _inside(xi, _hull2d(pts))


def _nearest_foot(xi, feet) -> str:
    return min(feet, key=lambda s: np.linalg.norm(
        np.asarray(feet[s].foot_pos) - xi))


def plan_footsteps(start_landmark, clip,
                   step_max: float = _STEP_MAX,
                   max_correction: int = 2) -> List[Footstep]:
    """Start LandMark + clip -> ordered footsteps incl. correction.

    Per transition segment the mover foot steps to the keyframe's
    foot xy (frame0).  Before each transition, physics gate: the
    LIP capture point xi = c + v/w must sit inside the CURRENT
    support hull — if not, emit a corrective capture step and
    RE-ANCHOR the virtual state (foot moved, momentum braked)
    before rechecking.  Corrective foot choice prefers the foot
    that was going to move anyway (least damage to clip intent).
    """
    steps: List[Footstep] = []
    trans = support_transitions(clip) or []
    if not trans:
        return steps

    poses = [(np.zeros(2), 0.0)]
    for k in range(1, len(clip.keyframes)):
        poses.append(compose_frame(*poses[-1],
                                   clip.keyframes[k].transfer))

    # virtual current state — updated as the plan emits steps
    com = np.asarray(start_landmark.com, np.float64)
    vel = (np.asarray(start_landmark.vel, np.float64)
           if start_landmark.vel is not None else np.zeros(3))
    feet_xy = {s: np.asarray(
        start_landmark.feet[s].foot_pos, np.float64).copy()
        for s in ("right", "left")}
    loaded = {s: start_landmark.feet[s].force_ratio > 0.05
              for s in ("right", "left")}

    def hull():
        from .capability_solver import _PATCH, _hull2d
        pts = [feet_xy[s] + np.asarray(_PATCH)
               for s in loaded if s and feet_xy[s] is not None]
        return _hull2d(np.vstack(pts)) if pts else None

    for t in trans:
        sup = t.support
        tgt_xy = to_world(
            clip.keyframes[t.seg].pos[
                _FIDX[t.mover]:_FIDX[t.mover] + 1],
            *poses[t.seg])[0, :2]

        # ---- physical gate: is current momentum catchable? ----
        from .capability_solver import _PATCH as _PATCH_g
        _PATCH = _PATCH_g
        n_fix = 0
        while n_fix < max_correction:
            z0 = max(com[2], 0.1)
            xi = com[:2] + vel[:2] / np.sqrt(_G / z0)
            h = hull()
            if h is not None and _inside(xi, h):
                break
            # momentum aligned with the intent needs no brace: if
            # the mover's own landing creates a hull containing xi,
            # the move itself IS the capture step
            probe = dict(feet_xy)
            probe[t.mover] = tgt_xy
            pts = [probe[s2] + np.asarray(_PATCH)
                   for s2 in probe if probe[s2] is not None]
            from .capability_solver import _hull2d
            if _inside(xi, _hull2d(np.vstack(pts))):
                break
            # capture step on the foot nearest xi; on a near-tie
            # prefer the mover (going to move anyway = least damage)
            cand = sorted(
                (s for s in feet_xy if feet_xy[s] is not None),
                key=lambda s: np.linalg.norm(feet_xy[s] - xi))
            foot = cand[0]
            if (t.mover in cand[1:]
                    and np.linalg.norm(feet_xy[t.mover] - xi)
                    < np.linalg.norm(feet_xy[foot] - xi) + 0.05):
                foot = t.mover
            d = xi - feet_xy[foot]
            n = np.linalg.norm(d)
            tgt = feet_xy[foot] + (d / n) * min(n, step_max)
            steps.append(Footstep(
                foot=foot, target=tgt, kind="corrective",
                reason=(f"capture overshoot {n:.2f}m -> {foot} "
                        f"braces at {np.round(xi, 2)}")))
            # re-anchor virtual state: foot moved, momentum braked
            feet_xy[foot] = tgt
            loaded[foot] = True
            vel = np.zeros(3)
            n_fix += 1

        # if a corrective displaced the SUPPORT foot this segment
        # expects, return it BEFORE the mover lifts — the move's
        # validity assumes the clip's support geometry
        sup_need = to_world(
            clip.keyframes[t.seg - 1].pos[
                _FIDX[sup]:_FIDX[sup] + 1],
            *poses[t.seg - 1])[0, :2]
        if np.linalg.norm(feet_xy[sup] - sup_need) > 0.05:
            steps.append(Footstep(
                foot=sup, target=sup_need, kind="recover",
                reason="restore support for clip intent"))
            feet_xy[sup] = sup_need

        # mover target only becomes a footstep if that keyframe's
        # foot is GROUNDED — a lifted z is a mid-air swing waypoint,
        # not a support point
        m_z = clip.keyframes[t.seg].pos[_FIDX[t.mover], 2]
        grounded = m_z <= LIFT_Z
        if not grounded:
            continue
        reach = np.linalg.norm(tgt_xy - feet_xy[sup])
        reason = f"seg{t.seg} mover"
        if reach > 2 * step_max:
            reason += f" [suspect: {reach:.2f}m from {sup}]"
        steps.append(Footstep(foot=t.mover, target=tgt_xy,
                              kind="move", reason=reason))
        feet_xy[t.mover] = tgt_xy
        loaded[t.mover] = True

    # recover: corrective steps may have displaced a foot the clip
    # requires planted — return it to the clip's final stance once
    # double support has resumed
    last = clip.keyframes[-1]
    w_last = to_world(last.pos, *poses[-1])
    for side in ("right", "left"):
        if last.pos[_FIDX[side], 2] > LIFT_Z:
            continue                        # clip doesn't want it down
        tgt = w_last[_FIDX[side], :2]
        if np.linalg.norm(feet_xy[side] - tgt) > 0.05:
            steps.append(Footstep(
                foot=side, target=tgt, kind="recover",
                reason="restore clip-required stance"))
            feet_xy[side] = tgt
    return steps




# ---------- scenario visualization (manual check) ---------------------------
if __name__ == "__main__":
    """Top-down scenario renders: foot patches, COM, capture point,
    planned footsteps.

        PYTHONPATH=. python3 -m \
            point5.humanoid21.executor.support_planner [outdir]
    """
    import sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MpPoly

    from point5.contracts import Keyframe, P5Clip, Transfer
    from point5.humanoid21.executor.com_solver import com_estimate
    from point5.humanoid21.executor.contracts import (FootSupport,
                                                      LandMark)
    from point5.humanoid21.stand_punch import (plan_stand_punch,
                                               stand_p5)

    outdir = Path(sys.argv[1] if len(sys.argv) > 1
                  else "/data1/dev/combatbench_executor/support_scen")
    outdir.mkdir(parents=True, exist_ok=True)

    P = stand_p5()

    def mk(com=(0.03, 0.0, 0.87), vel=(0, 0, 0), fr=0.5, fl=0.5):
        return LandMark(
            com=np.asarray(com, float), vel=np.asarray(vel, float),
            feet={"right": FootSupport(fr, P[3, :2].copy(),
                                       P[3, :2].copy()),
                  "left": FootSupport(fl, P[4, :2].copy(),
                                      P[4, :2].copy())})

    def punch():
        plan = plan_stand_punch(np.array([1.05, 0, 1.15]), "left")
        raw = list(plan.task.keyframes)
        return P5Clip(name="p", keyframes=tuple(
            [Keyframe(pos=stand_p5()),
             Keyframe(pos=raw[0].pos,
                      transfer=Transfer(0, 0, 0, 0.5))] + raw[1:]))

    def shift():
        p2 = P.copy()
        p2[3] += np.array([0, -0.3, 0])
        return P5Clip(name="s", keyframes=(
            Keyframe(pos=P),
            Keyframe(pos=p2, transfer=Transfer(0, 0, 0, 1.0))))

    def punch_r():
        plan = plan_stand_punch(np.array([1.05, 0, 1.15]), "right")
        raw = list(plan.task.keyframes)
        return P5Clip(name="pr", keyframes=tuple(
            [Keyframe(pos=stand_p5()),
             Keyframe(pos=raw[0].pos,
                      transfer=Transfer(0, 0, 0, 0.5))] + raw[1:]))

    def fwd_step():
        p2 = P.copy()
        p2[4] += np.array([0.3, 0, 0])          # left foot fwd
        return P5Clip(name="f", keyframes=(
            Keyframe(pos=P),
            Keyframe(pos=p2, transfer=Transfer(0, 0, 0, 0.8))))

    def back_step():
        p2 = P.copy()
        p2[3] += np.array([-0.3, 0, 0])         # right foot back
        return P5Clip(name="b", keyframes=(
            Keyframe(pos=P),
            Keyframe(pos=p2, transfer=Transfer(0, 0, 0, 0.8))))

    def render(name, lm, clip):
        steps = plan_footsteps(lm, clip)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_aspect("equal")
        placed = []                       # occupied label positions

        def _lab(pos, text, color):
            """Place a label clear of existing labels/markers,
            with a leader line back to pos."""
            dirs = [(0.03, 0.02), (0.03, -0.02), (-0.03, 0.02),
                    (-0.03, -0.02), (0.0, 0.04), (0.0, -0.04),
                    (0.05, 0.0), (-0.05, 0.0), (0.05, 0.04),
                    (-0.05, 0.04), (0.05, -0.04), (-0.05, -0.04),
                    (0.0, 0.07), (0.0, -0.07), (0.08, 0.0),
                    (-0.08, 0.0)]
            for d in dirs:
                lp = np.asarray(pos, float) + np.asarray(d)
                if all(np.linalg.norm(lp - q) > 0.035 for q in placed):
                    placed.append(lp)
                    ax.annotate(text, pos, xytext=lp, fontsize=9,
                                color=color,
                                arrowprops=dict(arrowstyle="-",
                                                color=color,
                                                lw=0.6, alpha=0.6))
                    return
            ax.annotate(text, pos, xytext=np.asarray(pos)
                        + [0.06, 0.06], fontsize=9, color=color)

        def _ring(pos, colors, edge):
            """Overlapping markers -> concentric rings, biggest out."""
            for j, c in enumerate(colors):
                ax.scatter(*pos, s=380 - 90 * j, facecolor="none",
                           edgecolor=c, linewidth=1.6, zorder=4 - j)

        # collect clip kf feet positions (frame0), grouped by spot
        poses = [(np.zeros(2), 0.0)]
        for k in range(1, len(clip.keyframes)):
            poses.append(compose_frame(
                *poses[-1], clip.keyframes[k].transfer))
        spots = {}                        # pos-key -> [(label,color)]
        for i, kf in enumerate(clip.keyframes):
            w = to_world(kf.pos, *poses[i]) if i else kf.pos
            for side, fi in (("R", 3), ("L", 4)):
                key = tuple(np.round(w[fi, :2], 2))
                spots.setdefault(key, []).append(
                    (w[fi, :2], f"k{i}:{side}", "gray"))
        for key, items in spots.items():
            pos = items[0][0]
            _ring(pos, ["gray"] * len(items), "gray")
            _lab(pos, " ".join(t for _, t, _ in items), "gray")

        # current support: black
        for side, f in lm.feet.items():
            if f.foot_pos is not None:
                key = tuple(np.round(f.foot_pos, 2))
                ax.scatter(*f.foot_pos, s=200, c="black", zorder=5)
                _lab(f.foot_pos, f"now:{side[0].upper()}", "black")

        # planned steps
        for i, st in enumerate(steps):
            col = {"corrective": "tab:orange", "move": "tab:green",
                   "recover": "tab:purple"}.get(st.kind, "tab:gray")
            ax.scatter(*st.target, s=260, facecolor="none",
                       edgecolor=col, linewidth=2.4, zorder=6)
            _lab(st.target,
                 f"{i + 1}:{st.foot[0].upper()}", col)
            cur = lm.feet[st.foot].foot_pos
            if cur is not None:
                ax.annotate("", xy=st.target, xytext=cur,
                            arrowprops=dict(arrowstyle="->",
                                            color=col, lw=1.2,
                                            alpha=0.6))
        ax.set_title(name)
        ax.set_xlabel("x fwd"); ax.set_ylabel("y left")
        fig.tight_layout()
        p = outdir / f"{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print("saved", p)

    render("01_still_punch", mk(), punch())
    render("02_drift_left", mk(vel=(0, 0.5, 0)), punch())
    render("03_drift_back", mk(vel=(-0.5, 0, 0)), punch())
    render("04_drift_right", mk(vel=(0, -0.5, 0)), punch())
    render("05_drift_fwd", mk(vel=(0.5, 0, 0)), punch())
    render("06_drift_diag", mk(vel=(-0.35, 0.35, 0)), punch())
    render("07_big_momentum", mk(vel=(0, 1.8, 0)), punch())
    render("08_single_support", mk(fr=1.0, fl=0.0), punch())
    render("09_single_drift", mk(vel=(0, 0.5, 0), fr=1.0, fl=0.0),
           punch())
    render("10_com_offset", mk(com=(0.0, -0.12, 0.87)), punch())
    render("11_fwd_step", mk(), fwd_step())
    render("12_back_step", mk(), back_step())
    render("13_punch_right", mk(), punch_r())
    render("14_crouch_drift", mk(com=(0.02, 0, 0.62), vel=(0, 0.4, 0)),
           punch())
    print("done ->", outdir)
