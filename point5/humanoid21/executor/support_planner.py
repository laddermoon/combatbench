"""Support planning (algorithm-1 branch).

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
        n_fix = 0
        while n_fix < max_correction:
            z0 = max(com[2], 0.1)
            xi = com[:2] + vel[:2] / np.sqrt(_G / z0)
            h = hull()
            if h is not None and _inside(xi, h):
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

        # feasibility note on the mover step
        reach = np.linalg.norm(tgt_xy - feet_xy[sup])
        reason = f"seg{t.seg} mover"
        if reach > 2 * step_max:
            reason += f" [suspect: {reach:.2f}m from {sup}]"
        steps.append(Footstep(foot=t.mover, target=tgt_xy,
                              kind="move", reason=reason))
        feet_xy[t.mover] = tgt_xy
        loaded[t.mover] = True
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

    def render(name, lm, clip):
        steps = plan_footsteps(lm, clip)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_aspect("equal")

        # 1) clip keyframe feet IN FRAME0 (through transfer chain):
        #    gray circles, labeled "k<i>:<R/L>"
        poses = [(np.zeros(2), 0.0)]
        for k in range(1, len(clip.keyframes)):
            poses.append(compose_frame(
                *poses[-1], clip.keyframes[k].transfer))
        for i, kf in enumerate(clip.keyframes):
            w = to_world(kf.pos, *poses[i]) if i else kf.pos
            for side, fi in (("R", 3), ("L", 4)):
                ax.scatter(*w[fi, :2], s=340, facecolor="none",
                           edgecolor="gray", linewidth=1.8)
                ax.annotate(f"k{i}:{side}", w[fi, :2],
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=8, color="gray")

        # 2) current support: black circles, "now:<R/L>"
        for side, f in lm.feet.items():
            if f.foot_pos is not None:
                ax.scatter(*f.foot_pos, s=200, c="black", zorder=5)
                ax.annotate(f"now:{side[0].upper()}", f.foot_pos,
                            xytext=(4, -14),
                            textcoords="offset points", fontsize=9)

        # 3) planned steps: colored circles, numbered + R/L
        for i, st in enumerate(steps):
            col = ("tab:orange" if st.kind == "corrective"
                   else "tab:green")
            ax.scatter(*st.target, s=220, facecolor="none",
                       edgecolor=col, linewidth=2.2, zorder=6)
            ax.annotate(f"{i + 1}:{st.foot[0].upper()}",
                        st.target, xytext=(5, 5),
                        textcoords="offset points", fontsize=10,
                        color=col, weight="bold")
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

    render("still_punch", mk(), punch())
    render("drift_left", mk(vel=(0, 0.5, 0)), punch())
    render("drift_back", mk(vel=(-0.5, 0, 0)), punch())
    render("drift_right", mk(vel=(0, -0.5, 0)), punch())
    render("single_support", mk(fr=1.0, fl=0.0), punch())
    render("side_shift", mk(), shift())
    print("done ->", outdir)
