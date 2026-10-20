"""StrikeTrackObserver — ego-anchored mocap tracking reward for strikes.

Strike segments in the command schedule (``move_schedule`` rows with
the full 10-field form) may contain strike segments
``[n, vx, vy, w, h, eff_type=+1, eff_side=±1, tx, ty, tz]``.  This
observer resolves which reference clip a strike segment maps to —

    side +1 (left)  : near → hdm05_3146_88_2895_127    (L jab)
                      far  → cmu_144_20_m_seg01        (L lunge, mirror)
    side -1 (right) : near → hdm05_3146_88_2895_127_m  (R jab, mirror)
                      far  → cmu_144_20_seg01          (R lunge)

with the style resolved by *nearest landing point* (``resolve_clip``):
the commanded target is compared against each side's two clip landings
in ego space (jab ~(0.44,±0.39,0.50), lunge ~(0.62,∓0.21,0.41), ~0.63 m
apart) — a near target maps to the jab, a far target to the lunge.

At a strike segment's onset the clip is anchored into the robot's ego
frame: the reference's frame-0 root xy/yaw is mapped onto the robot's
root xy/yaw at that step, so "forward" in the reference always means
"the direction the robot faces when the punch starts".  All DeepMimic
error terms then compare the live state against the anchored reference
(joint pose/vel are ego-invariant; keypoints are compared pelvis-
relative ego — both unchanged by the anchor; only root xy/yaw and the
reported world-frame kp error need the transform).

Outputs per step: ``reward, pose, vel, arm, kp, height, root, kp_err_m,
q_rmse_deg, in_strike, phase, frame, landed`` — ``in_strike`` marks
steps inside a resolved strike segment so downstream gating is exact.

``ext_only=True`` (skill-training "punch-out" phase) uses *monotone
progress tracking*: a per-segment progress pointer ``k`` starts at 0
and each step advances to the highest-reward reference frame within
``[k, k + lookahead]``.  The pointer never retreats — punching faster
advances the clock faster (speed unbounded by the clip), while falling
behind just keeps the pointer waiting; nothing penalizes being ahead.
Reaching the landing frame ``k_land`` is only *progress* — ``landed``
and the per-step hold bonus additionally require the live state to
actually match that frame (hand within ``LAND_HAND_ERR_M`` of its
reference point, composite score ≥ ``LAND_MIN_SCORE``), re-evaluated
every step.  Faster completion scores strictly better (speed =
time-at-target, plus γ-discount favours early progress).  Everything
after the segment is out of the scored window by construction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework.observer_plugin import BaseObserverPlugin
from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits
from mocap.humanoid21.tracking.ref_data import RefClip, DEFAULT_CLIPS_DIR

KP_SUBSET = ('torso', 'head', 'foot_left', 'foot_right',
             'hand_left', 'hand_right')

_HALF = np.array([
    0.5 * (joint_limits(n)[1] - joint_limits(n)[0])
    for n in CONTROLLED_JOINTS
])

W_POSE, W_VEL, W_KP, W_HEIGHT, W_ROOT = 0.35, 0.10, 0.25, 0.10, 0.20
W_HAND, SIGMA_HAND = 0.50, 0.45   # dense reach term on the punching hand
# Punching-arm joint-chain term: arm pose errors are diluted in the
# 21-dim pose mean, so a bent punching elbow is nearly free under
# r_pose alone — track the striking arm's joints separately.
W_ARM = 0.50
_JOINT_IDX = {n: i for i, n in enumerate(CONTROLLED_JOINTS)}
_ARM_JOINTS = {
    'hand_left': [_JOINT_IDX[n] for n in
                  ('shoulder1_left', 'shoulder2_left', 'elbow_left')
                  if n in _JOINT_IDX],
    'hand_right': [_JOINT_IDX[n] for n in
                   ('shoulder1_right', 'shoulder2_right', 'elbow_right')
                   if n in _JOINT_IDX],
}
# σ=0.45 keeps useful gradient across the whole workspace: at d=0.5 m
# r_hand≈0.30 with ∂r/∂d≈1.3/m; σ=0.15 would be ~0 gradient there
# (needle-in-haystack — it only pays once the hand is already close).

# Landing-hold bonus per step while the LIVE state verifiably matches
# the landing frame — makes earlier arrival strictly better (speed =
# time-at-target).
W_HOLD = 0.25
# Completion criterion for `landed` — the progress pointer reaching
# k_max is only progress, not success: "best match among candidates"
# stays meaningful even when every candidate matches badly.  Landing
# additionally requires (a) the punching hand within LAND_HAND_ERR_M of
# its reference point (successful runs land ~0.05 m; 0.10 m ≈ 2× slack,
# still inside the fist-face scale) and (b) a coherent composite frame
# score (catches "hand coincidentally at the ego point while the body
# is off-pose" — e.g. fallen).  Re-evaluated every step: drifting out
# of the landing pose stops the hold bonus (time-at-target literally).
LAND_HAND_ERR_M = 0.10
LAND_MIN_SCORE = 1.0      # frame reward max ≈ 2.40 (all terms = 1)

# (eff_side, is_near) → clip_id
_CLIP_MAP = {
    (+1, True): 'hdm05_3146_88_2895_127',      # left jab
    (-1, True): 'hdm05_3146_88_2895_127_m',    # right jab
    (+1, False): 'cmu_144_20_m_seg01',         # left lunge
    (-1, False): 'cmu_144_20_seg01',           # right lunge
}

# Candidate clip ids per side — [near_style(jab), far_style(lunge)]
_CLIPS_BY_SIDE = {
    +1: [_CLIP_MAP[(+1, True)], _CLIP_MAP[(+1, False)]],
    -1: [_CLIP_MAP[(-1, True)], _CLIP_MAP[(-1, False)]],
}

_LANDING: Dict[str, Tuple[np.ndarray, int]] = {}


def _yaw_of(quat_wxyz: np.ndarray) -> float:
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return float(r.as_euler('xyz')[2])


def _clip_meta(clip_id: str, clips_dir=DEFAULT_CLIPS_DIR
               ) -> Tuple[np.ndarray, int]:
    """(landing_xyz_ego, n_ctrl).  Landing = punching-hand keypoint at its
    max pelvis-relative forward extension, expressed in the reference's
    own frame-0 pelvis ego frame (the frame the ego anchor maps onto)."""
    if clip_id not in _LANDING:
        ref = RefClip(clip_id, clips_dir)
        kn = ref.keypoint_names
        pi = kn.index('pelvis')
        vl = np.linalg.norm(np.gradient(
            ref.keypoints[:, kn.index('hand_left'), :], axis=0), axis=1)
        vr = np.linalg.norm(np.gradient(
            ref.keypoints[:, kn.index('hand_right'), :], axis=0), axis=1)
        hand = 'hand_left' if vl.max() > vr.max() else 'hand_right'
        h = ref.keypoints[:, kn.index(hand), :]
        pel0 = ref.keypoints[0, pi, :]
        yaw0 = _yaw_of(ref.root_quat[0])
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        rel = h - pel0
        ex = c * rel[:, 0] - s * rel[:, 1]
        ey = s * rel[:, 0] + c * rel[:, 1]
        k = int(np.argmax(ex))
        _LANDING[clip_id] = (np.array([ex[k], ey[k], rel[k, 2]]),
                             ref.n_ctrl)
        _LAND_FRAME[clip_id] = k
    return _LANDING[clip_id]


_LAND_FRAME: Dict[str, int] = {}

# Launch-gate joint indices: every controlled joint except the ankles
# (ankle angles are excluded — foot pitch/roll adapt freely during
# the ready-stance match).
_GATE_JOINT_IDX = [i for i, n in enumerate(CONTROLLED_JOINTS)
                   if 'ankle' not in n]


def clip_land_frame(clip_id: str, clips_dir=DEFAULT_CLIPS_DIR) -> int:
    """Frame index of the clip's peak hand extension — the reference
    frame the strike is held at in ``ext_only`` tracking mode."""
    if clip_id not in _LAND_FRAME:
        _clip_meta(clip_id, clips_dir)   # fills _LAND_FRAME as a side effect
    return _LAND_FRAME[clip_id]


def punch_hand_name(ref: RefClip) -> str:
    """The clip's punching hand: the one with the higher speed peak."""
    kn = ref.keypoint_names
    vl = np.linalg.norm(np.gradient(
        ref.keypoints[:, kn.index('hand_left'), :], axis=0), axis=1)
    vr = np.linalg.norm(np.gradient(
        ref.keypoints[:, kn.index('hand_right'), :], axis=0), axis=1)
    return 'hand_left' if vl.max() > vr.max() else 'hand_right'


def resolve_clip(eff_side: int, target_xyz,
                 clips_dir=DEFAULT_CLIPS_DIR) -> str:
    """side + nearest-landing → clip_id ("近→刺拳, 远→弓步" emerges from
    the landing geometry: jab lands ~(0.44,±0.39,0.50), lunge
    ~(0.62,∓0.21,0.41) — 0.63 m apart in ego space)."""
    cands = _CLIPS_BY_SIDE[int(np.sign(eff_side)) or +1]
    tgt = np.asarray(target_xyz, dtype=np.float64)
    return min(cands, key=lambda c: float(
        np.linalg.norm(_clip_meta(c, clips_dir)[0] - tgt)))


def _rz(dyaw: float) -> np.ndarray:
    c, s = np.cos(dyaw), np.sin(dyaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class StrikeTrackObserver(BaseObserverPlugin):
    """Per-step strike tracking reward for one agent."""

    def __init__(self, agent_id: str = 'robot_a',
                 clips_dir: str = str(DEFAULT_CLIPS_DIR),
                 ext_only: bool = False,
                 lookahead: int = 2) -> None:
        self.agent_id = str(agent_id)
        self._clips_dir = clips_dir
        # ext_only (skill-training extension phase): monotone progress
        # tracking — see module docstring.  Default False keeps the
        # strict-clock full-clip tracking of the older experiments.
        self._ext_only = bool(ext_only)
        # lookahead: max ref-frame advance per step for progress
        # tracking (bounds how much of the wind-up may be skipped).
        self._la = int(lookahead)
        self._clips: Dict[str, RefClip] = {}
        self._segments: List[Dict[str, Any]] = []
        self._t = -1
        self._active: Optional[Dict[str, Any]] = None
        self._output: Dict[str, float] = self._zero()
        # Launch-pose gate (episode_options['launch_gate']):
        # {"score": θ, "frames": k} — strict alignment with the clip's
        # ready pose must be reached before progress tracking starts.
        self._gate: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    def _clip(self, clip_id: str) -> RefClip:
        if clip_id not in self._clips:
            self._clips[clip_id] = RefClip(clip_id, self._clips_dir)
        return self._clips[clip_id]

    @staticmethod
    def _zero() -> Dict[str, float]:
        return {'reward': 0.0, 'pose': 0.0, 'vel': 0.0, 'kp': 0.0,
                'height': 0.0, 'root': 0.0, 'hand': 0.0, 'arm': 0.0,
                'hand_err_m': 0.0, 'kp_err_m': 0.0, 'kp_err_rel_m': 0.0,
                'root_dxy_m': 0.0, 'root_dyaw_deg': 0.0,
                'q_rmse_deg': 0.0, 'in_strike': 0.0, 'phase': -1.0,
                'frame': -1.0, 'landed': 0.0, 'gate_qmax': 0.0,
                'hand_ex': 0.0, 'hand_ey': 0.0, 'hand_ez': 0.0,
                'tgt_ex': 0.0, 'tgt_ey': 0.0, 'tgt_ez': 0.0,
                'gated': 0.0, 'gate_t': -1.0}

    # ------------------------------------------------------------------
    # Episode setup — compile schedule rows, resolve clips
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx) -> None:
        self._t = -1
        self._segments = []
        self._active = None
        self._output = self._zero()
        sched = (ctx.episode_options.get('move_schedule') or {}).get(
            self.agent_id) or []
        self._gate = ctx.episode_options.get('launch_gate')
        t = 0
        clip_ov = (ctx.episode_options.get('strike_clip') or {}).get(
            self.agent_id)
        for seg in sched:
            n = int(seg[0])
            row = dict(a=t, b=t + n, clip=None, ref=None, k_ptr=0,
                       gated=False, gate_t=-1)
            if len(seg) > 6 and int(round(seg[5])) == +1:   # eff_type=hand
                side = int(round(seg[6]))
                # explicit clip binding (precision strikes: tgt is a
                # free point, NOT a clip selector) overrides nearest-
                # landing resolution.
                cid = clip_ov or resolve_clip(
                    side, seg[7:10], self._clips_dir)
                row['clip'] = cid
                row['ref'] = self._clip(cid)
                row['hand'] = punch_hand_name(row['ref'])
                row['k_max'] = (clip_land_frame(cid, self._clips_dir)
                                if self._ext_only else
                                row['ref'].n_ctrl - 1)
            self._segments.append(row)
            t += n

    # ------------------------------------------------------------------
    def on_post_action_step(self, ctx) -> None:
        self._t += 1
        self._output = self._compute(ctx)

    # ------------------------------------------------------------------
    def _compute(self, ctx) -> Dict[str, float]:
        seg = next((s for s in self._segments
                    if s['a'] <= self._t < s['b']), None)
        if seg is None or seg['ref'] is None:
            self._active = None
            return self._zero()

        ref: RefClip = seg['ref']
        live = self._live(ctx, seg, ref)
        if self._ext_only:
            # Launch-pose gate (STRIKE_LAUNCH_GATE.md): before progress
            # tracking may start, the live state must match some frame
            # of the launch window [0, k_launch] with score >= θ.
            # Until then the per-step reward IS the alignment score —
            # dense shaping toward the ready pose.  in_strike stays 1
            # so r_strike actor weight flows during the approach.
            if self._gate is not None and not seg['gated']:
                # Gate verdict — per-joint pose only (P2-1): every
                # non-ankle controlled joint must be within `frac` of
                # its FULL range vs the gate frame.  joint_pos_norm is
                # (pos − mid) / half_range, so "frac of full range" is
                # 2·frac in normalized units.
                #
                # Dense shaping uses the SAME semantics as the verdict —
                # per-joint position error only (no vel/kp/hand/…):
                # per-joint score exp(−(q_err/σ)²) with σ = 2·frac puts
                # e^−1 at the verdict boundary, so crossing joints
                # keep being pulled toward exact match and every joint
                # carries its own gradient.  The composite score is
                # kept only as a diagnostic — its vel term in
                # particular is semantically wrong here (we want the
                # fastest return to the ready pose, not clip-speed
                # matching).
                kf = int(self._gate.get('frames', 4))
                frac = float(self._gate.get('frac', 0.05))
                hi_g = min(kf, seg['k_max'])
                k, out = max(
                    ((c, self._frame_score(live, seg, ref, c))
                     for c in range(0, hi_g + 1)),
                    key=lambda p: p[1]['reward'])
                q_err = np.abs(
                    np.asarray(live['core']['joint_pos_norm'],
                               dtype=np.float64) - ref.q_norm[k])
                out['gate_qmax'] = float(q_err[_GATE_JOINT_IDX].max())
                out['reward'] = float(np.mean(np.exp(
                    -(q_err[_GATE_JOINT_IDX] / (2.0 * frac)) ** 2)))
                if out['gate_qmax'] <= 2.0 * frac:
                    seg['gated'] = True
                    seg['gate_t'] = float(self._t - seg['a'])
                    seg['k_ptr'] = k
                out['frame'] = float(k)
                out['phase'] = 0.0
                out['gated'] = float(seg['gated'])
                out['gate_t'] = seg['gate_t']
                return self._augment_tgt(ctx, seg, live, out)
            # Monotone progress tracking: advance the pointer to the
            # highest-reward frame within [k, k+lookahead]; never back.
            lo = seg['k_ptr']
            hi = min(lo + self._la, seg['k_max'])
            k, out = max(
                ((c, self._frame_score(live, seg, ref, c))
                 for c in range(lo, hi + 1)),
                key=lambda p: p[1]['reward'])
            seg['k_ptr'] = k
            if k >= seg['k_max']:
                # Pointer at the landing frame = progress, not success.
                # `landed` requires the live state to actually match
                # that frame; re-evaluated per step so the hold bonus
                # tracks time-at-target, not pointer position.
                landed = (out['hand_err_m'] <= LAND_HAND_ERR_M
                          and out['reward'] >= LAND_MIN_SCORE)
                if landed:
                    out['reward'] += W_HOLD
                    out['landed'] = 1.0
            out['frame'] = float(k)
            out['phase'] = k / max(1, seg['k_max'])
            out['gated'] = float(seg['gated'])
            out['gate_t'] = seg['gate_t']
            return self._augment_tgt(ctx, seg, live, out)

        k = min(self._t - seg['a'], seg['k_max'])
        out = self._frame_score(live, seg, ref, k)
        out['frame'] = float(k)
        out['phase'] = k / max(1, seg['k_max'])
        return self._augment_tgt(ctx, seg, live, out)

    def _augment_tgt(self, ctx, seg, live, out):
        """Precision-strike fields: punching-hand ego position and the
        DISPLAYED ego target (what the policy saw — world-anchored when
        ``tgt_anchor`` is on).  ||hand_ego − tgt_ego|| equals the world
        distance to the commanded point, so downstream reward building
        needs no anchor reconstruction."""
        hand = seg.get('hand')
        if hand in live['e_self']:
            hx, hy, hz = live['e_self'][hand]
            out['hand_ex'], out['hand_ey'], out['hand_ez'] = \
                float(hx), float(hy), float(hz)
        try:
            obs = ctx.accessor.get_observation()[self.agent_id]
            from baseline.humanoid21.hierarchy.command_simulator import (
                BODY_DIM, TGT_SCALE, TGT_X, TGT_Z)
            tgt = np.asarray(obs[BODY_DIM + TGT_X: BODY_DIM + TGT_Z + 1],
                             dtype=np.float64) * TGT_SCALE
            out['tgt_ex'], out['tgt_ey'], out['tgt_ez'] = \
                float(tgt[0]), float(tgt[1]), float(tgt[2])
        except Exception:
            pass
        return out

    def _live(self, ctx, seg, ref: RefClip) -> Dict[str, Any]:
        """Live-state bundle gathered once per step (shared by all
        candidate-frame scorings in progress mode)."""
        ds = ctx.accessor.get_derived_state()
        core = ctx.accessor.get_core_state()[self.agent_id]
        root_pos = np.asarray(core['root_pos'], dtype=np.float64)
        root_rot = np.asarray(core['root_rot'], dtype=np.float64)
        yaw_self = _yaw_of(root_rot)

        # --- ego anchor: set on first step inside the segment -----------
        if self._active is not seg:
            yaw_ref0 = _yaw_of(ref.root_quat[0])
            seg['anchor'] = dict(
                p0=root_pos[:2].copy(), yaw0=yaw_self,
                p_ref0=ref.root_pos[0][:2].copy(), yaw_ref0=yaw_ref0,
                M=_rz(yaw_self - yaw_ref0))
            self._active = seg

        static = ctx.accessor.get_static_data()
        kp_names = static.get(self.agent_id, {}).get(
            'keypoint_body_names', {})
        body_xpos = ds.get(self.agent_id, {}).get('body_xpos', {})
        p_self_pelvis = np.asarray(body_xpos[kp_names['pelvis']],
                                   dtype=np.float64)
        R_self_inv = R.from_quat(
            [root_rot[1], root_rot[2], root_rot[3], root_rot[0]]
        ).inv().as_matrix()
        # pelvis-relative ego position + world position of every scored
        # keypoint, computed once for all candidate frames
        e_self, p_self = {}, {}
        for n in KP_SUBSET:
            if n in kp_names:
                p = np.asarray(body_xpos[kp_names[n]], dtype=np.float64)
                e_self[n] = R_self_inv @ (p - p_self_pelvis)
                p_self[n] = p
        return dict(core=core, root_pos=root_pos, root_rot=root_rot,
                    yaw_self=yaw_self, anc=seg['anchor'],
                    e_self=e_self, p_self=p_self)

    def _frame_score(self, live, seg, ref: RefClip,
                     k: int) -> Dict[str, float]:
        """Full reward + metrics of the live state vs reference frame k."""
        core = live['core']
        root_pos, root_rot = live['root_pos'], live['root_rot']
        anc = live['anc']
        kn = ref.keypoint_names

        # --- pose / velocity -------------------------------------------
        q_err = np.asarray(core['joint_pos_norm'], dtype=np.float64) \
            - ref.q_norm[k]
        qv_err = np.asarray(core['joint_vel_norm'], dtype=np.float64) \
            - ref.qv_norm[k]
        r_pose = float(np.exp(-2.0 * np.mean(q_err ** 2)))
        r_vel = float(np.exp(-0.02 * np.mean(qv_err ** 2)))
        arm_idx = _ARM_JOINTS.get(seg.get('hand'))
        r_arm = (float(np.exp(-2.0 * np.mean(q_err[arm_idx] ** 2)))
                 if arm_idx else 0.0)
        q_rmse_deg = float(np.degrees(
            np.sqrt(np.mean((q_err * _HALF) ** 2))))

        # --- keypoints (pelvis-relative ego both sides — anchor-free) ---
        ref_kp = ref.keypoints[k]
        pi = kn.index('pelvis')
        p_ref_pelvis = ref_kp[pi]
        q_ref = ref.root_quat[k]
        R_ref_inv = R.from_quat(
            [q_ref[1], q_ref[2], q_ref[3], q_ref[0]]).inv().as_matrix()
        sq_rel, sq_world = [], []
        M = anc['M']
        hand_err = 0.0
        for n in KP_SUBSET:
            if n not in kn or n not in live['e_self']:
                continue
            e_self = live['e_self'][n]
            e_ref = R_ref_inv @ (ref_kp[kn.index(n)] - p_ref_pelvis)
            sq_rel.append(float(np.sum((e_self - e_ref) ** 2)))
            if n == seg.get('hand'):
                hand_err = float(np.linalg.norm(e_self - e_ref))
            # world-frame err vs anchored ref (report only)
            p_ref_world = np.concatenate(
                [anc['p0'] + M @ (ref_kp[kn.index(n)][:2]
                                 - ref.keypoints[0][pi][:2]),
                 [ref_kp[kn.index(n)][2]]])
            sq_world.append(float(np.sum(
                (live['p_self'][n] - p_ref_world) ** 2)))
        kp_mse = float(np.mean(sq_rel)) if sq_rel else 0.0
        r_kp = float(np.exp(-40.0 * kp_mse))
        kp_err_m = float(np.sqrt(np.mean(sq_world))) if sq_world else 0.0

        # --- dense reach term: punching hand to its reference position --
        # (same trajectory the displayed cmd target replays)
        r_hand = float(np.exp(-(hand_err / SIGMA_HAND) ** 2))

        # --- height -----------------------------------------------------
        z_err = float(root_pos[2] - ref.root_pos[k][2])
        r_height = float(np.exp(-20.0 * z_err ** 2))

        # --- root xy/yaw vs anchored ref --------------------------------
        p_anch = anc['p0'] + M @ (ref.root_pos[k][:2] - anc['p_ref0'])
        dxy = float(np.linalg.norm(root_pos[:2] - p_anch))
        yaw_anch = _yaw_of(ref.root_quat[k]) - anc['yaw_ref0'] \
            + anc['yaw0']
        dyaw = float((live['yaw_self'] - yaw_anch + np.pi)
                     % (2 * np.pi) - np.pi)
        r_root = 0.5 * float(np.exp(-3.0 * dxy ** 2)) \
            + 0.5 * float(np.exp(-1.0 * dyaw ** 2))

        # ext_only drops the velocity term entirely (P2-1 后续裁决):
        # "越快越好"的语义下奖励匹配 clip 速度是反向的——指针机制与
        # hold bonus 负责速度压力，速度项只会在"想快"时拖后腿。
        # r_vel 仍输出为诊断。非 ext_only（严格时钟跟踪）保留。
        w_vel = 0.0 if self._ext_only else W_VEL
        reward = (W_POSE * r_pose + w_vel * r_vel
                  + W_KP * r_kp + W_HEIGHT * r_height + W_ROOT * r_root
                  + W_HAND * r_hand + W_ARM * r_arm)
        return {
            'reward': reward, 'pose': r_pose, 'vel': r_vel, 'arm': r_arm,
            'kp': r_kp, 'height': r_height, 'root': r_root,
            'hand': r_hand, 'hand_err_m': hand_err,
            'kp_err_m': kp_err_m, 'kp_err_rel_m': float(np.sqrt(kp_mse)),
            'root_dxy_m': dxy, 'root_dyaw_deg': float(np.degrees(dyaw)),
            'q_rmse_deg': q_rmse_deg, 'in_strike': 1.0, 'landed': 0.0,
        }

    # ------------------------------------------------------------------
    def get_output(self) -> Dict[str, float]:
        return dict(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {'agent_id': self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'StrikeTrackObserver':
        return cls(**config)
