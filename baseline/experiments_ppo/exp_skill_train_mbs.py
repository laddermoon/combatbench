"""Skill training — preliminary discrete-skill stage (66-dim).

Post S3/S4 reset.  Warm-start from the accepted S2 checkpoint
(``hier_s2_balance_s42i_b`` @ u1635, recovery 0.948).  Composite
experiment: four env variants, one skill per episode, per-episode
reward-channel attachment.

Skills (one per episode; WarmupSteppingPlugin puts robots into a
mid-gait state at reset so every recorded frame IS the skill):
  move_fwd/back/left/right — max-speed cardinal translation (cmd ±1)
  turn_left/right          — max-speed in-place turn
  step                     — fallen init; zero cmd = standup+hold
  jab_l/r, heavy_l/r       — mocap-shape strike, two-phase targets

Reward channels (attached per skill family):
  r_potential / r_fall                     — all episodes
  r_move  = 2·v_along                      — move segs (mirror radial)
  r_drift = −v_off −0.5|w| −0.5|Δyaw|      — move eps (tangential+heading)
  r_turn  = 2·ω_signed −2|v_xy|            — turn segs
  r_hold  = −motion −(d_pos + d_yaw)       — idle eps (in-place hard lock)
  r_strike = mocap composite               — strike segs (in_strike gated)
  r_std_clip                               — idle eps only

Blueprints: baseline/humanoid21/hierarchy/blueprints/skill_*_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.end2end.stepping_state_machine import (
    fall_onset_mask,
)
from baseline.humanoid21.rewards.follow_opponent import (
    _centered_moving_average,
)
from baseline.humanoid21.hierarchy.command_simulator import V_MAX, W_MAX

from .exp_balance_step import _balance_explore_factor
from .exp_balance_pure import BalancePure


# ---------------------------------------------------------------------------
# Skill table — one entry per skill: family, schedule rows, episode length
# ---------------------------------------------------------------------------
_STRIKE_PAD = 3       # small tail after clip ends (hold final pose)

_JAB_LAND = (0.438, 0.391, 0.498)      # hdm05_3146_88_2895_127 landing
_HEAVY_LAND = (0.622, -0.213, 0.413)   # cmu_144_20_m_seg01 landing (left)
_JAB_NCTRL = 22
_HEAVY_NCTRL = 13


def _strike_rows(side: int, land: Tuple[float, float, float],
                 n_ctrl: int) -> List[list]:
    return [
        [n_ctrl + _STRIKE_PAD, 0.0, 0.0, 0.0, 0.0,
         +1, side, land[0], land[1] * (1 if side > 0 else -1), land[2]],
    ]


SKILLS: Dict[str, Dict[str, Any]] = {
    "move_fwd":   dict(fam="move",  dir_ego=(1.0, 0.0)),
    "move_back":  dict(fam="move",  dir_ego=(-1.0, 0.0)),
    "move_left":  dict(fam="move",  dir_ego=(0.0, 1.0)),
    "move_right": dict(fam="move",  dir_ego=(0.0, -1.0)),
    "turn_left":  dict(fam="turn",  w_sign=+1.0),
    "turn_right": dict(fam="turn",  w_sign=-1.0),
    "step":       dict(fam="idle"),
    "jab_l":      dict(fam="strike", side=+1, land=_JAB_LAND,
                       n_ctrl=_JAB_NCTRL),
    "jab_r":      dict(fam="strike", side=-1, land=_JAB_LAND,
                       n_ctrl=_JAB_NCTRL),
    "heavy_l":    dict(fam="strike", side=+1, land=_HEAVY_LAND,
                       n_ctrl=_HEAVY_NCTRL),
    "heavy_r":    dict(fam="strike", side=-1, land=_HEAVY_LAND,
                       n_ctrl=_HEAVY_NCTRL),
}

_IDLE_LEN = 50


def _skill_rows(skill: str) -> List[list]:
    """move_schedule rows for one skill (physical units, v2 semantics)."""
    s = SKILLS[skill]
    fam = s["fam"]
    if fam == "move":
        vx, vy = s["dir_ego"]
        return [[70, vx * V_MAX, vy * V_MAX, 0.0]]
    if fam == "turn":
        return [[70, 0.0, 0.0, s["w_sign"] * W_MAX]]
    if fam == "idle":
        return [[_IDLE_LEN, 0.0, 0.0, 0.0]]
    return _strike_rows(s["side"], s["land"], s["n_ctrl"])


def _skill_len(skill: str) -> int:
    return int(sum(int(r[0]) for r in _skill_rows(skill)))


_FAMILY_SKILLS: Dict[str, List[str]] = {}
for _k, _v in SKILLS.items():
    _FAMILY_SKILLS.setdefault(_v["fam"], []).append(_k)


class SkillTrainMBS(BalancePure):
    """Composite single-skill training on the S2 checkpoint."""

    name = "skill_train_mbs"

    # --- Network (66-dim CommandSimulator v2 contract) ---
    obs_dim: int = 66
    action_dim: int = 21
    actor_blueprint = "init_policy_mbs_66d.yaml"

    # --- Channels: per-episode subsets attached by skill family ---
    _channel_names = (
        "r_potential", "r_fall", "r_move", "r_drift",
        "r_turn", "r_hold", "r_strike", "r_std_clip",
    )
    _channel_gammas = {k: 0.99 for k in _channel_names}

    # --- Reward constants ---
    r_move_actor_weight: float = 3.0
    r_drift_actor_weight: float = 2.0
    r_turn_actor_weight: float = 3.0
    r_hold_actor_weight: float = 3.0
    r_strike_actor_weight: float = 4.0
    _clip_aw_scale: float = 2.0

    # --- Rollout: short episodes → more episodes per update ---
    episodes_per_update: int = 1536
    max_updates: int = 4000
    eval_episodes: int = 96

    # --- Family sampling weights ---
    FAMILY_WEIGHTS: Dict[str, float] = {
        "move": 0.30, "turn": 0.15, "strike": 0.35, "idle": 0.20,
    }
    # Per-skill oversampling: S2 gait has a right-side handedness,
    # weak left skills get extra gradient share.
    SKILL_SAMPLE_W: Dict[str, float] = {
        "turn_left": 2.5, "move_left": 1.5, "jab_l": 1.5,
    }

    _MOVE_OBS = {"robot_a": "move_track_a", "robot_b": "move_track_b"}
    _STRIKE_OBS = {"robot_a": "strike_track_a", "robot_b": "strike_track_b"}
    _CLIP_OBS = {"robot_a": "standard_clip_a", "robot_b": "standard_clip_b"}

    _BP_DIR = (
        Path(__file__).resolve().parent.parent
        / "humanoid21" / "hierarchy" / "blueprints"
    )
    _FAM_BP = {
        "move": "skill_move_env.yaml",
        "turn": "skill_turn_env.yaml",
        "strike": "skill_strike_env.yaml",
        "idle": "skill_idle_env.yaml",
    }

    # ------------------------------------------------------------------
    # Blueprints — one ParameterizedEnvBlueprint per family
    # ------------------------------------------------------------------

    def _env_pb(self):
        return self._fam_env_pb("move")   # interface-compat default

    @staticmethod
    def _fam_env_pb(fam: str):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        return ParameterizedEnvBlueprint.load(
            SkillTrainMBS._BP_DIR / SkillTrainMBS._FAM_BP[fam])

    # ------------------------------------------------------------------
    # Jobs — per-job family → env variant → per-robot skill + schedule
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=_balance_explore_factor)
        fams = list(self.FAMILY_WEIGHTS)
        probs = np.array([self.FAMILY_WEIGHTS[f] for f in fams])
        probs = probs / probs.sum()

        env_cache: Dict[Tuple[str, int], Any] = {}

        def _env(fam: str, max_steps: int):
            key = (fam, max_steps)
            if key not in env_cache:
                env_cache[key] = self._fam_env_pb(fam).materialize(
                    max_steps=max_steps)
            return env_cache[key]

        jobs = []
        if not stochastic:
            # Eval: round-robin all skills, paired within the same family.
            pairs = []
            for fam in fams:
                ks = _FAMILY_SKILLS[fam]
                for i in range(len(ks)):
                    pairs.append((ks[i], ks[(i + 1) % len(ks)]))
            for i in range(n_episodes):
                sa, sb = pairs[i % len(pairs)]
                fam = SKILLS[sa]["fam"]
                n = max(_skill_len(sa), _skill_len(sb))
                jobs.append(Job(
                    policy_a_bp=policy_bp, policy_b_bp=policy_bp,
                    env_bp=_env(fam, n), seed=int(base_seed + i),
                    episode_options={
                        "initial_distance": float(rng.uniform(
                            self.init_distance_min,
                            self.init_distance_max)),
                        "skills": {"robot_a": sa, "robot_b": sb},
                        "move_schedule": {
                            "robot_a": _skill_rows(sa),
                            "robot_b": _skill_rows(sb),
                        },
                        "warmup_policy_cls": policy_bp.cls,
                    },
                    sampling_a=sampling, sampling_b=sampling,
                    stochastic=False,
                ))
            return jobs

        for i in range(n_episodes):
            fam = str(rng.choice(fams, p=probs))
            ks = _FAMILY_SKILLS[fam]
            kw = np.array([self.SKILL_SAMPLE_W.get(k, 1.0) for k in ks])
            kw = kw / kw.sum()
            sa = str(rng.choice(ks, p=kw))
            sb = str(rng.choice(ks, p=kw))
            n = max(_skill_len(sa), _skill_len(sb))
            jobs.append(Job(
                policy_a_bp=policy_bp, policy_b_bp=policy_bp,
                env_bp=_env(fam, n), seed=int(base_seed + i),
                episode_options={
                    "initial_distance": float(rng.uniform(
                        self.init_distance_min, self.init_distance_max)),
                    "skills": {"robot_a": sa, "robot_b": sb},
                    "move_schedule": {
                        "robot_a": _skill_rows(sa),
                        "robot_b": _skill_rows(sb),
                    },
                    "warmup_policy_cls": policy_bp.cls,
                },
                sampling_a=sampling, sampling_b=sampling,
                stochastic=True,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Kinematics helpers (follow-style acausal smoothing)
    # ------------------------------------------------------------------

    @staticmethod
    def _kin(ep, agent_id: str, T: int):
        key = SkillTrainMBS._MOVE_OBS[agent_id]
        oo = ep.observer_outputs
        x = extract_per_step_field(oo, key, "root_x", T)
        y = extract_per_step_field(oo, key, "root_y", T)
        yaw = extract_per_step_field(oo, key, "yaw", T)
        h = extract_per_step_field(oo, key, "h_root", T)
        if x is None or y is None or yaw is None:
            return None
        return (np.stack([x, y], axis=1).astype(np.float64),
                np.asarray(yaw, dtype=np.float64),
                np.asarray(h, dtype=np.float64)
                if h is not None else np.zeros(T))

    @staticmethod
    def _smooth_vel(xy: np.ndarray):
        """(T,2) -> smoothed per-step displacement (clipped), = v·dt."""
        sm = _centered_moving_average(xy, 17)
        disp = np.empty_like(sm)
        disp[1:-1] = (sm[2:] - sm[:-2]) * 0.5
        disp[0] = sm[1] - sm[0]
        disp[-1] = sm[-1] - sm[-2]
        mag = np.linalg.norm(disp, axis=1)
        big = mag > 0.10            # 0.10 m/step = 2 m/s physical cap
        if big.any():
            disp[big] *= (0.10 / np.maximum(mag[big], 1e-9))[:, None]
        return disp

    # ------------------------------------------------------------------
    # Trajectories — per-skill channel attachment
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> List[Trajectory]:
        T_full = episode.num_frames
        if T_full == 0:
            return []
        T = episode.agent_frame_boundary.get(agent_id, T_full)
        if T == 0:
            return []

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)
        if obs_all is None or acts_all is None or fin_obs is None:
            return []
        obs_all = np.asarray(obs_all, dtype=np.float32)
        acts_all = np.asarray(acts_all, dtype=np.float32)

        skills = (episode.episode_options or {}).get("skills") or {}
        skill = str(skills.get(agent_id, "step"))
        fam = SKILLS.get(skill, SKILLS["step"])["fam"]

        oo = episode.observer_outputs
        phi = extract_per_step_field(oo, phi_key, "potential", T_full)
        phi = (np.clip(np.asarray(phi[:T], dtype=np.float32), 0.0, 1.0)
               if phi is not None else np.zeros(T, dtype=np.float32))

        channels: Dict[str, ChannelData] = {
            "r_potential": ChannelData(
                reward=(self.per_step_phi_coef * phi).astype(np.float32),
                is_terminated=False,
                actor_weight=np.full(
                    T, self.r_potential_actor_weight, dtype=np.float32),
            ),
            "r_fall": ChannelData(
                reward=self._r_fall(episode, agent_id, phi, T, T_full),
                is_terminated=False,
                actor_weight=np.full(
                    T, self.r_fall_actor_weight, dtype=np.float32),
            ),
        }

        kin = self._kin(episode, agent_id, T_full)

        if kin is not None and fam in ("move", "turn", "idle"):
            xy, yaw, _h = kin
            xy = xy[:T]
            yaw_u = np.unwrap(np.asarray(yaw[:T], dtype=np.float64))
            disp = self._smooth_vel(xy)              # (T,2) m/step
            v = disp * 20.0                          # m/s
            w = np.clip(np.gradient(yaw_u) * 20.0, -8.0, 8.0)  # rad/s
            v_xy = np.linalg.norm(v, axis=1)

            if fam == "move":
                yaw0 = yaw_u[0]
                c, s = np.cos(yaw0), np.sin(yaw0)
                de = np.asarray(SKILLS[skill]["dir_ego"])
                d_hat = np.array([c * de[0] - s * de[1],
                                  s * de[0] + c * de[1]])
                v_along = v @ d_hat
                v_off = np.linalg.norm(
                    v - v_along[:, None] * d_hat[None, :], axis=1)
                dyaw = np.abs((yaw_u - yaw0 + np.pi)
                              % (2 * np.pi) - np.pi)
                channels["r_move"] = ChannelData(
                    reward=(2.0 * v_along).astype(np.float32),
                    is_terminated=False,
                    actor_weight=(self.r_move_actor_weight
                                  * phi ** 2).astype(np.float32),
                )
                channels["r_drift"] = ChannelData(
                    reward=(-1.0 * v_off - 0.5 * np.abs(w)
                            - 1.5 * dyaw).astype(np.float32),
                    is_terminated=False,
                    actor_weight=(self.r_drift_actor_weight
                                  * phi ** 2).astype(np.float32),
                )
            elif fam == "turn":
                w_sign = float(SKILLS[skill]["w_sign"])
                w_signed = w * w_sign
                r_turn = (0.5 * np.clip(w_signed, -4.0, 4.0)
                          - 1.0 * v_xy).astype(np.float32)
                channels["r_turn"] = ChannelData(
                    reward=r_turn, is_terminated=False,
                    actor_weight=(self.r_turn_actor_weight
                                  * phi ** 2).astype(np.float32),
                )
            else:  # idle — in-place hard lock
                anchor = int(np.argmax(phi >= 0.8)) \
                    if (phi >= 0.8).any() else T - 1
                d_pos = np.clip(
                    np.linalg.norm(xy - xy[anchor], axis=1), 0.0, 3.0)
                d_yaw = np.abs((yaw_u - yaw_u[anchor] + np.pi)
                               % (2 * np.pi) - np.pi)
                r_hold = (-(0.3 * v_xy + 0.3 * np.abs(w))
                          - (d_pos + d_yaw)).astype(np.float32)
                channels["r_hold"] = ChannelData(
                    reward=r_hold, is_terminated=False,
                    actor_weight=(self.r_hold_actor_weight
                                  * phi ** 2).astype(np.float32),
                )

        if fam == "strike":
            sk = self._STRIKE_OBS[agent_id]
            rew = extract_per_step_field(oo, sk, "reward", T_full)
            ins = extract_per_step_field(oo, sk, "in_strike", T_full)
            r_strike = (np.asarray(rew[:T], dtype=np.float32)
                        if rew is not None else np.zeros(T, np.float32))
            in_strike = (np.asarray(ins[:T], dtype=np.float32)
                         if ins is not None else np.zeros(T, np.float32))
            channels["r_strike"] = ChannelData(
                reward=r_strike, is_terminated=False,
                actor_weight=(self.r_strike_actor_weight * phi ** 2
                              * in_strike).astype(np.float32),
            )

        if fam == "idle":
            ck = self._CLIP_OBS[agent_id]
            r_clip = np.zeros(T, dtype=np.float32)
            for field, wt in (("std_clip", 1.0), ("std_pose", 1.0),
                              ("std_cad", 0.5)):
                arr = extract_per_step_field(oo, ck, field, T_full)
                if arr is not None:
                    r_clip = r_clip + wt * np.clip(
                        np.asarray(arr[:T], dtype=np.float32), -1.0, 1.0)
            h_t = extract_per_step_field(oo, phi_key, "h_torso", T_full)
            h_t = (np.asarray(h_t[:T], dtype=np.float32)
                   if h_t is not None else np.zeros(T, np.float32))
            r_clip = r_clip + 0.5 * np.exp(-((h_t - 1.20) / 0.08) ** 2)
            channels["r_std_clip"] = ChannelData(
                reward=r_clip, is_terminated=False,
                actor_weight=(self._clip_aw_scale
                              * phi ** 2).astype(np.float32),
            )

        return [Trajectory(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    def _r_fall(self, episode, agent_id, phi, T, T_full):
        """S2 recipe minus the push-clean term (no pushes in skills)."""
        r = (-self.fall_penalty
             * fall_onset_mask(phi).astype(np.float32))
        wall_key = self._WALL_OBS.get(agent_id)
        if wall_key is not None and self.wall_penalty > 0:
            from baseline.humanoid21.end2end.wall_contact_observer import (
                sustained_wall_mask,
            )
            wc = extract_per_step_field(
                episode.observer_outputs, wall_key, "wall_contact", T_full)
            if wc is not None:
                lean = sustained_wall_mask(
                    wc, window=self.wall_window,
                    count_thresh=self.wall_count_thresh)
                r = r - (self.wall_penalty
                         * lean[:T].astype(np.float32)
                         * (phi >= 0.5).astype(np.float32))
        return r.astype(np.float32)

    # ------------------------------------------------------------------
    # Eval — per-skill metrics
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        agg: Dict[str, Dict[str, List[float]]] = {}
        n_fall = 0
        n_eps = 0
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            skills = (ep.episode_options or {}).get("skills") or {}
            for aid, _, phi_key in self._AGENT_OBS:
                skill = str(skills.get(aid, ""))
                if skill not in SKILLS:
                    continue
                fam = SKILLS[skill]["fam"]
                n_eps += 1
                T = ep.num_frames
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T)
                if phi is None:
                    continue
                phi = np.asarray(phi[:T], dtype=np.float32)
                fell = bool((phi < 0.3).sum() > 10)
                if fell:
                    n_fall += 1
                m = agg.setdefault(skill, {})

                kin = self._kin(ep, aid, T)
                if fam == "move" and kin is not None:
                    xy, yaw, _ = kin
                    yaw_u = np.unwrap(yaw)
                    disp = self._smooth_vel(xy) * 20.0
                    yaw0 = yaw_u[0]
                    c, s = np.cos(yaw0), np.sin(yaw0)
                    de = np.asarray(SKILLS[skill]["dir_ego"])
                    d_hat = np.array([c * de[0] - s * de[1],
                                      s * de[0] + c * de[1]])
                    v_along = disp @ d_hat
                    v_off = np.linalg.norm(
                        disp - v_along[:, None] * d_hat[None, :],
                        axis=1)
                    m.setdefault("v_along", []).append(
                        float(np.mean(v_along)))
                    m.setdefault("v_off", []).append(
                        float(np.mean(v_off)))
                    m.setdefault("dyaw", []).append(float(np.abs(
                        yaw_u[-1] - yaw0)))
                elif fam == "turn" and kin is not None:
                    xy, yaw, _ = kin
                    yaw_u = np.unwrap(yaw)
                    w_signed = (np.gradient(yaw_u) * 20.0
                                * SKILLS[skill]["w_sign"])
                    m.setdefault("w", []).append(
                        float(np.mean(w_signed)))
                    disp = self._smooth_vel(xy) * 20.0
                    m.setdefault("vxy", []).append(float(
                        np.mean(np.linalg.norm(disp, axis=1))))
                elif fam == "idle" and kin is not None:
                    xy, yaw, _ = kin
                    yaw_u = np.unwrap(yaw)
                    stood = (phi >= 0.8).any()
                    m.setdefault("stood", []).append(float(stood))
                    if stood:
                        a = int(np.argmax(phi >= 0.8))
                        m.setdefault("d_pos", []).append(float(
                            np.linalg.norm(xy[-1] - xy[a])))
                        m.setdefault("d_yaw", []).append(float(np.abs(
                            yaw_u[-1] - yaw_u[a])))
                elif fam == "strike":
                    sk = self._STRIKE_OBS[aid]
                    ins = extract_per_step_field(
                        ep.observer_outputs, sk, "in_strike", T)
                    if ins is not None and np.asarray(ins).sum() > 0:
                        ins = np.asarray(ins[:T], dtype=bool)
                        for f, out in (("kp_err_m", "kp"),
                                       ("q_rmse_deg", "q"),
                                       ("hand_err_m", "hand")):
                            arr = extract_per_step_field(
                                ep.observer_outputs, sk, f, T)
                            if arr is not None:
                                m.setdefault(out, []).append(float(
                                    np.asarray(arr[:T])[ins].mean()))

        info: Dict[str, Any] = {"n_eps": n_eps,
                                "falls": round(n_fall / max(n_eps, 1), 3)}
        fam_scores: Dict[str, float] = {}
        for skill, m in agg.items():
            for k, v in m.items():
                info[f"{skill}_{k}"] = round(
                    float(np.mean(v)), 3)
        va = [v for s, m in agg.items() if s.startswith("move")
              for v in m.get("v_along", [])]
        if va:
            fam_scores["move"] = float(np.clip(
                np.mean(va) / 0.6, 0.0, 1.0))
        ws = [v for s, m in agg.items() if s.startswith("turn")
              for v in m.get("w", [])]
        if ws:
            fam_scores["turn"] = float(np.clip(
                np.mean(ws) / 2.0, 0.0, 1.0))
        kp = [v for s, m in agg.items()
              if SKILLS[s]["fam"] == "strike" for v in m.get("kp", [])]
        if kp:
            fam_scores["strike"] = float(np.clip(
                1.0 - np.mean(kp) / 0.30, 0.0, 1.0))
        dp = [v for s, m in agg.items()
              if SKILLS[s]["fam"] == "idle" for v in m.get("d_pos", [])]
        if dp:
            fam_scores["idle"] = float(np.clip(
                1.0 - np.mean(dp) / 0.5, 0.0, 1.0))
        quality = (float(np.mean(list(fam_scores.values())))
                   if fam_scores else 0.0)
        info["quality"] = round(quality, 3)

        improved = quality > getattr(self, "_best_quality", -1.0) \
            and n_fall / max(n_eps, 1) < 0.2
        if improved:
            self._best_quality = quality
            self._last_best = update
        stop = (update - getattr(self, "_last_best", update)
                >= self._no_improvement_limit
                and update >= self._min_updates)
        return {"is_new_best": improved, "stop_training": stop,
                "info": info}


EXPERIMENT_CLASS = SkillTrainMBS
