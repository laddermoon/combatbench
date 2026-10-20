"""strike_kp_2phase — keypoint-position two-phase strike experiment.

Inherits CombatExperimentPPOBase ONLY (clean reimplementation; the
joint-angle strike_2phase line stays untouched).  Successor of
strike_2phase under the FIXED 66-dim observation / 21-dim action
contract (STRIKE_KEYPOINT_2PHASE_PLAN.md):

  Phase 1  — approach: reach the ready pose = clip frame-0 keypoints
             anchored to the robot's xy/yaw at strike onset.  Channel:
             −L_kp (weighted mean squared keypoint distance, 15 pts).
  Boundary — first frame with ALL 15 keypoints within tolerance
             (hands/feet 3 cm, rest 5 cm); +1 bonus on r_kp_p1 and the
             phase-1 trajectory ends there.
  Phase 2  — execution: same loss vs the LANDING-frame target,
             rigidly anchored so the rule-selected power foot stays at
             its actual transition position (the fixed target IS the
             foot constraint — no separate foot channel).  +1 on
             r_kp_p2 at the achieving frame; later frames dropped.

Balance rail: r_potential = 0.01·min(phi, 0.9) rides every trajectory.
Channels: r_potential, r_kp_p1, r_kp_p2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory)
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.hierarchy.command_simulator import V_MAX, W_MAX

from .base import CombatExperimentPPOBase

# ---------------------------------------------------------------------
# Strike skills — the whole experiment is these four commands.
# ---------------------------------------------------------------------
_STRIKE_STEPS = 30
_JAB_LAND = (0.438, 0.391, 0.498)      # hdm05_3146_88_2895_127 landing
_HEAVY_LAND = (0.622, -0.213, 0.413)   # cmu_144_20_m_seg01 landing
SKILLS: Dict[str, Dict[str, Any]] = {
    "jab_l":   dict(side=+1, land=_JAB_LAND),
    "jab_r":   dict(side=-1, land=_JAB_LAND),
    "heavy_l": dict(side=+1, land=_HEAVY_LAND),
    "heavy_r": dict(side=-1, land=_HEAVY_LAND),
}


def _strike_row(side: int, land: Tuple[float, float, float],
                n: int) -> list:
    return [n, 0.0, 0.0, 0.0, 0.0,
            +1, side, land[0], land[1] * (1 if side > 0 else -1),
            land[2]]


_P1 = ("r_kp_p1",)
_P2 = ("r_kp_p2",)
_WARMUP_SKILLS = (
    "move_fwd", "move_back", "move_left", "move_right",
    "turn_left", "turn_right", "step", *SKILLS,
)


def _warmup_row(skill: str) -> list:
    if skill in SKILLS:
        s = SKILLS[skill]
        return _strike_row(s["side"], s["land"], _STRIKE_STEPS)
    moves = {
        "move_fwd": (V_MAX, 0.0), "move_back": (-V_MAX, 0.0),
        "move_left": (0.0, V_MAX), "move_right": (0.0, -V_MAX),
    }
    if skill in moves:
        return [30, *moves[skill], 0.0]
    if skill in ("turn_left", "turn_right"):
        return [30, 0.0, 0.0,
                W_MAX if skill == "turn_left" else -W_MAX]
    if skill == "step":
        return [30, 0.0, 0.0, 0.0]
    raise ValueError(f"Unknown warmup skill: {skill}")


class StrikeKp2Phase(CombatExperimentPPOBase):
    """Keypoint-position two-phase strike (clean, fixed 66-dim)."""

    name = "strike_kp_2phase"

    # --- Network (66-dim CommandSimulator v2 contract — unchanged) ---
    obs_dim: int = 66
    action_dim: int = 21
    actor_blueprint = "init_policy_mbs_66d.yaml"

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_kp_p1", "r_kp_p2")
    _channel_gammas = {k: 0.99 for k in _channel_names}
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01
    r_potential_actor_weight: float = 1.0
    r_kp_p1_actor_weight: float = 4.0
    r_kp_p2_actor_weight: float = 4.0
    success_bonus: float = 1.0
    _PHI_CAP: float = 0.90

    # --- Skill sampling: mirror-side skills are the weak spot ---
    SKILL_SAMPLE_W: Dict[str, float] = {
        "jab_l": 1.0, "jab_r": 2.0, "heavy_l": 2.0, "heavy_r": 1.0,
    }

    # --- Rollout schedule ---
    agent_used = "both"
    episodes_per_update: int = 1536
    max_updates: int = 12000
    eval_interval: int = 5
    eval_episodes: int = 64
    video_eval_interval: int = 5

    # --- PPO tuning (same recipe as skill experiments) ---
    learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.03
    update_epochs: int = 4
    minibatch_size: int = 4096
    uncertainty_floor: float = 0.4
    uncertainty_coef: float = 1.0

    # --- Early stop ---
    _no_improvement_limit: int = 500
    _min_updates: int = 300
    _best_quality: float = -1.0
    _last_best: int = 0

    _AGENT_OBS = (
        ("robot_a", "standing_balance_a", "kp_track_a"),
        ("robot_b", "standing_balance_b", "kp_track_b"),
    )
    _BP = (Path(__file__).resolve().parent.parent
           / "humanoid21" / "hierarchy" / "blueprints"
           / "strike_kp_env.yaml")

    # ------------------------------------------------------------------
    # Blueprint / channels
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        return ParameterizedEnvBlueprint.load(self._BP)

    def reward_channels(self) -> tuple:
        return tuple(
            RewardChannel(name=k, gamma=self._channel_gammas[k],
                          gae_lambda=self._gae_lambda)
            for k in self._channel_names)

    # ------------------------------------------------------------------
    # Jobs: random non-target warmup (0-20 steps) + one 30-step strike
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec()
        skills = list(SKILLS)
        weights = np.array([self.SKILL_SAMPLE_W.get(s, 1.0)
                            for s in skills])
        weights = weights / weights.sum()
        env_bp = self._env_pb().materialize(max_steps=_STRIKE_STEPS)

        def _warmup(target: str) -> Dict[str, Any]:
            skill = str(rng.choice(
                [s for s in _WARMUP_SKILLS if s != target]))
            return {"skill": skill, "steps": int(rng.integers(0, 21)),
                    "schedule": [_warmup_row(skill)]}

        def _opts(sa: str, sb: str) -> Dict[str, Any]:
            s_a, s_b = SKILLS[sa], SKILLS[sb]
            return {
                "initial_distance": 2.0,
                "skills": {"robot_a": sa, "robot_b": sb},
                "move_schedule": {
                    "robot_a": [_strike_row(s_a["side"], s_a["land"],
                                            _STRIKE_STEPS)],
                    "robot_b": [_strike_row(s_b["side"], s_b["land"],
                                            _STRIKE_STEPS)],
                },
                "warmup": {"robot_a": _warmup(sa),
                           "robot_b": _warmup(sb)},
                "warmup_policy_cls": policy_bp.cls,
            }

        jobs = []
        for i in range(n_episodes):
            if stochastic:
                sa = str(rng.choice(skills, p=weights))
                sb = str(rng.choice(skills, p=weights))
            else:
                sa, sb = skills[i % 4], skills[(i + 2) % 4]
            jobs.append(Job(
                policy_a_bp=policy_bp, policy_b_bp=policy_bp,
                env_bp=env_bp, seed=int(base_seed + i),
                episode_options=_opts(sa, sb),
                sampling_a=sampling, sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Trajectories: split at p1_done / p2_done (first-True boundaries)
    # ------------------------------------------------------------------

    def build_trajectories(self, episodes) -> List[Trajectory]:
        trajs: List[Trajectory] = []
        for ep in episodes:
            for aid, phi_key, kp_key in self._AGENT_OBS:
                trajs.extend(self._traj(ep, aid, phi_key, kp_key))
        return trajs

    def _traj(self, ep, aid: str, phi_key: str,
              kp_key: str) -> List[Trajectory]:
        T_full = ep.num_frames
        if T_full == 0:
            return []
        T = ep.agent_frame_boundary.get(aid, T_full)
        obs = ep.observations.get(aid)
        acts = ep.actions.get(aid)
        fin = ep.final_observation.get(aid)
        if T == 0 or obs is None or acts is None or fin is None:
            return []
        obs = np.asarray(obs, dtype=np.float32)
        acts = np.asarray(acts, dtype=np.float32)

        oo = ep.observer_outputs

        def _f(obs_name: str, field: str, fill: float = 0.0):
            arr = extract_per_step_field(oo, obs_name, field, T_full)
            if arr is None:
                return np.full(T, fill, np.float32)
            return np.asarray(arr[:T], dtype=np.float32)

        phi = np.clip(_f(phi_key, "potential"), 0.0, 1.0)
        r_kp = _f(kp_key, "reward")
        p1 = np.flatnonzero(_f(kp_key, "p1_done") > 0.5)
        p2 = np.flatnonzero(_f(kp_key, "p2_done") > 0.5)
        p1_end = int(p1[0]) + 1 if p1.size else None
        p2_end = int(p2[0]) + 1 \
            if p1_end is not None and p2.size else None

        channels: Dict[str, ChannelData] = {
            "r_potential": ChannelData(
                reward=(self.per_step_phi_coef
                        * np.minimum(phi, self._PHI_CAP)),
                is_terminated=False,
                actor_weight=float(self.r_potential_actor_weight)),
            "r_kp_p1": ChannelData(
                reward=r_kp, is_terminated=False,
                actor_weight=float(self.r_kp_p1_actor_weight)),
            "r_kp_p2": ChannelData(
                reward=r_kp, is_terminated=False,
                actor_weight=float(self.r_kp_p2_actor_weight)),
        }

        def _piece(a: int, b: int, drop, success: bool):
            sub = {}
            for nm, ch in channels.items():
                if nm in drop:
                    continue
                reward = ch.reward[a:b].copy()
                if success and nm == ("r_kp_p1" if a == 0
                                      else "r_kp_p2"):
                    reward[-1] += float(self.success_bonus)
                sub[nm] = ChannelData(
                    reward=reward,
                    # Each phase is a finite task: reaching its target
                    # OR exhausting the budget ends it.  Only the shared
                    # balance channel continues across the internal
                    # phase-1 -> phase-2 boundary.
                    is_terminated=(nm != "r_potential" or a > 0 or b == T),
                    actor_weight=ch.actor_weight)
            sctx = self.extract_sampling_ctx(ep, aid, T)
            if sctx is not None:
                sctx = {k: v[a:b] for k, v in sctx.items()}
            last_ob = obs[b] if b < T else fin
            return Trajectory(obs=obs[a:b], actions=acts[a:b],
                              last_obs=np.asarray(last_ob,
                                                  dtype=np.float32),
                              channels=sub, importance=1.0,
                              sampling_ctx=sctx)

        if p1_end is None:
            return [_piece(0, T, _P2, False)]
        trajs = [_piece(0, p1_end, _P2, True)]
        if p1_end < T:
            trajs.append(_piece(p1_end, p2_end or T, _P1,
                                p2_end is not None))
        return trajs

    # ------------------------------------------------------------------
    # Eval: per-skill phase success / kp-error / foot diagnostics
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        per_skill: Dict[str, Dict[str, List[float]]] = {}
        n_fall = n_eps = 0
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            skills = (ep.episode_options or {}).get("skills") or {}
            for aid, phi_key, kp_key in self._AGENT_OBS:
                skill = str(skills.get(aid, ""))
                if skill not in SKILLS:
                    continue
                n_eps += 1
                T = ep.num_frames
                oo = ep.observer_outputs
                phi = extract_per_step_field(
                    oo, phi_key, "potential", T)
                if phi is not None and \
                        (np.asarray(phi[:T]) < 0.3).sum() > 10:
                    n_fall += 1
                m = per_skill.setdefault(
                    skill, {"p1": [], "p2": [], "p2_uncond": [],
                            "p1_t": [], "p2_t": [],
                            "p1_max": [], "p2_max": [],
                            "foot": [], "hand": []})

                def _arr(field):
                    arr = extract_per_step_field(
                        oo, kp_key, field, T)
                    return (None if arr is None
                            else np.asarray(arr[:T], dtype=np.float32))

                phase = _arr("phase")
                mx = _arr("kp_max_err_m")
                p1f = _arr("p1_done")
                p2f = _arr("p2_done")
                if phase is None or mx is None or p1f is None:
                    continue
                p1_mask = phase < 1.5
                p2_mask = phase > 1.5
                p1_ok = bool((p1f > 0.5).any())
                p2_ok = p2f is not None and bool((p2f > 0.5).any())
                m["p1"].append(float(p1_ok))
                if p1_mask.any():
                    m["p1_max"].append(float(mx[p1_mask].min()))
                    m["p1_t"].append(float(
                        np.flatnonzero(p1f > 0.5)[0])
                        if p1_ok else float(T))
                if p1_ok:
                    m["p2"].append(float(p2_ok))
                    m["p2_uncond"].append(float(p2_ok))
                    if p2_mask.any():
                        m["p2_max"].append(float(mx[p2_mask].min()))
                        t2 = np.flatnonzero(p2f > 0.5)
                        m["p2_t"].append(
                            float(t2[0]) if t2.size else float(T))
                    foot = _arr("foot_disp_m")
                    if foot is not None:
                        v = foot[p2_mask]
                        if v.size:
                            m["foot"].append(float(v.max()))
                else:
                    m["p2_uncond"].append(0.0)
                hand = _arr("hand_err_m")
                if hand is not None and hand.size:
                    m["hand"].append(float(hand[-1]))

        p1r, p2r, pmax = [], [], []
        for skill, m in per_skill.items():
            for k, v in m.items():
                if v:
                    info[f"{skill}_{k}"] = round(float(np.mean(v)), 3)
            if m["p1"]:
                p1r.append(np.mean(m["p1"]))
            if m["p2"]:
                p2r.append(np.mean(m["p2"]))
            if m["p1_max"]:
                pmax.append(np.mean(m["p1_max"]))
        info["falls"] = round(n_fall / max(n_eps, 1), 3)
        # quality: p1 proximity proxy until p2 samples exist, then
        # success-rate blend (p1 rate + conditional p2 rate).
        quality = (float(np.clip(1.0 - np.mean(pmax) / 0.4, 0.0, 1.0))
                   if pmax else 0.0)
        if p2r:
            quality = (0.25 * float(np.mean(p1r))
                       + 0.25 * quality
                       + 0.5 * float(np.mean(p2r)))
        info["quality"] = round(quality, 3)

        improved = quality > self._best_quality \
            and info["falls"] < 0.2
        if improved:
            self._best_quality = quality
            self._last_best = update
        stop = (update - self._last_best >= self._no_improvement_limit
                and update >= self._min_updates)
        return {"is_new_best": improved, "stop_training": stop,
                "info": info}

    def state(self) -> dict:
        return {**super().state(),
                "best_quality": self._best_quality,
                "last_best": self._last_best}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._best_quality = float(state.get("best_quality", -1.0))
        self._last_best = int(state.get("last_best", 0))


EXPERIMENT_CLASS = StrikeKp2Phase
