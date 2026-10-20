"""strike_2phase — clean, self-contained two-phase strike experiment.

Inherits CombatExperimentPPOBase ONLY (no skill_* chain).  Every
episode is a single strike command on the strike_2phase_env blueprint
(random skill warmup → one 30-step strike segment).

Scheme (STRIKE_2PHASE_PLAN.md):

  Phase 1  — approach: reach the ready pose = clip frame 0 anchored to
             the robot's current xy/yaw.  Channels: −MSE and −variance
             of the per-joint squared errors over the 17 non-ankle
             controlled joints (normalized units).
  Boundary — first step within 5% of frame 0; phase-1 MSE gets +1
             and its trajectory ends at this step (no curriculum).
  Phase 2  — execution: same potentials vs the LANDING frame
             (clip_land_frame), plus −rear-foot world displacement
             (anchored at the gate-pass frame). First step within 5%
             of landing gets +1 on phase-2 MSE; later steps are ignored.

Balance rail: a small dense r_potential (0.01·min(phi, 0.9)) rides every
trajectory — while phi ≥ 0.9 it is a constant bonus that does not shape
the strike action, and it only pulls the policy back when balance
degrades.

Channels: r_potential, r_strike_p1_mse, r_strike_p1_var,
          r_strike_p2_mse, r_strike_p2_var, r_strike_p2_foot.
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


_P1 = ("r_strike_p1_mse", "r_strike_p1_var")
_P2 = ("r_strike_p2_mse", "r_strike_p2_var", "r_strike_p2_foot")
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


class Strike2Phase(CombatExperimentPPOBase):
    """Strike-only two-phase reward split (clean reimplementation)."""

    name = "strike_2phase"

    # --- Network (66-dim CommandSimulator v2 contract) ---
    obs_dim: int = 66
    action_dim: int = 21
    actor_blueprint = "init_policy_mbs_66d.yaml"

    # --- Reward channels ---
    _channel_names = (
        "r_potential",
        "r_strike_p1_mse", "r_strike_p1_var",
        "r_strike_p2_mse", "r_strike_p2_var", "r_strike_p2_foot",
    )
    _channel_gammas = {k: 0.99 for k in _channel_names}
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01
    r_potential_actor_weight: float = 1.0
    r_strike_p1_mse_actor_weight: float = 4.0
    r_strike_p1_var_actor_weight: float = 4.0
    r_strike_p2_mse_actor_weight: float = 4.0
    r_strike_p2_var_actor_weight: float = 4.0
    r_strike_p2_foot_actor_weight: float = 8.0
    _PHI_CAP: float = 0.90

    # --- Fixed launch gate: all non-ankle joints <5% range vs frame 0 ---
    GATE_SPEC: Dict[str, float] = {"frames": 0, "frac": 0.05}

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
        ("robot_a", "standing_balance_a", "strike_track_a"),
        ("robot_b", "standing_balance_b", "strike_track_b"),
    )
    _BP = (Path(__file__).resolve().parent.parent
           / "humanoid21" / "hierarchy" / "blueprints"
           / "strike_2phase_env.yaml")

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
    # Jobs: one strike command per agent, fixed gate on every episode
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
                "launch_gate": dict(self.GATE_SPEC),
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
    # Trajectories: per-phase channel attachment + gate-boundary split
    # ------------------------------------------------------------------

    def build_trajectories(self, episodes) -> List[Trajectory]:
        trajs: List[Trajectory] = []
        for ep in episodes:
            for aid, phi_key, strike_key in self._AGENT_OBS:
                trajs.extend(self._traj(ep, aid, phi_key, strike_key))
        return trajs

    def _traj(self, ep, aid: str, phi_key: str,
              strike_key: str) -> List[Trajectory]:
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
        in_strike = _f(strike_key, "in_strike")
        gated = _f(strike_key, "gated")
        gate_qmax = _f(strike_key, "gate_qmax", np.inf)
        land_qmax = _f(strike_key, "land_qmax", np.inf)
        threshold = 2.0 * self.GATE_SPEC["frac"]
        ready = np.flatnonzero((in_strike > 0.5) &
                               (gated < 0.5) &
                               (gate_qmax <= threshold))
        gate_end = int(ready[0]) + 1 if ready.size else None
        landing = np.flatnonzero((in_strike > 0.5) &
                                 (gated > 0.5) &
                                 (land_qmax <= threshold))
        land_end = int(landing[0]) + 1 if gate_end is not None \
            and landing.size else None

        channels: Dict[str, ChannelData] = {
            "r_potential": ChannelData(
                reward=(self.per_step_phi_coef
                        * np.minimum(phi, self._PHI_CAP)),
                is_terminated=False,
                actor_weight=float(self.r_potential_actor_weight)),
        }
        for nm, fld, aw in (
                ("r_strike_p1_mse", "r_p1_mse",
                 self.r_strike_p1_mse_actor_weight),
                ("r_strike_p1_var", "r_p1_var",
                 self.r_strike_p1_var_actor_weight),
                ("r_strike_p2_mse", "r_p2_mse",
                 self.r_strike_p2_mse_actor_weight),
                ("r_strike_p2_var", "r_p2_var",
                 self.r_strike_p2_var_actor_weight),
                ("r_strike_p2_foot", "r_p2_foot",
                 self.r_strike_p2_foot_actor_weight)):
            channels[nm] = ChannelData(
                reward=_f(strike_key, fld),
                is_terminated=False,
                actor_weight=float(aw))

        def _piece(a: int, b: int, drop, success: bool):
            sub = {}
            for nm, ch in channels.items():
                if nm in drop:
                    continue
                reward = ch.reward[a:b].copy()
                if success and nm == ("r_strike_p1_mse" if a == 0
                                      else "r_strike_p2_mse"):
                    reward[-1] += 1.0
                sub[nm] = ChannelData(
                    reward=reward,
                    # Each strike phase is a finite task: reaching its
                    # target OR exhausting the 30-step budget ends it.
                    # Only the shared balance channel continues across
                    # an internal phase-1 -> phase-2 boundary.
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

        if gate_end is None:
            return [_piece(0, T, _P2, False)]
        trajs = [_piece(0, gate_end, _P2, True)]
        if gate_end < T:
            trajs.append(_piece(gate_end, land_end or T, _P1,
                                land_end is not None))
        return trajs

    # ------------------------------------------------------------------
    # Eval: per-skill gate/landing/foot diagnostics + best tracking
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        per_skill: Dict[str, Dict[str, List[float]]] = {}
        n_fall = n_eps = 0
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            skills = (ep.episode_options or {}).get("skills") or {}
            for aid, phi_key, strike_key in self._AGENT_OBS:
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
                    skill, {"gate_qmax": [], "ungated": [],
                            "gate_t": [], "land_t": [], "landed": [],
                            "land_qmax": [],
                            "hand": [], "foot": []})

                def _grab(field, dst, agg_fn):
                    arr = extract_per_step_field(
                        oo, strike_key, field, T)
                    if arr is None:
                        return
                    v = np.asarray(arr[:T], dtype=np.float32)
                    out = agg_fn(v)
                    if out is not None:
                        m[dst].append(float(out))

                gq = extract_per_step_field(
                    oo, strike_key, "gate_qmax", T)
                if gq is not None:
                    v = np.asarray(gq[:T])
                    gate_flags = extract_per_step_field(
                        oo, strike_key, "gated", T)
                    active = extract_per_step_field(
                        oo, strike_key, "in_strike", T)
                    if gate_flags is not None and active is not None:
                        v = v[(np.asarray(gate_flags[:T]) < 0.5)
                              & (np.asarray(active[:T]) > 0.5)]
                    else:
                        v = v[v > 0]
                    if v.size:
                        m["gate_qmax"].append(float(v.min()))
                g = extract_per_step_field(
                    oo, strike_key, "gated", T)
                if g is not None:
                    g_arr = np.asarray(g[:T])
                    gated_ep = bool(g_arr.max() > 0.5)
                    # The successful gate frame still emits gated=0;
                    # gate_t is latched on that frame, including T-1.
                    gt = extract_per_step_field(
                        oo, strike_key, "gate_t", T)
                    if gt is not None:
                        gated_ep = gated_ep or bool(
                            (np.asarray(gt[:T]) >= 0).any())
                    m["ungated"].append(float(not gated_ep))
                    if gated_ep:
                        _grab("gate_t", "gate_t",
                              lambda v: v[v >= 0].min()
                              if (v >= 0).any() else None)
                        _grab("foot_disp_m", "foot",
                              lambda v: v[v > 0].max()
                              if (v > 0).any() else None)
                        lq = extract_per_step_field(
                            oo, strike_key, "land_qmax", T)
                        if lq is not None:
                            v = np.asarray(lq[:T], dtype=np.float32)
                            v = v[g_arr > 0.5]
                            if v.size:
                                m["land_qmax"].append(float(v.min()))
                                m["landed"].append(float(
                                    (v <= 2.0 * self.GATE_SPEC["frac"]).any()))
                _grab("frame", "land_t",
                      lambda v: float(v.max()))
                _grab("hand_err_m", "hand", lambda v: float(v[-1])
                      if v.size else None)

        qmaxs, lands = [], []
        for skill, m in per_skill.items():
            for k, v in m.items():
                if v:
                    info[f"{skill}_{k}"] = round(float(np.mean(v)), 3)
            if m["gate_qmax"]:
                qmaxs.append(np.mean(m["gate_qmax"]))
            if m["landed"]:
                lands.append(np.mean(m["landed"]))
        info["falls"] = round(n_fall / max(n_eps, 1), 3)
        quality = (float(np.clip(1.0 - np.mean(qmaxs) / 0.4, 0.0, 1.0))
                   if qmaxs else 0.0)
        if lands:
            quality = 0.5 * quality + 0.5 * float(np.mean(lands))
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


EXPERIMENT_CLASS = Strike2Phase
