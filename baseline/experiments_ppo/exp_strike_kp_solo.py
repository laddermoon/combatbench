"""strike_kp_solo — single-skill, from-scratch keypoint strike.

Inherits CombatExperimentPPOBase ONLY.  Variant of the
keypoint two-phase strike design (STRIKE_KEYPOINT_2PHASE_PLAN.md)
testing one hypothesis: the locomotion prior baked into the S2->SF
lineage (in-place stepping habits) actively interferes with fast
striking, so this experiment trains ONE strike skill FROM SCRATCH
— no actor resume, no inherited gait prior.

  Target   — a single skill (TARGET_SKILL), both robots train it.
  Warmup   — initial-state diversity comes from a FROZEN
             demonstrator: the final exported policy of
             train_skill_full_mbs_ppo_20261020_104315
             (policy_exports/u06689), restricted to the 7
             non-strike skills (4 moves, 2 turns, step), 0-30
             steps.  The warmup policy does NOT follow the
             training export — the learner is from-scratch, so
             its own early policy cannot produce useful starts.
  Channels — r_potential, r_kp_p1, r_kp_p2 (same as kp_2phase).
  Balance gate — strike channels are advantage-weighted by
             phi**2 per frame (phi = r_potential input, [0,1]):
             unbalanced frames contribute ~nothing to the strike
             gradient; standing is learned first.  r_potential
             itself is raised to the same base weight as the
             strike channels.
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
# Single target skill — the whole experiment is this one command.
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
TARGET_SKILL: str = "jab_r"


def _strike_row(side: int, land: Tuple[float, float, float],
                n: int) -> list:
    return [n, 0.0, 0.0, 0.0, 0.0,
            +1, side, land[0], land[1] * (1 if side > 0 else -1),
            land[2]]


_P1 = ("r_kp_p1",)
_P2 = ("r_kp_p2",)
# Warmup skills: every locomotion skill EXCEPT strikes — the frozen
# SF policy supplies the start-state diversity; strike mid-poses are
# excluded because that prior is precisely what this experiment
# isolates away.
_WARMUP_SKILLS = (
    "move_fwd", "move_back", "move_left", "move_right",
    "turn_left", "turn_right", "step",
)
_WARMUP_MAX_STEPS = 30

# Frozen warmup demonstrator: final policy export of
# train_skill_full_mbs_ppo_20261020_104315 (last export, later than
# the last checkpoint u06685).  Hard-pinned so warmup start states
# never drift with the from-scratch learner's own exports.
_SF_EXPORT_DIR = (
    Path(__file__).resolve().parent.parent / "runs"
    / "train_skill_full_mbs_ppo_20261020_104315"
    / "policy_exports" / "u06689")
WARMUP_POLICY_CLS = (
    f"file:{_SF_EXPORT_DIR}/policy.py"
    ":ExportedStateMixtureBoundedStdTruncNormPolicy")


def _warmup_row(skill: str) -> list:
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


class StrikeKpSolo(CombatExperimentPPOBase):
    """Single-skill from-scratch keypoint strike (fixed 66-dim)."""

    name = "strike_kp_solo"

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
    # Standing is the bottleneck for a from-scratch policy: raise the
    # balance channel to the strike channels' base weight, and gate the
    # strike channels' actor weight by phi**2 per frame so unbalanced
    # frames contribute (almost) nothing to the strike gradient.
    r_potential_actor_weight: float = 4.0
    r_kp_p1_actor_weight: float = 4.0
    r_kp_p2_actor_weight: float = 4.0
    phi_gate_power: float = 2.0
    success_bonus: float = 1.0
    _PHI_CAP: float = 0.90

    # --- Rollout schedule ---
    agent_used = "both"
    episodes_per_update: int = 1536
    max_updates: int = 12000
    eval_interval: int = 5
    eval_episodes: int = 64
    video_eval_interval: int = 5

    # --- Exploration: per-frame explore_factor on stochastic rollout
    # (SamplingSpec → sampling_ctx; additive on the pre-sigmoid std
    # input, range [-1,1]; eval path is deterministic and unaffected).
    explore_factor: float = 0.5

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
    # Jobs: non-strike warmup (0-30 steps, frozen SF policy) + the
    # single 30-step target strike on both robots.
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=self.explore_factor)
        env_bp = self._env_pb().materialize(max_steps=_STRIKE_STEPS)
        s = SKILLS[TARGET_SKILL]
        strike_row = _strike_row(s["side"], s["land"], _STRIKE_STEPS)

        def _warmup() -> Dict[str, Any]:
            skill = str(rng.choice(_WARMUP_SKILLS))
            return {"skill": skill,
                    "steps": int(rng.integers(0, _WARMUP_MAX_STEPS + 1)),
                    "schedule": [_warmup_row(skill)]}

        def _opts() -> Dict[str, Any]:
            return {
                "initial_distance": 2.0,
                "skills": {"robot_a": TARGET_SKILL,
                           "robot_b": TARGET_SKILL},
                "move_schedule": {
                    "robot_a": [strike_row],
                    "robot_b": [strike_row],
                },
                "warmup": {"robot_a": _warmup(),
                           "robot_b": _warmup()},
                # Frozen SF export — NOT policy_bp.cls.
                "warmup_policy_cls": WARMUP_POLICY_CLS,
            }

        return [Job(
            policy_a_bp=policy_bp, policy_b_bp=policy_bp,
            env_bp=env_bp, seed=int(base_seed + i),
            episode_options=_opts(),
            sampling_a=sampling, sampling_b=sampling,
            stochastic=stochastic,
        ) for i in range(n_episodes)]

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
        gate = np.power(phi, self.phi_gate_power).astype(np.float32)
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
                actor_weight=float(self.r_kp_p1_actor_weight) * gate),
            "r_kp_p2": ChannelData(
                reward=r_kp, is_terminated=False,
                actor_weight=float(self.r_kp_p2_actor_weight) * gate),
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
                aw = ch.actor_weight
                if not np.isscalar(aw):
                    aw = aw[a:b]
                sub[nm] = ChannelData(
                    reward=reward,
                    # Each phase is a finite task: reaching its target
                    # OR exhausting the budget ends it.  Only the shared
                    # balance channel continues across the internal
                    # phase-1 -> phase-2 boundary.
                    is_terminated=(nm != "r_potential" or a > 0 or b == T),
                    actor_weight=aw)
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
    # Eval: single-skill phase success / kp-error / foot diagnostics
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
                if skill != TARGET_SKILL:
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
                            "p1_mean": [], "p2_mean": [],
                            "foot": [], "hand": []})

                def _arr(field):
                    arr = extract_per_step_field(
                        oo, kp_key, field, T)
                    return (None if arr is None
                            else np.asarray(arr[:T], dtype=np.float32))

                phase = _arr("phase")
                mx = _arr("kp_max_err_m")
                mn = _arr("kp_mean_err_m")
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
                    if mn is not None:
                        m["p1_mean"].append(float(mn[p1_mask].min()))
                    m["p1_t"].append(float(
                        np.flatnonzero(p1f > 0.5)[0])
                        if p1_ok else float(T))
                if p1_ok:
                    m["p2"].append(float(p2_ok))
                    m["p2_uncond"].append(float(p2_ok))
                    if p2_mask.any():
                        m["p2_max"].append(float(mx[p2_mask].min()))
                        if mn is not None:
                            m["p2_mean"].append(float(mn[p2_mask].min()))
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
        # Patience restarts on EVERY resume: a checkpoint saved under a
        # different promotion criterion/semantics would otherwise carry
        # a stale last_best and stop immediately.
        self._best_quality = -1.0
        self._last_best = 10**9


EXPERIMENT_CLASS = StrikeKpSolo
