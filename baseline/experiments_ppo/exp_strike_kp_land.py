"""strike_kp_land — single-skill, single-target landing pose.

Inherits CombatExperimentPPOBase ONLY.  Further simplification of
strike_kp_solo: NO ready-pose (Clip0) phase at all.  The task is a
single endpoint — from an arbitrary (warmup-driven) start state,
reach the clip LANDING pose directly.  Success = all-15-keypoint
mean error < 0.05 m AND worst point < 0.10 m against T2, which is
built at the first step with the power-foot keypoint pinned to its
actual position (the fixed target IS the foot-stay constraint).

Hypothesis under test: stripping the intermediate gate lets the
from-scratch policy learn "arrive at the punch pose" much faster.
Channels: r_potential (AW4) + r_kp_land (AW = 4·phi² per frame).
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
_STRIKE_STEPS = 60
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


# Warmup skills: every locomotion skill EXCEPT strikes (frozen SF
# demonstrator supplies start-state diversity).
_WARMUP_SKILLS = (
    "move_fwd", "move_back", "move_left", "move_right",
    "turn_left", "turn_right", "step",
)
_WARMUP_MAX_STEPS = 30

# Frozen warmup demonstrator: final policy export of
# train_skill_full_mbs_ppo_20261020_104315.
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


class StrikeKpLand(CombatExperimentPPOBase):
    """Single-skill from-scratch landing-pose strike (66-dim)."""

    name = "strike_kp_land"

    # --- Network (66-dim CommandSimulator v2 contract — unchanged) ---
    obs_dim: int = 66
    action_dim: int = 21
    actor_blueprint = "init_policy_mbs_66d.yaml"

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_kp_land")
    _channel_gammas = {k: 0.99 for k in _channel_names}
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01
    r_potential_actor_weight: float = 4.0
    r_kp_land_actor_weight: float = 4.0
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

    # --- Exploration ---
    explore_factor: float = 0.5

    # --- PPO tuning ---
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
           / "strike_kp_land_env.yaml")

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
    # Jobs
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
    # Trajectories: single segment; success frame gets +1 and ends it.
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
        # In land_only mode the observer scores every frame vs the
        # landing target; p2_done is the success flag.
        done = np.flatnonzero(_f(kp_key, "p2_done") > 0.5)
        end = int(done[0]) + 1 if done.size else T
        success = done.size > 0

        rew_pot = (self.per_step_phi_coef
                   * np.minimum(phi, self._PHI_CAP))[:end].copy()
        rew_kp = r_kp[:end].copy()
        if success:
            rew_kp[-1] += float(self.success_bonus)
        sctx = self.extract_sampling_ctx(ep, aid, T)
        if sctx is not None:
            sctx = {k: v[:end] for k, v in sctx.items()}
        last_ob = obs[end] if end < T else fin
        return [Trajectory(
            obs=obs[:end], actions=acts[:end],
            last_obs=np.asarray(last_ob, dtype=np.float32),
            channels={
                "r_potential": ChannelData(
                    reward=rew_pot, is_terminated=True,
                    actor_weight=float(self.r_potential_actor_weight)),
                "r_kp_land": ChannelData(
                    reward=rew_kp, is_terminated=True,
                    actor_weight=float(self.r_kp_land_actor_weight)
                    * gate[:end]),
            },
            importance=1.0, sampling_ctx=sctx)]

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        land, land_t, mx, mn, foot, hand = [], [], [], [], [], []
        n_fall = n_eps = 0
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            skills = (ep.episode_options or {}).get("skills") or {}
            for aid, phi_key, kp_key in self._AGENT_OBS:
                if str(skills.get(aid, "")) != TARGET_SKILL:
                    continue
                n_eps += 1
                T = ep.num_frames
                oo = ep.observer_outputs
                phi = extract_per_step_field(
                    oo, phi_key, "potential", T)
                if phi is not None and \
                        (np.asarray(phi[:T]) < 0.3).sum() > 10:
                    n_fall += 1

                def _arr(field):
                    arr = extract_per_step_field(
                        oo, kp_key, field, T)
                    return (None if arr is None
                            else np.asarray(arr[:T], dtype=np.float32))

                mx_a, mn_a = _arr("kp_max_err_m"), _arr("kp_mean_err_m")
                dn = _arr("p2_done")
                if mx_a is None or dn is None:
                    continue
                ok = bool((dn > 0.5).any())
                land.append(float(ok))
                t = np.flatnonzero(dn > 0.5)
                land_t.append(float(t[0]) if t.size else float(T))
                mx.append(float(mx_a.min()))
                if mn_a is not None:
                    mn.append(float(mn_a.min()))
                f = _arr("foot_disp_m")
                if f is not None and f.size:
                    foot.append(float(f.max()))
                h = _arr("hand_err_m")
                if h is not None and h.size:
                    hand.append(float(h.min()))

        if land:
            info[f"{TARGET_SKILL}_land"] = round(float(np.mean(land)), 3)
        for k, v in (("land_t", land_t), ("max", mx), ("mean", mn),
                     ("foot", foot), ("hand", hand)):
            if v:
                info[f"{TARGET_SKILL}_{k}"] = round(
                    float(np.mean(v)), 3)
        info["falls"] = round(n_fall / max(n_eps, 1), 3)
        quality = (float(np.clip(1.0 - np.mean(mx) / 0.4, 0.0, 1.0))
                   if mx else 0.0)
        if land:
            quality = 0.5 * quality + 0.5 * float(np.mean(land))
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
        self._best_quality = -1.0
        self._last_best = 10**9


EXPERIMENT_CLASS = StrikeKpLand
