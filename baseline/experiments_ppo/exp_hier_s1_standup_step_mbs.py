"""Hierarchy Step 1 — from-scratch standup + standard stepping (65-dim).

Per the revised bootstrip plan: NO warm-start.  The robot learns to get
up from a random fallen pose AND to settle into the standard dynamic
stepping state in a single training run, so the standing attractor it
forms *is* the cyclic stepping gait (avoids the planted-feet basin that
warm-started Stage 0 ran into: guard ~0.01, cycles ~0.03).

Channels (2):
    r_potential = 0.01 × φ(t),                       aw = 1.0 (fixed)
    r_std_clip  = clip_score + 0.3·pose + 0.5·cad,   aw = φ(t)²

The φ² gate (per user spec) suppresses the gait gradient while the
robot is down (φ≈0 → aw≈0) and opens it once standing (φ≈0.95 →
aw≈0.9); clip scores are already [0,1] and per-channel advantage
normalisation makes the raw scale unimportant.

clip_score = max_{clip,frame} exp(-||z_t - z_ref||² / 2σ²) in the
whitened 70-dim reduced-dynamics transition space — ~0.05 for a
planted-feet stance, ~0.5 for real stepping.  pose_score is the
far-field lure (raw weighted pose distance, σ=1.5).  cad is the
gait-cadence autocorrelation at the clip-derived half-period lag
(-s_t·s_{t-8}): 0 for planted standing, -1 for parking on one leg,
~+0.6 for cyclic stepping — the term that actually forces motion.

Blueprint: baseline/humanoid21/hierarchy/blueprints/hier_s1_standup_step_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from .exp_standup import Standup


class HierS1StandupStepMBS(Standup):
    """Standup + standard-clip stepping under the 65-dim command contract."""

    name = "hier_s1_standup_step_mbs"

    # --- Network ---
    obs_dim: int = 65
    action_dim: int = 21
    actor_blueprint = "init_policy_mbs_65d.yaml"

    # --- Reward channels ---
    _ch_pot = "r_potential"
    _ch_clip = "r_std_clip"
    _clip_aw_scale: float = 3.0          # peak actor weight when φ=1

    _CLIP_OBS = {"robot_a": "standard_clip_a", "robot_b": "standard_clip_b"}
    _FOOT_OBS = {"robot_a": "foot_state_a", "robot_b": "foot_state_b"}
    _AGENT_IDS = ("robot_a", "robot_b")

    # Track both criteria for best-checkpoint selection.
    _best_score: float = -1.0

    # ------------------------------------------------------------------
    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s1_standup_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    def reward_channels(self) -> tuple:
        return (
            RewardChannel(
                name=self._ch_pot, gamma=self._gamma,
                gae_lambda=self._gae_lambda),
            RewardChannel(
                name=self._ch_clip, gamma=self._gamma,
                gae_lambda=self._gae_lambda),
        )

    # ------------------------------------------------------------------
    def _build_agent_trajectory(self, episode, agent_id: str, obs_key: str):
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

        phi_arr = extract_per_step_field(
            episode.observer_outputs, obs_key, "potential", T_full,
        )
        if phi_arr is not None:
            phi_arr = np.clip(
                np.asarray(phi_arr[:T], dtype=np.float32), 0.0, 1.0)
            if phi_arr.size and self._ep_final_pots is not None:
                self._ep_final_pots.append(float(phi_arr[-1]))
        else:
            phi_arr = np.zeros(T, dtype=np.float32)

        clip_arr = extract_per_step_field(
            episode.observer_outputs, self._CLIP_OBS[agent_id],
            "std_clip", T_full,
        )
        pose_arr = extract_per_step_field(
            episode.observer_outputs, self._CLIP_OBS[agent_id],
            "std_pose", T_full,
        )
        cad_arr = extract_per_step_field(
            episode.observer_outputs, self._CLIP_OBS[agent_id],
            "std_cad", T_full,
        )
        clip_arr = (np.clip(np.asarray(clip_arr[:T], dtype=np.float32),
                            0.0, 1.0) if clip_arr is not None
                    else np.zeros(T, dtype=np.float32))
        pose_arr = (np.clip(np.asarray(pose_arr[:T], dtype=np.float32),
                            0.0, 1.0) if pose_arr is not None
                    else np.zeros(T, dtype=np.float32))
        cad_arr = (np.clip(np.asarray(cad_arr[:T], dtype=np.float32),
                           -1.0, 1.0) if cad_arr is not None
                   else np.zeros(T, dtype=np.float32))
        r_clip = clip_arr + 0.3 * pose_arr + 0.5 * cad_arr

        channels = {
            self._ch_pot: ChannelData(
                reward=(0.01 * phi_arr).astype(np.float32),
                is_terminated=False,
                actor_weight=np.ones(T, dtype=np.float32),
            ),
            self._ch_clip: ChannelData(
                reward=r_clip.astype(np.float32),
                is_terminated=False,
                actor_weight=(
                    self._clip_aw_scale * phi_arr * phi_arr
                ).astype(np.float32),
            ),
        }
        return [Trajectory(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    # ------------------------------------------------------------------
    # Eval — standup metrics + stepping quality on standing frames
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        success = result["info"].get("success", 0.0)

        clip_standing: List[float] = []
        cycles_list: List[float] = []
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id in self._AGENT_IDS:
                obs_key = self._AGENT_OBS[0][1] if agent_id == "robot_a" \
                    else self._AGENT_OBS[1][1]
                phi = extract_per_step_field(
                    ep.observer_outputs, obs_key, "potential", T)
                clip = extract_per_step_field(
                    ep.observer_outputs, self._CLIP_OBS[agent_id],
                    "std_clip", T)
                lc = extract_per_step_field(
                    ep.observer_outputs, self._FOOT_OBS[agent_id],
                    "left_foot_contact", T)
                rc = extract_per_step_field(
                    ep.observer_outputs, self._FOOT_OBS[agent_id],
                    "right_foot_contact", T)

                if phi is not None and clip is not None:
                    stand = np.asarray(phi[:T]) >= 0.9
                    if stand.any():
                        clip_standing.append(
                            float(np.asarray(clip[:T])[stand].mean()))

                if lc is not None and rc is not None and phi is not None:
                    standing = np.asarray(phi[:T]) >= 0.9
                    lc = np.asarray(lc[:T]) > 0.5
                    rc = np.asarray(rc[:T]) > 0.5
                    # count loaded-foot alternations while standing
                    diff = (lc.astype(np.float32) - rc.astype(np.float32))
                    diff[~standing] = 0.0
                    sig = np.sign(diff)
                    sig[diff == 0] = np.nan
                    valid = sig[~np.isnan(sig)]
                    flips = int(np.sum(valid[1:] * valid[:-1] < 0)) \
                        if len(valid) > 1 else 0
                    cycles_list.append(flips / 2.0)

        clip_mean = sum(clip_standing) / max(len(clip_standing), 1)
        cycles_mean = sum(cycles_list) / max(len(cycles_list), 1)

        # best requires the robot actually standing AND stepping
        score = (success if success >= 0.9 else 0.0) + clip_mean
        is_new_best = score > self._best_score
        if is_new_best:
            self._best_score = score

        result["is_new_best"] = is_new_best
        result["info"]["clip"] = round(clip_mean, 3)
        result["info"]["cycles"] = round(cycles_mean, 3)
        return result

    def state(self) -> dict:
        return {**super().state(), "best_score": self._best_score}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._best_score = float(state.get("best_score", -1.0))


EXPERIMENT_CLASS = HierS1StandupStepMBS
