"""Fight stage A: attack_prep recipe vs a frozen mirror opponent.

Warm-start from ``attack_prep_step`` (s42g — the accepted run: ~11-16
real alternating punches/episode while holding ~0.85m of the target).
The scripted RandomMovePlugin target is replaced by a **frozen copy**
of the accepted attack_prep export; both robots share the same policy
architecture (ego-centric obs — the export drives either side).

What changes vs attack_prep:

  - Opponent = frozen policy (``OPPONENT_POLICY_PATH``), driven through
    ``Job.policy_a/b_bp`` on whichever side is NOT the learner.  Its
    sampling spec is neutral (explore_factor=0) — plain sampling from
    its own learned σ, no exploration boost.
  - CombatScoringPlugin gives the HP ledger; DamageBreakdownRewarder
    exposes per-step dealt/taken deltas.
  - StandingTriggeredForcePlugin removed — the opponent's strikes are
    the disturbance now.
  - Two new channels:
      r_damage_dealt — damage dealt this step (HP), aw = +W_dealt × φ²
                       (must be standing to land real punches)
      r_damage_taken — damage taken this step (HP), aw = −W_taken
                       (constant: getting hit is bad even while down;
                       pressure to stand up fast under attack)
    Damage rewards are raw HP deltas (~0–15 per landed step) — a
    connect is worth far more than any shaping tick, which is the
    point: striking should dominate once contact is reliable.

  - best-of-run / early stop composite:
        quality = mean_net_damage + 0.2 × punches_per_ep
    gated by success ≥ 0.9.  Early on (net ≈ 0) the punch term keeps
    tracking form improvements; once punches land, damage dominates.

Blueprint: baseline/humanoid21/end2end/fight_step_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field

from .exp_attack_prep_step import AttackPrepStep


class FightStep(AttackPrepStep):
    """attack_prep_step + damage channels + frozen policy opponent."""

    name = "fight_step"

    # --- Reward channels: attack_prep's 12 + damage dealt/taken ---
    _channel_names = AttackPrepStep._channel_names + (
        "r_damage_dealt", "r_damage_taken",
    )
    _channel_gammas = {
        **AttackPrepStep._channel_gammas,
        "r_damage_dealt": 0.99,
        "r_damage_taken": 0.99,
    }

    # --- Damage actor weights ---
    DAMAGE_DEALT_WEIGHT: float = 1.0    # × φ² — land hits while standing
    DAMAGE_TAKEN_WEIGHT: float = 0.5    # constant −W — always penalize

    # --- Frozen mirror opponent (accepted attack_prep run) ---
    OPPONENT_POLICY_PATH: str = (
        "baseline/runs/attack_prep_s42g/policy/policy.py"
    )
    OPPONENT_POLICY_CLS: str = (
        "ExportedStateMixtureBoundedStdTruncNormPolicy"
    )

    # --- Fight tracking state ---
    _f_best_quality: float = -1.0
    _f_quality_ema: float | None = None
    _f_last_best_update: int = -1
    _net_damage_per_ep: float = 0.0

    _DMG_OBS = {
        "robot_a": "damage_breakdown_a",
        "robot_b": "damage_breakdown_b",
    }

    # ------------------------------------------------------------------
    # Blueprint
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "fight_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    def _opponent_bp(self):
        from envs.framework.policy import PolicyBlueprint
        path = (
            Path(__file__).resolve().parents[2] / self.OPPONENT_POLICY_PATH
        )
        return PolicyBlueprint(
            cls=f"file:{path}:{self.OPPONENT_POLICY_CLS}",
            config={},
        )

    # ------------------------------------------------------------------
    # Jobs — learner on one side, frozen opponent on the other
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        from .exp_balance_step import _balance_explore_factor

        env_bps: Dict[str, Any] = {
            aid: self._env_pb().materialize(
                max_steps=self.max_steps,
                agent_id=aid,
            )
            for aid in self._AGENT_IDS
        }
        rng = np.random.default_rng(base_seed)
        learner_sampling = SamplingSpec(
            explore_factor=_balance_explore_factor)
        oppo_sampling = SamplingSpec()  # neutral: sample own σ, ef=0
        oppo_bp = self._opponent_bp()

        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            if agent_id == "robot_a":
                pa, pb = policy_bp, oppo_bp
                sa, sb = learner_sampling, oppo_sampling
            else:
                pa, pb = oppo_bp, policy_bp
                sa, sb = oppo_sampling, learner_sampling
            jobs.append(Job(
                policy_a_bp=pa,
                policy_b_bp=pb,
                env_bp=env_bps[agent_id],
                seed=seed,
                episode_options={
                    "agent_id": agent_id,
                    "initial_distance": initial_distance,
                },
                sampling_a=sa,
                sampling_b=sb,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Trajectories — attack_prep recipe + damage channels (learner only)
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> List[Trajectory]:
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key,
        )
        if not trajs:
            return trajs

        T_full = episode.num_frames
        oo = episode.observer_outputs
        dmg_key = self._DMG_OBS.get(agent_id)

        dealt = extract_per_step_field(oo, dmg_key, "dealt", T_full)
        taken = extract_per_step_field(oo, dmg_key, "taken", T_full)
        phi = extract_per_step_field(oo, phi_key, "potential", T_full)

        if dealt is None or taken is None or phi is None:
            for t in trajs:
                T = t.obs.shape[0]
                z = np.zeros(T, dtype=np.float32)
                ref = t.channels["r_fall"]
                for key in ("r_damage_dealt", "r_damage_taken"):
                    t.channels[key] = ChannelData(
                        reward=z.copy(),
                        is_terminated=ref.is_terminated,
                        actor_weight=z.copy(),
                    )
            return trajs

        phi_arr = np.clip(np.asarray(phi, dtype=np.float32), 0.0, 1.0)
        phi2 = (phi_arr ** 2).astype(np.float32)
        w_dealt = (self.DAMAGE_DEALT_WEIGHT * phi2).astype(np.float32)
        w_taken = np.full(
            T_full, -self.DAMAGE_TAKEN_WEIGHT, dtype=np.float32)

        for t in trajs:
            T = t.obs.shape[0]
            ref = t.channels["r_fall"]
            t.channels["r_damage_dealt"] = ChannelData(
                reward=np.asarray(dealt[:T], dtype=np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=w_dealt[:T],
            )
            t.channels["r_damage_taken"] = ChannelData(
                reward=np.asarray(taken[:T], dtype=np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=w_taken[:T],
            )
        return trajs

    # ------------------------------------------------------------------
    # Eval — attack_prep metrics + damage/KO ledger
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        info = result["info"]

        dealt_eps: List[float] = []
        taken_eps: List[float] = []
        net_eps: List[float] = []
        ko_wins = 0
        ko_losses = 0
        hp_self_end: List[float] = []
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            agent_id = str(ep.episode_options.get("agent_id", "robot_a"))
            sfx = "a" if agent_id == "robot_a" else "b"
            opp_sfx = "b" if agent_id == "robot_a" else "a"
            m = ep.episode_metrics
            taken = float(m.get(f"damage_taken_{sfx}", 0.0))
            dealt = float(m.get(f"damage_taken_{opp_sfx}", 0.0))
            hp_self = float(m.get(f"health_{sfx}", 0.0))
            hp_opp = float(m.get(f"health_{opp_sfx}", 0.0))
            dealt_eps.append(dealt)
            taken_eps.append(taken)
            net_eps.append(dealt - taken)
            hp_self_end.append(hp_self)
            if hp_opp <= 0.0 < hp_self:
                ko_wins += 1
            elif hp_self <= 0.0 < hp_opp:
                ko_losses += 1

        dealt_m = float(np.mean(dealt_eps)) if dealt_eps else 0.0
        taken_m = float(np.mean(taken_eps)) if taken_eps else 0.0
        net_m = float(np.mean(net_eps)) if net_eps else 0.0
        self._net_damage_per_ep = net_m

        punches = self._punches_per_ep
        quality = net_m + 0.2 * punches
        improved = (
            info.get("success", 0.0) >= 0.9
            and quality > self._f_best_quality
        )
        if improved:
            self._f_best_quality = quality
        if self._f_quality_ema is None:
            self._f_quality_ema = quality
        else:
            self._f_quality_ema = (
                0.85 * self._f_quality_ema + 0.15 * quality
            )
        trend_improved = (
            info.get("success", 0.0) >= 0.9
            and self._f_quality_ema > self._f_best_quality
        )
        if trend_improved:
            self._f_best_quality = self._f_quality_ema
        if improved or trend_improved:
            self._f_last_best_update = update
        if self._f_last_best_update < 0:
            self._f_last_best_update = update

        no_improvement = update - self._f_last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )

        info.update({
            "dmg_dealt": round(dealt_m, 2),
            "dmg_taken": round(taken_m, 2),
            "dmg_net": round(net_m, 2),
            "ko_win": ko_wins,
            "ko_loss": ko_losses,
            "hp_self": round(
                float(np.mean(hp_self_end)) if hp_self_end else 0.0, 1),
        })
        return {
            "is_new_best": improved or trend_improved,
            "stop_training": stop_training,
            "info": info,
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            **super().state(),
            "f_best_quality": self._f_best_quality,
            "f_last_best_update": self._f_last_best_update,
            "net_damage_per_ep": self._net_damage_per_ep,
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._f_best_quality = -1.0
        self._f_quality_ema = None
        self._f_last_best_update = -1
        self._net_damage_per_ep = 0.0


EXPERIMENT_CLASS = FightStep
