"""Strike-once + retract stage on the attack_prep stack.

Stage intent (user design): replace the open-ended "keep punching a
moving target" loop with a single-shot episode — stand up from a
random fall, chase a pose-frozen kinematic target, land ONE hand
strike, settle its damage, then retract the arm.  Episode ends on
retract success (``strike_done``) or retract timeout.

Opponent: ``RandomPoseInitPlugin`` samples a perturbed
standing/leaning pose once per episode; ``PoseMovePlugin`` freezes it
and slides it at a per-episode random speed with slow random heading
drift — the learner sees varied target geometry and must reach it.

Settlement (``StrikeSettlementPlugin`` + ``StrikeStateObserver``):

* First hand→{head,torso} contact locks the striking hand.
* SETTLE (10 action steps): locked-hand damage Q accrues under the
  standard CombatScoring formula.
* RETRACT: score frozen; retract combo = hand→opp {head,torso}
  dist >= 0.5m AND elbow_norm >= 0.7 AND root height >= 1.0m, held
  3 consecutive steps → ``strike_done``; 60 steps without it →
  ``retract_timeout``.

Reward (new channel ``r_strike``, γ=0.99):

    settle frames      r[t] = 0.5 × step_damage      (first half,
                                                      paid per step)
    strike_done frame  r[t] += 0.5 × Q               (second half)
    retract_timeout    second half withheld (fail penalty = 0 in v1 —
                       the withheld half IS the punishment)
    actor_weight = 1.0 — the event reward IS the task, no gating.

Distance gating needs no new code: r_radial/r_tangential only pay
outside the hold radius and r_face ramps in below D_FACE — the
"follow outside / strike inside" blend is already in the inherited
channels.  The four arm-shaping channels stay ON as scaffolding (user
decision) so the warm-started punching behavior has continuity while
the sparse hit reward emerges; a fade experiment can follow.

best-of-run / early stop composite:
    quality = mean per-episode strike return
              = mean(0.5·Q + 0.5·Q·retracted)  over all eval episodes
    recorded only when success >= 0.9 (standup still solid).

Known limits (v1): the kinematic target is not knocked back; whiffed
swings before the first hit are not penalized; the policy has no
explicit "hit registered" observation (obs stays 99-dim so the
attack_prep checkpoint warm-starts cleanly) — retract has to be
learned from proprioception + context.

Blueprint: baseline/humanoid21/end2end/strike_step_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field

from .exp_attack_prep_step import AttackPrepStep
from .exp_balance_step import _balance_explore_factor


class StrikeStep(AttackPrepStep):
    """attack_prep recipe + single-strike settle/retract event reward."""

    name = "strike_step"
    agent_used = "random"

    # Resumed from the attack_prep run — headroom for the sparse
    # event reward to dominate shaping.
    max_updates: int = 10000

    # --- Strike channel ---
    _channel_names = AttackPrepStep._channel_names + ("r_strike",)
    _channel_gammas = {
        **AttackPrepStep._channel_gammas,
        "r_strike": 0.99,
    }

    STRIKE_ACTOR_WEIGHT: float = 1.0
    # v1: withheld second half is the only retract-timeout cost.
    RETRACT_FAIL_PENALTY: float = 0.0

    _STRIKE_OBS = {
        "robot_a": "strike_state_a",
        "robot_b": "strike_state_b",
    }

    # --- Tracking state ---
    _s_best_quality: float = -1.0
    _s_quality_ema: float | None = None
    _s_last_best_update: int = -1
    _mean_strike_return: float = 0.0

    # ------------------------------------------------------------------
    # Blueprint — attack_prep env + pose/move/settle plugins
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "strike_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    # ------------------------------------------------------------------
    # Trajectories — attack_prep recipe + r_strike (learner only)
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
        strike_key = self._STRIKE_OBS.get(agent_id)

        step_damage = extract_per_step_field(
            oo, strike_key, "step_damage", T_full)
        damage_total = extract_per_step_field(
            oo, strike_key, "damage_total", T_full)

        if step_damage is None or damage_total is None:
            for t in trajs:
                T = t.obs.shape[0]
                z = np.zeros(T, dtype=np.float32)
                ref = t.channels["r_fall"]
                t.channels["r_strike"] = ChannelData(
                    reward=z.copy(),
                    is_terminated=ref.is_terminated,
                    actor_weight=z.copy(),
                )
            return trajs

        term_reason = episode.agent_termination_reason.get(agent_id, "")

        # First half: 0.5 × locked-hand damage, paid on each settle
        # frame as it accrues.  step_damage is 0 outside SETTLE.
        r_strike = 0.5 * np.asarray(step_damage, dtype=np.float64)

        # Second half: 0.5 × Q on the retract-completion frame (the
        # terminal frame of a strike_done episode).
        if term_reason == "strike_done" and T_full > 0:
            r_strike[T_full - 1] += (
                0.5 * float(damage_total[T_full - 1])
            )
        elif term_reason == "retract_timeout" and T_full > 0:
            r_strike[T_full - 1] -= self.RETRACT_FAIL_PENALTY

        for t in trajs:
            T = t.obs.shape[0]
            ref = t.channels["r_fall"]
            t.channels["r_strike"] = ChannelData(
                reward=r_strike[:T].astype(np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=np.full(
                    T, self.STRIKE_ACTOR_WEIGHT, dtype=np.float32),
            )
        return trajs

    # ------------------------------------------------------------------
    # Jobs — FollowStep's recipe (speed now sampled inside the plugin)
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        env_bps: Dict[str, Any] = {
            aid: self._env_pb().materialize(
                max_steps=self.max_steps,
                agent_id=aid,
                oppo_agent_id=(
                    "robot_b" if aid == "robot_a" else "robot_a"
                ),
            )
            for aid in self._AGENT_IDS
        }
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=_balance_explore_factor)
        force = self.current_force
        dur_min, dur_max = self.current_duration_range
        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bps[agent_id],
                seed=seed,
                episode_options={
                    "agent_id": agent_id,
                    "initial_distance": initial_distance,
                    "impulse_params": {
                        agent_id: {
                            "force": force,
                            "duration_min": dur_min,
                            "duration_max": dur_max,
                            "body": "torso",
                            "seed": seed,
                        },
                    },
                },
                sampling_a=sampling,
                sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Eval — attack_prep metrics + strike ledger
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        info = result["info"]

        hits: List[float] = []
        q_hit: List[float] = []
        q_all: List[float] = []
        retracted: List[float] = []
        strike_returns: List[float] = []
        t2hits: List[float] = []
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            agent_id = str(
                ep.episode_options.get("agent_id", "robot_a"))
            m = ep.episode_metrics
            hit_step = float(m.get(f"{agent_id}_strike_hit_step", -1))
            q = float(m.get(f"{agent_id}_strike_damage_total", 0.0))
            done = (
                ep.agent_termination_reason.get(agent_id, "")
                == "strike_done"
            )
            hit = hit_step >= 0
            hits.append(float(hit))
            q_all.append(q if hit else 0.0)
            if hit:
                q_hit.append(q)
                t2hits.append(hit_step)
                retracted.append(float(done))
            strike_returns.append(0.5 * q + (0.5 * q if done else 0.0))

        hit_rate = float(np.mean(hits)) if hits else 0.0
        mean_q = float(np.mean(q_hit)) if q_hit else 0.0
        mean_q_all = float(np.mean(q_all)) if q_all else 0.0
        retract_rate = (
            float(np.mean(retracted)) if retracted else 0.0
        )
        mean_t2h = float(np.mean(t2hits)) if t2hits else 0.0
        mean_return = (
            float(np.mean(strike_returns)) if strike_returns else 0.0
        )
        self._mean_strike_return = mean_return

        quality = mean_return
        improved = (
            info.get("success", 0.0) >= 0.9
            and quality > self._s_best_quality
        )
        if improved:
            self._s_best_quality = quality
        if self._s_quality_ema is None:
            self._s_quality_ema = quality
        else:
            self._s_quality_ema = (
                0.85 * self._s_quality_ema + 0.15 * quality
            )
        trend_improved = (
            info.get("success", 0.0) >= 0.9
            and self._s_quality_ema > self._s_best_quality
        )
        if trend_improved:
            self._s_best_quality = self._s_quality_ema
        if improved or trend_improved:
            self._s_last_best_update = update
        if self._s_last_best_update < 0:
            self._s_last_best_update = update

        no_improvement = update - self._s_last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )

        info.update({
            "hit": round(hit_rate, 3),
            "Q": round(mean_q, 2),
            "Q_all": round(mean_q_all, 2),
            "retract": round(retract_rate, 3),
            "t2hit": round(mean_t2h, 1),
            "sret": round(mean_return, 2),
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
            "s_best_quality": self._s_best_quality,
            "s_last_best_update": self._s_last_best_update,
            "mean_strike_return": self._mean_strike_return,
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._s_best_quality = -1.0
        self._s_quality_ema = None
        self._s_last_best_update = -1
        self._mean_strike_return = 0.0


EXPERIMENT_CLASS = StrikeStep
