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
  dist >= 0.5m AND hand→own-torso dist <= 0.55m AND root height
  >= 1.0m, held 3 consecutive steps → ``strike_done``; 60 steps
  without it → ``retract_timeout``.

Reward (new channel ``r_strike``, γ=0.99):

    settle frames      r[t] = 0.5 × step_damage      (first half,
                                                      paid per step)
    strike_done frame  r[t] += 0.5 × Q               (second half)
    retract_timeout    second half withheld (fail penalty = 0 in v1 —
                       the withheld half IS the punishment)
    actor_weight = 1.0 — the event reward IS the task, no gating.

``r_retract`` (γ=0.99): dense retract-phase guidance.  Pure masking
turned out too sparse — s42b's retract_timeout RATE rose once the
slowed target stopped walking away by itself and the learner had to
actively create 0.5m separation with no reward telling it how.
Potential-difference shaping on RETRACT frames:

    φ = clip(d_opp/0.5, 0,1) + clip((0.8 − d_home)/0.25, 0,1) ∈ [0,2]
    r[t] = 0.3 · (φ[t] − φ[t−1])          (first retract frame uses
                                           the last SETTLE φ)

Parking can't farm it (Δφ=0 while static; oscillation nets ~0 over a
cycle) and the max total potential (0.6) stays below the withheld
half for any decent hit.

Phase gating: on RETRACT frames every shaping channel except r_fall
and r_strike is masked (reward=0, actor_weight=0).  Post-hit the only
remaining positive payoff is the withheld half — the inherited
follow/arm rewards would otherwise keep paying the policy for hugging
the target, which directly fights the retract combo (observed: s42a
episodes timed out pressed against the target at 0.14-0.27m).  r_fall
stays so falling during retract is still punished.

Distance gating needs no new code: r_radial/r_tangential only pay
outside the hold radius and r_face ramps in below D_FACE — the
"follow outside / strike inside" blend is already in the inherited
channels.  The four arm-shaping channels stay ON pre-hit as
scaffolding (user decision) so the warm-started punching behavior has
continuity while the sparse hit reward emerges.

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

    # --- Strike channels ---
    _channel_names = AttackPrepStep._channel_names + (
        "r_strike", "r_retract")
    _channel_gammas = {
        **AttackPrepStep._channel_gammas,
        "r_strike": 0.99,
        "r_retract": 0.99,
    }

    STRIKE_ACTOR_WEIGHT: float = 1.0
    # v1: withheld second half is the only retract-timeout cost.
    RETRACT_FAIL_PENALTY: float = 0.0

    # Retract potential shaping (must match strike_step_env.yaml /
    # StrikeSettlementPlugin defaults).
    RETRACT_W: float = 0.6
    _RET_SEP_REF: float = 0.5        # == retract_dist
    _RET_HOME_REF: float = 0.55      # == hand_home_dist
    _RET_EXTENDED_REF: float = 0.8   # ~fully extended arm

    # Phase-2 (RETRACT) frames: zero reward AND actor weight on every
    # shaping channel so nothing but the withheld second half (and the
    # r_fall fall penalty) is live post-settle.  Without this the
    # follow/hand-dist channels pay the policy for staying glued to
    # the target — the dominant s42a retract_timeout failure mode.
    _RETRACT_MASKED_CHANNELS = (
        "r_potential", "r_left_foot", "r_right_foot", "r_torso",
        "r_radial", "r_tangential", "r_face",
        "r_left_elbow", "r_right_elbow",
        "r_left_hand_dist", "r_right_hand_dist",
    )

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
        return self._apply_strike_channels(trajs, episode, agent_id)

    def _apply_strike_channels(
        self, trajs: List[Trajectory], episode, agent_id: str,
    ) -> List[Trajectory]:
        """Attach r_strike / r_retract + RETRACT masking to trajs.

        Split out so StrikeSparseStep can reuse it on top of the
        FollowStep trajectory recipe (no arm shaping).
        """
        T_full = episode.num_frames
        oo = episode.observer_outputs
        strike_key = self._STRIKE_OBS.get(agent_id)

        step_damage = extract_per_step_field(
            oo, strike_key, "step_damage", T_full)
        damage_total = extract_per_step_field(
            oo, strike_key, "damage_total", T_full)
        phase = extract_per_step_field(
            oo, strike_key, "phase", T_full)
        d_opp = extract_per_step_field(
            oo, strike_key, "d_opp", T_full)
        d_home = extract_per_step_field(
            oo, strike_key, "d_home", T_full)

        # RETRACT-phase masking: no shaping rewards after the settle
        # window — the withheld second half is the only prize left.
        retract_mask = None
        if phase is not None:
            retract_mask = np.asarray(phase) == 2.0   # PHASE_RETRACT
            for t in trajs:
                T = t.obs.shape[0]
                m = retract_mask[:T]
                for cname in self._RETRACT_MASKED_CHANNELS:
                    ch = t.channels.get(cname)
                    if ch is not None:
                        ch.reward[m] = 0.0
                        ch.actor_weight[m] = 0.0

        # r_retract: potential-difference shaping on RETRACT frames —
        # dense guidance toward the retract combo without a farmable
        # level reward (Δφ=0 while parked, oscillation nets ~0).
        r_retract = np.zeros(T_full, dtype=np.float64)
        if retract_mask is not None and d_opp is not None \
                and d_home is not None:
            phi = (
                np.clip(np.asarray(d_opp) / self._RET_SEP_REF, 0.0, 1.0)
                + np.clip(
                    (self._RET_EXTENDED_REF - np.asarray(d_home))
                    / (self._RET_EXTENDED_REF - self._RET_HOME_REF),
                    0.0, 1.0)
            )
            dphi = np.diff(phi, prepend=phi[0])
            r_retract = self.RETRACT_W * dphi * retract_mask

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
                t.channels["r_retract"] = ChannelData(
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
            t.channels["r_retract"] = ChannelData(
                reward=r_retract[:T].astype(np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=np.full(T, 1.0, dtype=np.float32),
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
