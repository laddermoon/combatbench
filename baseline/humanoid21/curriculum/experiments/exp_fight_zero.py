"""Fight-Zero experiment: 3-way composite policy (fight/recover/standup).

Reward strategy (5 keys):
  r_fall          — per-step survival bonus + terminal fall penalty
  r_distance      — PBRS distance potential (exp bump at d_strike)
  r_damage_dealt  — damage dealt to opponent (short gamma)
  r_damage_taken  — damage taken from opponent (penalty, long gamma)
  r_gate          — gate switch penalty (discourages losing balance)

gating_mode values:
  1.0  -> fight   (primary, only this is trained on)
  0.0  -> recover (frozen fallback)
  -2.0 -> standup (frozen fallback)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.ppo_trainer import (
    _extract_per_step_field,
)
from baseline.humanoid21.rewards.distance_potential import compute_dense_distance_reward
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


# ---------------------------------------------------------------------------
# Frozen fallback policy paths
# ---------------------------------------------------------------------------

_FALLBACK_POLICY_BP = PolicyBlueprint.load(
    "/data1/mono/things/combatbench/baseline/humanoid21/"
    "runs/curriculum_balance_recover_plus_v2_20260618_225956/policy_exports/"
    "u08845/policy_blueprint.yaml"
)
_STANDUP_POLICY_BP = PolicyBlueprint.load(
    "/data1/mono/things/combatbench/baseline/runs/"
    "train_standing_balance_4stage_dense_ppo_resume5k_20260730_211100/"
    "policy_exports/u05000/policy_blueprint.yaml"
)
_FOLLOW_POLICY_BP = PolicyBlueprint.load(
    "/data1/mono/things/combatbench/baseline/humanoid21/runs/"
    "curriculum_follow_v2_20260620_132447/"
    "policy_exports/u09236/policy_blueprint.yaml"
)
_GATING_MODEL_DIR = str(
    Path(__file__).resolve().parent.parent / "gating_model_v2_u08845_10w"
)


class FightZeroConfig(CombatExperimentBase):
    """Fight curriculum with 3-way composite policy (fight/recover/standup).

    The trained robot (robot_a) learns to fight the opponent (robot_b) using
    a FightMixedPolicyV2 that switches between a primary fight policy, a
    frozen recover policy, and a frozen standup policy based on gating MLP
    safety predictions and root height.
    """

    name = "fight_zero"
    weight_target_total: float = 200.0
    weight_cap: float = 10.0

    reward_keys = (
        "r_distance",
        "r_damage_dealt",
        "r_damage_taken",
        "r_gate",
    )
    gammas = {
        "r_distance": 0.99,
        "r_damage_dealt": 0.90,
        "r_damage_taken": 0.99,
        "r_gate": 0.99,
    }

    BLUEPRINT = "fight_zero_env.yaml"

    # --- Training schedule ---
    max_updates: int = 20000
    eval_interval: int = 2

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- PPO tuning ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 1024
    eval_episodes: int = 128

    # --- Reward shaping parameters ---
    gate_switch_penalty: float = -1.0

    # --- Distance potential parameters ---
    d_strike: float = 0.7
    d_max: float = 8.0

    # --- Damage reward scaling ---
    damage_dealt_scale: float = 1.0
    damage_taken_scale: float = 1.0

    # --- Fixed spawn distance ---
    INITIAL_DISTANCE: float = 2.0

    # --- Stateful scheduler ---
    _fight_ratio: float = 0.0
    _survival_rate: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id=agent_id,
            oppo_agent_id="robot_b" if agent_id == "robot_a" else "robot_a",
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a")

    # ---- Policy blueprint helpers -----------------------------------------

    @staticmethod
    def _make_mixed_bp(primary_bp: PolicyBlueprint) -> PolicyBlueprint:
        """Wrap *primary_bp* in FightMixedPolicyV2 with recover and standup fallbacks."""
        return PolicyBlueprint(
            cls="baseline.humanoid21.curriculum.fight_mixed_policy_v2:FightMixedPolicyV2",
            config={
                "primary_policy_bp": primary_bp.to_dict(),
                "fallback_policy_bp": _FALLBACK_POLICY_BP.to_dict(),
                "standup_policy_bp": _STANDUP_POLICY_BP.to_dict(),
                "gating_model_dir": _GATING_MODEL_DIR,
            },
        )

    # ---- Job construction -------------------------------------------------

    def _build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        mixed_bp = self._make_mixed_bp(policy_bp)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: self._materialize_env(aid)
            for aid in ("robot_a", "robot_b")
        }

        initial_distance = self.INITIAL_DISTANCE

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)

            if agent_id == "robot_a":
                p_a = mixed_bp
                p_b = _FOLLOW_POLICY_BP
            else:
                p_a = _FOLLOW_POLICY_BP
                p_b = mixed_bp

            jobs.append((
                p_a,
                p_b,
                env_bps[agent_id],
                seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        """Compare eval metrics: survival → fight_ratio → damage_dealt."""
        if not best_esum:
            return True
        for key in ("survived", "fight_ratio", "damage_dealt"):
            cur = esum.get(key, 0.0)
            best = best_esum.get(key, 0.0)
            if cur != best:
                return cur > best
        return False

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (0.01, 1.0, 1.0, 1.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Constant weights for single-stage fight_zero."""
        self._fight_ratio = float(eval_metrics.get("fight_ratio", 0.0))
        self._survival_rate = float(eval_metrics.get("survived", 0.0))
        return (0.01, 1.0, 1.0, 1.0)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        # r_distance: Dense distance potential from approach_velocity observer
        self_x = _extract_per_step_field(oo, "approach_velocity", "self_x", T)
        self_y = _extract_per_step_field(oo, "approach_velocity", "self_y", T)
        opp_x = _extract_per_step_field(oo, "approach_velocity", "opp_x", T)
        opp_y = _extract_per_step_field(oo, "approach_velocity", "opp_y", T)

        if all(v is not None for v in (self_x, self_y, opp_x, opp_y)):
            self_xy = np.stack([self_x, self_y], axis=1)
            opp_xy = np.stack([opp_x, opp_y], axis=1)
            r_distance = compute_dense_distance_reward(
                self_xy, opp_xy,
                d_strike=self.d_strike,
                d_max=self.d_max,
                gamma=self.gammas["r_distance"],
            )
        else:
            r_distance = np.zeros(T, dtype=np.float32)

        # r_damage_dealt / r_damage_taken: from DamageBreakdownRewarder
        r_dealt = _extract_per_step_field(oo, "damage", "dealt", T)
        if r_dealt is None:
            r_dealt = np.zeros(T, dtype=np.float32)
        r_taken = _extract_per_step_field(oo, "damage", "taken", T)
        if r_taken is None:
            r_taken = np.zeros(T, dtype=np.float32)

        r_damage_dealt = self.damage_dealt_scale * r_dealt
        r_damage_taken = -self.damage_taken_scale * r_taken  # negative = penalty

        # r_gate: boundary transition penalty (0 on normal fight steps,
        #         gate_switch_penalty on fight→fallback transitions,
        #         0 on all non-fight steps)
        r_gate = np.zeros(T, dtype=np.float32)

        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is not None and "gating_mode" in extras:
            gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
            length = min(len(gating_mode), T)

            for t in range(length - 1):
                # Fight → fallback transition: penalize
                if gating_mode[t] > 0.5 and gating_mode[t + 1] < 0.5:
                    r_gate[t] = self.gate_switch_penalty

        return {
            "r_distance": r_distance,
            "r_damage_dealt": r_damage_dealt,
            "r_damage_taken": r_damage_taken,
            "r_gate": r_gate,
        }

    # ---- Episode metrics --------------------------------------------------

    def prepare_training_segments(
        self, episode,
    ) -> List[Tuple[int, int, float]]:
        """Split episode at fallback boundaries, keeping only fight steps."""
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is None or "gating_mode" not in extras:
            w = min(self.weight_target_total / T, self.weight_cap)
            return [(0, T, w)]

        gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
        L = min(T, len(gating_mode))
        is_primary = gating_mode[:L] >= 0.5  # fight mode = 1.0

        segments: List[Tuple[int, int]] = []
        start = None
        for t in range(L):
            if is_primary[t]:
                if start is None:
                    start = t
            else:
                if start is not None:
                    segments.append((start, t))
                    start = None
        if start is not None:
            segments.append((start, L))

        return [
            (s, e, min(self.weight_target_total / (e - s), self.weight_cap))
            for s, e in segments
        ]

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Per-episode metrics for fight evaluation and logging."""
        T = episode.num_frames
        fell = any(p.startswith("imbalance") for p in episode.termination_proposals)

        oo = episode.observer_outputs
        self_x = _extract_per_step_field(oo, "approach_velocity", "self_x", T)
        self_y = _extract_per_step_field(oo, "approach_velocity", "self_y", T)
        opp_x = _extract_per_step_field(oo, "approach_velocity", "opp_x", T)
        opp_y = _extract_per_step_field(oo, "approach_velocity", "opp_y", T)

        mean_dist = 99.0
        min_dist = 99.0
        hold_ratio = 0.0

        if all(v is not None for v in (self_x, self_y, opp_x, opp_y)):
            raw_dist = np.sqrt((self_x - opp_x) ** 2 + (self_y - opp_y) ** 2)
            if len(raw_dist) > 0:
                mean_dist = float(np.mean(raw_dist))
                min_dist = float(np.min(raw_dist))
                hold_ratio = float(np.mean(raw_dist <= 1.1))

        # Damage from breakdown
        dealt_arr = _extract_per_step_field(oo, "damage", "dealt", T)
        taken_arr = _extract_per_step_field(oo, "damage", "taken", T)
        damage_dealt = float(np.sum(dealt_arr)) if dealt_arr is not None else 0.0
        damage_taken = float(np.sum(taken_arr)) if taken_arr is not None else 0.0

        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)

        fight_ratio = 1.0
        recover_ratio = 0.0
        standup_ratio = 0.0
        gating_switches = 0.0
        mean_p_safe = 1.0

        if extras is not None:
            if "gating_mode" in extras:
                gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
                if len(gating_mode) > 0:
                    fight_ratio = float(np.mean(gating_mode == 1.0))
                    recover_ratio = float(np.mean(gating_mode == 0.0))
                    standup_ratio = float(np.mean(gating_mode == -2.0))
                    gating_switches = float(np.sum(np.diff(gating_mode) != 0))

            if "p_safe" in extras:
                p_safe_arr = np.asarray(extras["p_safe"], dtype=np.float32).reshape(-1)
                if len(p_safe_arr) > 0:
                    mean_p_safe = float(np.mean(p_safe_arr))

        return {
            "survived": 0.0 if fell else 1.0,
            "fight_ratio": fight_ratio,
            "recover_ratio": recover_ratio,
            "standup_ratio": standup_ratio,
            "gating_switches": gating_switches,
            "mean_p_safe": mean_p_safe,
            "mean_dist": mean_dist,
            "min_dist": min_dist,
            "hold_ratio": hold_ratio,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
        }


    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "fight_ratio": round(self._fight_ratio, 3),
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "fight_ratio": self._fight_ratio,
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._fight_ratio = float(state.get("fight_ratio", 0.0))
        self._survival_rate = float(state.get("survival_rate", 0.0))


EXPERIMENT = FightZeroConfig()
