
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.common.rollout import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


# Paths resolved relative to the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_FALLBACK_POLICY_BP = PolicyBlueprint.load(
    "/data1/mono/things/combatbench/baseline/humanoid21/"
    "runs/curriculum_balance_recover_plus_v2_20260618_225956/policy_exports/"
    "u08845/policy_blueprint.yaml"
)
_FOLLOW_POLICY_BP = PolicyBlueprint.load(
    "/data1/mono/things/combatbench/baseline/humanoid21/runs/"
    "curriculum_follow_v2_20260620_132447/"
    "policy_exports/u09236/policy_blueprint.yaml"
)
_GATING_MODEL_DIR = str(
    Path(__file__).resolve().parent.parent / "gating_model_v2_u08845_10w"
)

class FightV2Config(CombatExperimentBase):
    """Fight curriculum experiment.

    The trained robot (robot_a) must learn to attack and fight the opponent
    (robot_b) while maintaining balance and utilizing distance-based fallback follow.
    The opponent policy is the frozen pre-trained follow policy.
    """

    name = "fight_v2"
    weight_target_total: float = 200.0
    weight_cap: float = 10.0
    reward_keys = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot", "r_radial", "r_tangential", "r_gate", "r_follow_gate", "r_damage")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
        "r_radial": 0.99,
        "r_tangential": 0.99,
        "r_gate": 0.99,
        "r_follow_gate": 0.99,
        "r_damage": 0.9,
    }

    BLUEPRINT = "fight_v2_env.yaml"

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

    # Small per-step survival bonus.
    per_step_survival_reward: float = 0.01
    # Penalty per step where MixedPolicy switches to fallback.
    gate_switch_penalty: float = -1.0
    follow_switch_penalty: float = -1.0

    # Fixed spawn distance (no randomization) for consistent curriculum metric.
    INITIAL_DISTANCE: float = 2.0

    # --- Curriculum: opponent movement speed per level (m/s) ---
    LEVEL_SPEEDS: Tuple[float, ...] = (0.0,)
    PROMOTE_HOLD_RATIO: float = 0.5
    PROMOTE_PATIENCE: int = 1

    # --- Stateful scheduler ---
    _level: int = 0
    _consecutive_pass: int = 0
    _hold_ratio: float = 0.0
    _survival_rate: float = 0.0
    _primary_ratio: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    @property
    def current_speed(self) -> float:
        """Current opponent movement speed (m/s), derived from level."""
        idx = max(0, min(self._level, len(self.LEVEL_SPEEDS) - 1))
        return float(self.LEVEL_SPEEDS[idx])

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
        """Wrap *primary_bp* in :class:`FightMixedPolicy` with standing and follow fallbacks."""
        return PolicyBlueprint(
            cls="baseline.humanoid21.curriculum.fight_mixed_policy:FightMixedPolicy",
            config={
                "primary_policy_bp": primary_bp.to_dict(),
                "follow_policy_bp": _FOLLOW_POLICY_BP.to_dict(),
                "fallback_policy_bp": _FALLBACK_POLICY_BP.to_dict(),
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

            # Map mixed_bp to the learning agent, and the pre-trained Follow policy to the opponent
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
        return (6.0, 1.0, 0.2, 0.2, 0.2, 0.2, 3.0, 1.0, 1.0, 1.0, 3.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Stateful scheduler weights update (keeps constant for fight single-stage)."""
        self._hold_ratio = float(eval_metrics.get("hold_ratio", 0.0))
        self._survival_rate = float(eval_metrics.get("survived", 0.0))
        self._primary_ratio = float(eval_metrics.get("fight_ratio", 0.0))
        return (6.0, 1.0, 0.2, 0.2, 0.2, 0.2, 3.0, 1.0, 1.0, 1.0, 3.0)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        # r_fall: per-step survival bonus + terminal signal
        fell = all(r.startswith("imbalance") for r in episode.agent_termination_reason.values())
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        # r_cross: cross-support balance reward
        r_cross = extract_per_step_scalar(oo, "cross_support", T)

        # Extract fields from the 'posture' observer
        joint_dev_arr = extract_per_step_field(episode.observer_outputs, "posture", "joint_deviation", T)
        joint_vel_arr = extract_per_step_field(episode.observer_outputs, "posture", "joint_vel", T)
        torso_tilt_arr = extract_per_step_field(episode.observer_outputs, "posture", "torso_tilt", T)
        foot_height_arr = extract_per_step_field(episode.observer_outputs, "posture", "foot_height", T)

        # Fallback if observer fields are missing
        if joint_dev_arr is None:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is None:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is None:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        # Vectorized calculation: small bonus when within normal range, linear penalty outside
        # 1. Joint deviation: normal <= 0.12, penalty factor 5.0
        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)

        # 2. Joint velocity: normal <= 0.5, penalty factor 1.0
        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)

        # 3. Torso tilt: normal <= 0.26 rad (~15 deg), penalty factor 3.0
        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)

        # 4. Foot height: normal <= 0.10m, penalty factor 5.0
        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)

        # r_radial / r_tangential: velocity decomposition
        from baseline.humanoid21.rewards.follow_opponent import compute_radial_tangential_rewards

        self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T)
        self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T)
        opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T)
        opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T)

        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            r_radial = np.zeros(T, dtype=np.float32)
            r_tangential = np.zeros(T, dtype=np.float32)
        else:
            self_xy = np.stack([self_x, self_y], axis=1)
            opp_xy = np.stack([opp_x, opp_y], axis=1)
            r_radial, r_tangential = compute_radial_tangential_rewards(self_xy, opp_xy)

        # r_damage: net damage (attack) reward
        r_damage = extract_per_step_scalar(oo, "damage", T)
        if r_damage is None:
            r_damage = np.zeros(T, dtype=np.float32)

        # r_gate / r_follow_gate: boundary transition penalties
        r_gate = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        r_follow_gate = np.full(T, self.per_step_survival_reward, dtype=np.float32)

        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is not None and "gating_mode" in extras:
            gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
            length = min(len(gating_mode), T)

            for t in range(length - 1):
                # Currently in primary "fight" mode
                if gating_mode[t] > 0.5:
                    if gating_mode[t+1] == 0.0:
                        # Switch to Recover
                        r_gate[t] = self.gate_switch_penalty
                    elif gating_mode[t+1] == -1.0:
                        # Switch to Follow
                        r_follow_gate[t] = self.follow_switch_penalty

                # Zero out fallback steps
                if gating_mode[t] < 0.5:
                    r_gate[t] = 0.0
                    r_follow_gate[t] = 0.0

            if length > 0 and gating_mode[length - 1] < 0.5:
                r_gate[length - 1] = 0.0
                r_follow_gate[length - 1] = 0.0

        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
            "r_radial": r_radial,
            "r_tangential": r_tangential,
            "r_gate": r_gate,
            "r_damage": r_damage,
            "r_follow_gate": r_follow_gate,
        }

    # ---- Episode metrics --------------------------------------------------

    def prepare_training_segments(
        self, episode,
    ) -> List[Tuple[int, int, float]]:
        """Split episode at fallback boundaries, keeping only primary (Fight) steps."""
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is None or "gating_mode" not in extras:
            w = min(self.weight_target_total / T, self.weight_cap)
            return [(0, T, w)]

        gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
        L = min(T, len(gating_mode))
        is_primary = gating_mode[:L] >= 0.5

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
        fell = all(r.startswith("imbalance") for r in episode.agent_termination_reason.values())

        oo = episode.observer_outputs
        self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T)
        self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T)
        opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T)
        opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T)

        mean_dist = 99.0
        min_dist = 99.0
        hold_ratio = 0.0

        if all(v is not None for v in (self_x, self_y, opp_x, opp_y)):
            raw_dist = np.sqrt((self_x - opp_x) ** 2 + (self_y - opp_y) ** 2)
            if len(raw_dist) > 0:
                mean_dist = float(np.mean(raw_dist))
                min_dist = float(np.min(raw_dist))
                hold_ratio = float(np.mean(raw_dist <= 1.1))

        # Net damage from r_damage
        r_damage = extract_per_step_scalar(oo, "damage", T)
        damage_dealt = float(np.sum(r_damage)) if r_damage is not None else 0.0

        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)

        fight_ratio = 1.0
        follow_ratio = 0.0
        recover_ratio = 0.0
        gating_switches = 0.0
        mean_p_safe = 1.0

        fall_on_fight = 0.0
        fall_on_follow = 0.0
        fall_on_recover = 0.0

        if extras is not None:
            if "gating_mode" in extras:
                gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
                if len(gating_mode) > 0:
                    fight_ratio = float(np.mean(gating_mode == 1.0))
                    follow_ratio = float(np.mean(gating_mode == -1.0))
                    recover_ratio = float(np.mean(gating_mode == 0.0))
                    gating_switches = float(np.sum(np.diff(gating_mode) != 0))

                    if fell:
                        if gating_mode[-1] == 1.0:
                            fall_on_fight = 1.0
                        elif gating_mode[-1] == -1.0:
                            fall_on_follow = 1.0
                        elif gating_mode[-1] == 0.0:
                            fall_on_recover = 1.0

            if "p_safe" in extras:
                p_safe = np.asarray(extras["p_safe"], dtype=np.float32).reshape(-1)
                if len(p_safe) > 0:
                    mean_p_safe = float(np.mean(p_safe))

        return {
            "survived": 0.0 if fell else 1.0,
            "level": float(self._level),
            "hold_ratio": hold_ratio,
            "fight_ratio": fight_ratio,
            "follow_ratio": follow_ratio,
            "recover_ratio": recover_ratio,
            "mean_dist": mean_dist,
            "min_dist": min_dist,
            "gating_switches": gating_switches,
            "mean_p_safe": mean_p_safe,
            "damage_dealt": damage_dealt,
            "fall_on_fight": fall_on_fight,
            "fall_on_follow": fall_on_follow,
            "fall_on_recover": fall_on_recover,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "opp_speed": round(self.current_speed, 3),
            "consecutive_pass": self._consecutive_pass,
            "hold_ratio": round(self._hold_ratio, 3),
            "survival_rate": round(self._survival_rate, 3),
            "fight_ratio": round(self._primary_ratio, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "hold_ratio": self._hold_ratio,
            "survival_rate": self._survival_rate,
            "fight_ratio": self._primary_ratio,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._hold_ratio = float(state.get("hold_ratio", 0.0))
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._primary_ratio = float(state.get("fight_ratio", 0.0))


# Singleton instance for the registry
EXPERIMENT = FightV2Config()
