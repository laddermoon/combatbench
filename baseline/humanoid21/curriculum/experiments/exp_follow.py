
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.ppo_trainer import (
    _extract_per_step_field,
    _extract_per_step_scalar,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


# Paths resolved relative to the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_RANDOM_POLICY_BP = PolicyBlueprint.load(
    _PROJECT_ROOT / "policy" / "blueprints" / "random.yaml"
)
_FALLBACK_POLICY_BP = PolicyBlueprint.load(
    "/data1/mono/things/combatbench/baseline/humanoid21/runs/"
    "curriculum_balance_recover_plus_20260612_103559/"
    "policy_exports/u10000/policy_blueprint.yaml"
)
_GATING_MODEL_DIR = str(
    Path(__file__).resolve().parent.parent / "gating_model_plus_mix_level"
)

class FollowConfig(ExperimentConfig):
    """Follow-opponent curriculum experiment.

    The trained robot (robot_a) must learn to follow a randomly-moving
    opponent (robot_b) while maintaining balance.  The opponent's movement
    is driven by :class:`RandomMovePlugin` inside the env; the opponent
    policy itself is a no-op random policy.

    Curriculum knob: the opponent's movement speed
    (``random_move_speed`` env parameter), indexed by ``LEVEL_SPEEDS``.
    """

    name = "follow"
    reward_keys = ("r_fall", "r_cross", "r_radial", "r_tangential", "r_gate")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_radial": 0.99,
        "r_tangential": 0.99,
        "r_gate": 0.99,
    }

    BLUEPRINT = "follow_env.yaml"

    max_updates: int = 20000

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
    eval_interval: int = 2

    # --- Video recording ---
    video_eval_interval: int = 2

    # Small per-step survival bonus.
    per_step_survival_reward: float = 0.01
    # Penalty per step where MixedPolicy switches to fallback.
    gate_switch_penalty: float = -1.0

    # Fixed spawn distance (no randomization) for consistent curriculum metric.
    INITIAL_DISTANCE: float = 2.0

    # --- Curriculum: opponent movement speed per level (m/s) ---
    LEVEL_SPEEDS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
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

    def _materialize_env(
        self, agent_id: str, random_move_speed: float,
    ) -> EnvBlueprint:
        return self._env_pb().materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id=agent_id,
            oppo_agent_id="robot_b" if agent_id == "robot_a" else "robot_a",
            random_move_speed=random_move_speed,
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a", self.current_speed)

    # ---- Policy blueprint helpers -----------------------------------------

    @staticmethod
    def _make_mixed_bp(primary_bp: PolicyBlueprint) -> PolicyBlueprint:
        """Wrap *primary_bp* in :class:`MixedPolicy` with a standing fallback."""
        return PolicyBlueprint(
            cls="baseline.humanoid21.curriculum.mixed_policy:MixedPolicy",
            config={
                "primary_policy_bp": primary_bp.to_dict(),
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
        speed = self.current_speed

        env_bps: Dict[str, EnvBlueprint] = {
            aid: self._materialize_env(aid, speed)
            for aid in ("robot_a", "robot_b")
        }

        initial_distance = self.INITIAL_DISTANCE

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            
            # Map mixed_bp to the learning agent, and _RANDOM_POLICY_BP to the opponent
            if agent_id == "robot_a":
                p_a = mixed_bp
                p_b = _RANDOM_POLICY_BP
            else:
                p_a = _RANDOM_POLICY_BP
                p_b = mixed_bp
                
            jobs.append((
                p_a, p_b,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        """Compare eval metrics: survival → primary_ratio → hold_ratio."""
        if not best_esum:
            return True
        for key in ("survived", "primary_ratio", "hold_ratio"):
            cur = esum.get(key, 0.0)
            best = best_esum.get(key, 0.0)
            if cur != best:
                return cur > best
        return False

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (6.0, 1.0, 3.0, 1.0, 1.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Advance the opponent speed level when hold_ratio is high enough."""
        hold_ratio = float(eval_metrics.get("hold_ratio", 0.0))
        self._hold_ratio = hold_ratio
        self._survival_rate = float(eval_metrics.get("survived", 0.0))
        self._primary_ratio = float(eval_metrics.get("primary_ratio", 0.0))

        if self._level < len(self.LEVEL_SPEEDS) - 1:
            if hold_ratio >= self.PROMOTE_HOLD_RATIO:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0

        return (6.0, 1.0, 3.0, 1.0, 1.0)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        # r_fall: per-step survival bonus + terminal signal
        fell = "imbalance" in episode.termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        # r_cross: cross-support balance reward
        r_cross = _extract_per_step_scalar(oo, "cross_support", T)

        # r_hold: in-zone hold reward
        #r_hold = _extract_per_step_field(oo, "in_zone_hold", "reward", T)
        #if r_hold is None:
        #    r_hold = np.zeros(T, dtype=np.float32)

        # r_radial / r_tangential: velocity decomposition (trainer-side post-processing)
        from baseline.humanoid21.rewards.follow_opponent import compute_radial_tangential_rewards

        self_x = _extract_per_step_field(oo, "approach_velocity", "self_x", T)
        self_y = _extract_per_step_field(oo, "approach_velocity", "self_y", T)
        opp_x = _extract_per_step_field(oo, "approach_velocity", "opp_x", T)
        opp_y = _extract_per_step_field(oo, "approach_velocity", "opp_y", T)

        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            r_radial = np.zeros(T, dtype=np.float32)
            r_tangential = np.zeros(T, dtype=np.float32)
        else:
            self_xy = np.stack([self_x, self_y], axis=1)
            opp_xy = np.stack([opp_x, opp_y], axis=1)
            r_radial, r_tangential = compute_radial_tangential_rewards(self_xy, opp_xy)

        # r_gate: penalty when MixedPolicy switches to fallback mode
        r_gate = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is not None and "gating_mode" in extras:
            gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
            length = min(len(gating_mode), T)
            is_primary = gating_mode[:length] >= 0.5
            for t in range(length - 1):
                if is_primary[t] and not is_primary[t+1]:
                    r_gate[t] = self.gate_switch_penalty
                elif not is_primary[t]:
                    r_gate[t] = 0.0
            if length > 0 and not is_primary[length - 1]:
                r_gate[length - 1] = 0.0

        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            #"r_hold": r_hold,
            "r_radial": r_radial,
            "r_tangential": r_tangential,
            "r_gate": r_gate,
        }

    # ---- Episode metrics --------------------------------------------------

    def segment_episode(self, episode) -> List[Tuple[int, int]]:
        """Split episode at fallback boundaries, keeping only primary steps.

        Steps where the gating model switched to the fallback (balance
        recovery) policy are excluded from training so the actor is never
        trained on actions it did not produce.
        """
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is None or "gating_mode" not in extras:
            return [(0, T)]

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

        return segments

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Per-episode metrics for eval comparison and logging."""
        T = episode.num_frames
        fell = "imbalance" in episode.termination_proposals

        # hold_ratio: fraction of steps within 1.1m of opponent, computed from
        # RAW (unsmoothed) positions recorded by the approach_velocity observer.
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

        # primary_ratio: fraction of steps where the approach (primary) policy
        # was active rather than the fallback standing policy.
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        
        primary_ratio = 1.0
        gating_switches = 0.0
        mean_p_safe = 1.0
        fallback_attempts = 0.0
        fallback_recoveries = 0.0
        fall_on_chaser = 0.0
        fall_on_fallback = 0.0
        
        if extras is not None:
            if "gating_mode" in extras:
                gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
                if len(gating_mode) > 0:
                    primary_ratio = float(np.mean(gating_mode >= 0.5))
                    gating_switches = float(np.sum(np.abs(np.diff(gating_mode)) > 0.5))
                    
                    is_fallback = gating_mode < 0.5
                    # transitions from primary (False) to fallback (True)
                    enters = int(np.sum(~is_fallback[:-1] & is_fallback[1:]))
                    if is_fallback[0]:
                        enters += 1
                    # transitions from fallback (True) to primary (False)
                    exits = int(np.sum(is_fallback[:-1] & ~is_fallback[1:]))
                    
                    fallback_attempts = float(enters)
                    fallback_recoveries = float(exits)
                    
                    if fell:
                        if len(gating_mode) > 0:
                            print(f"[DEBUG_FALL] T={T} len(gating_mode)={len(gating_mode)} last={gating_mode[-1]} fell={fell}", flush=True)
                            if gating_mode[-1] >= 0.5:
                                fall_on_chaser = 1.0
                            else:
                                fall_on_fallback = 1.0
            if "p_safe" in extras:
                p_safe = np.asarray(extras["p_safe"], dtype=np.float32).reshape(-1)
                if len(p_safe) > 0:
                    mean_p_safe = float(np.mean(p_safe))

        return {
            "survived": 0.0 if fell else 1.0,
            "level": float(self._level),
            "hold_ratio": hold_ratio,
            "primary_ratio": primary_ratio,
            "mean_dist": mean_dist,
            "min_dist": min_dist,
            "gating_switches": gating_switches,
            "mean_p_safe": mean_p_safe,
            "fallback_attempts": fallback_attempts,
            "fallback_recoveries": fallback_recoveries,
            "fall_on_chaser": fall_on_chaser,
            "fall_on_fallback": fall_on_fallback,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "opp_speed": round(self.current_speed, 3),
            "consecutive_pass": self._consecutive_pass,
            "hold_ratio": round(self._hold_ratio, 3),
            "survival_rate": round(self._survival_rate, 3),
            "primary_ratio": round(self._primary_ratio, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "hold_ratio": self._hold_ratio,
            "survival_rate": self._survival_rate,
            "primary_ratio": self._primary_ratio,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._hold_ratio = float(state.get("hold_ratio", 0.0))
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._primary_ratio = float(state.get("primary_ratio", 0.0))


# Singleton instance for the registry
EXPERIMENT = FollowConfig()
