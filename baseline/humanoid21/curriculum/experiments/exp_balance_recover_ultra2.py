
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.humanoid21.curriculum.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class BalanceRecoverUltra2Config(CombatExperimentBase):
    """P0 balance-recovery policy (IDEA.md step 2).

    Trained on top of the basic-standing policy. At every episode reset the
    robot's state is randomly perturbed (joint positions/velocities, root
    tilt, root linear/angular velocity); the robot must learn to recover
    balance from any such starting condition — this is the fallback policy.

    The curriculum is **progressive**: perturbations start small and grow
    stronger level by level. A single scalar ``scale`` in ``[0, 1]`` scales
    every full-strength magnitude in :pyattr:`PERTURB_FULL`. When the eval
    survival rate stays at/above :pyattr:`PROMOTE_SURVIVAL` for
    :pyattr:`PROMOTE_PATIENCE` consecutive evaluations, the next (harder)
    level is unlocked. The env blueprint file never changes across levels;
    only the perturbation parameters passed to ``materialize`` do.
    """

    name = "balance_recover_ultra2"
    reward_keys = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
    }

    max_steps = 200

    BLUEPRINT = "balance_recover_ultra2_env.yaml"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def video_env_blueprint(self):
        perturb = self._current_perturb_params()
        return self._env_pb().materialize(
            max_steps=self.max_steps,
            agent_id="robot_a",
            **perturb,
        )

    # --- PPO tuning (see training analysis) ---
    # Raise the log_std floor so the policy can't collapse to saturated,
    # near-deterministic actions — the main driver of the KL explosions /
    # exploding policy_loss observed in the first run.
    log_std_min: float = -1.8

    max_updates: int = 20000
    # Per-experiment PPO overrides: smaller actor LR + tighter KL/grad
    # clipping + fewer epochs to keep each PPO update from diverging.
    learning_rate: float = 3e-5      # was 1e-4: slow the actor down further to allow more epochs
    grad_clip_norm: float = 1.0      # was 1.0: tighter gradient clipping
    update_epochs: int = 4           # was 4: less policy drift per batch
    entropy_coef: float = 1.5e-3     # encourage exploration to prevent joint freeze

    # --- Rollout schedule ---
    episodes_per_update: int = 2048
    eval_episodes: int = 128
    eval_interval: int = 3
    # --- Video recording ---
    video_eval_interval: int = 3

    # --- Fall/recovery dense reward parameters ---
    fall_penalty: float = 1.0           # one-time penalty at fall moment
    fall_step_penalty: float = 0.01     # per-step penalty while fallen
    recovery_bonus: float = 1.0         # one-time bonus at recovery moment
    standing_step_bonus: float = 0.01   # per-step bonus while standing
    fall_debounce_steps: int = 5        # consecutive clean steps to confirm recovery / merge fall contacts

    # --- Progressive perturbation curriculum ---
    # Perturbation strength is controlled by init_steps (passed to
    # RandomFallenStatePlugin as max_phy_steps). More steps = robot falls
    # further = harder to recover. Range: 0 (standing, no fall) → 1000 (fully fallen).
    INIT_STEPS_MAX: int = 1000
    # 90 levels: init_steps from 0 to INIT_STEPS_MAX, linearly spaced.
    LEVEL_INIT_STEPS: Tuple[int, ...] = tuple(
        int(round(x)) for x in np.linspace(0, 1000, 90)
    )
    # Promote once survival >= threshold for N consecutive evaluations.
    PROMOTE_SURVIVAL: float = 0.9
    PROMOTE_PATIENCE: int = 1

    # --- Stateful scheduler ---
    _level: int = 0
    _consecutive_pass: int = 0
    _survival_rate: float = 0.0

    # --- Perturbation helpers ---
    @property
    def current_scale(self) -> float:
        """Returns init_steps for current level (for logging compatibility)."""
        idx = max(0, min(self._level, len(self.LEVEL_INIT_STEPS) - 1))
        return float(self.LEVEL_INIT_STEPS[idx])

    def _current_perturb_params(self) -> Dict[str, Any]:
        idx = max(0, min(self._level, len(self.LEVEL_INIT_STEPS) - 1))
        return {"init_steps": int(self.LEVEL_INIT_STEPS[idx])}

    def _sample_init_steps(self, rng: np.random.Generator) -> int:
        """Sample init_steps for a single episode.

        20% chance to use the current level's max init_steps (focus on
        current difficulty), 80% chance to uniformly sample from
        [0, current_max] to review easier recovery scenarios and prevent
        catastrophic forgetting.
        """
        idx = max(0, min(self._level, len(self.LEVEL_INIT_STEPS) - 1))
        current_max = int(self.LEVEL_INIT_STEPS[idx])
        if current_max == 0:
            return 0
        if rng.random() < 0.2:
            return current_max
        return int(rng.integers(0, current_max + 1))

    # --- Rollout job construction: inject scaled perturbation params ---
    def _build_perturbed_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        max_steps = self.max_steps
        env_pb = self._env_pb()
        rng = np.random.default_rng(base_seed)

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(
                    self.custom_config["rollout_distance_min"],
                    self.custom_config["rollout_distance_max"],
                )
            )
            init_steps = self._sample_init_steps(rng)
            env_bp = env_pb.materialize(
                max_steps=max_steps, agent_id=agent_id,
                init_steps=init_steps,
            )
            jobs.append((
                policy_bp, policy_bp,
                env_bp, seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_perturbed_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_perturbed_jobs(policy_bp, base_seed, self.eval_episodes)

    def compare_eval(self, esum, best_esum):
        """Compare eval metrics: prioritize higher level, then higher survival rate."""
        if not best_esum:
            return True
        # First: compare level (higher is better)
        level = esum.get("level", 0.0)
        best_level = best_esum.get("level", 0.0)
        if level != best_level:
            return level > best_level
        # Same level: compare survival rate
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Advance the perturbation level when the policy reliably recovers.

        Weights stay ``(1.0,)`` throughout; the curriculum knob is the
        perturbation scale, advanced once survival holds at/above
        ``PROMOTE_SURVIVAL`` for ``PROMOTE_PATIENCE`` consecutive evals.
        """
        survival_rate = float(eval_metrics.get("survived", 0.0))
        self._survival_rate = survival_rate

        if self._level < len(self.LEVEL_INIT_STEPS) - 1:
            if survival_rate >= self.PROMOTE_SURVIVAL:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0

        return (6.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    
    def initial_weights(self) -> Tuple[float, ...]:
        return (6.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    @staticmethod
    def _compute_fall_recovery_rewards(
        contact_arr: np.ndarray,
        T: int,
        fall_penalty: float,
        fall_step_penalty: float,
        recovery_bonus: float,
        standing_step_bonus: float,
        debounce_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Offline fall/recovery state machine.

        Returns (r_fall, is_fallen_mask) where:
        - r_fall: (T,) float32 reward array
        - is_fallen_mask: (T,) bool array, True for steps while robot is in fallen state

        States: STANDING → FALLEN → STANDING (with recovery confirmation)

        - STANDING: no non-foot contact → +standing_step_bonus per step
        - First non-foot contact → FALLEN, that step gets -fall_penalty
        - FALLEN: each step gets -fall_step_penalty
        - If debounce_steps consecutive steps without contact → recovery confirmed
          Recovery step (first clean step) gets +recovery_bonus, subsequent steps get +standing_step_bonus
        - If contact resumes before debounce_steps → reset counter, stay FALLEN (same fall session)
        """
        r_fall = np.zeros(T, dtype=np.float32)
        is_fallen_mask = np.zeros(T, dtype=bool)

        # If the robot starts already in contact (placed by RandomFallenStatePlugin),
        # begin in FALLEN state WITHOUT a fall penalty — the robot didn't fall on
        # its own, it was placed there by the environment.
        if T > 0 and bool(contact_arr[0]):
            state = "FALLEN"
            r_fall[0] = -fall_step_penalty
            is_fallen_mask[0] = True
        else:
            state = "STANDING"
            if T > 0:
                r_fall[0] = standing_step_bonus
        clean_counter = 0

        for t in range(1, T):
            is_contact = bool(contact_arr[t])

            if state == "STANDING":
                if is_contact:
                    state = "FALLEN"
                    r_fall[t] = -fall_penalty
                    is_fallen_mask[t] = True
                    clean_counter = 0
                else:
                    r_fall[t] = standing_step_bonus

            elif state == "FALLEN":
                if is_contact:
                    r_fall[t] = -fall_step_penalty
                    is_fallen_mask[t] = True
                    clean_counter = 0
                else:
                    clean_counter += 1
                    r_fall[t] = -fall_step_penalty
                    is_fallen_mask[t] = True

                    if clean_counter >= debounce_steps:
                        recovery_step = t - debounce_steps + 1
                        r_fall[recovery_step] = recovery_bonus
                        is_fallen_mask[recovery_step] = False
                        for tt in range(recovery_step + 1, t + 1):
                            r_fall[tt] = standing_step_bonus
                            is_fallen_mask[tt] = False
                        state = "STANDING"
                        clean_counter = 0

        return r_fall, is_fallen_mask

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Dense fall/recovery reward via offline state machine on contact data.

        r_fall: fall penalty + per-step fall penalty + recovery bonus + standing bonus
        r_cross: cross-support balance reward (zeroed while fallen)
        r_joint/r_vel/r_tilt/r_foot: posture rewards (zeroed while fallen)
        """
        T = episode.num_frames

        contact_arr = _extract_per_step_field(
            episode.observer_outputs, "fall_contact", "is_non_foot_grounded", T
        )
        if contact_arr is None:
            contact_arr = np.zeros(T, dtype=np.float32)

        r_fall, is_fallen_mask = self._compute_fall_recovery_rewards(
            contact_arr, T,
            fall_penalty=self.fall_penalty,
            fall_step_penalty=self.fall_step_penalty,
            recovery_bonus=self.recovery_bonus,
            standing_step_bonus=self.standing_step_bonus,
            debounce_steps=self.fall_debounce_steps,
        )

        r_cross = _extract_per_step_scalar(episode.observer_outputs, "cross_support", T)
        r_cross = np.where(is_fallen_mask, 0.0, r_cross)

        joint_dev_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_deviation", T)
        joint_vel_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_vel", T)
        torso_tilt_arr = _extract_per_step_field(episode.observer_outputs, "posture", "torso_tilt", T)
        foot_height_arr = _extract_per_step_field(episode.observer_outputs, "posture", "foot_height", T)

        if joint_dev_arr is None:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is None:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is None:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)
        r_joint = np.where(is_fallen_mask, 0.0, r_joint)

        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)
        r_vel = np.where(is_fallen_mask, 0.0, r_vel)

        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)
        r_tilt = np.where(is_fallen_mask, 0.0, r_tilt)

        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)
        r_foot = np.where(is_fallen_mask, 0.0, r_foot)

        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }


    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Rich fall/recovery diagnostics from contact data.

        Metrics:
        - survived: 1.0 if standing at end (no fall or recovered)
        - level: current curriculum level
        - n_falls: number of distinct fall events
        - n_recoveries: number of distinct recovery events
        - fallen_ratio: fraction of steps in FALLEN state
        - standing_ratio: fraction of steps in STANDING state
        - first_fall_step: step index of first fall (-1 if never fell)
        - last_recovery_step: step index of last recovery (-1 if never recovered)
        - ep_length: number of frames in episode
        """
        T = episode.num_frames
        contact_arr = _extract_per_step_field(
            episode.observer_outputs, "fall_contact", "is_non_foot_grounded", T
        )
        if contact_arr is None:
            contact_arr = np.zeros(T, dtype=np.float32)

        r_fall, is_fallen_mask = self._compute_fall_recovery_rewards(
            contact_arr, T,
            fall_penalty=self.fall_penalty,
            fall_step_penalty=self.fall_step_penalty,
            recovery_bonus=self.recovery_bonus,
            standing_step_bonus=self.standing_step_bonus,
            debounce_steps=self.fall_debounce_steps,
        )

        n_falls = int(np.sum(r_fall == -self.fall_penalty))
        n_recoveries = int(np.sum(r_fall == self.recovery_bonus))
        fallen_ratio = float(np.mean(is_fallen_mask)) if T > 0 else 0.0
        standing_ratio = 1.0 - fallen_ratio

        first_fall_step = -1.0
        fall_steps = np.where(r_fall == -self.fall_penalty)[0]
        if len(fall_steps) > 0:
            first_fall_step = float(fall_steps[0])

        last_recovery_step = -1.0
        recovery_steps = np.where(r_fall == self.recovery_bonus)[0]
        if len(recovery_steps) > 0:
            last_recovery_step = float(recovery_steps[-1])

        survived = 0.0 if (T > 0 and is_fallen_mask[-1]) else 1.0

        return {
            "survived": survived,
            "level": float(self._level),
            "n_falls": float(n_falls),
            "n_recoveries": float(n_recoveries),
            "fallen_ratio": round(fallen_ratio, 4),
            "standing_ratio": round(standing_ratio, 4),
            "first_fall_step": first_fall_step,
            "last_recovery_step": last_recovery_step,
            "ep_length": float(T),
        }

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "perturb_scale": round(self.current_scale, 3),
            "consecutive_pass": self._consecutive_pass,
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._survival_rate = float(state.get("survival_rate", 0.0))


# Singleton instance for the registry
EXPERIMENT = BalanceRecoverUltra2Config()
