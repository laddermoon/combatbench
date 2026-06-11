
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.ppo_trainer import _extract_per_step_scalar
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class BalanceRecoverConfig(ExperimentConfig):
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

    name = "balance_recover_plus"
    reward_keys = ("r_fall", "r_cross")
    gammas = {"r_fall": 0.99, "r_cross": 0.99}

    BLUEPRINT = "balance_recover_env.yaml"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def video_env_blueprint(self):
        perturb = self._current_perturb_params()
        return self._env_pb().materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id="robot_a",
            **perturb,
        )

    # --- PPO tuning (see training analysis) ---
    # Raise the log_std floor so the policy can't collapse to saturated,
    # near-deterministic actions — the main driver of the KL explosions /
    # exploding policy_loss observed in the first run.
    log_std_min: float = -1.8

    # Per-experiment PPO overrides: smaller actor LR + tighter KL/grad
    # clipping + fewer epochs to keep each PPO update from diverging.
    learning_rate: float = 3e-5      # was 1e-4: slow the actor down further to allow more epochs
    target_kl: float = 0.05          # was 0.05: early-stop sooner
    grad_clip_norm: float = 1.0      # was 1.0: tighter gradient clipping
    update_epochs: int = 4           # was 4: less policy drift per batch
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3     # encourage exploration to prevent joint freeze

    # --- Rollout schedule ---
    episodes_per_update: int = 1024
    eval_episodes: int = 128

    # Small per-step survival bonus (each alive step is worth this much).
    per_step_survival_reward: float = 0.01


    # TODO： 这里改成控制目标机器人移动的速度
    LEVEL_SCALES: Tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
    # Promote once survival >= threshold for N consecutive evaluations.
    PROMOTE_SURVIVAL: float = 0.9
    PROMOTE_PATIENCE: int = 1

    # --- Stateful scheduler ---
    _level: int = 0
    _consecutive_pass: int = 0
    _survival_rate: float = 0.0

    # --- Perturbation scale helpers ---
    @property
    def current_scale(self) -> float:
        idx = max(0, min(self._level, len(self.LEVEL_SCALES) - 1))
        return float(self.LEVEL_SCALES[idx])

    def _current_perturb_params(self) -> Dict[str, float]:
        scale = self.current_scale
        return {k: float(v) * scale for k, v in self.PERTURB_FULL.items()}

    # TODO： build job要改成 一个使用训练的混合策略PB， 一个使用随机策略（因为不会产生作用）
    # 随机策略使用 /data1/mono/things/combatbench/policy/blueprints/random.yaml
    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        
    # TODO： 同上
    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        

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

        if self._level < len(self.LEVEL_SCALES) - 1:
            if survival_rate >= self.PROMOTE_SURVIVAL:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0

        return (6.0, 1.0)

    
    def initial_weights(self) -> Tuple[float, ...]:
        return (6.0, 1.0)

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """r_fall: per-step survival bonus + terminal signal.
        r_cross: cross-support balance reward from CrossSupportBalanceRewarder.
        """
        fell = "imbalance" in termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        r_cross = _extract_per_step_scalar(observer_outputs, "cross_support", T)

        return {"r_fall": r_fall, "r_cross": r_cross}

    # TODO： 这里要做切分， 返回一个列表， 触发Imbalance或者触发切换到恢复模型都要惩罚
    # 需要解决的一个问题是如何获取到每一步是用哪个模型， ！ 对, 应该通过一个Observer来获取，这个Observer集成了GateModel， 每次把GateModel跑一遍
    # 这样虽然GateModel重复推理，但是问题也不大
    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        fell = "imbalance" in termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        r_cross = _extract_per_step_scalar(observer_outputs, "cross_support", T)
        # r_hold from in_zone_hold observer
        r_hold = _extract_per_step_field(observer_outputs, "in_zone_hold", "reward", T)
        if r_hold is None:
            r_hold = np.zeros(T, dtype=np.float32)

        # r_radial / r_tangential: trainer-side post-processing
        # from approach_velocity observer's recorded positions
        from baseline.humanoid21.rewards.follow_opponent import compute_approach_rewards

        self_x = _extract_per_step_field(observer_outputs, "approach_velocity", "self_x", T)
        self_y = _extract_per_step_field(observer_outputs, "approach_velocity", "self_y", T)
        opp_x = _extract_per_step_field(observer_outputs, "approach_velocity", "opp_x", T)
        opp_y = _extract_per_step_field(observer_outputs, "approach_velocity", "opp_y", T)

        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            r_radial = np.zeros(T, dtype=np.float32)
            r_tangential = np.zeros(T, dtype=np.float32)
        else:
            self_xy = np.stack([self_x, self_y], axis=1)
            opp_xy = np.stack([opp_x, opp_y], axis=1)
            r_radial, r_tangential = compute_approach_rewards(
                self_xy, opp_xy,
                debug=False,
            )

        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_hold": r_hold,
            "r_radial": r_radial,
            "r_tangential": r_tangential,
        }

    # TODO： 这里的计算逻辑应该改一下，主要去
    def compute_episode_metrics(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, float]:
        """``survived`` = 0 only if the robot fell (imbalance termination).

        Returns level/stage for eval comparison (higher level = better).
        """
        fell = "imbalance" in termination_proposals
        return {
            "survived": 0.0 if fell else 1.0,
            "level": float(self._level),  # higher level = harder perturbation = better
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
EXPERIMENT = BalanceRecoverConfig()
