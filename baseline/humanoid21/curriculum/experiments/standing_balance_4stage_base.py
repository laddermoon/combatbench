"""Base class for 4-stage standing-balance ablation: Delta vs Dense vs GS-1.1.

Dead-zone-free stage-dependent potential:
  Stage 1: Rollover (f_down orientation)
  Stage 2: Hands & feet approach ground, other parts lift off
  Stage 3: Minimize HORIZONTAL (XY) hand-midpoint to foot-midpoint distance
  Stage 4: Transfer load to feet + stand up (w_foot × h_score, torso height)

Three reward modes (identical otherwise):
  - "delta":   r_t = φ(t) - φ(t-1)
  - "dense":    r_t = (1-γ)·φ(t)
  - "gs_1p1":   r_t = 1.1·φ(t) - φ(t-1)

Uses RandomFallenStatePlugin for random fall initialization.
Every step is trainable (no mixed policy, no episode segmentation).

Known Risk Points (待观察):
  1. 瞬时跳 Stage 3/4：短暂腾空时 extra_count == 0 可瞬间进入高 stage，
     下一步掉回。delta 模式下产生奖励尖峰。
  2. Stage 4 → Stage 1 跳变：在 Stage 4 时如果其它部位触地且 f_score 低，
     直接从 Stage 4 跳到 Stage 1，potential 从 ~0.6+ 暴跌到 ~0.2 以下。
     delta 模式下尤其显著。
  3. Stage 4 顶部信号坍缩：站立后 Δφ ≈ 0，delta 模式可能探索死锁。
     dense 模式（保持奖励）预期表现最好。
  4. 反复重入：站起来 → 摔 → 再站起来可反复刷 delta 进步奖励。
     若 max_potential >> final_potential 则在刷。
  5. 摆臂被惩罚：起身时张开手臂平衡会推大 XY 距离，若跌破 D_GATE 会被
     踢出 Stage 4。D_GATE 已放宽到 0.6 留余量。
  6. H_CROUCH 未标定：若 min_h_torso 明显低于 H_CROUCH=0.35，h_score 会在
     底部被 clip 到 0 形成新死区，需调低 H_CROUCH。

History (已修复的设计错误):
  首版 Stage 3 用 3D 手-脚距离，使得 Stage 4 的维持条件（d_score ≥ D_GATE）
  与站立在几何上互斥（站立时手比脚高 ~1.3m → 3D d_score = 0）。结果机器人
  收敛到“深踯 + 双手垂到脚边”的姿态（potential 0.845），因为站起来会被踢回
  Stage 3（potential 0.40）。修复：d_score 改为只取 XY 平面。
  同时 h_score 从“去脚 COM 高度 + H_STAND=0.65”（实测站立 COM 为 0.97，
  参数严重错误）改为 torso 高度 + H_STAND=1.28（已验证），并去掉 p4 中的
  常数 0.5（原本仅“重心移到脚上”就能拿 80% potential，形成舒适高原）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.ppo_trainer import _extract_per_step_field
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class StandingBalance4StageBase(CombatExperimentBase):
    """Base class for 4-stage standing-balance reward-mode ablation."""

    reward_keys = ("r_potential",)
    gammas = {"r_potential": 0.99}

    BLUEPRINT = "standing_balance_4stage_env.yaml"
    reward_mode: str = "delta"  # overridden by subclasses

    max_updates: int = 1000

    # --- PPO tuning (aligned with rollover ablation report §1.1) ---
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- Stateful metrics ---
    _success_rate: float = 0.0

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 200,
        "potential_reward_scale": 1.0,
    }
    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            agent_id=agent_id,
            max_steps=self.custom_config["max_steps"],
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a")

    # ---- Job construction -------------------------------------------------

    def _build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        env_bp = self._materialize_env("robot_a")
        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            jobs.append((
                policy_bp, policy_bp, env_bp, int(base_seed + i),
                {"agent_id": "robot_a", "initial_distance": 2.0},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("max_potential", 0.0) > best_esum.get("max_potential", 0.0)

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0,)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        self._success_rate = float(eval_metrics.get("success", 0.0))
        return (1.0,)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Extract potential reward according to reward_mode."""
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = _extract_per_step_field(oo, "standing_balance", "potential", T)
        r = np.zeros(T, dtype=np.float32)

        if potentials is not None:
            pot_scale = float(self.custom_config.get("potential_reward_scale", 1.0))
            gamma = self.gammas["r_potential"]

            if self.reward_mode == "delta":
                # r_t = φ(t) - φ(t-1)
                r[1:] = pot_scale * (potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (potentials[0] - 0.0)
            elif self.reward_mode == "dense":
                # r_t = (1-γ)·φ(t) — direct dense potential reward
                r[:] = pot_scale * (1.0 - gamma) * potentials[:]
            elif self.reward_mode == "gs_1p1":
                # r_t = 1.1·φ(t) - φ(t-1) — generalized shaping with γ_s=1.1
                gs = 1.1
                r[1:] = pot_scale * (gs * potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (gs * potentials[0] - 0.0)
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        return {"r_potential": r}

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Compute metrics for success monitoring and curriculum progression."""
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = _extract_per_step_field(oo, "standing_balance", "potential", T)
        d_hf = _extract_per_step_field(oo, "standing_balance", "d_hf", T)
        stages = _extract_per_step_field(oo, "standing_balance", "stage", T)
        w_foot = _extract_per_step_field(oo, "standing_balance", "w_foot", T)
        h_torso = _extract_per_step_field(oo, "standing_balance", "h_torso", T)

        if potentials is not None and len(potentials) > 0:
            max_potential = float(np.max(potentials))
            final_potential = float(potentials[-1])
            avg_potential = float(np.mean(potentials))
        else:
            max_potential = 0.0
            final_potential = 0.0
            avg_potential = 0.0

        if d_hf is not None and len(d_hf) > 0:
            min_d_hf = float(np.min(d_hf))
        else:
            min_d_hf = 0.0

        if stages is not None and len(stages) > 0:
            max_stage = float(np.max(stages))
        else:
            max_stage = 0.0

        if w_foot is not None and len(w_foot) > 0:
            max_w_foot = float(np.max(w_foot))
        else:
            max_w_foot = 0.0

        if h_torso is not None and len(h_torso) > 0:
            max_h_torso = float(np.max(h_torso))
            # min over Stage-4 steps only: used to calibrate H_CROUCH.
            if stages is not None and len(stages) == len(h_torso):
                s4 = h_torso[stages >= 4.0]
                min_h_torso_s4 = float(np.min(s4)) if s4.size > 0 else 0.0
            else:
                min_h_torso_s4 = 0.0
        else:
            max_h_torso = 0.0
            min_h_torso_s4 = 0.0

        success = 1.0 if max_potential >= 0.9 else 0.0

        return {
            "success": success,
            "max_potential": max_potential,
            "final_potential": final_potential,
            "avg_potential": avg_potential,
            "min_d_hf": min_d_hf,
            "max_stage": max_stage,
            "max_w_foot": max_w_foot,
            "max_h_torso": max_h_torso,
            "min_h_torso_s4": min_h_torso_s4,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "success_rate": round(self._success_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "success_rate": self._success_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))
