"""V2 线 Stage 4 — 步令跟随（79 维，S4_PLAN.md / D16）。

任务核心：V2StepSimulator 双机步令流（时限预算到期结算），指令段
obs[67:78] = step_cur(5+age/budget)/step_next。姿态约束 r_std_clip 与
推力课程移除；臂脚本接管沿用 S3。

通道（6，全 gamma 0.99）：
    r_potential = 0.01·φ(t)                    aw=3.0
    r_fall      = 摔倒onset罚 + 持续靠墙罚      aw=2.0
    r_step_foot = 到期帧: d_mv<0.10 → +1        aw=1.5
    r_step_mv   = 到期帧: −w_mv·‖指令脚位−目标‖  aw=2.0
    r_step_st   = 到期帧: −w_st·‖stance脚−发令位‖ aw=2.0
    r_step_yaw  = 到期帧: −w_yaw·|yaw−目标yaw|   aw=2.0

步令通道稀疏（到期帧才非零），摔倒期照发——靠 aw 比
（步态 vs fall/potential）让策略学会"先保命再跟令"。

warm-start: --resume-from v2_s3_balance_arm_r3/checkpoints/
checkpoint_u02315.pt --reset-update
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.end2end.stepping_state_machine import (
    count_falls, fall_onset_mask,
)
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_s3_balance_arm import V2S3BalanceArm

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S4Step(V2S3BalanceArm):
    name = "v2_s4_step"

    # ---------------- 通道 ----------------
    _channel_names = ("r_potential", "r_fall",
                      "r_step_foot", "r_step_mv",
                      "r_step_st", "r_step_yaw")
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99,
        "r_step_foot": 0.99, "r_step_mv": 0.99,
        "r_step_st": 0.99, "r_step_yaw": 0.99,
    }
    _STEP_OBS = {"robot_a": "step_report_a", "robot_b": "step_report_b"}

    # aw（actor_weight 常数）
    step_foot_aw: float = 1.5
    step_mv_aw: float = 2.0
    step_st_aw: float = 2.0
    step_yaw_aw: float = 2.0
    # 惩罚系数（边界帧内）
    w_mv: float = 1.0    # /m
    w_st: float = 1.0    # /m
    w_yaw: float = 0.3   # /rad

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_step_env.yaml")

    # ---------------- 轨迹 ----------------
    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        foot_key: str,
        phi_key: str,
    ) -> List[Trajectory]:
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
            episode.observer_outputs, phi_key, "potential", T_full)
        phi_arr = (np.clip(np.asarray(phi_arr[:T], dtype=np.float32),
                           0.0, 1.0) if phi_arr is not None
                   else np.zeros(T, dtype=np.float32))

        r_potential = (self.per_step_phi_coef * phi_arr).astype(
            np.float32)

        # r_fall: 摔倒 onset 罚 + 持续靠墙罚（无推力 clean bonus）
        r_fall = (-self.fall_penalty
                  * fall_onset_mask(phi_arr).astype(np.float32))
        wall_key = self._WALL_OBS.get(agent_id)
        if wall_key is not None and self.wall_penalty > 0:
            from baseline.humanoid21.end2end.wall_contact_observer \
                import sustained_wall_mask
            wc = extract_per_step_field(
                episode.observer_outputs, wall_key, "wall_contact",
                T_full)
            if wc is not None:
                standing = phi_arr >= 0.5
                lean = sustained_wall_mask(
                    wc, window=self.wall_window,
                    count_thresh=self.wall_count_thresh)
                r_fall = r_fall - (self.wall_penalty
                                   * lean[:T].astype(np.float32)
                                   * standing[:T].astype(np.float32))

        # ---- 步令四通道：step_report 边界帧稀疏 ----
        sr_key = self._STEP_OBS.get(agent_id)

        def _fld(name: str) -> np.ndarray:
            if sr_key is None:
                return np.zeros(T, dtype=np.float32)
            a = extract_per_step_field(
                episode.observer_outputs, sr_key, name, T_full)
            if a is None:
                return np.zeros(T, dtype=np.float32)
            return np.asarray(a[:T], dtype=np.float32)

        ev = _fld("event")
        r_step_foot = (ev * _fld("foot_ok")).astype(np.float32)
        r_step_mv = (-self.w_mv * ev * _fld("d_mv")).astype(np.float32)
        r_step_st = (-self.w_st * ev * _fld("d_st")).astype(np.float32)
        r_step_yaw = (-self.w_yaw * ev * _fld("d_yaw")).astype(
            np.float32)

        aw_ones = lambda w: np.full(T, w, dtype=np.float32)
        channels = {
            "r_potential": ChannelData(
                reward=r_potential, is_terminated=False,
                actor_weight=aw_ones(self.r_potential_actor_weight)),
            "r_fall": ChannelData(
                reward=r_fall.astype(np.float32), is_terminated=False,
                actor_weight=aw_ones(self.r_fall_actor_weight)),
            "r_step_foot": ChannelData(
                reward=r_step_foot, is_terminated=False,
                actor_weight=aw_ones(self.step_foot_aw)),
            "r_step_mv": ChannelData(
                reward=r_step_mv, is_terminated=False,
                actor_weight=aw_ones(self.step_mv_aw)),
            "r_step_st": ChannelData(
                reward=r_step_st, is_terminated=False,
                actor_weight=aw_ones(self.step_st_aw)),
            "r_step_yaw": ChannelData(
                reward=r_step_yaw, is_terminated=False,
                actor_weight=aw_ones(self.step_yaw_aw)),
        }
        return [Trajectory(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots, all_falls = [], []
        n_ev = n_foot_ok = 0
        sum_mv = sum_st = sum_yaw = sum_budget = 0.0

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id, foot_key, phi_key in self._AGENT_OBS:
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T)
                mx = float(np.max(phi)) if phi is not None and len(
                    phi) else 0.0
                max_pots.append(mx)
                if phi is not None:
                    all_falls.append(count_falls(phi))
            for rid, sr_key in self._STEP_OBS.items():
                ev = extract_per_step_field(
                    ep.observer_outputs, sr_key, "event", T)
                if ev is None:
                    continue
                m = np.asarray(ev) > 0.5
                if not m.any():
                    continue
                n_ev += int(m.sum())
                n_foot_ok += int(np.sum(
                    np.asarray(extract_per_step_field(
                        ep.observer_outputs, sr_key, "foot_ok",
                        T))[m] > 0.5))
                sum_budget += float(np.sum(np.asarray(
                    extract_per_step_field(
                        ep.observer_outputs, sr_key, "budget", T))[m]))
                sum_mv += float(np.sum(np.asarray(
                    extract_per_step_field(
                        ep.observer_outputs, sr_key, "d_mv", T))[m]))
                sum_st += float(np.sum(np.asarray(
                    extract_per_step_field(
                        ep.observer_outputs, sr_key, "d_st", T))[m]))
                sum_yaw += float(np.sum(np.asarray(
                    extract_per_step_field(
                        ep.observer_outputs, sr_key, "d_yaw", T))[m]))

        n = max(len(max_pots), 1)
        success_rate = sum(m >= 0.9 for m in max_pots) / n
        falls_mean = sum(all_falls) / max(len(all_falls), 1)
        foot_rate = n_foot_ok / max(n_ev, 1)
        budget_mean = sum_budget / max(n_ev, 1)
        d_mv = sum_mv / max(n_ev, 1)
        d_st = sum_st / max(n_ev, 1)
        d_yaw = sum_yaw / max(n_ev, 1)
        quality = (foot_rate
                   - 0.5 * min(d_mv / 0.5, 1.0)
                   - 0.5 * min(d_st / 0.3, 1.0)
                   - 0.3 * min(d_yaw / 1.0, 1.0))

        # best 追踪（BalancePure 同构：quality+EMA+潜在 best）
        if self._b_quality_ema is None:
            self._b_quality_ema = quality
        else:
            self._b_quality_ema = (0.85 * self._b_quality_ema
                                   + 0.15 * quality)
        improved = (success_rate >= 0.9
                    and quality > self._b_best_quality)
        if improved:
            self._b_best_quality = quality
        trend_improved = (success_rate >= 0.9
                          and self._b_quality_ema > self._b_best_quality)
        if trend_improved:
            self._b_best_quality = self._b_quality_ema
        if improved or trend_improved:
            self._b_last_best_update = update
        if self._b_last_best_update < 0:
            self._b_last_best_update = update
        pot_mean = sum(max_pots) / n
        improved_pot = pot_mean > self._best_potential
        if improved_pot:
            self._best_potential = pot_mean

        no_improvement = update - self._b_last_best_update
        stop_training = (no_improvement >= self._no_improvement_limit
                         and update >= self._min_updates)

        return {
            "is_new_best": improved or improved_pot or trend_improved,
            "stop_training": stop_training,
            "info": {
                "max_pot": round(pot_mean, 3),
                "success": round(success_rate, 3),
                "falls": round(falls_mean, 2),
                "step_ev": n_ev,
                "foot_rate": round(foot_rate, 3),
                "budget": round(budget_mean, 1),
                "d_mv": round(d_mv, 3),
                "d_st": round(d_st, 3),
                "d_yaw": round(d_yaw, 3),
                "quality": round(quality, 3),
            },
        }


EXPERIMENT_CLASS = V2S4Step
