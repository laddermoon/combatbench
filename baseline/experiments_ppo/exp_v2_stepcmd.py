"""V1 步令实验 — D20 残差编码步令跟随（STEPCMD_V1_PLAN.md）：79 维。

任务核心：V2StepCmdSimulator 每机 0.5s/10 帧固定周期连续指令流，
obs[67:73] = step_cur = [foot, Δx, Δy, Δψ, 0, 0]（动脚踝→世界锚
目标的逐帧残差，本体系；保持令 foot=0、Δψ=对发令朝向漂移）。
无站立门控（帧0施令）、无推力、无臂脚本；r_std_clip 移除。

通道（5，全 gamma 0.99）：
    r_potential = 0.01·φ（h_score 1.10m 饱和 → φ≤0.889） aw=3.0
    r_fall      = 摔倒onset罚 + 持续靠墙罚                  aw=2.0
    r_step_mv   = −‖Δx,Δy‖（obs[68:70] 逐帧）              aw=1.0
    r_step_st   = −max(0, res_st − 0.03)（observer live）  aw=0.5
    r_step_yaw  = −max(0, |Δψ| − 0.05)（obs[70] 逐帧）     aw=0.5

积分语义：逐帧残差 = ASAP（每晚一帧 = +error·dt），无速度通道。
时序对齐：observer 字段在 action step 末量测（post-state t），
故 obs 残差列后移一帧配对——reward[t] 用 obs[t+1]/fin_obs 的残差。

eval：d_mv/d_st/d_yaw 终态、t_arrive 首达、hold 占比、滑步 watch、
falls/max_pot；warm-start S2 u1680（--reset-update）。

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 python3 -B \
      baseline/framework/train.py --experiment v2_stepcmd --algo ppo \
      --resume-from baseline/runs/v2_s2_balance/checkpoints/checkpoint_u01680.pt \
      --reset-update --run-name v2_stepcmd_r1 --background
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

from .exp_v2_s2_balance import V2S2Balance

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"

# obs[67:73] = step_cur：[foot(67), Δx(68), Δy(69), Δψ(70), 0, 0]
_COL_MVXY = slice(68, 70)
_COL_YAW = 70
_COL_FOOT = 67


class V2StepCmd(V2S2Balance):
    name = "v2_stepcmd"

    # ---------------- 通道 ----------------
    _channel_names = ("r_potential", "r_fall",
                      "r_step_mv", "r_step_st", "r_step_yaw")
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99,
        "r_step_mv": 0.99, "r_step_st": 0.99, "r_step_yaw": 0.99,
    }
    _STEP_OBS = {"robot_a": "stepcmd_report_a",
                 "robot_b": "stepcmd_report_b"}

    # aw（actor_weight 常数）
    step_mv_aw: float = 1.0
    step_st_aw: float = 0.5
    step_yaw_aw: float = 0.5
    # 惩罚系数 / 死区
    w_mv: float = 1.0     # /m
    w_st: float = 1.0     # /m（死区原始累计，不归一化）
    w_yaw: float = 1.0    # /rad
    st_tol: float = 0.03  # stance 漂移死区 m
    yaw_tol: float = 0.05  # yaw 死区 rad

    # V1 可行性验证：关早停，人工盯（同 S3/S4 惯例）
    _no_improvement_limit: int = 10 ** 9
    # h_cap=1.10 后 φ_max≈0.889 < 0.9——success 阈值降到封顶值的
    # ~96%（蹲姿 φ≈0.64、摔倒 <0.3 仍远低于此）
    _success_thresh: float = 0.85

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_stepcmd_env.yaml")

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

        # ---- 步令三通道（逐帧积分） ----
        # post-state 对齐：reward[t] 对应 action t 之后状态的残差
        # = obs[t+1]（末帧用 final_observation）。
        obs_t1 = np.concatenate(
            [obs_all[1:T],
             np.asarray(fin_obs, np.float32)[None, :]], axis=0)
        d_xy = np.linalg.norm(obs_t1[:, _COL_MVXY], axis=1)
        d_psi = np.abs(obs_t1[:, _COL_YAW])
        r_step_mv = (-self.w_mv * d_xy).astype(np.float32)
        r_step_yaw = (-self.w_yaw
                      * np.maximum(0.0, d_psi - self.yaw_tol)
                      ).astype(np.float32)

        sr_key = self._STEP_OBS.get(agent_id)
        res_st = extract_per_step_field(
            episode.observer_outputs, sr_key, "res_st", T_full)
        res_st = (np.asarray(res_st[:T], dtype=np.float32)
                  if res_st is not None
                  else np.zeros(T, dtype=np.float32))
        r_step_st = (-self.w_st
                     * np.maximum(0.0, res_st - self.st_tol)
                     ).astype(np.float32)

        aw_ones = lambda w: np.full(T, w, dtype=np.float32)
        channels = {
            "r_potential": ChannelData(
                reward=r_potential, is_terminated=False,
                actor_weight=aw_ones(self.r_potential_actor_weight)),
            "r_fall": ChannelData(
                reward=r_fall.astype(np.float32), is_terminated=False,
                actor_weight=aw_ones(self.r_fall_actor_weight)),
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
        n_ev = n_hold = n_arr = 0
        sum_mv = sum_st = sum_yaw = sum_arr = 0.0
        skate_n = skate_d = 0

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

                def _fld(name: str) -> np.ndarray:
                    a = extract_per_step_field(
                        ep.observer_outputs, sr_key, name, T)
                    return np.asarray(a)[m] if a is not None \
                        else np.zeros(int(m.sum()))

                hold = _fld("hold") > 0.5
                arr = _fld("t_arrive") >= 0
                n_ev += int(m.sum())
                n_hold += int(hold.sum())
                step_m = ~hold                       # 动脚令子集
                n_arr += int((arr & step_m).sum())
                sum_arr += float(step_m.sum())
                sum_mv += float(np.sum(_fld("d_mv")[step_m]))
                sum_st += float(np.sum(_fld("d_st")))
                sum_yaw += float(np.sum(_fld("d_yaw")))

                # 滑步 watch：动脚令期间，接触中的指令脚 + 残差仍大
                obs = ep.observations.get(rid)
                foot_key = ("foot_state_a" if rid == "robot_a"
                            else "foot_state_b")
                lc = extract_per_step_field(
                    ep.observer_outputs, foot_key,
                    "left_foot_contact", T)
                rc = extract_per_step_field(
                    ep.observer_outputs, foot_key,
                    "right_foot_contact", T)
                if obs is not None and lc is not None \
                        and rc is not None:
                    o = np.asarray(obs, np.float32)[:T]
                    foot = o[:, _COL_FOOT]
                    res = np.linalg.norm(o[:, _COL_MVXY], axis=1)
                    contact = np.where(
                        foot < -0.5, np.asarray(lc)[:T],
                        np.asarray(rc)[:T])
                    mv_m = (np.abs(foot) > 0.5) & (res > 0.05)
                    skate_d += int(mv_m.sum())
                    skate_n += int((mv_m & (contact > 0.5)).sum())

        n = max(len(max_pots), 1)
        success_rate = sum(m >= self._success_thresh
                           for m in max_pots) / n
        falls_mean = sum(all_falls) / max(len(all_falls), 1)
        n_step = max(int(sum_arr), 1)                # 动脚令数
        arr_rate = n_arr / n_step
        d_mv = sum_mv / n_step
        d_st = sum_st / max(n_ev, 1)
        d_yaw = sum_yaw / max(n_ev, 1)
        hold_rate = n_hold / max(n_ev, 1)
        skate_frac = skate_n / max(skate_d, 1)
        quality = (arr_rate
                   - 0.5 * min(d_mv / 0.3, 1.0)
                   - 0.5 * min(d_st / 0.3, 1.0)
                   - 0.3 * min(d_yaw / 1.0, 1.0))

        # best 追踪（同构：quality+EMA+潜在 best）
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
                "hold": round(hold_rate, 3),
                "arr_rate": round(arr_rate, 3),
                "d_mv": round(d_mv, 3),
                "d_st": round(d_st, 3),
                "d_yaw": round(d_yaw, 3),
                "skate": round(skate_frac, 3),
                "quality": round(quality, 3),
            },
        }


EXPERIMENT_CLASS = V2StepCmd
