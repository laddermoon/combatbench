"""STEPBUDGET V1 — 方向+时间预算步令（STEPBUDGET_V1_PLAN.md）：79 维。

obs[67:73] = step_cur = [foot, ux, uy, Δψ, 剩余预算/20, 0]
obs[73:79] = step_nxt = [foot, ux, uy, Δψ, 总预算/20, 0]
步长 = 执行结果（预算 × 脚速，站姿包络 L2W 封顶）；朝向 = 绝对目标。
指令流 = Clip 随机组合 + 20% 逐步随机（step_budget_gen）。

通道（7，全 gamma 0.99；字段来自 stepbudget_report live，晋升前量测
→ reward[t] 直接对齐 action t，无需 obs 后移）：
    r_potential = 0.01·φ（h_cap 1.10）                          aw=3.0
    r_fall      = 摔倒 onset + 持续靠墙                          aw=2.0
    r_progress  = w_prog·ΔΦ − w_viol·viol_end（Φ=包络内沿向投影；
                  包络违规只判结算帧终态站姿）     aw=1.0
    r_lateral   = −w_lat·max(0, lat − lat_tol)（动脚令）          aw=0.5
    r_land      = w_land·[预算末帧动脚着地]                       aw=0.5
    r_yaw       = −w_yaw·yaw_w·max(0, |Δψ| − 0.05)               aw=0.5
    r_pivot     = −w_pv·max(0, pv_dr − pv_tol)（10cm/4cm 切换）   aw=0.5

起训：S2 balance u1680（--reset-update）——指令输入语义全变。

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=<N> python3 -B \\
      baseline/framework/train.py --experiment v2_stepbudget --algo ppo \\
      --resume-from baseline/runs/v2_s2_balance/checkpoints/checkpoint_u01680.pt \\
      --reset-update --run-name v2_stepbudget_r1 --background
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.end2end.stepping_state_machine import (
    count_falls, fall_onset_mask,
)
from baseline.humanoid21.v1.step_budget_gen import CLIP_NAMES
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_stepcmd import V2StepCmd

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"
_DT = 0.05                     # s / action frame（25 × 2ms）
# 方向分桶：front |rel|<45° / side 45–135° / back >135°
_DIR_B = (("f", 0.0, 45.0), ("s", 45.0, 135.0), ("b", 135.0, 181.0))
# 预算分桶（帧）
_BUD_B = ((1, 3, 6), (2, 7, 10), (3, 11, 14), (4, 15, 20))


class V2StepBudget(V2StepCmd):
    name = "v2_stepbudget"

    _channel_names = ("r_potential", "r_fall", "r_progress",
                      "r_lateral", "r_land", "r_yaw", "r_pivot")
    _channel_gammas = {k: 0.99 for k in _channel_names}
    _STEP_OBS = {"robot_a": "stepbudget_report_a",
                 "robot_b": "stepbudget_report_b"}

    aw_progress: float = 1.0
    aw_lateral: float = 0.5
    aw_land: float = 0.5
    aw_yaw: float = 0.5
    aw_pivot: float = 0.5
    w_prog: float = 10.0       # /m 包络内沿向推进
    w_viol: float = 10.0       # /m 终态出包络距离（每步一次）
    w_lat: float = 1.0         # /m
    lat_tol: float = 0.03
    w_land: float = 2.0        # 每步一次
    w_yaw: float = 1.0         # /rad
    yaw_tol: float = 0.05
    w_pv: float = 1.0          # /m

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_stepbudget_env.yaml")

    # ---------------- 轨迹 ----------------
    def _base_rewards(self, episode, agent_id, phi_key, T, T_full):
        phi = extract_per_step_field(
            episode.observer_outputs, phi_key, "potential", T_full)
        phi = (np.clip(np.asarray(phi[:T], np.float32), 0.0, 1.0)
               if phi is not None else np.zeros(T, np.float32))
        r_pot = (self.per_step_phi_coef * phi).astype(np.float32)
        r_fall = -self.fall_penalty * fall_onset_mask(phi).astype(
            np.float32)
        wall_key = self._WALL_OBS.get(agent_id)
        if wall_key is not None and self.wall_penalty > 0:
            from baseline.humanoid21.end2end.wall_contact_observer \
                import sustained_wall_mask
            wc = extract_per_step_field(
                episode.observer_outputs, wall_key, "wall_contact",
                T_full)
            if wc is not None:
                lean = sustained_wall_mask(
                    wc, window=self.wall_window,
                    count_thresh=self.wall_count_thresh)
                r_fall = r_fall - (self.wall_penalty
                                   * lean[:T].astype(np.float32)
                                   * (phi >= 0.5)[:T].astype(np.float32))
        return r_pot, r_fall.astype(np.float32)

    def _build_agent_trajectory(self, episode, agent_id: str,
                                foot_key: str, phi_key: str
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
        obs_all = np.asarray(obs_all, np.float32)
        acts_all = np.asarray(acts_all, np.float32)

        r_pot, r_fall = self._base_rewards(
            episode, agent_id, phi_key, T, T_full)
        key = self._STEP_OBS[agent_id]

        def f(name: str) -> np.ndarray:
            a = extract_per_step_field(
                episode.observer_outputs, key, name, T_full)
            if a is None:
                raise RuntimeError(f"stepbudget 字段缺失: {key}.{name}")
            return np.asarray(a[:T], np.float32)

        mv = f("mv")
        r_progress = self.w_prog * f("prog") - self.w_viol * f("viol_end")
        r_lateral = -self.w_lat * mv * np.maximum(
            0.0, f("lat") - self.lat_tol)
        r_land = self.w_land * f("land")
        r_yaw = -self.w_yaw * f("yaw_w") * np.maximum(
            0.0, f("yaw_e") - self.yaw_tol)
        r_pivot = -self.w_pv * np.maximum(0.0, f("pv_dr") - f("pv_tol"))

        aw = lambda w: np.full(T, w, np.float32)
        mk = lambda r, w: ChannelData(
            reward=np.asarray(r, np.float32), is_terminated=False,
            actor_weight=aw(w))
        channels = {
            "r_potential": mk(r_pot, self.r_potential_actor_weight),
            "r_fall": mk(r_fall, self.r_fall_actor_weight),
            "r_progress": mk(r_progress, self.aw_progress),
            "r_lateral": mk(r_lateral, self.aw_lateral),
            "r_land": mk(r_land, self.aw_land),
            "r_yaw": mk(r_yaw, self.aw_yaw),
            "r_pivot": mk(r_pivot, self.aw_pivot),
        }
        return [Trajectory(
            obs=obs_all[:T], actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, np.float32),
            channels=channels, importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots, falls = [], []
        recs: Dict[str, List[np.ndarray]] = {}
        names = ("hold", "zdir_r", "budget", "clip", "rel", "stride",
                 "s_max", "dir_err", "lat_r", "landed", "viol_end_r",
                 "pv_dr_r", "yaw_e_r", "pv_air")
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for _aid, _fk, phi_key in self._AGENT_OBS:
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T)
                if phi is not None and len(phi):
                    max_pots.append(float(np.max(phi)))
                    falls.append(count_falls(phi))
            for _rid, key in self._STEP_OBS.items():
                ev = extract_per_step_field(
                    ep.observer_outputs, key, "event", T)
                if ev is None:
                    continue
                m = np.asarray(ev) > 0.5
                if not m.any():
                    continue
                for nm in names:
                    a = extract_per_step_field(
                        ep.observer_outputs, key, nm, T)
                    recs.setdefault(nm, []).append(np.asarray(a)[m])
        R = {k: np.concatenate(v) for k, v in recs.items()}
        info: Dict[str, Any] = {}
        n = max(len(max_pots), 1)
        success = sum(p >= self._success_thresh for p in max_pots) / n
        info["max_pot"] = round(sum(max_pots) / n, 3)
        info["success"] = round(success, 3)
        info["falls"] = round(sum(falls) / max(len(falls), 1), 2)
        quality = -1.0
        if R:
            hold = R["hold"] > 0.5
            mvm = ~hold
            dirm = mvm & (R["zdir_r"] < 0.5)
            zm = mvm & (R["zdir_r"] > 0.5)
            mean = lambda a, mm: float(a[mm].mean()) if mm.any() else 0.0
            info["step_ev"] = int(len(hold))
            info["land"] = round(mean(R["landed"], mvm), 3)
            info["stride"] = round(mean(R["stride"], dirm), 3)
            info["cpl"] = round(mean(
                np.clip(R["stride"] / np.maximum(R["s_max"], 1e-3),
                        -1, 1.5), dirm), 3)
            info["dir_err"] = round(mean(R["dir_err"], dirm), 3)
            info["lat"] = round(mean(R["lat_r"], dirm), 3)
            info["zdir_dev"] = round(mean(R["lat_r"], zm), 3)
            info["viol"] = round(mean(
                (R["viol_end_r"] > 0.02).astype(float), mvm), 3)
            info["pv_dr"] = round(mean(R["pv_dr_r"], mvm), 3)
            info["hold_dr"] = round(mean(R["pv_dr_r"], hold), 3)
            info["yaw_e"] = round(float(R["yaw_e_r"].mean()), 3)
            info["pv_air"] = round(mean(R["pv_air"], mvm), 3)
            # 预算→步长曲线 + 各方向最快步速
            arel = np.degrees(np.abs(R["rel"]))
            for dn, lo, hi in _DIR_B:
                dm = dirm & (arel >= lo) & (arel < hi)
                best_v, best_b = 0.0, 0
                for bi, blo, bhi in _BUD_B:
                    mm = dm & (R["budget"] >= blo) & (R["budget"] <= bhi)
                    s = mean(R["stride"], mm)
                    info[f"L{dn}{bi}"] = round(s, 3)
                    if mm.any():
                        v = float((R["stride"][mm]
                                   / (R["budget"][mm] * _DT)).mean())
                        if v > best_v:
                            best_v, best_b = v, bi
                info[f"v{dn}"] = round(best_v, 3)
                info[f"vb{dn}"] = best_b
            # 按 clip 分桶：动脚令着地率
            for ci, cn in enumerate(CLIP_NAMES):
                cm = mvm & (R["clip"] == ci)
                if cm.any():
                    info[f"c_{cn}"] = round(mean(R["landed"], cm), 2)
            quality = (info["land"]
                       + 0.5 * min(info["cpl"], 1.0)
                       - 0.5 * min(info["dir_err"] / 0.5, 1.0)
                       - 0.3 * min(info["yaw_e"] / 0.5, 1.0)
                       - 0.5 * info["viol"])
        info["quality"] = round(quality, 3)

        if self._b_quality_ema is None:
            self._b_quality_ema = quality
        else:
            self._b_quality_ema = 0.85 * self._b_quality_ema + 0.15 * quality
        improved = success >= 0.9 and quality > self._b_best_quality
        if improved:
            self._b_best_quality = quality
        if improved or self._b_last_best_update < 0:
            self._b_last_best_update = update
        pot_mean = sum(max_pots) / n
        improved_pot = pot_mean > self._best_potential
        if improved_pot:
            self._best_potential = pot_mean
        stop = (update - self._b_last_best_update
                >= self._no_improvement_limit
                and update >= self._min_updates)
        return {"is_new_best": improved or improved_pot,
                "stop_training": stop, "info": info}


EXPERIMENT_CLASS = V2StepBudget
