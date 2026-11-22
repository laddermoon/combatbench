"""STEPTVAR V1 — 随机时长步令（STEPTVAR_V1_PLAN.md）：79 维。

= v2_stepcmd 任务框架，sim 换 V2StepTvarSimulator：指令周期
逐令 U(1,20) 帧且 obs 不可见——唯一鲁棒策略=ASAP。
5 通道原样。eval = stepcmd 全字段 + 时长分桶/驻留/半空诊断。

分桶（非保持令）：dur≤5 短令 / 6-15 中令 / ≥16 长令；
hold_err = 长令且已首达者的终末 d_mv（驻留质量）；
air_frac = st_z>0.20 的事件占比（半空发令频率）；
res0 = 发令初始残差（preview 预摆位度量）。

warm-start: v2_stepcmd_r1 最新 ckpt --reset-update。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from baseline.framework.rollout import extract_per_step_field
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_stepcmd import V2StepCmd

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"

_ST_Z_AIR = 0.20          # 轴心脚 z 阈值：>0.20 ≈ 半空发令


class V2StepTvar(V2StepCmd):
    name = "v2_steptvar"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_steptvar_env.yaml")

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        info = res["info"]

        # 按时长分桶：idx 0:短(≤5) 1:中(6-15) 2:长(≥16)
        n_b = np.zeros(3)
        arr_b = np.zeros(3)
        dm_b = np.zeros(3)
        hold_n = hold_cnt = 0.0
        air_n = 0
        ev_all = 0
        res0_s = res0_n = 0.0
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
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
                dur = _fld("dur")
                ta = _fld("t_arrive")
                dmv = _fld("d_mv")
                res0 = _fld("res0")
                stz = _fld("st_z")
                ev_all += int(m.sum())
                air_n += int((stz > _ST_Z_AIR).sum())
                step_m = ~hold
                for b, lo, hi in ((0, 0, 5), (1, 6, 15), (2, 16, 99)):
                    mm = step_m & (dur >= lo) & (dur <= hi)
                    n_b[b] += int(mm.sum())
                    arr_b[b] += float((ta[mm] >= 0).sum())
                    dm_b[b] += float(dmv[mm].sum())
                    res0_s += float(res0[mm].sum())
                    res0_n += int(mm.sum())
                # 长令且已首达 → 终末残差 = 驻留质量
                hq = step_m & (dur >= 16) & (ta >= 0)
                hold_cnt += float(dmv[hq].sum())
                hold_n += int(hq.sum())

        info["arr_s"] = round(arr_b[0] / max(n_b[0], 1), 3)
        info["arr_m"] = round(arr_b[1] / max(n_b[1], 1), 3)
        info["arr_l"] = round(arr_b[2] / max(n_b[2], 1), 3)
        info["dmv_s"] = round(dm_b[0] / max(n_b[0], 1), 3)
        info["dmv_m"] = round(dm_b[1] / max(n_b[1], 1), 3)
        info["dmv_l"] = round(dm_b[2] / max(n_b[2], 1), 3)
        info["hold_err"] = round(hold_cnt / max(hold_n, 1), 3)
        info["air_frac"] = round(air_n / max(ev_all, 1), 3)
        info["res0"] = round(res0_s / max(res0_n, 1), 3)
        return res


EXPERIMENT_CLASS = V2StepTvar
