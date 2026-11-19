"""V2 臂令跟随仿真器（D17）：臂目标只进观测+奖励，不覆写动作。

继承 V2StepSimulator（79 维 obs），但 `step_enabled` 默认 False——
本实验是 S3 血统（平衡+臂令+推力），不背 S4 步令任务；需臂令+步令
联合实验时 yaml 显式 `step_enabled: true` 即可打开。与 S3 臂接管的
唯一差别：`arm_override=False`——arm_tgt 指令流照常生成/写入
seg[1:7]，但 act[15:21] 完全归策略；跟随压力改由奖励通道表达
（"可忽略指令"：站稳时跟随划算，失衡时策略自己压掉）。

跟随误差（每 action step 结算一次，obs 承诺目标即当期脚本目标）：
    err_r = ‖act[15:18] − tgt[0:3]‖   （右臂 3 关节，norm 空间）
    err_l = ‖act[18:21] − tgt[3:6]‖   （左臂）

误差经 get_derived_state(["arm_follow"]) 供 observer 读取
（accessor 白名单合规通道，同 step_reports 模式）。flag 维保留
原义（近期站稳标记），仅作观测信息位、不再门控任何事。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .v2_step_simulator import V2StepSimulator

_ROBOTS = ("robot_a", "robot_b")


class V2ArmCmdSimulator(V2StepSimulator):
    """V2StepSimulator + 臂令跟随误差上报（不覆写臂动作）。"""

    def __init__(self, *args: Any, **kw: Any) -> None:
        kw.setdefault("arm_override", False)
        kw.setdefault("step_enabled", False)
        super().__init__(*args, **kw)

    # -------------------------------------------------------- action 结算 --
    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        """先按 obs 承诺的 arm_tgt 结算跟随误差，再推进状态。"""
        if self._arm_on:
            for rid in _ROBOTS:
                st = self._st.get(rid)
                if st is None:
                    continue
                raw = action.get(rid)
                if raw is None:
                    st["arm_err"] = None
                    continue
                if hasattr(raw, "detach"):
                    raw = raw.detach().cpu().numpy()
                act = np.asarray(raw, np.float32)
                tgt = st["script"].current
                st["arm_err"] = (
                    float(np.linalg.norm(act[15:18] - tgt[0:3])),
                    float(np.linalg.norm(act[18:21] - tgt[3:6])),
                )
        super().set_action(action)

    # ------------------------------------------------- observer 接口 ------
    def arm_follow(self, rid: str) -> Optional[Dict[str, float]]:
        """本步臂跟随报告（内部/测试用）；observer 走 derived_state。"""
        st = self._st.get(rid)
        if st is None:
            return None
        err = st.get("arm_err")
        if err is None:
            return None
        return {"err_r": float(err[0]), "err_l": float(err[1]),
                "flag": float(st["flag"])}

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        """扩 DATASPEC：arm_follow = {rid: {err_r,err_l,flag}|None}。"""
        _EXTRA = "arm_follow"
        if fields is None:
            out = super().get_derived_state()
            out[_EXTRA] = {r: self.arm_follow(r) for r in _ROBOTS}
            return out
        fields = list(fields)
        base = [f for f in fields if f != _EXTRA]
        out = (super().get_derived_state(base) if base else {})
        if _EXTRA in fields:
            out[_EXTRA] = {r: self.arm_follow(r) for r in _ROBOTS}
        return out


__all__ = ["V2ArmCmdSimulator"]
