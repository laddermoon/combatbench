"""V2 训练仿真器：79 维 obs = 57 本体 + 22 指令段。

22 维指令段布局见 DECISION_LOG D8：
    flag(1) | arm_tgt(6) | waist_tgt(3) | step_cur(6) | step_next(6)

阶段语义：
- S1/S2（arm_script=False）：整段恒零——flag=0 即"臂输出未被
  替换"（策略自控双臂）；arm/waist/step 字段为预留占位。
- S3（arm_script=True）：双臂随机脚本接管。调度+观测+覆写全部
  在本类内完成（D6 同源契约：obs 承诺 = 实际施加）：

    * get_observation 纯读：seg = [flag | arm_tgt | 0×15]，无副作用，
      任意时刻/频次调用都返回"下一步将施加"的承诺。
    * set_action 是唯一状态推进点（EnvRuntime 每 action step 恰好
      调一次）：先按当前 flag/tgt 覆写 act[15:21]（兑现 obs[t] 承诺），
      再更新 flag 状态机 + 推进调度器（为 obs[t+1] 生产承诺）。
      flag 判据复用推力插件站立状态机：h>flag_height 计时+1，
      连续 >grace 帧低于阈值清零，满 settle 步 → flag=1。
      覆写发生在 env 内，轨迹存策略原始输出（变换 MDP 上标准 PPO，
      臂维成死动作维→有界漂移；eval/demo 同 env 生效零差异）。
    * flag=0 时 arm_tgt 仍填脚本当前目标——语义统一+可预判接管。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator

from .arm_script import ArmScript

BODY_DIM = 57      # obs[0:57]：本体感知42 + 全局状态13 + 触觉2
CMD_DIM = 22       # D8 指令段
OBS_DIM = BODY_DIM + CMD_DIM   # 79

_FLAG_IDX, _TGT_IDX = 0, slice(1, 7)   # 指令段内偏移：flag | arm_tgt
_ARM_ACT = slice(15, 21)               # action 中臂关节段
_ROBOTS = ("robot_a", "robot_b")


class V2Simulator(Humanoid21Simulator):
    """剥对手 39 维 + 22 维指令段（可选随机臂脚本）的 79 维仿真器。"""

    def __init__(self, *args: Any,
                 arm_script: bool = False,
                 arm_override: bool = True,
                 arm_hold_range: Tuple[int, int] = (2, 20),
                 arm_tgt_range: Tuple[float, float] = (-1.0, 1.0),
                 arm_release_prob: float = 0.0,
                 flag_height: float = 1.10,
                 flag_grace: int = 5,
                 flag_settle: int = 20,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._arm_on = bool(arm_script)
        # arm_override=False → arm_tgt 仅作指令观测+跟随奖励输入，
        # 不覆写 act[15:21]（D17 臂令跟随；默认 True 保持 S3 行为）
        self._arm_override = bool(arm_override)
        self._hold_range = tuple(arm_hold_range)
        self._tgt_range = tuple(arm_tgt_range)
        self._release_prob = float(arm_release_prob)
        self._flag_h = float(flag_height)
        self._flag_grace = int(flag_grace)
        self._flag_settle = int(flag_settle)
        # 每机 {timer, below, flag, script}（reset 时初始化）
        self._st: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------- reset --
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)
        if not self._arm_on:
            return
        base = 0 if seed is None else int(seed) & 0xFFFFFFFF
        for i, rid in enumerate(_ROBOTS):
            rng = np.random.default_rng([base, 0xA8 + i])
            st = {"timer": 0, "below": 0,
                  # flag_settle<=0 → 恒 1（令恒激活语义，armsoft V1）
                  "flag": 1.0 if self._flag_settle <= 0 else 0.0,
                  "script": ArmScript(rng, self._hold_range,
                                      self._tgt_range,
                                      self._release_prob)}
            st["script"].step()          # obs[0] 即显示真实脚本目标
            self._st[rid] = st

    # ---------------------------------------------------------- 指令段 -----
    def get_observation(self) -> Dict[str, Any]:
        if "_obs_v2" not in self._data_cache:
            base = super().get_observation()          # 96 维
            out: Dict[str, Any] = {}
            for rid, o in base.items():
                seg = np.zeros(CMD_DIM, np.float32)
                if self._arm_on:
                    st = self._st[rid]
                    seg[_FLAG_IDX] = st["flag"]
                    seg[_TGT_IDX] = st["script"].current
                out[rid] = np.concatenate(
                    [np.asarray(o, np.float32)[:BODY_DIM], seg])
            self._data_cache["_obs_v2"] = out
        return self._data_cache["_obs_v2"]

    # -------------------------------------------------------- action 覆写 --
    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        """兑现 obs 承诺（flag=1 → act[15:21]=arm_tgt），再推进状态。"""
        if self._arm_on and self._arm_override:
            patched: Dict[str, Optional[Any]] = dict(action)
            for rid in _ROBOTS:
                st = self._st.get(rid)
                if st is None or st["flag"] < 0.5:
                    continue
                raw = patched.get(rid)
                if raw is None:
                    continue
                if hasattr(raw, "detach"):
                    raw = raw.detach().cpu().numpy()
                act = np.asarray(raw, np.float32).copy()
                cur = st["script"].current
                rel = st["script"].released
                if not rel[0]:
                    act[15:18] = cur[0:3]
                if not rel[1]:
                    act[18:21] = cur[3:6]
                patched[rid] = act
            action = patched
        super().set_action(action)
        if not self._arm_on:
            return
        # ---- 推进（每 action step 恰好一次）：flag 状态机 + 调度器 ----
        core = self.get_core_state()
        for rid in _ROBOTS:
            st = self._st.get(rid)
            if st is None:
                continue
            h = float(np.asarray(core[rid]["root_pos"], np.float64)[2])
            if h > self._flag_h:
                st["timer"] += 1
                st["below"] = 0
            else:
                st["below"] += 1
                if st["below"] > self._flag_grace:
                    st["timer"] = 0
            st["flag"] = 1.0 if st["timer"] >= self._flag_settle else 0.0
            st["script"].step()
