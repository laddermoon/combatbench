"""V2 训练仿真器：79 维 obs = 57 本体 + 22 指令段（当前全零）。

22 维指令段布局见 DECISION_LOG D8：
    flag(1) | arm_tgt(6) | waist_tgt(3) | step_cur(6) | step_next(6)

S1 阶段整段恒零——flag=0 语义即"臂输出未被替换"（策略自控双臂，
起身需要）；arm/waist/step 字段为接管与指令的预留占位，待相应
阶段接入脚本源时再填充，不提前实现（D8「需要时再复杂」原则）。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator

BODY_DIM = 57      # obs[0:57]：本体感知42 + 全局状态13 + 触觉2
CMD_DIM = 22       # D8 指令段（现恒零）
OBS_DIM = BODY_DIM + CMD_DIM   # 79


class V2Simulator(Humanoid21Simulator):
    """剥对手 39 维 + 追加 22 维零指令段的 79 维观测仿真器。"""

    def get_observation(self) -> Dict[str, Any]:
        if "_obs_v2" not in self._data_cache:
            base = super().get_observation()          # 96 维
            zeros = np.zeros(CMD_DIM, np.float32)
            self._data_cache["_obs_v2"] = {
                rid: np.concatenate(
                    [np.asarray(o, np.float32)[:BODY_DIM], zeros])
                for rid, o in base.items()}
        return self._data_cache["_obs_v2"]
