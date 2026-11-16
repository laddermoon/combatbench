"""V1 技能训练仿真器：快照起点 + 技能指令注入。

继承 ``CommandSimulator``（66 维 obs = 57 本体 + 9 指令槽），reset() 时
追加消费两个 episode_options：

- ``options["v1_snapshot"]`` = ``{robot_id: {core_state 字段}}``
  把该 agent 重置到快照状态（起点池条目）；未提供的 agent 走
  ``initial_pose_*`` 默认复位。
- ``options["v1_cmd"]`` = ``{robot_id: [9 维指令]}``
  cmd[0:6] = D1 六维技能码，cmd[6:9] = 0（D2 约定）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from baseline.humanoid21.hierarchy.command_simulator import CommandSimulator


class V1SkillSimulator(CommandSimulator):
    """CommandSimulator + v1 快照/技能指令 episode options。"""

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)
        opts = options or {}
        for rid, st in (opts.get("v1_snapshot") or {}).items():
            self.set_core_state({
                rid: {k: np.asarray(v, np.float64)
                      for k, v in st.items()}})
        for rid, cmd in (opts.get("v1_cmd") or {}).items():
            self.set_command(rid, np.asarray(cmd, np.float64))
