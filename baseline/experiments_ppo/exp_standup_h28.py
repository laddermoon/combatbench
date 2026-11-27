"""Humanoid28 4-stage standup — exp_standup 的 humanoid28 复现版。

与 humanoid21 版唯一的本质差异是观测/动作维度（110/28 vs 96/21）
和蓝图目录（baseline/humanoid28/blueprints/）。rewarder 已按 humanoid28
机身重新标定 H_STAND=1.1186（见 baseline/humanoid28/rewards/）。

用途：验证 humanoid28 环境端到端训练通路可用性。
"""
from __future__ import annotations

from .exp_standup import Standup


class StandupH28(Standup):
    name = "standup_h28"
    blueprint_subdir = "humanoid28"

    # --- Network ---
    obs_dim: int = 110
    action_dim: int = 28


EXPERIMENT_CLASS = StandupH28
