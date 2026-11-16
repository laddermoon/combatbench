"""V1 技能契约包 —— 无状态技能规约基类 + 第一阶段四技能。

契约实例无 episode 状态；运行态、复杂注册机制随需要再引入。
"""

from .base import Clip, Skill, end_support_score
from .step import StepSkill, step_l, step_r
from .jab import JabSkill, jab_l, jab_r

SKILLS = {
    "step_l": step_l,
    "step_r": step_r,
    "jab_l": jab_l,
    "jab_r": jab_r,
}

__all__ = [
    "Clip", "Skill", "end_support_score",
    "StepSkill", "step_l", "step_r",
    "JabSkill", "jab_l", "jab_r",
    "SKILLS",
]
