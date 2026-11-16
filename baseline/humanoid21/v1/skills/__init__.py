"""V1 技能契约包 —— 无状态技能规约基类 + 共享步边界服务。

技能实现放在本包子模块中，经 ``register_skill`` 登记；消费方
（训练实验、可行性查询、上层战术代码）经 ``get_skill`` 取契约实例。
契约实例无 episode 状态，可全局共享。
"""

from typing import Dict

from .base import (AnchorMatch, ContextSpec, Skill, SkillRuntime,
                   SkillScore, SkillState, TerminalSpec, cpose_distance,
                   FOOT_LEFT, FOOT_RIGHT)
from .boundary import StepBoundaryTracker, StepEvent

__all__ = [
    "AnchorMatch", "ContextSpec", "Skill", "SkillRuntime", "SkillScore",
    "SkillState", "TerminalSpec", "cpose_distance",
    "FOOT_LEFT", "FOOT_RIGHT",
    "StepBoundaryTracker", "StepEvent",
    "register_skill", "get_skill", "list_skills", "SKILL_REGISTRY",
]

SKILL_REGISTRY: Dict[str, Skill] = {}


def register_skill(skill: Skill) -> Skill:
    """登记技能契约实例（实例无状态，登记即共享）。"""
    if not skill.name:
        raise ValueError("skill.name is empty")
    SKILL_REGISTRY[skill.name] = skill
    return skill


def get_skill(name: str) -> Skill:
    return SKILL_REGISTRY[name]


def list_skills() -> Dict[str, Skill]:
    return dict(SKILL_REGISTRY)
