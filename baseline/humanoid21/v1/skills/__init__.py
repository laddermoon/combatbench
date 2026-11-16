"""V1 技能契约包 —— 无状态技能规约基类。

契约实例无 episode 状态；运行态、注册等机制随消费方出现时再引入。
"""

from .base import AnchorMatch, Skill, cpose_distance

__all__ = ["AnchorMatch", "Skill", "cpose_distance"]
