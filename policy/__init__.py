"""
CombatBench Policy Module

Available Policies:
    - BaseCombatPolicy: Abstract base class defining the interface

Policy Loading:
    Use load_policy() to load policies from directory:

    ```python
    from combatbench.policy import load_policy

    # Load from directory (auto-detect first BaseCombatPolicy)
    policy = load_policy("my_policy")

    # Load with specific class
    policy = load_policy("my_policy.policy.MyCombatPolicy")

    # Load with parameters
    policy = load_policy("my_policy.policy.MyCombatPolicy?lr=0.01&epochs=100")
    ```
"""

from .base import BaseCombatPolicy
from .load_util import load_policy, load_policy_from_dir

__all__ = [
    "BaseCombatPolicy",
    "load_policy",
    "load_policy_from_dir",
]
