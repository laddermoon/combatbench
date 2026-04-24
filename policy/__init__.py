"""CombatBench policy package.

The canonical policy contract is :class:`envs.framework.policy.Policy` —
this package provides a dynamic loader (:func:`load_policy`) plus a few
reference implementations under sibling sub-packages (``random/``,
``standing/``, ...).

Loading::

    from combatbench.policy import load_policy

    # Load from directory (auto-detect first Policy subclass)
    policy = load_policy("my_policy")

    # Load a specific class
    policy = load_policy("my_policy.policy:MyPolicy")

    # Load with query-string kwargs
    policy = load_policy("my_policy.policy:MyPolicy?lr=0.01&epochs=100")
"""

from envs.framework.policy import Policy

from .load_util import load_policy, load_policy_from_dir

__all__ = [
    "Policy",
    "load_policy",
    "load_policy_from_dir",
]
