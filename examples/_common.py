"""Shared helpers for the ``examples/`` scripts.

Only one non-trivial utility lives here — :func:`build_humanoid21_runtime` —
so each example can stay focused on the pedagogical idea and not on
"how do I wire up a humanoid21 runtime for the 20th time".

Everything else (policies, plugins, observers) is inlined inside each
example on purpose, to keep them readable top-to-bottom.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Render MuJoCo off-screen so examples run on headless boxes (GPU server,
# CI). Set BEFORE importing anything that pulls in MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Make ``combatbench.<pkg>`` imports work when running the files directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.framework import EnvRuntime  # noqa: E402
from envs.humanoid21 import make_env  # noqa: E402
from envs.humanoid21.plugins import CombatScoringPlugin  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parent / "out"


def example_out_dir(name: str) -> Path:
    """Return (and create) ``examples/out/<name>/`` for writing artifacts."""
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_humanoid21_runtime(
    *,
    match_duration: float = 5.0,
    control_frequency: int = 20,
    initial_health_a: float = 100.0,
    initial_health_b: float = 100.0,
    extra_plugins: Optional[List[Any]] = None,
    observer_plugins: Optional[Dict[str, Any]] = None,
) -> EnvRuntime:
    """Build a combat-ready humanoid21 runtime.

    Defaults to a **short 5-second match** so examples finish quickly; the
    evaluation example (06) overrides this to the full 30-second rule.

    Parameters
    ----------
    match_duration:
        Episode duration in seconds. ``max_steps`` is derived via
        ``match_duration * control_frequency``.
    control_frequency:
        Policy decision rate in Hz (rule: 20Hz).
    initial_health_a, initial_health_b:
        Starting HP. Required by :class:`CombatScoringPlugin` to compute
        damage / winner / termination in the standard combat ruleset.
    extra_plugins:
        Additional :class:`BasePlugin` instances to mount *after*
        :class:`CombatScoringPlugin`. Use this to layer curriculum /
        early-termination / observer plugins on top.
    observer_plugins:
        Extra observer plugins. Humanoid21's default ``robot_a_obs`` /
        ``robot_b_obs`` are always included; anything you pass here is
        merged on top.
    """
    plugins: List[Any] = [
        CombatScoringPlugin(
            initial_health_a=initial_health_a,
            initial_health_b=initial_health_b,
        ),
    ]
    if extra_plugins:
        plugins.extend(extra_plugins)

    return make_env(
        match_duration=match_duration,
        control_frequency=control_frequency,
        plugins=plugins,
        observer_plugins=observer_plugins,
    )
