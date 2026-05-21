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
from typing import Any, List, Optional

import numpy as np
from gymnasium import spaces

# Render MuJoCo off-screen so examples run on headless boxes (GPU server,
# CI). Set BEFORE importing anything that pulls in MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Make ``combatbench.<pkg>`` imports work when running the files directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.framework import EnvRuntime  # noqa: E402
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parent / "out"

# Humanoid21 constants
ACTION_DIM = 21
OBS_DIM = 96


def example_out_dir(name: str) -> Path:
    """Return (and create) ``examples/out/<name>/`` for writing artifacts."""
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_humanoid21_runtime(
    *,
    match_duration: float = 5.0,
    initial_distance: float = 2.0,
    extra_plugins: Optional[List[Any]] = None,
) -> EnvRuntime:
    """Build a combat-ready humanoid21 runtime using ParameterizedEnvBlueprint.

    Defaults to a **short 5-second match** so examples finish quickly; the
    evaluation example (06) overrides this to the full 30-second rule.

    Note: control frequency is fixed at 20Hz (phy_steps_per_action=25) per
    the combat rules. Do not change this value.

    Parameters
    ----------
    match_duration:
        Episode duration in seconds. ``max_steps`` is derived as
        ``match_duration * 20`` (20 = control frequency in Hz).
    initial_distance:
        Initial distance between the two robots in meters.
    extra_plugins:
        Additional :class:`BasePlugin` instances to mount.

    Returns
    -------
    EnvRuntime
        A configured runtime ready for reset/step calls.
    """
    # Load the parameterized blueprint
    blueprint_path = Path(__file__).parent.parent / "envs" / "humanoid21" / "blueprint.yaml"
    pb = ParameterizedEnvBlueprint.load(blueprint_path)

    # Calculate max_steps from match_duration (control frequency is fixed at 20Hz)
    control_frequency = 20  # Fixed per combat rules
    max_steps = int(match_duration * control_frequency)

    # Build runtime with parameter overrides
    runtime = pb.build(
        max_steps=max_steps,
        initial_distance=initial_distance,
        debug_plugins=list(extra_plugins) if extra_plugins else [],
    )

    # Attach action_space and observation_space for compatibility
    runtime.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
    )
    runtime.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )

    return runtime
