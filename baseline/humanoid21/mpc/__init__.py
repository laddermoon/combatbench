"""Simulation-feedback MPC baselines for Humanoid21."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from .standing_mpc import FeedbackGains, StandingMPCConfig, StandingMPCPolicy

__all__ = ["FeedbackGains", "StandingMPCConfig", "StandingMPCPolicy"]
