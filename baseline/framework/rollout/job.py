"""Re-export ``Job`` and ``EiSpec`` from the core framework layer.

These concepts have been promoted to :mod:`envs.framework.job` because
``explore_intensity`` is already part of the core policy execution
interface (``Policy.act``, ``EnvRuntime.step``, ``EpisodeRunner``).
This module remains for import-path stability.
"""
from envs.framework.job import EiSpec, Job

__all__ = ["EiSpec", "Job"]
