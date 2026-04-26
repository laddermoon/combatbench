"""Top-level picklable fixtures for parallel-rollout tests.

Placed at a stable import path (``baseline.common.rollout._parallel_probe``)
so that spawned worker processes can re-import the factories. Anonymous
closures / test-file-local functions would fail to pickle under the
``spawn`` start method — which is the default on macOS and required on
CUDA builds to avoid forked-CUDA crashes.

Importing this at runtime has no side effect; the only cost is when a
test actually instantiates the factories.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make ``envs.framework.tests.conftest`` importable from worker processes
# (spawned workers don't inherit sys.path from pytest).
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_FRAMEWORK_TESTS = _REPO / "envs" / "framework" / "tests"
if str(_FRAMEWORK_TESTS) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_TESTS))

from envs.framework.env_runtime import EnvRuntime
from envs.framework.runtime_plugin import BaseObserverPlugin
from conftest import MockSimulator  # type: ignore[import-not-found]

from baseline.common.policies import TanhGaussianMLPPolicy, TorchPolicyAdapter


OBS_DIM = 5
ACTION_DIM = 21
HIDDEN = 8
MAX_STEPS = 3


class _QposObserver(BaseObserverPlugin):
    def __init__(self) -> None:
        self._output = np.zeros(OBS_DIM, dtype=np.float32)

    def on_pre_episode(self, ctx) -> None:
        self._output = ctx.accessor.get_core_state()["qpos"][:OBS_DIM].astype(np.float32)

    def on_post_action_step(self, ctx) -> None:
        self._output = ctx.accessor.get_core_state()["qpos"][:OBS_DIM].astype(np.float32)

    def get_output(self) -> np.ndarray:
        return self._output.copy()


class _ConstRewardObserver(BaseObserverPlugin):
    def __init__(self) -> None:
        self._v = 1.0

    def on_pre_episode(self, ctx) -> None:
        pass

    def on_post_action_step(self, ctx) -> None:
        pass

    def get_output(self) -> float:
        return float(self._v)


def make_runtime() -> EnvRuntime:
    """Top-level runtime factory (picklable under ``spawn``)."""
    return EnvRuntime(
        simulator=MockSimulator(),
        observer_plugins={
            "robot_a_obs": _QposObserver(),
            "robot_a_reward": _ConstRewardObserver(),
            "robot_b_obs": _QposObserver(),
            "robot_b_reward": _ConstRewardObserver(),
        },
        max_steps=MAX_STEPS,
        phy_steps_per_action=1,
    )


def make_adapter() -> TorchPolicyAdapter:
    """Top-level policy factory — torch seed pinned for reproducibility."""
    torch.manual_seed(0)
    actor = TanhGaussianMLPPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN)
    return TorchPolicyAdapter(actor=actor, deterministic=True)


def build_forged_state_dict(seed: int) -> dict:
    """Build a state_dict for a distinct-weight actor so state_dict hot-reload
    has an observable effect. CPU-side, detached — ready to ship to workers.
    """
    torch.manual_seed(int(seed))
    actor = TanhGaussianMLPPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN)
    return {k: v.detach().cpu() for k, v in actor.state_dict().items()}
