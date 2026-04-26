"""Tests for ``TorchPolicyAdapter``."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pytest
import torch
from torch import nn

# Project root for relative imports / temp policy dir export.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.common.policies import (
    CriticMLP,
    TanhGaussianMLPPolicy,
    TorchPolicyAdapter,
)


def _make_actor(obs_dim: int = 4, action_dim: int = 2, hidden_dim: int = 8) -> TanhGaussianMLPPolicy:
    torch.manual_seed(0)
    return TanhGaussianMLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)


class _MissingActNumpy(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


class TestActContract:
    def test_act_returns_float32_action(self):
        actor = _make_actor()
        adapter = TorchPolicyAdapter(actor=actor, deterministic=True)
        action = adapter.act(np.zeros(4, dtype=np.float32))
        assert action.dtype == np.float32
        assert action.shape == (2,)

    def test_act_with_extras_carries_log_prob_when_stochastic(self):
        actor = _make_actor()
        adapter = TorchPolicyAdapter(actor=actor, deterministic=False)
        action, extras = adapter.act_with_extras(np.zeros(4, dtype=np.float32))
        assert action.shape == (2,)
        assert "log_prob" in extras
        assert isinstance(extras["log_prob"], float)

    def test_deterministic_drops_log_prob(self):
        actor = _make_actor()
        adapter = TorchPolicyAdapter(actor=actor, deterministic=True)
        _, extras = adapter.act_with_extras(np.zeros(4, dtype=np.float32))
        # In deterministic mode TanhGaussianMLPPolicy.act_numpy returns log_prob=None
        # so the adapter must not surface it.
        assert "log_prob" not in extras

    def test_actor_without_act_numpy_rejected(self):
        with pytest.raises(TypeError, match="act_numpy"):
            TorchPolicyAdapter(actor=_MissingActNumpy())


class TestExtrasWithCritic:
    def test_value_propagated_when_critic_attached(self):
        actor = _make_actor()
        critic = CriticMLP(obs_dim=4, hidden_dim=8)
        adapter = TorchPolicyAdapter(actor=actor, critic=critic, deterministic=True)
        _, extras = adapter.act_with_extras(np.zeros(4, dtype=np.float32))
        assert "value" in extras
        assert isinstance(extras["value"], float)

    def test_no_value_when_no_critic(self):
        actor = _make_actor()
        adapter = TorchPolicyAdapter(actor=actor, deterministic=True)
        _, extras = adapter.act_with_extras(np.zeros(4, dtype=np.float32))
        assert "value" not in extras


class TestHotReload:
    def test_load_state_dict_changes_outputs(self):
        actor = _make_actor()  # seeds torch to 0 internally
        adapter = TorchPolicyAdapter(actor=actor, deterministic=True)
        obs = np.ones(4, dtype=np.float32)
        before = adapter.act(obs)

        # Build a clearly different actor — bypass _make_actor() because
        # that helper always reseeds torch to 0 (would yield identical
        # weights and the test would silently pass on a no-op).
        torch.manual_seed(123)
        other = TanhGaussianMLPPolicy(obs_dim=4, action_dim=2, hidden_dim=8)
        adapter.load_state_dict(actor_state_dict=other.state_dict())

        after = adapter.act(obs)
        assert not np.allclose(before, after), "Hot-reload had no observable effect."

    def test_load_critic_state_dict_requires_critic(self):
        adapter = TorchPolicyAdapter(actor=_make_actor(), deterministic=True)
        with pytest.raises(ValueError, match="no critic"):
            adapter.load_state_dict(critic_state_dict={})


class TestSetDeterministic:
    def test_runtime_toggle(self):
        actor = _make_actor()
        adapter = TorchPolicyAdapter(actor=actor, deterministic=False)
        assert adapter.deterministic is False
        adapter.set_deterministic(True)
        assert adapter.deterministic is True


class TestExport:
    def test_export_writes_policy_dir(self, tmp_path: Path):
        actor = _make_actor(obs_dim=4, action_dim=2, hidden_dim=8)
        adapter = TorchPolicyAdapter(actor=actor, deterministic=True)
        out_dir = tmp_path / "exported"
        adapter.export(out_dir, extra_payload={"note": "unit-test"})
        assert (out_dir / "model.pt").exists()
        assert (out_dir / "policy.py").exists()
        payload = torch.load(out_dir / "model.pt", map_location="cpu")
        assert payload["obs_dim"] == 4
        assert payload["action_dim"] == 2
        assert payload["hidden_dim"] == 8
        assert payload["note"] == "unit-test"
