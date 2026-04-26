"""Tests for ``ppo_loss``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.common.algos import PPOLossOutput, ppo_loss


def _make_inputs(b: int = 8, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        log_probs_old=torch.randn(b, generator=g),
        log_probs_new=torch.randn(b, generator=g),
        advantages=torch.randn(b, generator=g),
        values_old=torch.randn(b, generator=g),
        values_new=torch.randn(b, generator=g),
        returns=torch.randn(b, generator=g),
        entropy=torch.rand(b, generator=g),
    )


class TestShapesAndOutputs:
    def test_returns_PPOLossOutput_with_expected_fields(self):
        out = ppo_loss(**_make_inputs())
        assert isinstance(out, PPOLossOutput)
        # All scalar.
        for name in ("loss", "policy_loss", "value_loss", "entropy",
                     "approx_kl", "clip_fraction", "explained_variance"):
            assert getattr(out, name).dim() == 0, name

    def test_loss_carries_grad(self):
        x = _make_inputs()
        x["log_probs_new"] = x["log_probs_new"].clone().requires_grad_(True)
        out = ppo_loss(**x)
        out.loss.backward()
        assert x["log_probs_new"].grad is not None

    def test_entropy_none_zeros_term(self):
        x = _make_inputs()
        x["entropy"] = None
        out = ppo_loss(**x, entropy_coef=10.0)
        assert out.entropy.item() == 0.0


class TestShapeValidation:
    def test_shape_mismatch_raises(self):
        x = _make_inputs(b=8)
        x["advantages"] = torch.randn(7)
        with pytest.raises(ValueError, match="Shape mismatch"):
            ppo_loss(**x)

    def test_2d_inputs_rejected(self):
        x = _make_inputs(b=8)
        x["log_probs_new"] = x["log_probs_new"].unsqueeze(-1)
        x["log_probs_old"] = x["log_probs_old"].unsqueeze(-1)
        x["advantages"] = x["advantages"].unsqueeze(-1)
        x["values_old"] = x["values_old"].unsqueeze(-1)
        x["values_new"] = x["values_new"].unsqueeze(-1)
        x["returns"] = x["returns"].unsqueeze(-1)
        x["entropy"] = x["entropy"].unsqueeze(-1)
        with pytest.raises(ValueError, match="1-D"):
            ppo_loss(**x)


class TestClippingBehavior:
    def test_clip_fraction_zero_when_ratio_within_range(self):
        # log_probs identical → ratio=1 → no clipping.
        x = _make_inputs()
        x["log_probs_new"] = x["log_probs_old"].clone()
        out = ppo_loss(**x, clip_range=0.2)
        assert out.clip_fraction.item() == 0.0

    def test_clip_fraction_one_when_ratio_far_from_one(self):
        # Force log_probs_new = log_probs_old + 5 → ratio ~ 148 → all clipped.
        x = _make_inputs()
        x["log_probs_new"] = x["log_probs_old"] + 5.0
        out = ppo_loss(**x, clip_range=0.2)
        assert out.clip_fraction.item() == 1.0

    def test_value_clip_none_skips_value_clipping(self):
        # When value_clip is None, value loss is just MSE / 2.
        x = _make_inputs()
        x["values_old"] = x["values_new"].clone()  # zero clip-delta
        with_clip = ppo_loss(**x, value_clip=0.2)
        no_clip = ppo_loss(**x, value_clip=None)
        # When values_old == values_new, both forms yield same value loss.
        assert with_clip.value_loss.item() == pytest.approx(
            no_clip.value_loss.item(), abs=1e-5,
        )


class TestApproxKL:
    def test_kl_zero_when_policies_identical(self):
        x = _make_inputs()
        x["log_probs_new"] = x["log_probs_old"].clone()
        out = ppo_loss(**x)
        assert abs(out.approx_kl.item()) < 1e-6

    def test_kl_positive_when_policies_differ(self):
        x = _make_inputs()
        x["log_probs_new"] = x["log_probs_old"] + 0.5
        out = ppo_loss(**x)
        assert out.approx_kl.item() > 0


class TestExplainedVariance:
    def test_perfect_value_estimate_yields_one(self):
        x = _make_inputs()
        x["values_new"] = x["returns"].clone()
        out = ppo_loss(**x, value_clip=None)
        assert out.explained_variance.item() == pytest.approx(1.0, abs=1e-5)

    def test_constant_returns_yields_nan(self):
        # When var(returns) == 0, EV is undefined.
        x = _make_inputs()
        x["returns"] = torch.zeros_like(x["returns"])
        out = ppo_loss(**x, value_clip=None)
        assert torch.isnan(out.explained_variance)


class TestNormalizeAdvantages:
    def test_off_keeps_raw_advantages(self):
        x = _make_inputs()
        x["advantages"] = torch.full((8,), 5.0)
        # log_probs identical → ratio=1 → policy_loss = -mean(adv) = -5
        x["log_probs_new"] = x["log_probs_old"].clone()
        out = ppo_loss(**x, normalize_advantages=False, value_coef=0.0,
                       entropy_coef=0.0)
        assert out.policy_loss.item() == pytest.approx(-5.0, abs=1e-5)

    def test_on_centers_advantages(self):
        # When normalized, all-equal advantages reduce to zeros (since
        # std=0 + eps), so policy loss should be near zero.
        x = _make_inputs()
        x["advantages"] = torch.full((8,), 5.0)
        x["log_probs_new"] = x["log_probs_old"].clone()
        out = ppo_loss(**x, normalize_advantages=True, value_coef=0.0,
                       entropy_coef=0.0)
        assert abs(out.policy_loss.item()) < 1e-3
