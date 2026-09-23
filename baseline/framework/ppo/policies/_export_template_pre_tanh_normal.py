"""Self-contained inference-only export template for PreTanhNormalPolicy.

Written into the export directory as ``policy.py`` by
``PreTanhNormalPolicy.to_blueprint()``.  Must not import any
``baseline.*`` or ``envs.*`` modules — it is standalone.

Implements:  z ~ N(mu, sigma_e^2), a = tanh(z), shared per-dim sigma,
coverage-radial explore mapping, float32 stored-action scoring with
the same fail-loud guards as the training-side implementation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

# ------------------------------------------------------------------
# Constants — MUST match the training-side implementation exactly.
# ------------------------------------------------------------------

_COVERAGE_LOG_STD_STAR = -0.1513403077614502   # log(sigma*)
_EXPLORE_K = math.log(3.0)

_ACTION_MARGIN = 2.0 ** -20
_ACTION_SAFE = 1.0 - _ACTION_MARGIN
_Z_SAFE = 0.5 * math.log((2.0 - _ACTION_MARGIN) / _ACTION_MARGIN)
_TAIL_RISK_BUDGET = 1e-12

_SQRT_2 = math.sqrt(2.0)
_LN_2PI_HALF = 0.5 * math.log(2.0 * math.pi)
_LN_2 = math.log(2.0)
_LN_2_SQRT_PI = math.log(2.0 * math.sqrt(math.pi))

_EI_TOLERANCE = 1e-6


def _atanh_stable(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))


def _log_jac(z_hat: torch.Tensor) -> torch.Tensor:
    az = z_hat.abs()
    return 2.0 * (_LN_2 - az - torch.nn.functional.softplus(-2.0 * az))


class _PreTanhNormalInferenceNet(nn.Module):
    """MLP + shared per-dim log_std — same param names as training side."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    # --- distribution helpers (float64) ---

    def _policy_params(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net(obs).double(), self.log_std.double()

    @staticmethod
    def _check_ei(explore_factor: Any) -> None:
        if isinstance(explore_factor, torch.Tensor):
            bad = (explore_factor < -1.0 - _EI_TOLERANCE) | (
                explore_factor > 1.0 + _EI_TOLERANCE
            )
            if bool(bad.any()):
                raise ValueError(
                    f"explore_factor out of [-1, 1]: "
                    f"min={float(explore_factor.min())}, "
                    f"max={float(explore_factor.max())}"
                )
        elif abs(float(explore_factor)) > 1.0 + _EI_TOLERANCE:
            raise ValueError(
                f"explore_factor out of [-1, 1]: {explore_factor}"
            )

    def _explored_params(
        self,
        mu: torch.Tensor,
        r: torch.Tensor,
        explore_factor: Any = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_ei(explore_factor)
        if isinstance(explore_factor, torch.Tensor):
            c = torch.exp(
                -explore_factor.double() * _EXPLORE_K
            ).unsqueeze(-1)
        else:
            c = math.exp(-float(explore_factor) * _EXPLORE_K)
        mu_e = mu * c
        r_e = _COVERAGE_LOG_STD_STAR + c * (r - _COVERAGE_LOG_STD_STAR)
        return mu_e, r_e

    @staticmethod
    def _action_uncertainty(
        mu: torch.Tensor, r: torch.Tensor,
    ) -> torch.Tensor:
        logcosh = torch.logaddexp(2.0 * mu, -2.0 * mu) - _LN_2
        inner = torch.exp(2.0 * r) + logcosh
        log_u = _LN_2_SQRT_PI + r - torch.nn.functional.softplus(inner)
        return log_u.exp().mean(dim=-1).clamp(0.0, 1.0)

    @staticmethod
    def _tail_risk(
        mu_e: torch.Tensor, sigma_e: torch.Tensor,
    ) -> torch.Tensor:
        arg_hi = (_Z_SAFE - mu_e) / (_SQRT_2 * sigma_e)
        arg_lo = (_Z_SAFE + mu_e) / (_SQRT_2 * sigma_e)
        return 0.5 * (torch.erfc(arg_hi) + torch.erfc(arg_lo))

    def _check_support(
        self, mu_e: torch.Tensor, sigma_e: torch.Tensor,
    ) -> None:
        p_tail = self._tail_risk(mu_e, sigma_e)
        worst = float(p_tail.max())
        if worst > _TAIL_RISK_BUDGET:
            raise RuntimeError(
                f"PreTanhNormalPolicy: effective distribution exceeds "
                f"the verified inversion range — max per-dim "
                f"P(|z|>{_Z_SAFE:.4f}) = {worst:.3e} > "
                f"{_TAIL_RISK_BUDGET:.0e}."
            )

    @staticmethod
    def _check_actions(actions: torch.Tensor) -> None:
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions contain non-finite values")
        worst = float(actions.abs().max())
        if worst > _ACTION_SAFE:
            raise RuntimeError(
                f"PreTanhNormalPolicy: action |a|={worst:.9f} exceeds "
                f"the verified inversion range {_ACTION_SAFE:.9f}."
            )

    def _action_log_prob(
        self,
        actions: torch.Tensor,
        mu_e: torch.Tensor,
        r_e: torch.Tensor,
    ) -> torch.Tensor:
        a = actions.double()
        z_hat = _atanh_stable(a)
        sigma_e = r_e.exp()
        z = (z_hat - mu_e) / sigma_e
        per_dim = (
            -0.5 * z * z - r_e - _LN_2PI_HALF - _log_jac(z_hat)
        )
        return per_dim.sum(dim=-1)

    # --- inference ---

    def forward(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, r = self._policy_params(obs)
        mu_e, r_e = self._explored_params(mu, r, explore_factor)
        return mu_e, r_e.exp()

    def sample_action(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, r = self._policy_params(obs)
        mu_e, r_e = self._explored_params(mu, r, explore_factor)
        sigma_e = r_e.exp()
        self._check_support(mu_e, sigma_e)

        z = mu_e + sigma_e * torch.randn_like(mu_e)
        action = torch.tanh(z).float()
        self._check_actions(action)

        log_prob = self._action_log_prob(action, mu_e, r_e)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self._policy_params(obs)
        return torch.tanh(mu).float()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        self._check_actions(actions)
        mu, r = self._policy_params(obs)
        mu_e, r_e = self._explored_params(mu, r, explore_factor)
        return self._action_log_prob(actions, mu_e, r_e)


class ExportedPreTanhNormalPolicy:
    """Inference-side policy loaded from the exported model.pt payload."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            here = Path(__file__).resolve().parent
            model_path = str(here / "model.pt")

        payload = torch.load(model_path, map_location="cpu")

        required_top = [
            "format_version", "policy_class", "arch", "state_dict",
            "distribution_kind", "uncertainty_kind", "exploration_kind",
        ]
        for key in required_top:
            if key not in payload:
                raise ValueError(
                    f"ExportedPreTanhNormalPolicy: model payload missing "
                    f"required key '{key}'"
                )
        if payload["format_version"] != 1:
            raise ValueError(
                f"ExportedPreTanhNormalPolicy: unsupported "
                f"format_version {payload['format_version']}"
            )
        if payload["policy_class"] != "PreTanhNormalPolicy":
            raise ValueError(
                f"ExportedPreTanhNormalPolicy: expected policy_class "
                f"'PreTanhNormalPolicy', got "
                f"'{payload['policy_class']}'"
            )

        for kind_key, expected in (
            ("distribution_kind", "tanh_diagonal_normal_shared_std_v1"),
            ("uncertainty_kind", "marginal_renyi2_width_v1"),
            ("exploration_kind", "coverage_radial_logscale_v1"),
        ):
            if payload[kind_key] != expected:
                raise ValueError(
                    f"ExportedPreTanhNormalPolicy: {kind_key} mismatch — "
                    f"expected '{expected}', got '{payload[kind_key]}'"
                )

        arch = payload["arch"]
        self.obs_dim = int(arch["obs_dim"])
        self.action_dim = int(arch["action_dim"])
        self.hidden_dim = int(arch["hidden_dim"])

        self._net = _PreTanhNormalInferenceNet(
            self.obs_dim, self.action_dim, self.hidden_dim,
        )
        state_dict = payload["state_dict"]
        expected_keys = set(self._net.state_dict().keys())
        if set(state_dict.keys()) != expected_keys:
            missing = expected_keys - set(state_dict.keys())
            extra = set(state_dict.keys()) - expected_keys
            raise ValueError(
                f"ExportedPreTanhNormalPolicy: state_dict keys mismatch "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )
        self._net.load_state_dict(state_dict)
        self._net.eval()

    @property
    def device(self) -> torch.device:
        return next(self._net.parameters()).device

    # --------------------------------------------------------------
    # Policy contract
    # --------------------------------------------------------------

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            action = self._net.deterministic_action(obs_tensor)
        return action.squeeze(0).numpy().astype(np.float32), None

    def sample(
        self,
        observation: Any,
        *,
        explore_factor: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self._net.sample_action(
                obs_tensor, explore_factor=explore_factor,
            )
        action_np = action.squeeze(0).numpy().astype(np.float32)
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {"log_prob": float(log_prob.item())}

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._net.evaluate_actions(
                obs, actions, explore_factor,
            )
