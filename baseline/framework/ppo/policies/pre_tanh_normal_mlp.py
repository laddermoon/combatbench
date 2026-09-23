"""PreTanhNormalPolicy — diagonal Gaussian in pre-tanh space.

The distribution lives on the latent z; the action is a = tanh(z):

    z_d ~ N(mu_d(s), sigma_d^2),    a_d = tanh(z_d)

- mu comes straight from the MLP head (no output tanh); sigma is a
  shared per-dim parameter vector (NOT state-dependent, NOT a scalar).
- explore_factor moves (mu, log sigma) radially toward the maximum
  action-coverage point theta* = (0, r*): mu_e = c*mu,
  r_e = r* + c*(r - r*), c = 3^(-e).  Increasing e is guaranteed to
  non-decrease the per-dim action-space L2 effective width (strictly
  increasing off the optimum) — it is a coverage controller, not a
  latent-noise temperature.
- ActorEval.uncertainty is the closed-form marginal Renyi-2 width
  U_d = 2*sqrt(pi)*sigma_d / (1 + exp(sigma_d^2)*cosh(2*mu_d)),
  evaluated on the *policy* parameters (no explore scaling).
- The action data path is unchanged (float32 actions only, no latent
  channel).  Sampling converts z -> tanh -> float32 once and scores
  THAT stored action; evaluate_actions inverts the stored action.
  Both paths fail loudly outside the verified inversion range instead
  of clipping or resampling.

See DESIGN_pre_tanh_normal.md for the full specification.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from envs.framework.policy import Policy, PolicyBlueprint

from baseline.framework.ppo import ActorEval, TrainablePolicy
from baseline.framework.ppo.stochastic_policy import StochasticPolicy

__all__ = [
    "PreTanhNormalPolicy",
]

# Unique maximizer of the per-dim action-space L2 width for
# a = tanh(N(mu, sigma^2)): solves 2*x*sigmoid(x) = 1, x = sigma^2.
_COVERAGE_LOG_STD_STAR = -0.1513403077614502   # r* = log(sigma*)
_COVERAGE_U_MAX = 0.9849840996324593            # U at (mu=0, sigma*)

_EXPLORE_K = math.log(3.0)      # c(e) = 3^(-e)

# Scoreable action range: |a32| <= 1 - 2^-20  (z_safe ~= 7.28).
# Actions outside are outside the verified inversion range — fail loud.
_ACTION_MARGIN = 2.0 ** -20
_ACTION_SAFE = 1.0 - _ACTION_MARGIN
_Z_SAFE = 0.5 * math.log((2.0 - _ACTION_MARGIN) / _ACTION_MARGIN)
# Per-dim sampling tail budget P(|Z| > z_safe) under the effective
# distribution.  Parameters exceeding it are unsupported, not clipped.
_TAIL_RISK_BUDGET = 1e-12

_SQRT_2 = math.sqrt(2.0)
_LN_2PI_HALF = 0.5 * math.log(2.0 * math.pi)
_LN_2 = math.log(2.0)
_LN_2_SQRT_PI = math.log(2.0 * math.sqrt(math.pi))

_EI_TOLERANCE = 1e-6            # float32 slack on the [-1, 1] ei range


def _build_export_policy_code(template_name: str) -> str:
    """Return the source of the ``policy.py`` embedded in export dirs.

    Self-contained template — no ``baseline.*`` / ``envs.*`` imports.
    See DESIGN_pre_tanh_normal.md §10.
    """
    template_path = Path(__file__).resolve().parent / template_name
    return template_path.read_text(encoding="utf-8")


def _atanh_stable(a: torch.Tensor) -> torch.Tensor:
    """atanh(a) = 0.5 * (log1p(a) - log1p(-a)), accurate near ±1."""
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))


def _log_jac(z_hat: torch.Tensor) -> torch.Tensor:
    """log(1 - tanh(z)^2) = log sech^2(z), stable for large |z|.

    log sech^2(z) = 2 * (log 2 - |z| - softplus(-2|z|)).
    """
    az = z_hat.abs()
    return 2.0 * (_LN_2 - az - torch.nn.functional.softplus(-2.0 * az))


class PreTanhNormalPolicy(nn.Module, TrainablePolicy, Policy):
    """Diagonal pre-tanh Gaussian with shared per-dim sigma.

    Export metadata lives in class attributes so subclasses (e.g. the
    state-dependent-σ variant) can reuse ``to_blueprint`` unchanged —
    they only need to point the attributes at their own payload kinds
    and export template.
    """

    _POLICY_CLASS = "PreTanhNormalPolicy"
    _DISTRIBUTION_KIND = "tanh_diagonal_normal_shared_std_v1"
    _UNCERTAINTY_KIND = "marginal_renyi2_width_v1"
    _EXPLORATION_KIND = "coverage_radial_logscale_v1"
    _EXPORTED_CLASS = "ExportedPreTanhNormalPolicy"
    _EXPORT_TEMPLATE = "_export_template_pre_tanh_normal.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ):
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
        self.log_std = nn.Parameter(
            torch.full((action_dim,), -1.0, dtype=torch.float32)
        )

        self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        """Runtime device of this policy's parameters (P0-4)."""
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Distribution helpers (all in float64 — see DESIGN §7/§8)
    # ------------------------------------------------------------------

    def _policy_params(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mu, r) both float64, shapes (B,D) and (D,)."""
        mu = self.net(obs).double()
        r = self.log_std.double()
        return mu, r

    @staticmethod
    def _check_ei(explore_factor: Any) -> None:
        """explore_factor must stay within [-1, 1] — no silent clamp."""
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
        """Radial move toward the coverage optimum in (mu, r).

        c = 3^(-e); mu_e = c*mu; r_e = r* + c*(r - r*).
        e<0 extrapolates away from the optimum.  Returns (B,D) tensors.
        """
        self._check_ei(explore_factor)
        if isinstance(explore_factor, torch.Tensor):
            c = torch.exp(
                -explore_factor.double() * _EXPLORE_K
            ).unsqueeze(-1)                      # (B,1)
        else:
            c = math.exp(-float(explore_factor) * _EXPLORE_K)
        mu_e = mu * c
        r_e = _COVERAGE_LOG_STD_STAR + c * (r - _COVERAGE_LOG_STD_STAR)
        return mu_e, r_e

    @staticmethod
    def _action_uncertainty(
        mu: torch.Tensor, r: torch.Tensor,
    ) -> torch.Tensor:
        """Per-dim L2 effective width, mean over dims -> (B,).

        U_d = 2*sqrt(pi)*sigma / (1 + exp(sigma^2)*cosh(2*mu))
        log U_d = log(2*sqrt(pi)) + r
                  - softplus(exp(2r) + logcosh(2*mu))
        """
        logcosh = torch.logaddexp(2.0 * mu, -2.0 * mu) - _LN_2
        inner = torch.exp(2.0 * r) + logcosh
        log_u = _LN_2_SQRT_PI + r - torch.nn.functional.softplus(inner)
        return log_u.exp().mean(dim=-1).clamp(0.0, 1.0)

    @staticmethod
    def _tail_risk(
        mu_e: torch.Tensor, sigma_e: torch.Tensor,
    ) -> torch.Tensor:
        """P(|Z| > z_safe) per (b, d) under N(mu_e, sigma_e^2).

        Uses erfc for accurate small tails.
        """
        arg_hi = (_Z_SAFE - mu_e) / (_SQRT_2 * sigma_e)
        arg_lo = (_Z_SAFE + mu_e) / (_SQRT_2 * sigma_e)
        return 0.5 * (torch.erfc(arg_hi) + torch.erfc(arg_lo))

    def _check_support(
        self, mu_e: torch.Tensor, sigma_e: torch.Tensor,
    ) -> None:
        """Fail if the effective distribution has too much mass outside
        the verified inversion range.  No clip, no resample."""
        p_tail = self._tail_risk(mu_e, sigma_e)
        worst = float(p_tail.max())
        if worst > _TAIL_RISK_BUDGET:
            raise RuntimeError(
                f"PreTanhNormalPolicy: effective distribution exceeds "
                f"the verified inversion range — max per-dim "
                f"P(|z|>{_Z_SAFE:.4f}) = {worst:.3e} > "
                f"{_TAIL_RISK_BUDGET:.0e}.  Parameters are outside the "
                f"supported numeric region; see DESIGN_pre_tanh_normal.md "
                f"§8."
            )

    @staticmethod
    def _check_actions(actions: torch.Tensor) -> None:
        """Stored actions must be finite and inside the scoreable range."""
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions contain non-finite values")
        worst = float(actions.abs().max())
        if worst > _ACTION_SAFE:
            raise RuntimeError(
                f"PreTanhNormalPolicy: action |a|={worst:.9f} exceeds "
                f"the verified inversion range {_ACTION_SAFE:.9f} "
                f"(margin {_ACTION_MARGIN:.3e}).  Stored actions at the "
                f"float32 boundary cannot be inverted reliably — "
                f"see DESIGN_pre_tanh_normal.md §8."
            )

    def _action_log_prob(
        self,
        actions: torch.Tensor,
        mu_e: torch.Tensor,
        r_e: torch.Tensor,
    ) -> torch.Tensor:
        """log p_A(a) = sum_d [ log q(z_hat) - log(1 - a^2) ] -> (B,)."""
        a = actions.double()
        z_hat = _atanh_stable(a)
        sigma_e = r_e.exp()
        z = (z_hat - mu_e) / sigma_e
        per_dim = (
            -0.5 * z * z - r_e - _LN_2PI_HALF - _log_jac(z_hat)
        )
        return per_dim.sum(dim=-1)

    def forward(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mu_e, sigma_e) under the effective distribution — (B,D) float64."""
        mu, r = self._policy_params(obs)
        mu_e, r_e = self._explored_params(mu, r, explore_factor)
        return mu_e, r_e.exp()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_action(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a=tanh(z), quantize to the stored float32 action,
        score THAT action.  Returns (action_f32, log_prob_f64)."""
        mu, r = self._policy_params(obs)
        mu_e, r_e = self._explored_params(mu, r, explore_factor)
        sigma_e = r_e.exp()
        self._check_support(mu_e, sigma_e)

        z = mu_e + sigma_e * torch.randn_like(mu_e)
        action = torch.tanh(z).float()          # the stored action
        self._check_actions(action)

        log_prob = self._action_log_prob(action, mu_e, r_e)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """act = tanh(mu) — the per-dim action median."""
        mu, _ = self._policy_params(obs)
        return torch.tanh(mu).float()

    # ------------------------------------------------------------------
    # Evaluation (training-side)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        explore_factor: torch.Tensor,
        *,
        want_stats: bool = False,
    ) -> ActorEval:
        """Score stored actions under the e-mapped distribution and
        compute the policy-distribution L2 U."""
        self._check_actions(actions)
        mu, r = self._policy_params(obs)
        mu_e, r_e = self._explored_params(mu, r, explore_factor)

        log_prob = self._action_log_prob(actions, mu_e, r_e)
        if not bool(torch.isfinite(log_prob).all()):
            raise RuntimeError(
                "PreTanhNormalPolicy: non-finite log_prob — parameters "
                "or inputs are outside the supported numeric region"
            )
        uncertainty = self._action_uncertainty(mu, r)

        stats: Optional[Dict[str, float]] = None
        if want_stats:
            stats = self._build_stats(
                mu, r, mu_e, r_e, uncertainty,
            )

        return ActorEval(
            log_prob=log_prob,
            uncertainty=uncertainty,
            stats=stats,
        )

    def _build_stats(
        self,
        mu: torch.Tensor,
        r: torch.Tensor,
        mu_e: torch.Tensor,
        r_e: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, float]:
        """Whole-batch diagnostics — only under want_stats."""
        with torch.no_grad():
            sigma = r.exp()                                  # (D,)
            sigma_e = r_e.exp()                              # (B,D)
            u_eff = self._action_uncertainty(mu_e, r_e)
            coverage_distance = torch.sqrt(
                mu * mu + (r - _COVERAGE_LOG_STD_STAR) ** 2
            )
            z99 = 0.5 * math.log(199.0)      # atanh(0.99)
            p_near = 0.5 * (
                torch.erfc((z99 - mu) / (_SQRT_2 * sigma))
                + torch.erfc((z99 + mu) / (_SQRT_2 * sigma))
            )
            p_near_e = 0.5 * (
                torch.erfc((z99 - mu_e) / (_SQRT_2 * sigma_e))
                + torch.erfc((z99 + mu_e) / (_SQRT_2 * sigma_e))
            )
            return {
                "uncertainty": float(uncertainty.mean().item()),
                "effective_uncertainty": float(u_eff.mean().item()),
                "latent_std_mean": float(sigma.mean().item()),
                "latent_std_min": float(sigma.min().item()),
                "latent_std_max": float(sigma.max().item()),
                "effective_latent_std_mean": float(sigma_e.mean().item()),
                "effective_latent_std_min": float(sigma_e.min().item()),
                "effective_latent_std_max": float(sigma_e.max().item()),
                "latent_mean_abs": float(mu.abs().mean().item()),
                "effective_latent_mean_abs": float(
                    mu_e.abs().mean().item()
                ),
                "coverage_distance": float(coverage_distance.mean().item()),
                "near_boundary_probability": float(p_near.mean().item()),
                "effective_near_boundary_probability": float(
                    p_near_e.mean().item()
                ),
                "effective_unsafe_tail_max": float(
                    self._tail_risk(mu_e, sigma_e).max().item()
                ),
            }

    # ------------------------------------------------------------------
    # Policy contract
    # ------------------------------------------------------------------

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Deterministic action — tanh(mu), the per-dim median."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(
            obs_array, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    # ------------------------------------------------------------------
    # StochasticPolicy contract
    # ------------------------------------------------------------------

    def sample(
        self,
        observation: Any,
        *,
        explore_factor: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Stochastic action — coverage-mapped pre-tanh sample."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(
            obs_array, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.sample_action(
                obs_tensor, explore_factor=explore_factor,
            )
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {
            "log_prob": float(log_prob.item()),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_blueprint(
        self, dest_path: Optional[str] = None,
    ) -> "PolicyBlueprint":
        """Export a deployable, self-contained PolicyBlueprint."""
        if dest_path is None:
            dest_path = tempfile.mkdtemp(prefix="policy_export_")
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {
            k: v.detach().cpu() for k, v in self.state_dict().items()
        }
        payload = {
            "format_version": 1,
            "policy_class": self._POLICY_CLASS,
            "arch": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
            },
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "distribution_kind": self._DISTRIBUTION_KIND,
            "uncertainty_kind": self._UNCERTAINTY_KIND,
            "exploration_kind": self._EXPLORATION_KIND,
            "state_dict": state_dict,
            "state_dict_keys": sorted(state_dict.keys()),
        }
        torch.save(payload, policy_dir / "model.pt")

        policy_code = _build_export_policy_code(self._EXPORT_TEMPLATE)
        (policy_dir / "policy.py").write_text(policy_code, encoding="utf-8")

        manifest = {
            "format_version": 1,
            "policy_class": self._POLICY_CLASS,
            "arch": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
            },
            "distribution_kind": self._DISTRIBUTION_KIND,
            "uncertainty_kind": self._UNCERTAINTY_KIND,
            "exploration_kind": self._EXPLORATION_KIND,
            "files": ["model.pt", "policy.py", "MANIFEST.json"],
            "exported_class": self._EXPORTED_CLASS,
        }
        (policy_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

        policy_py_path = policy_dir / "policy.py"
        return PolicyBlueprint(
            cls=f"file:{policy_py_path}:{self._EXPORTED_CLASS}",
        )
