"""MixtureTruncatedNormalPolicy — mixture of truncated normals on [-1, 1].

A mixture of K diagonal truncated-normal components defined directly on
the action space:

    p(a|s) = Σ_k π_k(s) · Π_d TN(a_d; μ_kd(s), σ_kd(s), [-1,1])

A single component index k is drawn per step and generates the whole
action vector (shared component index — NOT per-dimension mixing).

- explore_factor scales only component σ: σ_eff = σ × exp(ei·ln3);
  mixture weights and means are untouched (ei=-1 is not deterministic).
- log_prob sums dims inside each component, then logsumexp over
  components — the full mixture density, never the selected component's
  conditional density.
- uncertainty U is the normalized marginal Rényi-2 effective width
  U_d = 1/(2·∫p_d²), averaged over dims — closed-form via pairwise
  component overlap integrals, no sampling or grid search.
- act() returns the highest-weight component's mean vector (no
  weighted averaging of modes).

See DESIGN_mixture_truncated_normal.md for the full specification.
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

from baseline.framework.ppo.policies.truncated_normal_mlp import (
    _ACTION_HIGH,
    _ACTION_LOW,
    _EXPLORE_K,
    _LOG_STD_SAFE_MAX,
    _LOG_STD_SAFE_MIN,
    _SQRT_2,
)

__all__ = [
    "MixtureTruncatedNormalPolicy",
]

_LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)
# erfinv domain guard: q must stay strictly inside (-1, 1).
_ERFINV_EPS = 1e-7
# Sampling boundary guard — same convention as the other policies.
_ACTION_EPS = 1e-6


def _build_export_policy_code() -> str:
    """Return the source of the ``policy.py`` embedded in export dirs.

    The export is self-contained — it reads from a real template file
    (``_export_template_mixture_truncnorm.py``) that has no imports from
    ``baseline.*`` or ``envs.*``.  See TruncatedNormalPolicy.to_blueprint
    for the full rationale (P0-6).
    """
    template_path = (
        Path(__file__).resolve().parent
        / "_export_template_mixture_truncnorm.py"
    )
    return template_path.read_text(encoding="utf-8")


class MixtureTruncatedNormalPolicy(nn.Module, TrainablePolicy, Policy):
    """Mixture of K diagonal truncated-normal components on [-1, 1].

    Head layout (single Linear, component-major):
        logits        (K,)
        raw_mean      (K·D,) — view (K, D), mean = tanh(raw_mean)
        raw_log_std   (K·D,) — view (K, D), σ = exp(clamp(raw, ±20))

    σ carries no business bounds — only the same ±20 numerical safety
    clamp as the other truncated-normal policies.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_components: int = 3,
        component_init_noise: float = 0.02,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if int(num_components) < 1:
            raise ValueError(
                f"num_components must be >= 1, got {num_components}"
            )
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_components = int(num_components)
        self.component_init_noise = float(component_init_noise)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        K, D = self.num_components, self.action_dim
        self.head = nn.Linear(hidden_dim, K + 2 * K * D)
        self._init_head()

        self.to(torch.device(device))

        # Per-policy RNG — same contract as TruncatedNormalPolicy:
        # per-episode ``reset(seed)`` reseeds this instance's own stream.
        self._gen = torch.Generator(device=self.device)
        self._last_seed: Optional[int] = None

    def _ensure_gen(self, device: torch.device) -> torch.Generator:
        """Return the policy RNG, migrating it if the device changed."""
        if self._gen.device != device:
            last = self._last_seed
            self._gen = torch.Generator(device=device)
            if last is not None:
                self._gen.manual_seed(last)
        return self._gen

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed this policy's private RNG with the per-episode seed."""
        self._last_seed = None if seed is None else int(seed)
        if seed is not None:
            self._ensure_gen(self.device).manual_seed(int(seed))

    def _init_head(self) -> None:
        """Init: uniform weights, σ ≡ e⁻¹, broken component symmetry.

        - logits rows → w=0, b=0 → π_k = 1/K.
        - mean block → shared base map (component 0's default init) plus
          small independent per-component noise of std
          ``component_init_noise`` — components start near-identical but
          not identical, so symmetric-gradient deadlock is avoided
          without a large initial multimodality.
        - log_std block → w=0, b=-1 → σ ≡ e⁻¹ for every component and
          dim, matching the other policies' init.
        """
        K, D, H = self.num_components, self.action_dim, self.hidden_dim
        with torch.no_grad():
            self.head.weight[:K].zero_()
            self.head.bias[:K].zero_()
            if K > 1:
                mw = self.head.weight[K:K + K * D].view(K, D, H)
                mb = self.head.bias[K:K + K * D].view(K, D)
                base_w = mw[0].clone()
                base_b = mb[0].clone()
                mw.copy_(
                    base_w.unsqueeze(0)
                    + torch.randn_like(mw) * self.component_init_noise
                )
                mb.copy_(
                    base_b.unsqueeze(0)
                    + torch.randn_like(mb) * self.component_init_noise
                )
            self.head.weight[K + K * D:].zero_()
            self.head.bias[K + K * D:].fill_(-1.0)

    @property
    def device(self) -> torch.device:
        """Runtime device of this policy's parameters (P0-4)."""
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def _head_forward(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One trunk+head pass → (log_pi, mean, policy_sigma).

        Shapes: log_pi (B,K), mean (B,K,D), policy_sigma (B,K,D).
        ``policy_sigma`` is unscaled — explore_factor is applied on top
        by :meth:`_effective_sigma`.
        """
        K, D = self.num_components, self.action_dim
        out = self.head(self.trunk(obs))
        logits = out[..., :K]
        raw_mean = out[..., K:K + K * D].reshape(-1, K, D)
        raw_log_std = out[..., K + K * D:].reshape(-1, K, D)
        log_pi = torch.log_softmax(logits, dim=-1)
        mean = torch.tanh(raw_mean)
        sigma = torch.clamp(
            raw_log_std, _LOG_STD_SAFE_MIN, _LOG_STD_SAFE_MAX,
        ).exp()
        return log_pi, mean, sigma

    def _explore_scale(self, explore_factor: Any = 0.0) -> Any:
        """scale = exp(ei·ln3): ei=0→1, +1→3, -1→1/3."""
        if isinstance(explore_factor, torch.Tensor):
            return torch.exp(explore_factor * _EXPLORE_K)
        return math.exp(float(explore_factor) * _EXPLORE_K)

    def _effective_sigma(
        self, policy_sigma: torch.Tensor, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        """σ for sampling / log_prob (includes explore scale)."""
        scale = self._explore_scale(explore_factor)
        if isinstance(scale, torch.Tensor):
            return policy_sigma * scale.view(-1, 1, 1)  # (B,K,D)
        return policy_sigma * scale

    @staticmethod
    def _log_trunc_Z(
        mean: torch.Tensor, sigma: torch.Tensor,
    ) -> torch.Tensor:
        """log Z of the truncation normalization, erf-sum form.

        Z = Φ(b) − Φ(a) = 0.5·[erf((1−μ)/(√2σ)) + erf((1+μ)/(√2σ))].

        For μ ∈ [-1, 1] both erf arguments are ≥ 0, so Z is a sum of two
        non-negative numbers — no cancellation at large σ (unlike
        subtracting two CDF values that both approach 1).
        """
        erf_hi = torch.erf((_ACTION_HIGH - mean) / (_SQRT_2 * sigma))
        erf_lo = torch.erf((_ACTION_HIGH + mean) / (_SQRT_2 * sigma))
        Z = 0.5 * (erf_hi + erf_lo)
        # Legit Z can be far below 1e-8 (e.g. μ at boundary, σ huge) —
        # clamp only at float-tiny as a pure underflow net.
        Z = torch.clamp_min(Z, torch.finfo(Z.dtype).tiny)
        return torch.log(Z)

    def _mixture_log_prob(
        self,
        actions: torch.Tensor,
        mean: torch.Tensor,
        sigma: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        """log p(a) = logsumexp_k(log π_k + Σ_d log t_kd(a_d)) → (B,)."""
        a = actions.unsqueeze(1)  # (B,1,D)
        log_Z = self._log_trunc_Z(mean, sigma)  # (B,K,D)
        z = (a - mean) / sigma
        comp_lp = (
            -0.5 * z * z - torch.log(sigma) - _LOG_2PI_HALF - log_Z
        ).sum(dim=-1)  # (B,K)
        return torch.logsumexp(comp_lp + log_pi, dim=-1)  # (B,)

    # ------------------------------------------------------------------
    # Uncertainty: normalized marginal Rényi-2 effective width
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_log_overlap(
        mean: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """log I_ij = log ∫ t_i(x)·t_j(x) dx for all pairs → (B,K,K,D).

        Gaussian product identity with truncation:

            I_ij = C_ij · Z_* / (Z_i · Z_j)
            C_ij = exp(-(μ_i−μ_j)²/(2(v_i+v_j))) / √(2π(v_i+v_j))
            v_*  = v_i v_j / (v_i+v_j),  m_* = (μ_i v_j + μ_j v_i)/(v_i+v_j)
            Z_*  = Z(m_*, √v_*)
        """
        mu_i = mean.unsqueeze(2)              # (B,K,1,D)
        mu_j = mean.unsqueeze(1)              # (B,1,K,D)
        v_i = sigma.pow(2).unsqueeze(2)       # (B,K,1,D)
        v_j = sigma.pow(2).unsqueeze(1)       # (B,1,K,D)

        v_sum = v_i + v_j                     # (B,K,K,D)
        v_star = v_i * v_j / v_sum
        m_star = (mu_i * v_j + mu_j * v_i) / v_sum

        log_C = (
            -0.5 * (mu_i - mu_j).pow(2) / v_sum
            - _LOG_2PI_HALF - 0.5 * torch.log(v_sum)
        )
        log_Z_star = MixtureTruncatedNormalPolicy._log_trunc_Z(
            m_star, v_star.sqrt(),
        )
        log_Z_i = MixtureTruncatedNormalPolicy._log_trunc_Z(mu_i, v_i.sqrt())
        log_Z_j = MixtureTruncatedNormalPolicy._log_trunc_Z(mu_j, v_j.sqrt())
        return log_C + log_Z_star - log_Z_i - log_Z_j  # (B,K,K,D)

    def _marginal_uncertainty(
        self,
        log_pi: torch.Tensor,
        mean: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """U_d = 1/(2·∫p_d²), mean over dims → (B,).

        R_d = Σ_ij π_i π_j I_ij computed in log space via logsumexp.
        U ∈ (0,1] by Cauchy–Schwarz; the final clamp only absorbs
        sub-ulp rounding.
        """
        log_I = self._pairwise_log_overlap(mean, sigma)  # (B,K,K,D)
        lp_i = log_pi[:, :, None, None]       # (B,K,1,1)
        lp_j = log_pi[:, None, :, None]       # (B,1,K,1)
        log_terms = lp_i + lp_j + log_I       # (B,K,K,D)
        B = mean.shape[0]
        log_R = torch.logsumexp(
            log_terms.reshape(B, -1, mean.shape[-1]), dim=1,
        )                                     # (B,D)
        U_d = torch.exp(-math.log(2.0) - log_R)
        return U_d.mean(dim=-1).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_action(
        self, obs: torch.Tensor, *, explore_factor: Any = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick one component (shared across dims), inverse-CDF sample.

        Returns (action, mixture log_prob summed over dims).
        """
        log_pi, mean, policy_sigma = self._head_forward(obs)
        sigma = self._effective_sigma(policy_sigma, explore_factor)
        B, K, D = mean.shape

        # Component selection — one index for the whole action vector.
        if K == 1:
            # Skip the RNG draw: the mixture is degenerate, and keeping
            # the stream identical to the single-component policies makes
            # the seeded degenerate-equivalence test exact.
            idx = torch.zeros(B, dtype=torch.long, device=mean.device)
        else:
            idx = torch.multinomial(
                log_pi.exp(), 1, generator=self._ensure_gen(mean.device),
            ).squeeze(-1)
        sel = idx.view(-1, 1, 1).expand(-1, 1, D)
        mu_k = mean.gather(1, sel).squeeze(1)      # (B,D)
        sg_k = sigma.gather(1, sel).squeeze(1)     # (B,D)

        # Inverse-CDF in erf space (no cancellation at large σ):
        #   A = erf(α/√2), B = erf(β/√2), q = (1-u)A + uB ∈ (A,B)
        #   a = μ + √2·σ·erfinv(q)
        erf_a = torch.erf((_ACTION_LOW - mu_k) / (_SQRT_2 * sg_k))
        erf_b = torch.erf((_ACTION_HIGH - mu_k) / (_SQRT_2 * sg_k))
        u = torch.rand(
            mu_k.shape,
            generator=self._ensure_gen(mu_k.device),
            device=mu_k.device,
            dtype=mu_k.dtype,
        )
        q = (1.0 - u) * erf_a + u * erf_b
        q = q.clamp(-1.0 + _ERFINV_EPS, 1.0 - _ERFINV_EPS)
        action = mu_k + _SQRT_2 * sg_k * torch.erfinv(q)
        action = torch.clamp(
            action, _ACTION_LOW + _ACTION_EPS, _ACTION_HIGH - _ACTION_EPS,
        )

        log_prob = self._mixture_log_prob(action, mean, sigma, log_pi)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Highest-weight component's mean vector (argmax on ties →
        smallest index, matching torch.argmax semantics)."""
        log_pi, mean, _ = self._head_forward(obs)
        idx = log_pi.argmax(dim=-1)  # (B,)
        sel = idx.view(-1, 1, 1).expand(-1, 1, self.action_dim)
        return mean.gather(1, sel).squeeze(1)

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
        """Score actions under the full mixture and compute U.

        log_prob uses effective σ (with explore scale) so the PPO
        importance ratio is correct; U uses policy σ (unscaled) and is
        action-independent.
        """
        log_pi, mean, policy_sigma = self._head_forward(obs)
        eff_sigma = self._effective_sigma(policy_sigma, explore_factor)

        actions_c = torch.clamp(
            actions, _ACTION_LOW + _ACTION_EPS, _ACTION_HIGH - _ACTION_EPS,
        )
        log_prob = self._mixture_log_prob(actions_c, mean, eff_sigma, log_pi)
        uncertainty = self._marginal_uncertainty(log_pi, mean, policy_sigma)

        stats: Optional[Dict[str, float]] = None
        if want_stats:
            stats = self._build_stats(
                log_pi, mean, policy_sigma, eff_sigma, uncertainty,
            )

        return ActorEval(
            log_prob=log_prob,
            uncertainty=uncertainty,
            stats=stats,
        )

    def _build_stats(
        self,
        log_pi: torch.Tensor,
        mean: torch.Tensor,
        policy_sigma: torch.Tensor,
        eff_sigma: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, float]:
        """Whole-batch diagnostics — called only under want_stats.

        Definitions per DESIGN_mixture_truncated_normal.md §11.
        """
        K = self.num_components
        with torch.no_grad():
            pi = log_pi.exp()                                # (B,K)
            # π-weighted σ per dim — the marginal scale seen by actions.
            w_sigma = (pi.unsqueeze(-1) * policy_sigma).sum(1)  # (B,D)
            w_sigma_eff = (pi.unsqueeze(-1) * eff_sigma).sum(1)
            weight_ent = -(pi * log_pi).sum(-1)               # (B,)
            stats: Dict[str, float] = {
                "uncertainty": float(uncertainty.mean().item()),
                "std_mean": float(w_sigma.mean().item()),
                "eff_std_mean": float(w_sigma_eff.mean().item()),
                # Extremes over all (b,k,d) — includes low-weight
                # components; not a coverage measure.
                "std_min": float(policy_sigma.min().item()),
                "std_max": float(policy_sigma.max().item()),
                "mixture_weight_entropy": float(weight_ent.mean().item()),
                "effective_components": float(
                    weight_ent.exp().mean().item()
                ),
                "max_component_weight": float(
                    pi.max(dim=-1).values.mean().item()
                ),
                # Per-(k,d) std over the batch axis, averaged — measures
                # state-dependence of σ, not spread across components.
                "sigma_state_std": float(
                    policy_sigma.std(dim=0, correction=0).mean().item()
                ),
            }
            for k in range(K):
                stats[f"component_weight_{k}"] = float(pi[:, k].mean().item())
            if K > 1:
                stats["component_overlap"] = self._overlap_stat(
                    mean, policy_sigma,
                )
        return stats

    def _overlap_stat(
        self, mean: torch.Tensor, sigma: torch.Tensor,
    ) -> float:
        """Mean normalized pairwise overlap I_ij/√(I_ii·I_jj), i<j.

        ∈ [0, 1] by Cauchy–Schwarz on densities; 1 = identical component
        marginals.  Only called under want_stats (no_grad).
        """
        K = self.num_components
        mu_i = mean.unsqueeze(2)
        mu_j = mean.unsqueeze(1)
        v_i = sigma.pow(2).unsqueeze(2)
        v_j = sigma.pow(2).unsqueeze(1)
        v_sum = v_i + v_j
        v_star = v_i * v_j / v_sum
        m_star = (mu_i * v_j + mu_j * v_i) / v_sum
        log_C = (
            -0.5 * (mu_i - mu_j).pow(2) / v_sum
            - _LOG_2PI_HALF - 0.5 * torch.log(v_sum)
        )
        log_Z_star = self._log_trunc_Z(m_star, v_star.sqrt())
        log_Z_i = self._log_trunc_Z(mu_i, v_i.sqrt())
        log_Z_j = self._log_trunc_Z(mu_j, v_j.sqrt())
        I = torch.exp(log_C + log_Z_star - log_Z_i - log_Z_j)  # (B,K,K,D)

        # Diagonal I_ii — direct closed form:
        # C_ii = 1/(2σ√π), m*=μ, v*=v/2 → Z(μ, σ/√2)
        log_I_ii = (
            -torch.log(sigma) - 0.5 * math.log(4.0 * math.pi)
            + self._log_trunc_Z(mean, sigma / _SQRT_2)
            - 2.0 * self._log_trunc_Z(mean, sigma)
        )
        I_ii = log_I_ii.exp()                                 # (B,K,D)

        i_idx, j_idx = torch.triu_indices(
            K, K, offset=1, device=mean.device,
        )
        denom = (I_ii[:, i_idx] * I_ii[:, j_idx]).sqrt().clamp_min(
            torch.finfo(I_ii.dtype).tiny,
        )
        overlap = I[:, i_idx, j_idx, :] / denom               # (B,P,D)
        return float(overlap.mean().item())

    # ------------------------------------------------------------------
    # Policy contract (deterministic default behaviour)
    # ------------------------------------------------------------------

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Deterministic action — mean of the highest-weight component."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(
            obs_array, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    # ------------------------------------------------------------------
    # StochasticPolicy contract (sampling with exploration control)
    # ------------------------------------------------------------------

    def sample(
        self,
        observation: Any,
        *,
        explore_factor: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Stochastic action — mixture sample with explore_factor."""
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
        """Export to a deployable, self-contained PolicyBlueprint.

        Writes ``model.pt`` + ``policy.py`` (self-contained, no imports
        from ``baseline.*``) + ``MANIFEST.json`` into ``dest_path`` and
        returns a blueprint that rebuilds the policy via the
        ``ExportedMixtureTruncNormPolicy`` class.
        """
        if dest_path is None:
            dest_path = tempfile.mkdtemp(prefix="policy_export_")
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {
            k: v.detach().cpu() for k, v in self.state_dict().items()
        }
        payload = {
            "format_version": 1,
            "policy_class": "MixtureTruncatedNormalPolicy",
            "distribution_kind": "mixture_truncated_normal_v1",
            "std_source": "state",
            "std_parameterization": "log_std_v1",
            "uncertainty_kind": "marginal_renyi2_width_v1",
            "exploration_kind": "log_std_multiplicative_v1",
            "arch": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_components": self.num_components,
            },
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_components": self.num_components,
            "state_dict": state_dict,
            "state_dict_keys": sorted(state_dict.keys()),
        }
        torch.save(payload, policy_dir / "model.pt")

        policy_code = _build_export_policy_code()
        (policy_dir / "policy.py").write_text(policy_code, encoding="utf-8")

        manifest = {
            "format_version": 1,
            "policy_class": "MixtureTruncatedNormalPolicy",
            "distribution_kind": "mixture_truncated_normal_v1",
            "std_source": "state",
            "std_parameterization": "log_std_v1",
            "exploration_kind": "log_std_multiplicative_v1",
            "arch": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_components": self.num_components,
            },
            "uncertainty_kind": "marginal_renyi2_width_v1",
            "files": ["model.pt", "policy.py", "MANIFEST.json"],
            "exported_class": "ExportedMixtureTruncNormPolicy",
        }
        (policy_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

        policy_py_path = policy_dir / "policy.py"
        return PolicyBlueprint(
            cls=f"file:{policy_py_path}:ExportedMixtureTruncNormPolicy",
        )
