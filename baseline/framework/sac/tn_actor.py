"""SAC-native eight-cell truncated-normal actor family (D09/D41).

One configurable implementation covers the 2×2×2 grid:

    arch = f"{family}{sigma_source}{sigma_bound}"
    family:       's' = single component, 'm' = mixture of K components
    sigma_source: '0' = shared parameter, '1' = state-dependent head
    sigma_bound:  '0' = unbounded exp(log σ), '1' = bounded sigmoid-v map

All cells share the N-SAC-01 kernel (:mod:`tn_kernel`): the policy is a
diagonal truncated Normal *directly on the action cube* (-1, 1)^D —
no tanh squash.  μ = tanh(mean head) is the TN centre, σ is the raw
Normal scale parameter.

Contract (A4.3):
    distribution(obs)            → {mu[B,K,D], sigma[B,K,D], logits[B,K]}
    expectation_samples(obs, u)  → actions[B,K,M,D], log_prob[B,K,M],
                                   integration_weights[B,K,M]
    sample_behavior(obs, spec)   → β action + extras (e mapping applied)
    deterministic_action(obs)    → μ (single) or argmax-p_k component μ
    uncertainty(obs, kind)       → mean over dims of 1/(2·width) measure

RNG: one private torch.Generator, saved/restored in state_payload; the
collection/eval/training streams must not share it.
"""
from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from envs.framework.policy import PolicyBlueprint

from . import tn_kernel as K
from .collection import SACBehaviorSpec

# ---------------------------------------------------------------------------
# Architecture registry (SAC-R1-D41)
# ---------------------------------------------------------------------------

# arch → (is_mixture, sigma_source, sigma_bounded)
ARCH_SPECS: Dict[str, Tuple[bool, str, bool]] = {
    "s00": (False, "shared", False),
    "s01": (False, "shared", True),
    "s10": (False, "state", False),
    "s11": (False, "state", True),
    "m00": (True, "shared", False),
    "m01": (True, "shared", True),
    "m10": (True, "state", False),
    "m11": (True, "state", True),
}

# All eight cells are enabled (mixture path verified in P4-MIX-1).
_ENABLED_ARCHS = frozenset(ARCH_SPECS)

SIGMA_MIN = 0.05
SIGMA_MAX = 2.0
INIT_STD = math.exp(-1.0)
_LOG_STD_CLAMP = 20.0  # numerical safety clamp for unbounded log σ
_EI_TOL = 1e-6


def _bounded_geometry(
    sigma_min: float, sigma_max: float, init_std: float,
) -> Tuple[float, float, float, float]:
    """(r_min, delta_r, p0, v_init) for the bounded sigmoid σ map."""
    r_min = math.log(sigma_min)
    r_max = math.log(sigma_max)
    delta_r = r_max - r_min
    p0 = (math.log(init_std) - r_min) / delta_r
    v_init = math.log(p0) - math.log1p(-p0)
    return r_min, delta_r, p0, v_init


def _default_explore_alpha(delta_r: float, p0: float) -> float:
    """α so d(log σ)/de at init equals ln 3 (the unbounded 3^e slope)."""
    return math.log(3.0) / (delta_r * p0 * (1.0 - p0))


class TNActor(nn.Module):
    """Eight-cell TN actor; ``arch`` selects the cell (D41)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        arch: str = "s01",
        hidden_dim: int = 256,
        n_components: int = 3,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
        init_std: float = INIT_STD,
        explore_alpha: Optional[float] = None,
        component_init_noise: float = 0.02,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if arch not in ARCH_SPECS:
            raise ValueError(
                f"unknown SAC actor arch {arch!r}; allowed={sorted(ARCH_SPECS)}"
            )
        if arch not in _ENABLED_ARCHS:
            raise ValueError(
                f"SAC actor arch {arch!r} is not enabled; "
                f"enabled={sorted(_ENABLED_ARCHS)}"
            )
        self.is_mixture, self.sigma_source, self.sigma_bounded = ARCH_SPECS[arch]
        self.policy_arch = f"tn_{arch}"
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_components = int(n_components) if self.is_mixture else 1
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.init_std = float(init_std)
        self.component_init_noise = float(component_init_noise)
        if not (0.0 < self.sigma_min < self.init_std < self.sigma_max):
            raise ValueError(
                "require 0 < sigma_min < init_std < sigma_max, got "
                f"{sigma_min=} {init_std=} {sigma_max=}"
            )
        self._r_min, self._delta_r, p0, v_init = _bounded_geometry(
            self.sigma_min, self.sigma_max, self.init_std,
        )
        self._v_init = v_init
        if explore_alpha is None:
            explore_alpha = _default_explore_alpha(self._delta_r, p0)
        if not (math.isfinite(explore_alpha) and explore_alpha > 0.0):
            raise ValueError(f"explore_alpha must be finite and > 0")
        self.explore_alpha = float(explore_alpha)

        self._build_net()

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Network layout
    # ------------------------------------------------------------------

    def _build_net(self) -> None:
        D, H = self.action_dim, self.hidden_dim
        Kc = self.num_components
        if self.sigma_source == "shared":
            if not self.is_mixture:
                self.net = nn.Sequential(
                    nn.Linear(self.obs_dim, H), nn.Tanh(),
                    nn.Linear(H, H), nn.Tanh(),
                    nn.Linear(H, D),
                )
            else:
                self.trunk = nn.Sequential(
                    nn.Linear(self.obs_dim, H), nn.Tanh(),
                    nn.Linear(H, H), nn.Tanh(),
                )
                # head: [logits K | raw_mean K·D] — shared σ is a parameter
                self.head = nn.Linear(H, Kc + Kc * D)
                self._init_mixture_head(state_sigma=False)
        else:  # state σ
            self.trunk = nn.Sequential(
                nn.Linear(self.obs_dim, H), nn.Tanh(),
                nn.Linear(H, H), nn.Tanh(),
            )
            if not self.is_mixture:
                # head: [raw_mean D | σ_ctrl D]
                self.head = nn.Linear(H, 2 * D)
                self._init_state_head()
            else:
                # head: [logits K | raw_mean K·D | σ_ctrl K·D]
                self.head = nn.Linear(H, Kc + 2 * Kc * D)
                self._init_mixture_head(state_sigma=True)

        if self.sigma_source == "shared":
            ctrl = self._sigma_ctrl_init().expand(Kc * D).contiguous()
            if not self.is_mixture:
                self.std_ctrl = nn.Parameter(ctrl.view(D))
            else:
                self.std_ctrl = nn.Parameter(ctrl.view(Kc, D))

    def _sigma_ctrl_init(self) -> torch.Tensor:
        """Scalar init for the shared σ control parameter."""
        if self.sigma_bounded:
            return torch.tensor(self._v_init, dtype=torch.float32)
        return torch.tensor(math.log(self.init_std), dtype=torch.float32)

    def _init_state_head(self) -> None:
        """σ half: zero weight + bias so σ(obs) ≡ init_std at step 0."""
        d = self.action_dim
        bias = self._v_init if self.sigma_bounded else math.log(self.init_std)
        with torch.no_grad():
            self.head.weight[d:, :].zero_()
            self.head.bias[d:].fill_(bias)

    def _init_mixture_head(self, *, state_sigma: bool) -> None:
        """Init logits uniform, component means jittered, σ = init_std."""
        Kc, D = self.num_components, self.action_dim
        bias = self._v_init if self.sigma_bounded else math.log(self.init_std)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            # Component means: small independent jitter breaks symmetry.
            mean_rows = self.head.bias[Kc:Kc + Kc * D]
            mean_rows.copy_(
                torch.randn(Kc * D) * self.component_init_noise
            )
            if state_sigma:
                self.head.bias[Kc + Kc * D:].fill_(bias)
            else:
                # σ half absent from head — nothing to init there.
                pass

    # ------------------------------------------------------------------
    # σ maps
    # ------------------------------------------------------------------

    def _sigma_from_ctrl(self, ctrl: torch.Tensor) -> torch.Tensor:
        if self.sigma_bounded:
            return torch.exp(
                self._r_min + self._delta_r * torch.sigmoid(ctrl)
            )
        return torch.exp(ctrl.clamp(-_LOG_STD_CLAMP, _LOG_STD_CLAMP))

    def _apply_explore(self, ctrl: torch.Tensor, e: float) -> torch.Tensor:
        """e∈[-1,1]: bounded → v + α·e; unbounded → log σ + e·ln3."""
        self._check_e(e)
        if self.sigma_bounded:
            return ctrl + self.explore_alpha * float(e)
        return ctrl + float(e) * math.log(3.0)

    @staticmethod
    def _check_e(e: Any) -> None:
        ef = float(e)
        if not math.isfinite(ef) or abs(ef) > 1.0 + _EI_TOL:
            raise ValueError(f"explore_factor out of [-1, 1]: {e}")

    # ------------------------------------------------------------------
    # A4.3 contract
    # ------------------------------------------------------------------

    def distribution(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return π parameters: mu[B,K,D], sigma[B,K,D], logits[B,K]."""
        obs = torch.as_tensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        D, Kc = self.action_dim, self.num_components

        if not self.is_mixture:
            if self.sigma_source == "shared":
                mu = torch.tanh(self.net(obs)).unsqueeze(1)  # [B,1,D]
                ctrl = self.std_ctrl.view(1, 1, D)
            else:
                out = self.head(self.trunk(obs))
                raw_mean, v = out.split(D, dim=-1)
                mu = torch.tanh(raw_mean).unsqueeze(1)
                ctrl = v.unsqueeze(1)
            sigma = self._sigma_from_ctrl(ctrl.expand(B, 1, D))
            logits = torch.zeros(B, 1, device=obs.device, dtype=mu.dtype)
            ctrl = ctrl.expand(B, 1, D)
        else:
            out = self.head(self.trunk(obs))
            logits = out[:, :Kc]
            raw_mean = out[:, Kc:Kc + Kc * D].view(B, Kc, D)
            mu = torch.tanh(raw_mean)
            if self.sigma_source == "shared":
                ctrl = self.std_ctrl.view(1, Kc, D).expand(B, Kc, D)
            else:
                ctrl = out[:, Kc + Kc * D:].view(B, Kc, D)
            sigma = self._sigma_from_ctrl(ctrl)
        return {"mu": mu, "sigma": sigma, "logits": logits, "ctrl": ctrl}

    def _distribution_with_e(
        self, obs: torch.Tensor, e: float,
    ) -> Dict[str, torch.Tensor]:
        """Policy distribution with the β explore shift applied to σ."""
        dist = self.distribution(obs)
        if float(e) == 0.0:
            return dist
        return {
            **dist,
            "sigma": self._sigma_from_ctrl(
                self._apply_explore(dist["ctrl"], e)
            ),
        }

    def expectation_samples(
        self,
        obs: torch.Tensor,
        u_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enumerated reparameterized samples for trainer consumption.

        Args:
            obs: [B, obs_dim]
            u_noise: [B, K, M, D] uniform noise on [0,1] (constant input,
                no grad).  For single-component cells K=1.

        Returns:
            actions [B,K,M,D], joint mixture log_prob [B,K,M],
            integration_weights [B,K,M] = softmax(logits)_k / M
            (differentiable — NOT detached; this carries the Q-only
            logits gradient per A4.4).
        """
        dist = self.distribution(obs)
        mu, sigma, logits = dist["mu"], dist["sigma"], dist["logits"]
        B, Kc, D = mu.shape
        if u_noise.dim() != 4:
            raise ValueError(
                f"u_noise must be [B,K,M,D], got shape {tuple(u_noise.shape)}"
            )
        if u_noise.shape[0] != B or u_noise.shape[1] != Kc or u_noise.shape[3] != D:
            raise ValueError(
                f"u_noise shape {tuple(u_noise.shape)} incompatible with "
                f"distribution [B={B},K={Kc},D={D}]"
            )
        M = u_noise.shape[2]
        mu_e = mu[:, :, None, :].expand(B, Kc, M, D)
        sigma_e = sigma[:, :, None, :].expand(B, Kc, M, D)
        actions = K.sample_from_uniform(mu_e, sigma_e, u_noise)

        # Joint mixture density at every enumerated action:
        # logp(a_km) = logsumexp_j( logit_j + Σ_d log TN_jd(a_km,d) )
        a5 = actions[:, :, :, None, :]                     # [B,K,M,1,D]
        mu_j = mu[:, None, None, :, :]                     # [B,1,1,K,D]
        sig_j = sigma[:, None, None, :, :]
        lp = K.log_prob(
            a5.expand(B, Kc, M, Kc, D),
            mu_j.expand(B, Kc, M, Kc, D),
            sig_j.expand(B, Kc, M, Kc, D),
        ).sum(dim=-1)                                      # [B,K,M,K]
        logp = torch.logsumexp(
            torch.log_softmax(logits, dim=-1)[:, None, None, :] + lp, dim=-1
        )                                                  # [B,K,M]
        weights = (
            torch.softmax(logits, dim=-1)[:, :, None] / float(M)
        ).expand(B, Kc, M)
        return actions, logp, weights

    def sample_behavior(
        self,
        obs: torch.Tensor,
        spec: SACBehaviorSpec,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sample a β action.  Mixture draws a categorical component index
        per row (whole-vector component, not per-dim)."""
        e = float(spec.explore_factor)
        if spec.mode == "deterministic":
            if abs(e) > _EI_TOL:
                raise ValueError(
                    "deterministic behavior requires explore_factor == 0"
                )
            return self.deterministic_action(obs), {
                "component_id": -1, "explore_factor": 0.0,
                "mode": "deterministic", "log_prob": None,
            }
        self._check_e(e)
        dist = self._distribution_with_e(obs, e)
        mu, sigma, logits = dist["mu"], dist["sigma"], dist["logits"]
        B, Kc, D = mu.shape
        u = torch.rand(
            (B, D), generator=self._rng, dtype=torch.float32,
        ).to(mu.device)
        if self.is_mixture:
            comp = torch.multinomial(
                torch.softmax(logits.detach(), dim=-1).cpu(),
                1, generator=self._rng,
            ).squeeze(1).to(mu.device)                      # [B]
            idx = comp.view(B, 1, 1).expand(B, 1, D)
            mu_k = mu.gather(1, idx).squeeze(1)
            sigma_k = sigma.gather(1, idx).squeeze(1)
        else:
            comp = torch.zeros(B, dtype=torch.long, device=mu.device)
            mu_k, sigma_k = mu[:, 0], sigma[:, 0]
        action = K.sample_from_uniform(mu_k, sigma_k, u)
        # Diagnostic joint log_prob of the drawn action under π (e=0 view
        # is what the trainer sees; β log_prob is reported separately).
        lp_pi = self.log_prob_of(obs, action).detach()
        return action, {
            "component_id": comp.cpu(),
            "explore_factor": e,
            "mode": spec.mode,
            "log_prob": lp_pi,
            "sigma_eff_mean": float(sigma_k.mean().item()),
        }

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        if self.is_mixture:
            comp = dist["logits"].argmax(dim=-1)            # [B]
            idx = comp.view(-1, 1, 1).expand(-1, 1, self.action_dim)
            return dist["mu"].gather(1, idx).squeeze(1)
        return dist["mu"][:, 0]

    def uncertainty(self, obs: torch.Tensor, kind: str) -> torch.Tensor:
        """Differentiable marginal effective-width U (mean over dims).

        kind="peak":  U_d = σ√(2π)Z/2   (single-cell native measure)
        kind="l2":    U_d = 1/(2∫p_d²) — closed form via pairwise TN
                      overlap integrals (mixture-native measure).
        For mixture "peak" the max of the marginal is evaluated at the
        component means — an approximation used for diagnostics only.
        """
        if kind not in ("peak", "l2"):
            raise ValueError(f"unknown uncertainty kind {kind!r}")
        dist = self.distribution(obs)
        mu, sigma = dist["mu"], dist["sigma"]               # [B,K,D]
        logits = dist["logits"]
        p = torch.softmax(logits, dim=-1)                   # [B,K]
        Kc = mu.shape[1]

        if kind == "peak":
            if not self.is_mixture:
                z = K.normalizer(mu[:, 0], sigma[:, 0])
                u_d = sigma[:, 0] * math.sqrt(2 * math.pi) * z / 2.0
            else:
                # Diagnostic approximation: mixture marginal evaluated at
                # each component mean, max over candidates.
                a_c = mu[:, :, None, :]                    # [B,J,1,D]
                comp_lp = K.log_prob(
                    a_c.expand(-1, -1, Kc, -1),            # [B,J,K,D]
                    mu[:, None, :, :].expand(-1, Kc, -1, -1),
                    sigma[:, None, :, :].expand(-1, Kc, -1, -1),
                )
                dens = torch.exp(
                    torch.logsumexp(
                        torch.log(p)[:, None, :, None] + comp_lp, dim=2
                    )
                )                                          # [B,J,D]
                u_d = 1.0 / (2.0 * dens.max(dim=1).values)
            return u_d.mean(dim=-1)

        # kind == "l2": ∫ p_d² over (-1,1)
        # Pairwise overlap I_kj[d] = ∫ TN_kd · TN_jd dx.
        mu_i = mu[:, :, None, :].expand(-1, -1, Kc, -1)
        mu_j = mu[:, None, :, :].expand(-1, Kc, -1, -1)
        s_i = sigma[:, :, None, :].expand(-1, -1, Kc, -1)
        s_j = sigma[:, None, :, :].expand(-1, Kc, -1, -1)
        s2 = s_i**2 + s_j**2
        m_p = (mu_i * s_j**2 + mu_j * s_i**2) / s2
        s_p = torch.sqrt(s_i**2 * s_j**2 / s2)
        log_pair = (
            -0.5 * (mu_i - mu_j) ** 2 / s2
            - 0.5 * torch.log(2 * math.pi * s2)
            + K.log_normalizer(m_p, s_p)
            - K.log_normalizer(mu_i, s_i)
            - K.log_normalizer(mu_j, s_j)
        )
        overlap = torch.exp(log_pair)                       # [B,K,K,D]
        w = (p[:, :, None] * p[:, None, :])[:, :, :, None]  # [B,K,K,1]
        int_p2 = (w * overlap).sum(dim=(1, 2))              # [B,D]
        u_d = 1.0 / (2.0 * int_p2.clamp_min(1e-30))
        return u_d.mean(dim=-1)

    # ------------------------------------------------------------------
    # Compatibility path (K=M=1 squeeze) — trainer keeps calling this
    # until P4-TRAIN-2 migrates it to expectation_samples.
    # ------------------------------------------------------------------

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0] if obs.dim() > 1 else 1
        u = torch.rand(
            (B, self.num_components, 1, self.action_dim),
            generator=self._rng, dtype=torch.float32,
        ).to(obs.device)
        actions, logp, _w = self.expectation_samples(obs, u)
        return actions[:, 0, 0], logp[:, 0, 0]

    def log_prob_of(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Joint mixture log-density of an externally supplied action."""
        dist = self.distribution(obs)
        mu, sigma, logits = dist["mu"], dist["sigma"], dist["logits"]
        B, Kc, D = mu.shape
        a = action.to(mu.dtype)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        lp = K.log_prob(
            a[:, None, :].expand(B, Kc, D), mu, sigma,
        ).sum(dim=-1)                                        # [B,K]
        return torch.logsumexp(
            torch.log_softmax(logits, dim=-1) + lp, dim=-1
        )

    # ------------------------------------------------------------------
    # RNG & identity
    # ------------------------------------------------------------------

    def _noise(self, shape, device, dtype) -> torch.Tensor:
        return torch.rand(shape, generator=self._rng, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

    def rng_state(self) -> torch.Tensor:
        return self._rng.get_state()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self._rng.set_state(state)

    def policy_fingerprint(self) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            h.update(name.encode("utf-8"))
            h.update(tensor.detach().cpu().numpy().tobytes())
        h.update(self.policy_arch.encode("utf-8"))
        h.update(str(self.obs_dim).encode("ascii"))
        h.update(str(self.action_dim).encode("ascii"))
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Persistence / export
    # ------------------------------------------------------------------

    def state_payload(self) -> Dict[str, Any]:
        return {
            "policy_arch": self.policy_arch,
            "arch": self._arch,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_components": self.num_components,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "init_std": self.init_std,
            "explore_alpha": self.explore_alpha,
            "component_init_noise": self.component_init_noise,
            "sigma_source": self.sigma_source,
            "sigma_bounded": self.sigma_bounded,
            "kernel_version": "tn_kernel_v1",
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "rng_state": self._rng.get_state(),
        }

    @property
    def _arch(self) -> str:
        fam = "m" if self.is_mixture else "s"
        src = "1" if self.sigma_source == "state" else "0"
        bnd = "1" if self.sigma_bounded else "0"
        return f"{fam}{src}{bnd}"

    def to_blueprint(
        self,
        dest_path: Optional[str] = None,
        *,
        stochastic: bool = False,
        explore_factor: float = 0.0,
    ) -> PolicyBlueprint:
        self._check_e(explore_factor)
        if not stochastic and abs(float(explore_factor)) > _EI_TOL:
            raise ValueError("deterministic export cannot carry explore_factor")
        dest = Path(dest_path) if dest_path is not None else Path(
            tempfile.mkdtemp(prefix="sac_tn_")
        )
        self.export_policy_artifacts(
            dest, stochastic=stochastic, explore_factor=explore_factor,
        )
        return PolicyBlueprint(
            cls="baseline.framework.sac.tn_actor:TNRuntimePolicy",
            config={
                "model_path": str(dest / "model.pt"),
                "stochastic": bool(stochastic),
                "explore_factor": float(explore_factor),
            },
        )

    def export_policy_artifacts(
        self,
        dest_dir: Path,
        *,
        stochastic: bool = False,
        explore_factor: float = 0.0,
    ) -> None:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_payload(), dest / "model.pt")
        PolicyBlueprint(
            cls="baseline.framework.sac.tn_actor:TNRuntimePolicy",
            config={
                "model_path": str(dest / "model.pt"),
                "stochastic": bool(stochastic),
                "explore_factor": float(explore_factor),
            },
        ).save(dest / "policy_blueprint.yaml")


class TNRuntimePolicy:
    """Runtime policy loaded from an exported TN ``model.pt``."""

    def __init__(
        self,
        model_path: str,
        stochastic: bool = False,
        explore_factor: float = 0.0,
        seed: int = 0,
        **_ignored: Any,
    ) -> None:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        arch = str(payload.get("policy_arch", ""))
        if not arch.startswith("tn_") or payload.get("kernel_version") != "tn_kernel_v1":
            raise ValueError(
                f"unsupported TN runtime payload: policy_arch={arch!r} "
                f"kernel={payload.get('kernel_version')!r}"
            )
        self.stochastic = bool(stochastic)
        self.explore_factor = float(explore_factor)
        TNActor._check_e(self.explore_factor)
        if not self.stochastic and abs(self.explore_factor) > _EI_TOL:
            raise ValueError(
                "deterministic runtime policy cannot carry explore_factor"
            )
        self._actor = TNActor(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            arch=str(payload["arch"]),
            hidden_dim=int(payload["hidden_dim"]),
            n_components=int(payload["num_components"]),
            sigma_min=float(payload["sigma_min"]),
            sigma_max=float(payload["sigma_max"]),
            init_std=float(payload["init_std"]),
            explore_alpha=float(payload["explore_alpha"]),
            component_init_noise=float(payload["component_init_noise"]),
            seed=int(seed),
        )
        self._actor.load_state_dict(payload["state_dict"])
        if "rng_state" in payload:
            self._actor.set_rng_state(payload["rng_state"])
        self._actor.eval()

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            if self.stochastic:
                spec = SACBehaviorSpec(
                    mode="stochastic", explore_factor=self.explore_factor,
                )
                action, info = self._actor.sample_behavior(obs, spec)
                extra = {
                    "log_prob": (
                        float(info["log_prob"].item())
                        if info["log_prob"] is not None else None
                    ),
                    "explore_factor": info["explore_factor"],
                    "component_id": (
                        info["component_id"].item()
                        if hasattr(info["component_id"], "item") else -1
                    ),
                } if want_extra else None
            else:
                action = self._actor.deterministic_action(obs)
                extra = {"log_prob": None, "explore_factor": 0.0} if want_extra else None
        return action.squeeze(0).cpu().numpy().astype(np.float32), extra

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._actor._rng.manual_seed(int(seed))

    def close(self) -> None:
        return None


__all__ = [
    "ARCH_SPECS",
    "SIGMA_MIN",
    "SIGMA_MAX",
    "INIT_STD",
    "TNActor",
    "TNRuntimePolicy",
]
