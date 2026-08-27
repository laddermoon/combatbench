"""Shared base class for tanh-squashed stochastic policies.

All new policy families (state-dependent Gaussian, low-rank Gaussian,
mixture of Gaussians, normalizing flow) share the same tanh-squashing
math and the same ``Policy`` / ``TrainablePolicy`` protocol glue.
This module centralizes that shared logic so it is implemented and
validated once, not duplicated (and mis-duplicated) per family.

The base class owns:
  * ``act`` / ``act_numpy`` / ``set_deterministic`` — single-step
    inference glue for the ``Policy`` ABC.
  * ``evaluate_actions`` — the ``TrainablePolicy`` contract, including
    the tanh Jacobian correction (``-log(1 - tanh(raw)² + ε)``).
  * ``set_exploration`` — temperature / entropy_coef bookkeeping.
  * ``to_blueprint`` — export via :mod:`baseline.common.policies.export_generic`.

Subclasses implement four hooks, all in **raw space** (pre-tanh):

  * ``_raw_sample(obs) -> (raw_action, extras)`` — rsample from the
    raw distribution.  ``extras`` is an optional dict carried to the
    regularizer/stats hook (e.g. MoG component indices for
    ``frame_modes``-compatible diagnostics).
  * ``_raw_log_prob(obs, raw_action) -> (raw_log_prob, extras)`` —
    log-density of ``raw_action`` under the raw distribution.
  * ``_raw_mode(obs) -> raw_action`` — deterministic raw action.
  * ``_regularizer_and_stats(obs, raw_action, raw_log_prob, want_stats,
    sample_extras, score_extras) -> (regularizer, stats)`` —
    family-owned entropy-like regularizer (already signed/scaled) and
    diagnostics dict.

The tanh Jacobian is applied in exactly one place (here), so it cannot
be gotten wrong per-family.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from envs.framework.policy import Policy, PolicyBlueprint

from baseline.framework.ppo import ActorEval, ExplorationSpec


# Epsilon for the tanh Jacobian correction:  log(1 - tanh(x)² + ε).
# Must match TanhGaussianMLPPolicy exactly so the degenerate-equivalence
# test (test_policy_families.py) can compare log_prob to 1e-6.
_TANH_JAC_EPS = 1e-6

# Clamping bound for actions before atanh:  atanh is undefined at ±1.
# Must match TanhGaussianMLPPolicy's 0.999999.
_ATANH_CLAMP = 0.999999


class TanhSquashedPolicyBase(nn.Module, Policy):
    """Generic tanh-squashed policy base.

    Subclasses must implement the four raw-space hooks listed in the
    module docstring.  They must also set ``self.obs_dim``,
    ``self.action_dim``, and call ``super().__init__()`` with the
    appropriate arguments.

    The constructor stores exploration state (``_log_std_offset``,
    ``_entropy_coef``, ``_temperature``) as plain floats, not buffers,
    so they are owned by the experiment's schedule and never restored
    from a checkpoint's ``state_dict`` — matching
    :class:`TanhGaussianMLPPolicy`'s convention.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        entropy_coef: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self._deterministic = bool(deterministic)
        # Exploration state — plain floats, not buffers (see class docstring).
        self._entropy_coef = float(entropy_coef)
        self._temperature = float(temperature)

    # ------------------------------------------------------------------
    # Subclass hooks (raw space, pre-tanh)
    # ------------------------------------------------------------------

    def _raw_sample(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """rsample a raw (pre-tanh) action from the policy's distribution.

        Returns:
            (raw_action, extras) where extras is an optional dict
            threaded to the regularizer/stats hook.  May be None.
        """
        raise NotImplementedError

    def _raw_log_prob(
        self, obs: torch.Tensor, raw_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Log-density of raw_action under the raw distribution at obs.

        Returns:
            (raw_log_prob, extras) where raw_log_prob is shape (B,).
        """
        raise NotImplementedError

    def _raw_mode(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic raw (pre-tanh) action (the distribution's mode)."""
        raise NotImplementedError

    def _regularizer_and_stats(
        self,
        obs: torch.Tensor,
        raw_action: torch.Tensor,
        raw_log_prob: torch.Tensor,
        want_stats: bool,
        sample_extras: Optional[Dict[str, Any]],
        score_extras: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """Compute the family-owned regularizer and optional stats.

        The regularizer is **already signed and scaled** — the framework
        adds it verbatim to the actor loss.  For an entropy bonus this
        is ``-entropy_coef * entropy.mean()``.

        Returns:
            (regularizer, stats) where either may be None.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Export config (subclass provides constructor kwargs for round-trip)
    # ------------------------------------------------------------------

    def export_config(self) -> Dict[str, Any]:
        """Return full constructor kwargs for export round-trip.

        Subclasses must override this to include all architecture
        parameters (e.g. K, rank, num_layers) needed to reconstruct
        the module on load.
        """
        raise NotImplementedError

    @property
    def export_class_path(self) -> str:
        """Dotted path to the policy class for export, e.g.
        ``baseline.common.policies.state_gaussian_mlp:StateGaussianMLPPolicy``.

        Subclasses must override this.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Tanh Jacobian (single source of truth)
    # ------------------------------------------------------------------

    @staticmethod
    def _tanh_jacobian(clipped_actions: torch.Tensor) -> torch.Tensor:
        """Compute the tanh Jacobian correction:  -log(1 - tanh² + ε).

        This is the log|d atanh(a)/d a| term for a = tanh(raw).
        Summed over the action dimension by the caller.
        """
        return -torch.log(1.0 - clipped_actions.pow(2) + _TANH_JAC_EPS)

    # ------------------------------------------------------------------
    # Sampling / deterministic action
    # ------------------------------------------------------------------

    def sample_action(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a (post-tanh) action and compute its log_prob.

        Returns:
            (action, log_prob) where action is (B, action_dim) in
            [-1, 1] and log_prob is (B,).

        Computation order matches TanhGaussianMLPPolicy: per-dimension
        (raw_log_prob - jacobian), then sum, when
        ``_raw_log_prob_per_dim`` is available.  Otherwise falls back
        to separate sums.
        """
        raw_action, sample_extras = self._raw_sample(obs)
        action = torch.tanh(raw_action)
        clipped = torch.clamp(action, -_ATANH_CLAMP, _ATANH_CLAMP)
        if hasattr(self, "_raw_log_prob_per_dim"):
            raw_lp_per_dim, _ = self._raw_log_prob_per_dim(obs, raw_action)
            jac_per_dim = self._tanh_jacobian(clipped)
            log_prob = (raw_lp_per_dim + jac_per_dim).sum(dim=-1)
        else:
            raw_log_prob, _ = self._raw_log_prob(obs, raw_action)
            log_prob = raw_log_prob + self._tanh_jacobian(clipped).sum(dim=-1)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return tanh(mode) as the deterministic action."""
        return torch.tanh(self._raw_mode(obs))

    # ------------------------------------------------------------------
    # TrainablePolicy: evaluate_actions
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        frame_modes: Optional[torch.Tensor] = None,
        want_stats: bool = False,
    ) -> ActorEval:
        """Score actions under the current parameters.

        Implements the TrainablePolicy contract.  The tanh Jacobian is
        applied here (single source of truth), so subclasses only need
        to provide the raw-space log_prob.

        The computation order matches TanhGaussianMLPPolicy exactly:
        per-dimension (raw_log_prob - jacobian), then sum over the
        action dimension.  This is deliberately bit-identical to the
        baseline so the degenerate-equivalence test is meaningful.
        """
        clipped_actions = torch.clamp(actions, -_ATANH_CLAMP, _ATANH_CLAMP)
        raw_actions = torch.atanh(clipped_actions)
        # _raw_log_prob returns (B,) — already summed over action dims.
        # For bit-identical matching with the baseline, we need the
        # per-dimension version.  Subclasses that want to support the
        # degenerate-equivalence test should also implement
        # _raw_log_prob_per_dim, which returns (B, action_dim).
        if hasattr(self, "_raw_log_prob_per_dim"):
            raw_lp_per_dim, score_extras = self._raw_log_prob_per_dim(obs, raw_actions)
            jac_per_dim = self._tanh_jacobian(clipped_actions)
            log_prob = (raw_lp_per_dim + jac_per_dim).sum(dim=-1)
        else:
            raw_log_prob, score_extras = self._raw_log_prob(obs, raw_actions)
            log_prob = raw_log_prob + self._tanh_jacobian(clipped_actions).sum(dim=-1)

        # Regularizer + stats.
        #
        # For the regularizer, we prefer the subclass's closed-form
        # entropy (returned by _regularizer_and_stats) when available.
        # This is critical for diagonal Gaussian families (① and ②)
        # where the score-function estimate becomes unreliable when σ
        # is small: -log_prob(rsample()) → 0 as σ → 0, so the
        # regularizer gradient vanishes and σ collapses to zero.
        # The closed-form entropy Normal.entropy() = 0.5*log(2πeσ²)
        # has gradient 1/σ which correctly pushes σ up.
        #
        # For families without closed-form entropy (③ MoG, ④ flow),
        # _regularizer_and_stats returns None for the regularizer,
        # and we fall back to the score-function estimate.
        #
        # We always draw a fresh sample for stats (want_stats path),
        # but only compute the score-function regularizer as fallback.
        regularizer = None
        stats: Optional[Dict[str, float]] = None
        need_sample = want_stats or self._entropy_coef != 0.0
        if need_sample:
            # First, try the subclass's closed-form regularizer.
            if self._entropy_coef != 0.0:
                # Call _regularizer_and_stats with want_stats=False
                # to get just the regularizer (no sample needed for
                # closed-form entropy).
                reg_cf, _ = self._regularizer_and_stats(
                    obs, None, None, False, None, None,
                )
                if reg_cf is not None:
                    regularizer = reg_cf

            # Draw a fresh sample for stats and/or score-function fallback.
            reg_raw_action, reg_sample_extras = self._raw_sample(obs)
            reg_raw_log_prob, _ = self._raw_log_prob(obs, reg_raw_action)

            # If no closed-form regularizer, use score-function estimate.
            if self._entropy_coef != 0.0 and regularizer is None:
                entropy_estimate = -reg_raw_log_prob.mean()
                regularizer = -self._entropy_coef * entropy_estimate

            if want_stats:
                with torch.no_grad():
                    stats = self._compute_stats(
                        obs, raw_actions, log_prob,
                        reg_raw_action, reg_raw_log_prob,
                        score_extras, reg_sample_extras,
                    )

        return ActorEval(
            log_prob=log_prob,
            regularizer=regularizer,
            stats=stats,
        )

    def _compute_stats(
        self,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
        raw_log_prob: torch.Tensor,
        reg_raw_action: torch.Tensor,
        reg_raw_log_prob: torch.Tensor,
        score_extras: Optional[Dict[str, Any]],
        sample_extras: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute stats dict.  Delegates to subclass hook with all
        available quantities.  Subclasses override
        ``_regularizer_and_stats`` for the primary path; this method
        is a thin wrapper that ensures the entropy estimate is included.
        """
        # Get family-specific stats from the subclass.
        _, family_stats = self._regularizer_and_stats(
            obs, reg_raw_action, reg_raw_log_prob,
            want_stats=True,
            sample_extras=sample_extras,
            score_extras=score_extras,
        )
        stats: Dict[str, float] = dict(family_stats or {})
        # Always include the entropy estimate (score-function).
        stats["entropy"] = float((-reg_raw_log_prob.mean()).item())
        return stats

    # ------------------------------------------------------------------
    # TrainablePolicy: set_exploration
    # ------------------------------------------------------------------

    def set_exploration(self, spec: ExplorationSpec) -> Dict[str, float]:
        """Apply an ExplorationSpec; return the effective config.

        Honoured fields:
        - ``temperature``: stored as ``self._temperature``.  Subclasses
          use it in their raw distribution construction.
        - ``entropy_coef``: stored as ``self._entropy_coef``.  Used in
          ``evaluate_actions`` to build the regularizer.

        Ignored fields: ``entropy_target``, ``clip_eps``, ``target_kl``,
        ``policy_extras``.
        """
        if spec.temperature is not None:
            temperature = float(spec.temperature)
            if temperature <= 0.0:
                raise ValueError(f"temperature must be > 0, got {temperature}")
            self._temperature = temperature
        if spec.entropy_coef is not None:
            self._entropy_coef = float(spec.entropy_coef)
        return {
            "entropy_coef": self._entropy_coef,
            "temperature": self._temperature,
        }

    # ------------------------------------------------------------------
    # Policy ABC: act / set_deterministic
    # ------------------------------------------------------------------

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Single-step inference."""
        action_np, log_prob = self.act_numpy(
            observation, device=self.device, deterministic=self._deterministic
        )
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {"log_prob": float(log_prob)}

    def set_deterministic(self, deterministic: bool) -> None:
        """Toggle stochastic vs deterministic action sampling."""
        self._deterministic = bool(deterministic)

    def act_numpy(
        self, obs: np.ndarray, device: torch.device, deterministic: bool,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Numpy-flavoured single-step inference."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.deterministic_action(obs_tensor)
                log_prob = None
            else:
                action, log_prob = self.sample_action(obs_tensor)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if log_prob is None:
            return action_np, None
        return action_np, float(log_prob.item())

    # ------------------------------------------------------------------
    # TrainablePolicy: to_blueprint (via export_generic)
    # ------------------------------------------------------------------

    def to_blueprint(
        self, dest_path: Optional[str] = None, *, stochastic: bool = False,
    ) -> PolicyBlueprint:
        """Export this policy to a deployable PolicyBlueprint.

        Uses :mod:`baseline.common.policies.export_generic` for a
        family-agnostic export path with ``strict=True`` reload.
        """
        import tempfile

        if dest_path is None:
            dest_path = tempfile.mkdtemp(prefix="policy_export_")
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        from .export_generic import export_generic_policy_artifacts

        export_generic_policy_artifacts(
            actor=self,
            policy_dir=policy_dir,
            policy_class_path=self.export_class_path,
            config=self.export_config(),
            stochastic=stochastic,
            extra_payload={
                "temperature": self._temperature,
                "entropy_coef": self._entropy_coef,
            },
        )

        policy_py_path = policy_dir / "policy.py"
        return PolicyBlueprint(
            cls=f"file:{policy_py_path}:ExportedPolicy",
            config={"stochastic": stochastic},
        )
