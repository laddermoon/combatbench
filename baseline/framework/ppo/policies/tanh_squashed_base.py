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
  * ``to_blueprint`` — export via :mod:`baseline.framework.ppo.policies.export_generic`.

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

from baseline.framework.ppo import ActorEval


# Epsilon for the tanh Jacobian correction:  log(1 - tanh(x)² + ε).
# Must match TanhGaussianMLPPolicy exactly so the degenerate-equivalence
# test (test_policy_families.py) can compare log_prob to 1e-6.
_TANH_JAC_EPS = 1e-6

# Clamping bound for actions before atanh:  atanh is undefined at ±1.
# Must match TanhGaussianMLPPolicy's 0.999999.
_ATANH_CLAMP = 0.999999


# ---------------------------------------------------------------------------
# Temporally correlated exploration noise (OU process)
#
# The base class can optionally apply a raw-space shift to exploration
# samples, drawn from an Ornstein-Uhlenbeck (AR(1)) process.  This
# produces temporally correlated noise whose power spectrum is
# concentrated at low frequencies, unlike the white (independent-per-step)
# noise that a diagonal Gaussian produces by default.
#
# The shift is applied as a translation in raw (pre-tanh) space:
#
#   sampling:  z ~ p(·|o);   raw = z + s;   a = tanh(raw)
#   scoring:   raw = atanh(a) - s;   log_prob = p.log_prob(raw) + jac
#
# This is exact for ANY raw distribution p (Gaussian, mixture, flow, ...)
# because translating a density does not change its form — only its
# evaluation point.  The subclass hooks (_raw_sample, _raw_log_prob, ...)
# are completely unchanged.
#
# The shift s_t = noise_scale * x_t is recorded per-step in rollout
# extras and threaded through to evaluate_actions, so the training side
# never needs to know any OU parameter — it only subtracts the recorded
# shift.  This eliminates the class of bugs where rollout-side and
# training-side OU parameters disagree and silently corrupt log_prob.
# ---------------------------------------------------------------------------


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
        noise_tau_steps: float = 0.0,
        noise_scale: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self._deterministic = bool(deterministic)
        # Exploration state — plain floats, not buffers (see class docstring).
        self._entropy_coef = float(entropy_coef)
        self._temperature = float(temperature)
        # OU exploration noise state.  noise_scale=0 disables the feature
        # and makes the policy bit-identical to one without OU support.
        self._noise_tau_steps = float(noise_tau_steps)
        self._noise_scale = float(noise_scale)
        self._ou_x: Optional[np.ndarray] = None
        self._ou_rng: Optional[np.random.Generator] = None
        self._ou_a: float = 0.0
        self._ou_innov: float = 1.0
        self._update_ou_params()

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

        .. deprecated::
            This hook is from the old ``ActorEval.regularizer`` design.
            New policies should override :meth:`_entropy_and_stats`
            instead.  This method is kept for backward compatibility
            with subclasses that have not yet been migrated.

        The regularizer is **already signed and scaled** — the framework
        adds it verbatim to the actor loss.  For an entropy bonus this
        is ``-entropy_coef * entropy.mean()``.

        Returns:
            (regularizer, stats) where either may be None.
        """
        raise NotImplementedError

    def _entropy_and_stats(
        self,
        obs: torch.Tensor,
        want_stats: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute per-obs normalized entropy and optional stats.

        This is the new hook for the ``ActorEval.entropy`` design.
        Returns ``(entropy_norm, stats)`` where ``entropy_norm`` is a
        ``(B,)`` tensor of normalized entropy in [0, 1].

        The default implementation calls :meth:`_regularizer_and_stats`
        for backward compatibility with unmigrated subclasses.  It
        draws a fresh sample to get a score-function entropy estimate
        when the subclass does not provide closed-form entropy.

        Subclasses should override this to return closed-form entropy
        when available (e.g. diagonal Gaussian: ``Normal.entropy()``).

        Returns:
            (entropy_norm, stats) where entropy_norm is ``(B,)`` and
            stats is an optional dict.
        """
        # Default: sample-based estimate via _regularizer_and_stats.
        # This is a fallback for unmigrated subclasses.
        raw_action, sample_extras = self._raw_sample(obs)
        raw_log_prob, _ = self._raw_log_prob(obs, raw_action)
        # Score-function entropy estimate (per-obs).
        entropy_raw = -raw_log_prob  # (B,)
        # Normalize using log_std_min/max if available.
        if hasattr(self, "log_std_min") and hasattr(self, "log_std_max"):
            import math
            H_max = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_max)
            H_min = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_min)
            entropy_norm = (entropy_raw - H_min) / (H_max - H_min)
        else:
            # No normalization reference — return raw estimate clamped.
            entropy_norm = torch.clamp(entropy_raw / 10.0, 0.0, 1.0)

        stats = None
        if want_stats:
            _, stats = self._regularizer_and_stats(
                obs, raw_action, raw_log_prob, True,
                sample_extras, None,
            )
            if stats is None:
                stats = {"entropy_raw": float(entropy_raw.mean().item())}
            else:
                stats = dict(stats)
                stats.setdefault("entropy_raw", float(entropy_raw.mean().item()))

        return entropy_norm, stats

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
        ``baseline.framework.ppo.policies.state_gaussian_mlp:StateGaussianMLPPolicy``.

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
    # OU exploration noise
    # ------------------------------------------------------------------

    def _update_ou_params(self) -> None:
        """Recompute AR(1) coefficients from ``_noise_tau_steps``.

        The process is parameterised as a unit-variance AR(1):

            x_{t+1} = a * x_t + sqrt(1 - a^2) * xi,   xi ~ N(0, I)

        where ``a = exp(-1 / tau)``.  This keeps ``Var(x) = 1`` for any
        ``tau > 0``, so ``noise_scale`` is directly comparable to the
        policy's native sigma (both are in raw-space units).
        """
        if self._noise_tau_steps > 0.0:
            self._ou_a = float(np.exp(-1.0 / self._noise_tau_steps))
            self._ou_innov = float(np.sqrt(1.0 - self._ou_a ** 2))
        else:
            # tau=0 → pure white noise (a=0).  Still unit variance.
            self._ou_a = 0.0
            self._ou_innov = 1.0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset OU state for a new episode.

        Zeroes ``_ou_x`` and (re)seeds the numpy RNG.  Called by the
        episode runner at every episode boundary, so each agent's OU
        process starts fresh and is reproducible from the per-agent seed.
        """
        self._ou_x = np.zeros(self.action_dim, dtype=np.float32)
        if seed is not None:
            self._ou_rng = np.random.default_rng(int(seed))

    def _next_noise_shift(self) -> Optional[np.ndarray]:
        """Step the OU process and return the raw-space shift ``s_t``.

        Returns ``None`` when OU is disabled (``noise_scale == 0``),
        so callers can treat ``None`` as "no shift" without branching
        on the policy's configuration.

        The returned array has shape ``(action_dim,)`` and is in
        raw (pre-tanh) space units.  It is recorded in rollout extras
        and subtracted during ``evaluate_actions`` scoring.
        """
        if self._noise_scale == 0.0:
            return None
        if self._ou_x is None:
            self._ou_x = np.zeros(self.action_dim, dtype=np.float32)
        if self._ou_rng is None:
            self._ou_rng = np.random.default_rng()
        xi = self._ou_rng.standard_normal(self.action_dim).astype(np.float32)
        self._ou_x = self._ou_a * self._ou_x + self._ou_innov * xi
        return (self._noise_scale * self._ou_x).astype(np.float32)

    # ------------------------------------------------------------------
    # Sampling / deterministic action
    # ------------------------------------------------------------------

    def sample_action(
        self, obs: torch.Tensor,
        *,
        noise_shift: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a (post-tanh) action and compute its log_prob.

        Returns:
            (action, log_prob) where action is (B, action_dim) in
            [-1, 1] and log_prob is (B,).

        When ``noise_shift`` is provided (shape ``(B, action_dim)`` or
        broadcastable), it is added to the raw sample *before* tanh.
        The log_prob is computed for the **unshifted** sample ``z``
        (i.e. ``_raw_log_prob(obs, z)``), which is the correct density
        of the shifted distribution evaluated at the shifted point:
        ``p(z | o) = p(raw - s | o)``.  This matches
        ``evaluate_actions(..., noise_shift=s)`` which computes
        ``_raw_log_prob(obs, atanh(a) - s)``.

        Computation order matches TanhGaussianMLPPolicy: per-dimension
        (raw_log_prob - jacobian), then sum, when
        ``_raw_log_prob_per_dim`` is available.  Otherwise falls back
        to separate sums.
        """
        raw_sample, sample_extras = self._raw_sample(obs)
        if noise_shift is not None:
            raw_action = raw_sample + noise_shift
        else:
            raw_action = raw_sample
        action = torch.tanh(raw_action)
        clipped = torch.clamp(action, -_ATANH_CLAMP, _ATANH_CLAMP)
        # Score the unshifted sample z = raw_action - noise_shift.
        # When noise_shift is None, z == raw_action (no-op).
        scored = raw_sample if noise_shift is not None else raw_action
        if hasattr(self, "_raw_log_prob_per_dim"):
            raw_lp_per_dim, _ = self._raw_log_prob_per_dim(obs, scored)
            jac_per_dim = self._tanh_jacobian(clipped)
            log_prob = (raw_lp_per_dim + jac_per_dim).sum(dim=-1)
        else:
            raw_log_prob, _ = self._raw_log_prob(obs, scored)
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
        noise_shift: Optional[torch.Tensor] = None,
        want_stats: bool = False,
    ) -> ActorEval:
        """Score actions under the current parameters.

        Implements the TrainablePolicy contract.  The tanh Jacobian is
        applied here (single source of truth), so subclasses only need
        to provide the raw-space log_prob.

        When ``noise_shift`` is provided (shape ``(B, action_dim)``),
        it is subtracted from ``atanh(action)`` before scoring, so the
        log_prob matches the shifted distribution used at rollout time.
        This is the training-side counterpart of ``sample_action``'s
        ``noise_shift`` parameter — the shift is a *fact recorded at
        rollout time*, not a quantity to re-infer.

        The computation order matches TanhGaussianMLPPolicy exactly:
        per-dimension (raw_log_prob - jacobian), then sum over the
        action dimension.  This is deliberately bit-identical to the
        baseline so the degenerate-equivalence test is meaningful.
        """
        clipped_actions = torch.clamp(actions, -_ATANH_CLAMP, _ATANH_CLAMP)
        raw_actions = torch.atanh(clipped_actions)
        if noise_shift is not None:
            # Recover the unshifted sample: z = atanh(a) - s
            raw_actions = raw_actions - noise_shift
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

        # Entropy + stats.
        #
        # The policy returns a per-obs normalized entropy in [0, 1].
        # The framework computes the entropy floor loss from this.
        # See DESIGN_unified_exploration_control.md.
        entropy, stats = self._entropy_and_stats(obs, want_stats)

        return ActorEval(
            log_prob=log_prob,
            entropy=entropy,
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
        # Include the score-function entropy estimate only when the
        # subclass did not already provide a closed-form "entropy" stat.
        # This lets diagonal-Gaussian families report Normal.entropy()
        # (which has a non-vanishing gradient as σ → 0) while mixture /
        # flow families that lack closed-form entropy get the
        # score-function estimate as fallback.
        if "entropy" not in stats:
            stats["entropy"] = float((-reg_raw_log_prob.mean()).item())
        return stats

    # ------------------------------------------------------------------
    # TrainablePolicy: set_exploration
    # ------------------------------------------------------------------

    def set_exploration(self, explore_intensity) -> None:
        """Apply an exploration directive.

        Accepts either a float (new interface) or an
        :class:`~baseline.framework.ppo.experiment.ExplorationSpec`
        (old interface, for backward compat with unmigrated subclasses).

        **New interface** — ``explore_intensity: float`` ∈ [0, 1]:
        - ``0.5`` = neutral (temperature=1.0, no change)
        - ``→ 0`` = compress σ (temperature < 1.0)
        - ``→ 1`` = expand σ (temperature > 1.0)

        **Old interface** — ``ExplorationSpec``:
        - ``temperature``: stored as ``self._temperature``
        - ``noise_tau_steps``: OU correlation time
        - ``noise_scale``: OU shift steady-state std
        - ``entropy_coef``: ignored (entropy floor is framework-side now)
        """
        # Backward compat: accept ExplorationSpec.
        if not isinstance(explore_intensity, (int, float)):
            spec = explore_intensity
            if hasattr(spec, 'temperature') and spec.temperature is not None:
                self._temperature = float(spec.temperature)
            if hasattr(spec, 'noise_tau_steps') and spec.noise_tau_steps is not None:
                self._noise_tau_steps = float(spec.noise_tau_steps)
                self._update_ou_params()
            if hasattr(spec, 'noise_scale') and spec.noise_scale is not None:
                self._noise_scale = float(spec.noise_scale)
            return

        # New interface: symmetric mapping centered at 0.5.
        # 0.5 → temperature=1.0 (neutral), 0 → compress, 1 → expand.
        import math
        offset = (float(explore_intensity) - 0.5) * 2.0  # EXPLORE_SPAN=2.0
        self._temperature = float(math.exp(offset))

    # ------------------------------------------------------------------
    # Policy ABC: act / set_deterministic
    # ------------------------------------------------------------------

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Single-step inference.

        When stochastic and OU is enabled, steps the OU process and
        applies the shift to the raw sample.  The shift is included in
        the returned extras dict (key ``"noise_shift"``) so the rollout
        can record it for exact log_prob recomputation during training.
        """
        noise_shift = None if self._deterministic else self._next_noise_shift()
        action_np, log_prob = self.act_numpy(
            observation, device=self.device,
            deterministic=self._deterministic,
            noise_shift=noise_shift,
        )
        if not want_extra or log_prob is None:
            return action_np, None
        extras: Dict[str, Any] = {"log_prob": float(log_prob)}
        if noise_shift is not None:
            extras["noise_shift"] = np.asarray(noise_shift, dtype=np.float32)
        return action_np, extras

    def set_deterministic(self, deterministic: bool) -> None:
        """Toggle stochastic vs deterministic action sampling."""
        self._deterministic = bool(deterministic)

    def act_numpy(
        self, obs: np.ndarray, device: torch.device, deterministic: bool,
        *,
        noise_shift: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Numpy-flavoured single-step inference."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        shift_tensor = (
            torch.as_tensor(noise_shift, dtype=torch.float32, device=device).unsqueeze(0)
            if noise_shift is not None else None
        )
        with torch.no_grad():
            if deterministic:
                action = self.deterministic_action(obs_tensor)
                log_prob = None
            else:
                action, log_prob = self.sample_action(
                    obs_tensor, noise_shift=shift_tensor,
                )
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

        Uses :mod:`baseline.framework.ppo.policies.export_generic` for a
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
                "noise_tau_steps": self._noise_tau_steps,
                "noise_scale": self._noise_scale,
            },
        )

        policy_py_path = policy_dir / "policy.py"
        return PolicyBlueprint(
            cls=f"file:{policy_py_path}:ExportedPolicy",
            config={"stochastic": stochastic},
        )
