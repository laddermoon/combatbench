"""Continuous-control actor backbone (Tanh-squashed Gaussian).

Checkpoint / export IO lives in :mod:`baseline.framework.ppo.policies.checkpoint`
and is re-exported from this module for backward compatibility with scripts
that imported it from here (pre-PR1). New code should import directly from
:mod:`baseline.framework.ppo.policies.checkpoint`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from envs.framework.policy import Policy, PolicyBlueprint

# The actor-side data types of the TrainablePolicy contract. Importing
# them here is safe (ppo.experiment depends only on envs.framework, so
# there is no cycle) and keeps a single definition of the contract.
from baseline.framework.ppo import ActorEval

from .checkpoint import (
    DEFAULT_EXPORT_ACTOR_HIDDEN_DIM,
    build_actor_export_payload,
    build_export_policy_code,
    export_actor_policy_artifacts,
    export_policy_artifacts_from_checkpoint,
)

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 1.0

# Numerical safety bounds for log_std.  These are NOT business bounds —
# they only prevent exp(log_std) from overflowing or underflowing.  Normal
# training never approaches ±20.  See DESIGN_migration_tanh_gaussian.md §3.
_LOG_STD_SAFE_MIN = -20.0  # exp(-20) ≈ 2e-9
_LOG_STD_SAFE_MAX = 20.0   # exp(20) ≈ 5e8

__all__ = [
    "DEFAULT_LOG_STD_MIN",
    "DEFAULT_LOG_STD_MAX",
    "DEFAULT_EXPORT_ACTOR_HIDDEN_DIM",
    "TanhGaussianMLPPolicy",
    "build_actor_export_payload",
    "build_export_policy_code",
    "export_actor_policy_artifacts",
    "export_policy_artifacts_from_checkpoint",
]


class TanhGaussianMLPPolicy(nn.Module, Policy):
    """Generic continuous-control policy backbone for Box-like actions.

    Implements the framework :class:`envs.framework.policy.Policy` ABC
    directly so no adapter is required. The ``device`` and ``deterministic``
    flags control inference behaviour; ``to_blueprint`` exports a
    deployable :class:`PolicyBlueprint` pointing to
    :class:`ExportedMLPPolicy`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.device = torch.device(device)
        self._deterministic = bool(deterministic)
        # Exploration state — plain float, not a buffer, so it is owned
        # by the experiment's schedule and never restored from a
        # checkpoint's state_dict.  set_exploration sets this; 0.0 means
        # neutral (policy uses its learned σ as-is, i.e. ei=0.5).
        self._log_std_offset = 0.0
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0, dtype=torch.float32))
        if model_path is not None:
            payload = torch.load(model_path, map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                raise RuntimeError(f"Missing keys loading TanhGaussianMLPPolicy: {missing}")
            if unexpected:
                print(f"[TanhGaussianMLPPolicy] unexpected keys on load: {unexpected}", flush=True)
            self.to(self.device)

    def effective_log_std(self) -> torch.Tensor:
        """Return the ``(action_dim,)`` log-sigma used for sampling.

        Only a numerical-safety clamp at ±20 is applied — no business
        bounds.  ``log_std_min`` / ``log_std_max`` are normalization
        reference points, not hard limits.  See
        ``DESIGN_migration_tanh_gaussian.md`` §3.
        """
        return torch.clamp(
            self.log_std + self._log_std_offset,
            _LOG_STD_SAFE_MIN,
            _LOG_STD_SAFE_MAX,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.effective_log_std()
        return mean, log_std.expand_as(mean)

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        frame_modes: Optional[torch.Tensor] = None,
        noise_shift: Optional[torch.Tensor] = None,
        want_stats: bool = False,
    ) -> "ActorEval":
        """Score ``actions`` under the current parameters.

        Implements the :class:`~baseline.framework.ppo.experiment.TrainablePolicy`
        contract.  ``frame_modes`` and ``noise_shift`` are accepted and
        ignored — this baseline backbone has no sub-network routing and
        no OU exploration support.  Use :class:`FixedSigmaGaussianMLPPolicy`
        for OU-enabled training from a baseline checkpoint.

        ``log_prob`` uses the effective σ (policy σ + explore offset) so
        the PPO importance ratio is correct.  ``entropy`` uses the
        policy's original σ (without explore offset) so it reflects the
        policy's own certainty, not the temporary exploration noise.
        See ``DESIGN_migration_tanh_gaussian.md`` §2.
        """
        # log_prob: effective σ (policy σ + explore offset)
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(clipped_actions)
        mean, eff_log_std = self.forward(obs)
        dist = Normal(mean, eff_log_std.exp())
        log_prob = (dist.log_prob(raw_actions)
                    - torch.log(1.0 - clipped_actions.pow(2) + 1e-6)).sum(dim=-1)

        # entropy: policy's original σ (no explore offset)
        policy_log_std = self.log_std
        entropy_raw = Normal(mean, policy_log_std.exp()).entropy().sum(dim=-1)
        H_max = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_max)
        H_min = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_min)
        entropy_norm = (entropy_raw - H_min) / (H_max - H_min)

        stats: Optional[Dict[str, float]] = None
        if want_stats:
            with torch.no_grad():
                policy_std = policy_log_std.exp()
                eff_std = eff_log_std.exp()
                stats = {
                    "entropy_raw": float(entropy_raw.mean().item()),
                    "std_mean": float(policy_std.mean().item()),
                    "eff_std_mean": float(eff_std.mean().item()),
                    "std_min": float(policy_std.min().item()),
                    "std_max": float(policy_std.max().item()),
                    # Fraction of pre-tanh means in the saturated region.
                    "tanh_sat_frac": float(
                        (mean.abs() > 2.0).float().mean().item()
                    ),
                }
        return ActorEval(
            log_prob=log_prob,
            entropy=entropy_norm,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Exploration contract
    # ------------------------------------------------------------------

    # Span of the exploration offset in log-std space.
    # At ei=0 or ei=1, offset = ∓EXPLORE_SPAN / 2, so σ is scaled by
    # exp(∓EXPLORE_SPAN / 2).  With span=2.0 this gives ≈ 0.37x ~ 2.72x,
    # a practical range that avoids extreme saturation.
    EXPLORE_SPAN = 2.0

    def set_exploration(self, explore_intensity: float) -> None:
        """Apply an exploration directive.

        ``explore_intensity`` ∈ [0, 1] is a symmetric temperature-like
        control centered at 0.5:

        - ``0.5`` = neutral (offset=0, policy uses its learned σ as-is)
        - ``→ 0`` = compress (offset < 0, σ shrinks; ei=0 → σ × ~0.37)
        - ``→ 1`` = expand (offset > 0, σ grows; ei=1 → σ × ~2.72)

        The offset is additive on log_std: ``effective_log_std = log_std
        + offset``, which is mathematically equivalent to multiplying σ
        by ``exp(offset)`` (a temperature scaling).  Operating in log
        space keeps the offset linear in entropy and numerically stable.

        **Warning**: ``ei=0`` nearly collapses sampling noise.  Only use
        it deliberately; the safe default is ``0.5``.
        """
        self._log_std_offset = (float(explore_intensity) - 0.5) * self.EXPLORE_SPAN

    # ------------------------------------------------------------------
    # Policy contract
    # ------------------------------------------------------------------
    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Single-step inference.

        Returns ``(action, None)`` when ``want_extra=False``.
        When ``want_extra=True`` and the policy is stochastic, the
        returned dict contains ``log_prob``.
        """
        action_np, log_prob = self.act_numpy(
            observation, device=self.device, deterministic=self._deterministic
        )
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {"log_prob": float(log_prob)}

    def set_deterministic(self, deterministic: bool) -> None:
        """Toggle stochastic vs deterministic action sampling."""
        self._deterministic = bool(deterministic)

    def to_blueprint(
        self, dest_path: Optional[str] = None, *, stochastic: bool = False,
    ) -> "PolicyBlueprint":
        """Export this policy to a deployable :class:`PolicyBlueprint`.

        Writes ``model.pt`` + ``policy.py`` (standalone, no repo deps) into
        ``dest_path`` and returns a blueprint that rebuilds the policy via
        the generated ``ExportedMLPPolicy`` class. When ``dest_path`` is
        ``None`` a temporary directory is used.

        Args:
            stochastic: If True, the exported blueprint uses stochastic
                sampling (for training rollouts).  If False (default),
                it uses deterministic mean actions (for evaluation).
        """
        import tempfile

        if dest_path is None:
            dest_path = tempfile.mkdtemp(prefix="policy_export_")
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Export full artifacts: model.pt + standalone policy.py + blueprint.yaml
        from .checkpoint import export_actor_policy_artifacts

        export_actor_policy_artifacts(
            actor=self,
            policy_dir=policy_dir,
            stochastic=stochastic,
        )

        # Return blueprint pointing to the generated standalone policy.py
        policy_py_path = policy_dir / "policy.py"
        return PolicyBlueprint(
            cls=f"file:{policy_py_path}:ExportedMLPPolicy",
            config={"stochastic": stochastic},
        )

    # ------------------------------------------------------------------
    # Numpy-flavoured inference (kept for backward compat with trainers)
    # ------------------------------------------------------------------
    def act_numpy(self, obs: np.ndarray, device: torch.device, deterministic: bool) -> tuple[np.ndarray, Optional[float]]:
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


