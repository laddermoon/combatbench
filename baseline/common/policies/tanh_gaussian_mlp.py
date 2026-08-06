"""Continuous-control actor backbone (Tanh-squashed Gaussian).

Checkpoint / export IO has moved to :mod:`baseline.common.policies.checkpoint`
and is re-exported from this module for backward compatibility with scripts
that imported it from here (pre-PR1). New code should import directly from
:mod:`baseline.common.policies.checkpoint`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from envs.framework.policy import Policy, PolicyBlueprint

from .checkpoint import (
    DEFAULT_EXPORT_ACTOR_HIDDEN_DIM,
    build_actor_export_payload,
    build_export_policy_code,
    export_actor_policy_artifacts,
    export_policy_artifacts_from_checkpoint,
)

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 1.0

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

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
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

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(clipped_actions)
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw_actions) - torch.log(1.0 - clipped_actions.pow(2) + 1e-6)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob.sum(dim=-1), entropy

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


