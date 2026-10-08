"""S01 actor: single-component Gaussian with bounded shared σ.

This is the canonical first production actor for SAC.  It is independent
of PPO policy classes and exports a SAC-native runtime policy blueprint.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from envs.framework.policy import PolicyBlueprint


LOG_PROB_EPS = 1e-6


class S01Actor(nn.Module):
    """Tanh-squashed Gaussian actor with one shared bounded σ parameter."""

    policy_arch = "s01_shared_sigma"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        log_std_min: float = -4.0,
        log_std_max: float = 0.0,
        init_log_std: float = -0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if int(obs_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        if not float(log_std_min) < float(log_std_max):
            raise ValueError("log_std_min must be < log_std_max")
        if not float(log_std_min) <= float(init_log_std) <= float(log_std_max):
            raise ValueError("init_log_std must be inside [log_std_min, log_std_max]")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.init_log_std = float(init_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.log_std = nn.Parameter(torch.tensor(self.init_log_std, dtype=torch.float32))
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

    def _noise(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(
            shape,
            generator=self._rng,
            device="cpu",
            dtype=dtype,
        ).to(device)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return mean, std

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiably sample tanh-squashed action and joint log-density."""
        mean, std = self.forward(obs)
        raw = mean + std * self._noise(mean.shape, mean.device, mean.dtype)
        action = torch.tanh(raw)
        log_prob = torch.distributions.Normal(mean, std).log_prob(raw)
        correction = torch.log(1.0 - action.pow(2) + LOG_PROB_EPS)
        return action, (log_prob - correction).sum(dim=-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate joint log-density and deterministic entropy proxy."""
        mean, std = self.forward(obs)
        clipped = actions.clamp(-1.0 + LOG_PROB_EPS, 1.0 - LOG_PROB_EPS)
        raw = torch.atanh(clipped)
        log_prob = torch.distributions.Normal(mean, std).log_prob(raw)
        correction = torch.log(1.0 - clipped.pow(2) + LOG_PROB_EPS)
        entropy = torch.distributions.Normal(mean, std).entropy().sum(dim=-1)
        return (log_prob - correction).sum(dim=-1), entropy

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def policy_fingerprint(self) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            h.update(name.encode("utf-8"))
            h.update(tensor.detach().cpu().numpy().tobytes())
        h.update(self.policy_arch.encode("utf-8"))
        h.update(str(self.obs_dim).encode("ascii"))
        h.update(str(self.action_dim).encode("ascii"))
        return h.hexdigest()

    def state_payload(self) -> Dict[str, Any]:
        return {
            "policy_arch": self.policy_arch,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
            "init_log_std": self.init_log_std,
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }

    def to_blueprint(
        self,
        dest_path: Optional[str] = None,
        *,
        stochastic: bool = False,
    ) -> PolicyBlueprint:
        dest = Path(dest_path) if dest_path is not None else Path(tempfile.mkdtemp(prefix="sac_s01_"))
        self.export_policy_artifacts(dest, stochastic=stochastic)
        return PolicyBlueprint(
            cls="baseline.framework.sac.s01_actor:S01RuntimePolicy",
            config={
                "model_path": str(dest / "model.pt"),
                "stochastic": bool(stochastic),
            },
        )

    def export_policy_artifacts(
        self,
        dest_dir: Path,
        *,
        stochastic: bool = False,
    ) -> None:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_payload(), dest / "model.pt")
        bp = PolicyBlueprint(
            cls="baseline.framework.sac.s01_actor:S01RuntimePolicy",
            config={
                "model_path": str(dest / "model.pt"),
                "stochastic": bool(stochastic),
            },
        )
        bp.save(dest / "policy_blueprint.yaml")


class S01RuntimePolicy:
    """Runtime policy loaded from an exported S01 ``model.pt``."""

    def __init__(
        self,
        model_path: str,
        stochastic: bool = False,
        seed: int = 0,
        **_ignored: Any,
    ) -> None:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if payload.get("policy_arch") != S01Actor.policy_arch:
            raise ValueError(
                f"unsupported S01 runtime policy_arch {payload.get('policy_arch')!r}"
            )
        self.stochastic = bool(stochastic)
        self._actor = S01Actor(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            log_std_min=float(payload["log_std_min"]),
            log_std_max=float(payload["log_std_max"]),
            init_log_std=float(payload.get("init_log_std", -0.5)),
            seed=int(seed),
        )
        self._actor.load_state_dict(payload["state_dict"])
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
                action, log_prob = self._actor.sample_action(obs)
                extra = {"log_prob": float(log_prob.item())} if want_extra else None
            else:
                action = self._actor.deterministic_action(obs)
                extra = {"log_prob": None} if want_extra else None
        return action.squeeze(0).cpu().numpy().astype(np.float32), extra

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._actor._rng.manual_seed(int(seed))

    def close(self) -> None:
        return None


__all__ = ["S01Actor", "S01RuntimePolicy", "LOG_PROB_EPS"]
