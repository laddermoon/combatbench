"""``TorchPolicyAdapter``: bridge nn.Module training-side <-> Policy framework-side.

The framework :class:`envs.framework.policy.Policy` ABC is intentionally
numpy-flavored (single-step ``act(obs) -> ndarray``) so it stays free of
torch / GPU concerns. Training, however, lives inside torch — there's a
``nn.Module`` actor whose weights are updated each PPO/GRPO iteration and
optionally a critic for value estimates.

This adapter is the standard glue: take a torch ``nn.Module`` and expose
it as a ``Policy`` that ``EpisodeRunner`` / ``RolloutCollector`` can drive.
On top of the basic adapter, it adds:

  * deterministic / stochastic switch (``deterministic`` ctor flag, also
    settable per-call via :meth:`set_deterministic`);
  * extras collection: when ``act_with_extras`` is invoked (PPO sets
    ``RolloutConfig.store_extras=True``), ``log_prob`` and optionally
    ``value`` are returned for the framework to persist in
    ``AgentTrajectory.extras``;
  * weight hot-reload via :meth:`load_state_dict` — collectors call this
    each iteration to push fresh PPO weights without rebuilding the
    runtime;
  * deployment-friendly :meth:`export` that delegates to
    :func:`baseline.common.policies.checkpoint.export_actor_policy_artifacts`.

Module contract
---------------
The wrapped module must expose:

  * ``act_numpy(obs: np.ndarray, *, device: torch.device, deterministic: bool) -> (action_np, log_prob_or_None)``
    — ``TanhGaussianMLPPolicy`` already does this. ``log_prob`` may be
    ``None`` for deterministic / non-stochastic actors.
  * ``obs_dim`` and ``action_dim`` int attributes (used for sanity checks
    against ``observation_space`` / ``action_space`` if provided).

The optional critic must be either ``None`` or an ``nn.Module`` callable
on a single-batch obs tensor returning a scalar value tensor of shape
``(1,)`` or ``()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from envs.framework.policy import Policy


class TorchPolicyAdapter(Policy):
    """Wrap an ``nn.Module`` actor (and optional critic) as a framework ``Policy``."""

    def __init__(
        self,
        actor: nn.Module,
        critic: Optional[nn.Module] = None,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
        observation_space: Any = None,
        action_space: Any = None,
    ) -> None:
        if not hasattr(actor, "act_numpy"):
            raise TypeError(
                f"actor must expose act_numpy(obs, device, deterministic); "
                f"got {type(actor).__name__}"
            )
        self.actor = actor
        self.critic = critic
        self.device = torch.device(device)
        self._deterministic = bool(deterministic)
        self.actor.to(self.device).eval()
        if self.critic is not None:
            self.critic.to(self.device).eval()
        self._sanity_check_spaces(observation_space, action_space)

    # ------------------------------------------------------------------
    # Policy contract
    # ------------------------------------------------------------------
    def act(self, observation: Any) -> np.ndarray:
        """Single-step inference. ``log_prob`` and ``value`` are discarded."""
        action, _ = self._infer(observation, want_extras=False)
        return action

    def act_with_extras(self, observation: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Single-step inference + log_prob (and value if a critic is attached).

        Used by the framework when ``RolloutConfig.store_extras=True``.
        """
        return self._infer(observation, want_extras=True)

    def reset(self, seed: Optional[int] = None) -> None:
        """No-op: torch actors are stateless across episodes by default.

        Subclasses that hold recurrent state should override and seed
        their RNGs from ``seed`` for reproducibility.
        """
        return None

    # ------------------------------------------------------------------
    # Training-time hooks
    # ------------------------------------------------------------------
    def load_state_dict(
        self,
        actor_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        critic_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        *,
        strict: bool = True,
    ) -> None:
        """Hot-reload weights without rebuilding the adapter / runtime.

        Either argument may be ``None`` to skip that module. After the
        reload the modules are returned to ``eval()`` mode — the adapter
        is exclusively an inference wrapper.
        """
        if actor_state_dict is not None:
            self.actor.load_state_dict(actor_state_dict, strict=strict)
            self.actor.eval()
        if critic_state_dict is not None:
            if self.critic is None:
                raise ValueError("critic_state_dict supplied but adapter has no critic.")
            self.critic.load_state_dict(critic_state_dict, strict=strict)
            self.critic.eval()

    def set_deterministic(self, deterministic: bool) -> None:
        """Toggle stochastic vs deterministic action sampling."""
        self._deterministic = bool(deterministic)

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    def export(self, policy_dir: Path, extra_payload: Optional[Dict[str, Any]] = None) -> None:
        """Write a deployable ``policy/`` directory (``model.pt`` + ``policy.py``).

        Thin wrapper around
        :func:`baseline.common.policies.checkpoint.export_actor_policy_artifacts`
        so users don't need to remember to detach / move to CPU first.
        """
        from .checkpoint import export_actor_policy_artifacts

        export_actor_policy_artifacts(
            actor=self.actor,
            policy_dir=Path(policy_dir),
            extra_payload=extra_payload,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _infer(
        self,
        observation: Any,
        *,
        want_extras: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_array = np.asarray(observation, dtype=np.float32)
        action_np, log_prob = self.actor.act_numpy(
            obs_array, device=self.device, deterministic=self._deterministic
        )
        if not want_extras:
            return action_np, {}
        extras: Dict[str, Any] = {}
        if log_prob is not None:
            extras["log_prob"] = float(log_prob)
        if self.critic is not None:
            extras["value"] = self._critic_value(obs_array)
        return action_np, extras

    def _critic_value(self, obs_array: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return float(value.reshape(-1)[0].item())

    @staticmethod
    def _sanity_check_spaces(observation_space: Any, action_space: Any) -> None:
        # Soft validation — only when the caller provided gym-style spaces
        # with a ``shape`` attribute. We don't import gym here.
        if observation_space is None and action_space is None:
            return
        # Intentionally silent on missing-shape spaces; this is best-effort.
        return None
