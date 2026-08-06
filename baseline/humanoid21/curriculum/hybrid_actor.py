"""Hybrid standup + balance actor with deterministic uprightness-based routing.

Contains:
  * HybridActor — trainable nn.Module implementing TrainablePolicy protocol.
    Holds two TanhGaussianMLPPolicy sub-networks (standup_net, balance_net).
    evaluate_actions() routes each sample to the correct sub-network based
    on uprightness computed from the observation.

  * HybridRolloutPolicy — envs.framework.policy.Policy for rollout collection.
    Maintains a stateful mode (standup/balance) with hysteresis and routes
    act() calls to the appropriate sub-network.

Uprightness is computed from local_orientation in the 96-dim observation:
  obs[42:48] = [r00, r10, r20, r01, r11, r21]  (first 2 columns of rot mat)
  uprightness = r22 = r00*r11 - r10*r01 = obs[42]*obs[46] - obs[43]*obs[45]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy
from baseline.common.policies.checkpoint import build_export_policy_code
from envs.framework.policy import Policy, PolicyBlueprint

# Observation indices for uprightness computation
_OBS_R00 = 42
_OBS_R10 = 43
_OBS_R01 = 45
_OBS_R11 = 46


def compute_uprightness(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Compute uprightness (cos of torso tilt angle) from 96-dim observation.

    uprightness = r00*r11 - r10*r01  (third column of rotation matrix, z-component)
    """
    if isinstance(obs, torch.Tensor):
        return obs[:, _OBS_R00] * obs[:, _OBS_R11] - obs[:, _OBS_R10] * obs[:, _OBS_R01]
    return obs[..., _OBS_R00] * obs[..., _OBS_R11] - obs[..., _OBS_R10] * obs[..., _OBS_R01]


class HybridActor(nn.Module):
    """Trainable hybrid actor with two sub-networks and deterministic routing.

    Implements the TrainablePolicy protocol:
      - evaluate_actions(obs, actions) -> (log_prob, entropy)
      - to_blueprint(dest_path) -> PolicyBlueprint
      - parameters() (inherited from nn.Module)

    Routing is based on uprightness: samples with uprightness >= switch_threshold
    go to balance_net, others go to standup_net. This is deterministic and
    differentiable (no gradient through routing itself).
    """

    def __init__(
        self,
        obs_dim: int = 96,
        action_dim: int = 21,
        hidden_dim: int = 256,
        log_std_min: float = -4.0,
        log_std_max: float = 0.0,
        switch_uprightness: float = 0.97,
        standup_model_path: Optional[str] = None,
        balance_model_path: Optional[str] = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.switch_uprightness = float(switch_uprightness)

        self.standup_net = TanhGaussianMLPPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=device,
        )
        self.balance_net = TanhGaussianMLPPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=device,
        )

        # Load pretrained weights if provided
        if standup_model_path is not None:
            payload = torch.load(standup_model_path, map_location="cpu")
            sd = payload.get("actor_state_dict", payload.get("state_dict", payload))
            self.standup_net.load_state_dict(sd, strict=False)
            print(f"[HybridActor] Loaded standup weights from {standup_model_path}", flush=True)

        if balance_model_path is not None:
            payload = torch.load(balance_model_path, map_location="cpu")
            sd = payload.get("actor_state_dict", payload.get("state_dict", payload))
            self.balance_net.load_state_dict(sd, strict=False)
            print(f"[HybridActor] Loaded balance weights from {balance_model_path}", flush=True)

        self.to(device)

    @property
    def log_std(self) -> torch.Tensor:
        """Return concatenated log_std from both networks for PPO diagnostics."""
        return torch.cat([self.standup_net.log_std, self.balance_net.log_std])

    def _route_mask(self, obs: torch.Tensor) -> torch.Tensor:
        """Return boolean mask: True = balance, False = standup."""
        upright = obs[:, _OBS_R00] * obs[:, _OBS_R11] - obs[:, _OBS_R10] * obs[:, _OBS_R01]
        return upright >= self.switch_uprightness

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
        *, frame_modes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given obs/actions pairs.

        Routes each sample to the appropriate sub-network.  If
        ``frame_modes`` is provided, use it directly (0.0=balance,
        1.0=standup) instead of computing uprightness from the
        observation.  This ensures routing consistency with the
        rollout policy's hysteresis mode.
        """
        if frame_modes is not None:
            # mode=1.0 → standup, mode=0.0 → balance
            balance_mask = frame_modes < 0.5
        else:
            balance_mask = self._route_mask(obs)  # (N,)
        standup_mask = ~balance_mask

        log_probs = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
        entropies = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)

        if standup_mask.any():
            lp_s, ent_s = self.standup_net.evaluate_actions(
                obs[standup_mask], actions[standup_mask]
            )
            log_probs[standup_mask] = lp_s
            entropies[standup_mask] = ent_s

        if balance_mask.any():
            lp_b, ent_b = self.balance_net.evaluate_actions(
                obs[balance_mask], actions[balance_mask]
            )
            log_probs[balance_mask] = lp_b
            entropies[balance_mask] = ent_b

        return log_probs, entropies

    def to_blueprint(
        self, dest_path: str, *, stochastic: bool = False,
    ) -> PolicyBlueprint:
        """Export both sub-networks and a custom policy.py that routes at inference.

        Writes model.pt (with both state dicts) and a standalone policy.py
        that implements the hybrid routing logic.

        Args:
            stochastic: If True, the exported blueprint uses stochastic
                sampling (for training rollouts).  If False (default),
                it uses deterministic mean actions (for evaluation).
        """
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Save combined model
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "switch_uprightness": self.switch_uprightness,
            "standup_state_dict": {
                k: v.detach().cpu() for k, v in self.standup_net.state_dict().items()
            },
            "balance_state_dict": {
                k: v.detach().cpu() for k, v in self.balance_net.state_dict().items()
            },
        }
        torch.save(payload, policy_dir / "model.pt")

        # Write standalone policy.py with routing logic
        policy_code = _build_hybrid_export_policy_code()
        with (policy_dir / "policy.py").open("w", encoding="utf-8") as f:
            f.write(policy_code)

        blueprint = PolicyBlueprint(
            cls=f"file:{policy_dir / 'policy.py'}:ExportedHybridPolicy",
            config={"stochastic": stochastic},
        )
        blueprint.save(policy_dir / "policy_blueprint.yaml")

        return blueprint

    def export_policy_artifacts(
        self,
        policy_dir: str | Path,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write model.pt + policy.py + policy_blueprint.yaml for deployment.

        Compatible with the training loop's best-of-run snapshot logic.
        """
        policy_dir = Path(policy_dir)
        policy_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "switch_uprightness": self.switch_uprightness,
            "standup_state_dict": {
                k: v.detach().cpu() for k, v in self.standup_net.state_dict().items()
            },
            "balance_state_dict": {
                k: v.detach().cpu() for k, v in self.balance_net.state_dict().items()
            },
        }
        if extra_payload:
            payload.update(extra_payload)
        torch.save(payload, policy_dir / "model.pt")

        policy_code = _build_hybrid_export_policy_code()
        with (policy_dir / "policy.py").open("w", encoding="utf-8") as f:
            f.write(policy_code)

        blueprint = PolicyBlueprint(
            cls=f"file:{policy_dir / 'policy.py'}:ExportedHybridPolicy",
            config={"stochastic": False},
        )
        blueprint.save(policy_dir / "policy_blueprint.yaml")


def _build_hybrid_export_policy_code() -> str:
    """Return the source code for the exported hybrid policy.py."""
    return '''"""Hybrid standup+balance policy with uprightness-based routing."""
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy
from envs.framework.policy import Policy

_OBS_R00 = 42
_OBS_R10 = 43
_OBS_R01 = 45
_OBS_R11 = 46


def _compute_uprightness(obs: np.ndarray) -> float:
    return float(obs[_OBS_R00] * obs[_OBS_R11] - obs[_OBS_R10] * obs[_OBS_R01])


class ExportedHybridPolicy(Policy):
    """Runtime-loadable hybrid policy with two sub-networks.

    Routes to standup_net when uprightness < switch_threshold,
    to balance_net otherwise. Maintains stateful mode with hysteresis:
    once in balance mode, only switches back to standup when uprightness
    drops below fall_threshold.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stochastic: bool = False,
        switch_uprightness: float = 0.97,
        fall_uprightness: float = 0.30,
        **_ignored: Any,
    ):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.pt"
        payload = torch.load(payload_path, map_location="cpu")

        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        hidden_dim = int(payload["hidden_dim"])
        self.switch_uprightness = float(payload.get("switch_uprightness", switch_uprightness))
        self.fall_uprightness = float(fall_uprightness)
        self.stochastic = bool(stochastic)

        self.standup_net = TanhGaussianMLPPolicy(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        self.standup_net.load_state_dict(payload["standup_state_dict"], strict=False)
        self.standup_net.eval()

        self.balance_net = TanhGaussianMLPPolicy(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        self.balance_net.load_state_dict(payload["balance_state_dict"], strict=False)
        self.balance_net.eval()

        self._mode = "standup"

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        obs_array = np.asarray(observation, dtype=np.float32)
        upright = _compute_uprightness(obs_array)

        # State machine with hysteresis
        if self._mode == "standup" and upright >= self.switch_uprightness:
            self._mode = "balance"
        elif self._mode == "balance" and upright < self.fall_uprightness:
            self._mode = "standup"

        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        net = self.balance_net if self._mode == "balance" else self.standup_net
        with torch.no_grad():
            if self.stochastic:
                action, _ = net.sample_action(obs_tensor)
            else:
                action = net.deterministic_action(obs_tensor)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        extra = {"mode": 1.0 if self._mode == "standup" else 0.0, "uprightness": upright}
        return action_np, extra

    def reset(self, seed: Optional[int] = None) -> None:
        self._mode = "standup"
'''


class HybridRolloutPolicy(Policy):
    """Rollout policy wrapping a HybridActor blueprint with stateful routing.

    Used during training rollouts. Loads the exported hybrid policy from
    a blueprint directory and maintains the standup/balance mode state.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stochastic: bool = False,
        switch_uprightness: float = 0.97,
        fall_uprightness: float = 0.30,
        **_ignored: Any,
    ):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.pt"
        payload = torch.load(payload_path, map_location="cpu")

        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        hidden_dim = int(payload["hidden_dim"])
        self.switch_uprightness = float(payload.get("switch_uprightness", switch_uprightness))
        self.fall_uprightness = float(fall_uprightness)
        self.stochastic = bool(stochastic)

        self.standup_net = TanhGaussianMLPPolicy(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        self.standup_net.load_state_dict(payload["standup_state_dict"], strict=False)
        self.standup_net.eval()

        self.balance_net = TanhGaussianMLPPolicy(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        self.balance_net.load_state_dict(payload["balance_state_dict"], strict=False)
        self.balance_net.eval()

        self._mode = "standup"

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        obs_array = np.asarray(observation, dtype=np.float32)
        upright = float(
            obs_array[_OBS_R00] * obs_array[_OBS_R11]
            - obs_array[_OBS_R10] * obs_array[_OBS_R01]
        )

        # State machine with hysteresis
        if self._mode == "standup" and upright >= self.switch_uprightness:
            self._mode = "balance"
        elif self._mode == "balance" and upright < self.fall_uprightness:
            self._mode = "standup"

        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        net = self.balance_net if self._mode == "balance" else self.standup_net
        with torch.no_grad():
            if self.stochastic:
                action, log_prob = net.sample_action(obs_tensor)
            else:
                action = net.deterministic_action(obs_tensor)
                log_prob = None

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

        extra = {
            "mode": 1.0 if self._mode == "standup" else 0.0,
            "uprightness": upright,
        }
        if log_prob is not None:
            extra["log_prob"] = float(log_prob.item())

        if not want_extra:
            return action_np, None
        return action_np, extra

    def reset(self, seed: Optional[int] = None) -> None:
        self._mode = "standup"
