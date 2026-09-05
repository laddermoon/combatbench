"""Generic, family-agnostic policy export / import.

This module replaces the Gaussian-specific :mod:`checkpoint.py` path for
new policy families.  The key differences from ``checkpoint.py`` are:

1. **``strict=True`` state-dict reload.**  ``checkpoint.py`` uses
   ``strict=False``, which silently swallows missing/unexpected keys
   — exactly the failure mode that made the ``log_std_min`` export bug
   invisible.  For new families with shape hyperparameters (K, rank,
   flow depth), a wrong hyperparameter means a wrong module structure,
   which means a wrong state-dict shape, which must surface as a loud
   crash, not a silent partial load.

2. **Full constructor config in the payload.**  The family's
   ``export_config()`` returns every constructor kwarg needed to
   rebuild the module (obs_dim, action_dim, hidden_dim, K, rank,
   num_layers, scale_max, log_std_min, log_std_max, ...).  The
   generated ``policy.py`` uses these to instantiate the correct class
   before loading weights.

3. **Family-agnostic code generation.**  The generated ``policy.py``
   imports the family's class from the repo (via
   ``policy_class_path``) and instantiates it with ``config``.  No
   hardcoded class name, no hardcoded constructor kwargs.

The existing ``checkpoint.py`` is left untouched for
``TanhGaussianMLPPolicy``'s continued use.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from envs.framework.policy import PolicyBlueprint


def build_generic_export_payload(
    actor: nn.Module,
    *,
    policy_class_path: str,
    config: Dict[str, Any],
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``model.pt`` payload for a generic policy export.

    Args:
        actor: The live policy module (must have ``state_dict()``).
        policy_class_path: Dotted path like
            ``"baseline.framework.ppo.policies.mog_tanh_mlp:MoGTanhMLPPolicy"``.
        config: Full constructor kwargs needed to rebuild the module.
        extra_payload: Optional additional scalar fields (e.g.
            ``temperature``, ``entropy_coef``).

    Returns:
        Payload dict suitable for ``torch.save``.
    """
    payload: Dict[str, Any] = dict(extra_payload or {})
    payload["policy_class_path"] = str(policy_class_path)
    payload["config"] = dict(config)
    payload["state_dict"] = {
        key: value.detach().cpu()
        for key, value in actor.state_dict().items()
    }
    return payload


def build_generic_export_policy_code() -> str:
    """Return the source of the ``policy.py`` for generic exports.

    The generated module defines ``ExportedPolicy`` that:
    1. Loads the ``model.pt`` payload.
    2. Imports the policy class from ``policy_class_path``.
    3. Instantiates it with ``payload["config"]``.
    4. Loads ``payload["state_dict"]`` with ``strict=True``.
    5. Applies ``temperature`` and OU noise params from the payload.
    6. Delegates ``act`` to the inner policy so OU stepping, extras,
       and stochastic/deterministic dispatch are handled in one place.
    7. Forwards ``reset`` to the inner policy so OU state is zeroed
       at episode boundaries.
    """
    return '''"""Policy module - generic export (auto-generated)."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import importlib

import numpy as np
import torch

from envs.framework.policy import Policy


class ExportedPolicy(Policy):
    """Runtime-loadable policy backed by a generic model.pt checkpoint.

    Reconstructs the policy from the recorded class path and constructor
    config, then loads weights with strict=True so any architecture
    mismatch surfaces as a loud crash rather than a silent partial load.

    ``act`` delegates to the inner policy so that OU noise stepping,
    extras collection, and stochastic/deterministic dispatch are all
    handled by the policy's own ``act`` method — there is no duplicate
    sampling logic here that could drift out of sync.

    ``reset`` forwards to the inner policy so OU state (and any other
    per-episode RNG state) is zeroed at episode boundaries.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stochastic: bool = False,
        **_ignored: Any,
    ):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.pt"
        payload = torch.load(payload_path, map_location="cpu")

        # Resolve and instantiate the policy class.
        class_path = payload["policy_class_path"]
        module_path, cls_name = class_path.split(":")
        module = importlib.import_module(module_path.strip())
        cls = getattr(module, cls_name.strip())

        config = dict(payload["config"])
        # OU noise parameters are still baked into the exported policy
        # (they are architecture-level, not per-step exploration state).
        if "noise_tau_steps" in payload:
            config.setdefault("noise_tau_steps", float(payload["noise_tau_steps"]))
        if "noise_scale" in payload:
            config.setdefault("noise_scale", float(payload["noise_scale"]))

        self._policy = cls(**config)

        # strict=True: any key mismatch is a loud crash, not a silent
        # partial load.  This is the whole point of the generic export
        # path — a wrong K / rank / num_layers on reload must fail
        # immediately, not produce a silently wrong policy.
        self._policy.load_state_dict(payload["state_dict"], strict=True)
        self._policy.eval()
        self._policy.set_deterministic(not stochastic)
        self.stochastic = bool(stochastic)

    def act(
        self,
        observation: Any,
        explore_intensity: float = 0.5,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        # Delegate to the inner policy's act so OU stepping, extras,
        # and stochastic/deterministic dispatch are handled in one
        # place.  This avoids a second copy of the sampling logic that
        # could silently diverge from the policy's own act method.
        return self._policy.act(
            observation, explore_intensity=explore_intensity, want_extra=want_extra,
        )

    def reset(self, seed: Optional[int] = None) -> None:
        # Forward to the inner policy so OU state is zeroed at episode
        # boundaries and the per-agent seed reseeds its RNG.
        self._policy.reset(seed)
'''


def export_generic_policy_artifacts(
    actor: nn.Module,
    policy_dir: Path,
    *,
    policy_class_path: str,
    config: Dict[str, Any],
    stochastic: bool = False,
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write ``model.pt`` + ``policy.py`` + ``policy_blueprint.yaml``.

    Args:
        actor: The live policy module.
        policy_dir: Destination directory (created if needed).
        policy_class_path: Dotted path to the policy class.
        config: Full constructor kwargs for round-trip.
        stochastic: Whether the exported blueprint uses stochastic sampling.
        extra_payload: Optional additional payload fields.
    """
    policy_dir.mkdir(parents=True, exist_ok=True)
    export_payload = build_generic_export_payload(
        actor=actor,
        policy_class_path=policy_class_path,
        config=config,
        extra_payload=extra_payload,
    )
    torch.save(export_payload, policy_dir / "model.pt")
    policy_code = build_generic_export_policy_code()
    with (policy_dir / "policy.py").open("w", encoding="utf-8") as handle:
        handle.write(policy_code)

    policy_py_path = policy_dir / "policy.py"
    blueprint = PolicyBlueprint(
        cls=f"file:{policy_py_path}:ExportedPolicy",
        config={
            "stochastic": stochastic,
        },
    )
    blueprint.save(policy_dir / "policy_blueprint.yaml")
