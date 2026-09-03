"""Training-side <-> deployment-side policy IO.

Single source of truth for:
  * ``build_actor_export_payload(actor, extra_payload=...)``:
    Produces the ``model.pt`` payload written into a ``policy/`` directory.
  * ``build_export_policy_code()``:
    Returns the ``policy.py`` source embedded into an exported
    ``policy/`` directory; this is the code that ``load_policy(...)``
    imports and instantiates at deployment time.
  * ``export_actor_policy_artifacts(actor, policy_dir, extra_payload=...)``:
    Writes both ``model.pt`` and ``policy.py`` under ``policy_dir``.
  * ``export_policy_artifacts_from_checkpoint(model_path, policy_dir, ...)``:
    Same end result but sourced from an on-disk training checkpoint
    instead of a live ``nn.Module``.

These used to live at the tail of ``tanh_gaussian_mlp.py``. They are
moved here (PR1 / B5) so every baseline / algorithm can share a single
checkpoint contract without dragging the backbone definition as a
dependency. The backbone file re-exports the same names for backward
compatibility.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from envs.framework.policy import PolicyBlueprint

DEFAULT_EXPORT_ACTOR_HIDDEN_DIM = 256


def build_export_policy_code() -> str:
    """Return the source of the ``policy.py`` embedded in ``policy/`` dirs.

    The produced module defines ``ExportedMLPPolicy`` that reuses
    :class:`baseline.common.policies.tanh_gaussian_mlp.TanhGaussianMLPPolicy`
    from the repo. Requires the repo to be on ``sys.path`` (e.g., via
    PYTHONPATH=. when running).

    This eliminates code duplication by importing the training-time class
    directly rather than re-implementing the architecture.
    """
    return '''"""Policy module - imports from repo to reuse TanhGaussianMLPPolicy."""
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

# Import from repo - requires baseline/ to be on sys.path
from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy
from envs.framework.policy import Policy


class ExportedMLPPolicy(Policy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Uses :class:`TanhGaussianMLPPolicy` from the training repo for
    consistent architecture and behavior.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stochastic: bool = False,
        **_ignored: Any,
    ):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.pt"
        payload = torch.load(payload_path, map_location="cpu")

        hidden_dim = int(payload.get("hidden_dim", payload.get("actor_hidden_dim", 256)))

        # Reuse training-time policy class (no code duplication).
        # log_std_min/max/offset come from the payload so rollout sampling
        # matches the sigma the trainer scores actions with; the defaults
        # are only a fallback for pre-existing checkpoints.
        self._policy = TanhGaussianMLPPolicy(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=hidden_dim,
            log_std_min=float(payload.get("log_std_min", -4.0)),
            log_std_max=float(payload.get("log_std_max", 0.0)),
        )
        self._policy.load_state_dict(payload["state_dict"], strict=False)
        self._policy.eval()
        self.stochastic = bool(stochastic)

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Return action for given observation."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.stochastic:
                action, _ = self._policy.sample_action(obs_tensor)
            else:
                action = self._policy.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def reset(self, seed: Optional[int] = None) -> None:
        """Optional: reseed RNG for reproducible rollouts."""
        if seed is not None:
            torch.manual_seed(seed)
        return None


# Backward compatibility alias
Policy = ExportedMLPPolicy
'''


def build_actor_export_payload(
    actor: nn.Module,
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``model.pt`` payload for a trained actor.

    Infers ``obs_dim`` / ``action_dim`` / ``hidden_dim`` from standard
    attributes when present, falling back to inspecting the first / last
    Linear layers of ``actor.net`` (the convention used by
    :class:`TanhGaussianMLPPolicy`). ``log_std`` is deliberately stripped
    — deployment only needs the deterministic (mean-tanh) path.
    """
    export_payload: Dict[str, Any] = dict(extra_payload or {})
    export_payload["obs_dim"] = int(getattr(actor, "obs_dim", None) or actor.net[0].in_features)
    export_payload["action_dim"] = int(getattr(actor, "action_dim", None) or actor.net[-1].out_features)
    export_payload["hidden_dim"] = int(getattr(actor, "hidden_dim", None) or actor.net[0].out_features)
    export_payload["actor_hidden_dim"] = int(export_payload["hidden_dim"])
    # Sampling-distribution parameters must travel with the weights.
    # Omitting them made the exported policy fall back to the module
    # defaults (-4.0 / 0.0) while training clamped at the experiment's
    # own bounds. Since ``log_std`` is stored unclamped, a floor of -2.5
    # on the training side and -4.0 on the rollout side means rollout
    # sampled with a *smaller* sigma than the sigma used to compute
    # ``old_log_prob`` — a silent violation of the on-policy assumption
    # that showed up nowhere in the logs.
    export_payload["log_std_min"] = float(getattr(actor, "log_std_min", -4.0))
    export_payload["log_std_max"] = float(getattr(actor, "log_std_max", 0.0))
    export_payload["state_dict"] = {
        key: value.detach().cpu()
        for key, value in actor.state_dict().items()
    }
    return export_payload


def export_actor_policy_artifacts(
    actor: nn.Module,
    policy_dir: Path,
    extra_payload: Optional[Mapping[str, Any]] = None,
    stochastic: bool = False,
) -> None:
    """Write ``model.pt`` + ``policy.py`` + ``policy_blueprint.yaml`` into ``policy_dir``.

    End result is a directory compatible with :func:`policy.load_util.load_policy`
    and :class:`envs.framework.policy.PolicyBlueprint`.
    """
    policy_dir.mkdir(parents=True, exist_ok=True)
    export_payload = build_actor_export_payload(actor=actor, extra_payload=extra_payload)
    torch.save(export_payload, policy_dir / "model.pt")
    policy_code = build_export_policy_code()
    with (policy_dir / "policy.py").open("w", encoding="utf-8") as handle:
        handle.write(policy_code)

    # Export PolicyBlueprint YAML pointing to the standalone policy.py.
    # Uses "file:" prefix so it works without repo on sys.path.
    policy_py_path = policy_dir / "policy.py"
    blueprint = PolicyBlueprint(
        cls=f"file:{policy_py_path}:ExportedMLPPolicy",
        config={
            "stochastic": stochastic,
        },
    )
    blueprint.save(policy_dir / "policy_blueprint.yaml")


def export_policy_artifacts_from_checkpoint(
    model_path: Path,
    policy_dir: Path,
    default_hidden_dim: int = DEFAULT_EXPORT_ACTOR_HIDDEN_DIM,
) -> None:
    """Like :func:`export_actor_policy_artifacts` but source = on-disk checkpoint.

    Used by trainers that keep a rolling ``model.pt`` on disk and want
    to publish a deployable policy directory without rebuilding the
    actor in memory.
    """
    policy_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(model_path, map_location="cpu")
    export_payload = dict(payload)
    export_payload["state_dict"] = {
        key: value.detach().cpu()
        for key, value in payload["state_dict"].items()
    }
    export_payload["hidden_dim"] = int(payload.get("actor_hidden_dim", payload.get("hidden_dim", default_hidden_dim)))
    export_payload["actor_hidden_dim"] = int(export_payload["hidden_dim"])
    torch.save(export_payload, policy_dir / "model.pt")
    policy_code = build_export_policy_code()
    with (policy_dir / "policy.py").open("w", encoding="utf-8") as handle:
        handle.write(policy_code)
