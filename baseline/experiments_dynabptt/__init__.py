"""DynaBPTT experiment registry — auto-discovers ``exp_*.py`` files.

Each experiment file exports ``EXPERIMENT_CLASS`` — a duck-typed class
consumed by ``baseline.framework.dynabptt.train``:

  name, agent_ids, state_dim, action_dim, obs_dim, critic_hidden_dim,
  wm_hidden, wm_loss_weights, aux_names, channels ([{"name","gamma"}]),
  init_distance_min/max, max_steps
  env_bp() -> EnvBlueprint (materialized, pre-injection)
  build_actor(device) -> TruncatedNormalPolicy
  aux_keys(agent_id) -> {aux_name: (observer_name, field|None)}
  channel_reward(name, aux_tensors) -> torch.Tensor
  channel_weight(name, aux_tensors) -> torch.Tensor or float
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Type

_REGISTRY: Dict[str, Type] = {}


def _discover() -> None:
    pkg_dir = Path(__file__).parent
    for f in sorted(pkg_dir.glob("exp_*.py")):
        try:
            mod = importlib.import_module(f".{f.stem}", package=__package__)
        except Exception as exc:
            import sys
            print(f"[experiments_dynabptt] skip {f.stem}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        exp_cls = getattr(mod, "EXPERIMENT_CLASS", None)
        if exp_cls is not None:
            _REGISTRY[exp_cls.name] = exp_cls


def get_dynabptt_experiment(name: str, **kwargs):
    if not _REGISTRY:
        _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dynabptt experiment {name!r}. "
            f"Available: {list_dynabptt_experiments()}"
        )
    return _REGISTRY[name](**kwargs)


def list_dynabptt_experiments() -> List[str]:
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY.keys())
