"""Offline replay — re-run ppo_update on a snapshot with full debug recording.

S2: Strictly reuses production code (P2 from DESIGN_debug_system.md):
``build_trajectories``, ``debug_arrays``, ``PPOBuffer``, ``ppo_update``.
No GAE/PPO logic is copied.  The replay deep-copies actor + critics so
repeated runs don't mutate the snapshot.

Self-verification (``verify_against_log``): when the snapshot was
captured with ``--episodes all``, the replayed ``UpdateStats`` must
match the training log's ``__RAW_STATS__`` for that update, field by
field (float tolerance).  Mismatch = the snapshot is incomplete or
there's undetected non-determinism — a bug that must be fixed first.

See ``DESIGN_debug_system.md`` §4.3.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline.framework.ppo.experiment import (
    ExperimentPPO,
    PPOParams,
    UpdateStats,
)
from baseline.framework.ppo.trainer import PPOBuffer, ppo_update, set_seed
from baseline.framework.rollout import EpisodeCollection

from .sink import NpzSink
from .snapshot import MANIFEST_FILENAME


@dataclass
class ReplayResult:
    """Result of ``replay_snapshot``."""
    snapshot_dir: Path
    update: int
    episodes_mode: str
    n_episodes: int
    n_frames: int
    stats: UpdateStats
    debug_arrays: Dict[str, np.ndarray]
    replay_dir: Path


@dataclass
class FieldComparison:
    """One field's comparison result."""
    name: str
    log_value: Any
    replay_value: Any
    passed: bool
    note: str = ""


@dataclass
class VerificationResult:
    """Result of ``verify_against_log``."""
    snapshot_dir: Path
    update: int
    episodes_mode: str
    comparable: bool
    fields: List[FieldComparison] = field(default_factory=list)
    n_passed: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    verdict: str = ""  # "pass" | "fail" | "not_comparable"


def _load_manifest(snapshot_dir: Path) -> Dict[str, Any]:
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} at {manifest_path} — not a snapshot dir"
        )
    with open(manifest_path) as f:
        return json.load(f)


def _load_experiment(snapshot_dir: Path, manifest: Dict[str, Any]) -> ExperimentPPO:
    """Reconstruct experiment from manifest via the registry."""
    exp_name = manifest.get("experiment_name")
    if not exp_name:
        raise KeyError("manifest missing 'experiment_name'")
    from baseline.experiments_ppo import get_ppo_experiment
    return get_ppo_experiment(exp_name)


def _set_rng_state(rng_state: Dict[str, Any]) -> None:
    """Restore torch + numpy + cuda RNG state."""
    if "torch_cpu" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def replay_snapshot(
    snapshot_dir: Path,
    *,
    device: Optional[torch.device] = None,
    experiment: Optional[ExperimentPPO] = None,
) -> ReplayResult:
    """Re-run ppo_update on a snapshot with full debug recording.

    Strictly reuses production code (P2): ``build_trajectories``,
    ``debug_arrays``, ``PPOBuffer``, ``ppo_update``.  Actor + critics
    are deep-copied so the snapshot is not mutated.

    Args:
        snapshot_dir: Path to ``<run_dir>/debug/u{u:05d}/``.
        device: Torch device (default: CPU).
        experiment: Pre-loaded experiment (optional; loaded from
            manifest if not provided).

    Returns:
        :class:`ReplayResult` with stats + debug_arrays + paths.
    """
    snapshot_dir = Path(snapshot_dir)
    if device is None:
        device = torch.device("cpu")

    manifest = _load_manifest(snapshot_dir)
    update = manifest["update"]
    episodes_mode = manifest.get("episodes_mode", "subset")
    include_full_grad = manifest.get("include_full_grad", False)

    # --- Load experiment ---
    if experiment is None:
        experiment = _load_experiment(snapshot_dir, manifest)

    # --- Load episodes ---
    episodes_dir = snapshot_dir / "episodes"
    if not episodes_dir.exists() or not any(episodes_dir.iterdir()):
        raise FileNotFoundError(
            f"No episodes at {episodes_dir} — snapshot is incomplete"
        )
    coll = EpisodeCollection.load(episodes_dir)
    episodes = list(coll)

    # --- Build fresh actor + critics, then load θ_old weights ---
    actor = experiment.build_actor(device)
    actor_state_path = snapshot_dir / "actor_state.pt"
    if actor_state_path.exists():
        actor.load_state_dict(torch.load(actor_state_path, map_location=device))
    else:
        raise FileNotFoundError(
            f"No actor_state.pt at {actor_state_path} — cannot reconstruct θ_old"
        )

    reward_channels = experiment.reward_channels()
    reward_keys = tuple(ch.name for ch in reward_channels)

    critics: Dict[str, torch.nn.Module] = {}
    for ch in reward_channels:
        critics[ch.name] = experiment.build_critic(ch.name, device)
    critics_state = torch.load(
        snapshot_dir / "critics.pt", map_location=device, weights_only=False,
    )
    for k, v in critics.items():
        if k in critics_state:
            v.load_state_dict(critics_state[k])

    # --- Deep-copy actor + critics (don't mutate snapshot) ---
    actor_copy = copy.deepcopy(actor)
    critics_copy = {k: copy.deepcopy(v) for k, v in critics.items()}

    # --- Restore RNG state ---
    rng_state = torch.load(
        snapshot_dir / "rng_state.pt", map_location="cpu", weights_only=False,
    )
    _set_rng_state(rng_state)

    # --- Build optimizers (needed by ppo_update but not used for replay
    # correctness — we only care about the forward pass + stats, not the
    # weight update.  But ppo_update calls optimizer.step(), so we need
    # real optimizers.  They operate on the deep-copies, not the originals.)
    pp = experiment.ppo_params()
    cp = experiment.common_params()
    actor_optimizer = torch.optim.Adam(
        actor_copy.parameters(), lr=cp.learning_rate,
    )
    critic_optimizers = {
        k: torch.optim.Adam(v.parameters(), lr=cp.critic_learning_rate)
        for k, v in critics_copy.items()
    }

    # --- Real production calls (P2: no logic duplication) ---
    trajectories = experiment.build_trajectories(episodes)
    debug_arrays = experiment.debug_arrays(episodes, trajectories)

    buf = PPOBuffer(
        trajectories=trajectories,
        actor=actor_copy,
        device=device,
        reward_keys=reward_keys,
    )

    # --- Sink for per-frame debug recording ---
    replay_dir = snapshot_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    sink = NpzSink(replay_dir)

    # --- Restore RNG state again right before ppo_update ---
    # (build_trajectories / PPOBuffer may have consumed RNG; the snapshot
    # captured RNG state right before ppo_update, so we restore it here
    # to match the training-time ppo_update RNG sequence.)
    _set_rng_state(rng_state)

    stats = ppo_update(
        actor=actor_copy,
        critics=critics_copy,
        actor_optimizer=actor_optimizer,
        critic_optimizers=critic_optimizers,
        buf=buf,
        reward_channels=reward_channels,
        pp=pp,
        grad_clip_norm=cp.grad_clip_norm,
        device=device,
        use_confidence=True,
        exploration=experiment.exploration(update),
        debug_sink=sink,
        include_full_grad=include_full_grad,
    )
    sink.close()

    # --- Save debug_arrays + stats ---
    if debug_arrays:
        np.savez_compressed(
            replay_dir / "debug_arrays.npz", **debug_arrays,
        )
    with open(replay_dir / "stats.json", "w") as f:
        json.dump(stats.to_log_dict(), f, indent=2, default=str)

    n_frames = sum(len(t.obs) for t in trajectories) if trajectories else 0

    return ReplayResult(
        snapshot_dir=snapshot_dir,
        update=update,
        episodes_mode=episodes_mode,
        n_episodes=len(episodes),
        n_frames=n_frames,
        stats=stats,
        debug_arrays=debug_arrays,
        replay_dir=replay_dir,
    )


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

_FLOAT_TOLERANCE_REL = 1e-5
_FLOAT_TOLERANCE_ABS = 1e-6

# Fields that are inherently non-deterministic or structural (not
# field-by-field comparable).
_SKIP_FIELDS = {
    "epoch_kl_stats",  # list of dicts — compared by length + mean only
    "diagnostics",     # human-readable strings
}
# epoch_kl_stats is compared specially (length + first element mean).


def _parse_log_stats(run_dir: Path, target_update: int) -> Optional[Dict[str, Any]]:
    """Find the ``__RAW_STATS__`` line for ``target_update`` in train.log."""
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return None
    pattern = re.compile(r"__RAW_STATS__\s*(\{.*\})")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if data.get("update") == target_update:
                return data
    return None


def _compare_field(name: str, log_val: Any, replay_val: Any) -> FieldComparison:
    """Compare one field between log and replay."""
    # Skip structural / non-comparable fields.
    if name in _SKIP_FIELDS:
        return FieldComparison(name, log_val, replay_val, True, "skipped (structural)")

    # Both must be present.
    if log_val is None and replay_val is None:
        return FieldComparison(name, None, None, True, "both None")
    if log_val is None or replay_val is None:
        return FieldComparison(name, log_val, replay_val, False, "one is None")

    # Numeric comparison.
    if isinstance(log_val, (int, float)) and isinstance(replay_val, (int, float)):
        diff = abs(float(log_val) - float(replay_val))
        scale = max(1.0, abs(float(log_val)), abs(float(replay_val)))
        if diff <= _FLOAT_TOLERANCE_REL * scale + _FLOAT_TOLERANCE_ABS:
            return FieldComparison(name, log_val, replay_val, True)
        return FieldComparison(
            name, log_val, replay_val, False,
            f"diff={diff:.2e} > tol={_FLOAT_TOLERANCE_REL * scale:.2e}",
        )

    # String comparison.
    if isinstance(log_val, str) and isinstance(replay_val, str):
        return FieldComparison(
            name, log_val, replay_val, log_val == replay_val,
        )

    # List comparison (element-wise for numbers).
    if isinstance(log_val, list) and isinstance(replay_val, list):
        if len(log_val) != len(replay_val):
            return FieldComparison(
                name, log_val, replay_val, False,
                f"length {len(log_val)} vs {len(replay_val)}",
            )
        if not log_val:
            return FieldComparison(name, [], [], True)
        # Compare element-wise with float tolerance.
        all_ok = True
        max_diff = 0.0
        for lv, rv in zip(log_val, replay_val):
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                d = abs(float(lv) - float(rv))
                s = max(1.0, abs(float(lv)), abs(float(rv)))
                if d > _FLOAT_TOLERANCE_REL * s + _FLOAT_TOLERANCE_ABS:
                    all_ok = False
                    max_diff = max(max_diff, d)
            elif lv != rv:
                all_ok = False
                break
        if all_ok:
            return FieldComparison(name, log_val, replay_val, True)
        return FieldComparison(name, log_val, replay_val, False, f"max_diff={max_diff:.2e}")

    # Dict comparison (recurse on values).
    if isinstance(log_val, dict) and isinstance(replay_val, dict):
        if set(log_val.keys()) != set(replay_val.keys()):
            return FieldComparison(
                name, list(log_val.keys()), list(replay_val.keys()), False,
                "key mismatch",
            )
        all_ok = True
        for k in log_val:
            sub = _compare_field(f"{name}.{k}", log_val[k], replay_val[k])
            if not sub.passed:
                all_ok = False
                break
        return FieldComparison(name, "dict", "dict", all_ok)

    # Fallback: equality.
    return FieldComparison(name, log_val, replay_val, log_val == replay_val)


def verify_against_log(
    snapshot_dir: Path,
    run_dir: Path,
) -> VerificationResult:
    """Verify replayed stats against the training log.

    Only meaningful when ``episodes_mode == "all"`` — subset snapshots
    are explicitly not comparable (DESIGN §4.2).

    Args:
        snapshot_dir: Path to the snapshot directory.
        run_dir: Path to the training run directory (contains train.log).

    Returns:
        :class:`VerificationResult` with per-field pass/fail.
    """
    snapshot_dir = Path(snapshot_dir)
    run_dir = Path(run_dir)

    manifest = _load_manifest(snapshot_dir)
    update = manifest["update"]
    episodes_mode = manifest.get("episodes_mode", "subset")

    if episodes_mode != "all":
        return VerificationResult(
            snapshot_dir=snapshot_dir,
            update=update,
            episodes_mode=episodes_mode,
            comparable=False,
            verdict="not_comparable",
        )

    # Load replayed stats.
    stats_path = snapshot_dir / "replay" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"No replay/stats.json at {stats_path} — run replay first"
        )
    with open(stats_path) as f:
        replay_stats = json.load(f)

    # Find the matching log entry.
    log_data = _parse_log_stats(run_dir, update)
    if log_data is None:
        return VerificationResult(
            snapshot_dir=snapshot_dir,
            update=update,
            episodes_mode=episodes_mode,
            comparable=True,
            verdict="fail",
            fields=[FieldComparison(
                "__log__", None, None, False,
                f"No __RAW_STATS__ entry for update {update} in {run_dir}/train.log",
            )],
            n_failed=1,
        )

    log_stats = log_data.get("stats", log_data)

    # Compare field by field.
    fields: List[FieldComparison] = []
    n_passed = 0
    n_failed = 0
    n_skipped = 0

    # Union of keys.
    all_keys = set(log_stats.keys()) | set(replay_stats.keys())

    # epoch_kl_stats: special comparison (length + mean of first element).
    if "epoch_kl_stats" in all_keys:
        log_ekls = log_stats.get("epoch_kl_stats", [])
        rep_ekls = replay_stats.get("epoch_kl_stats", [])
        if isinstance(log_ekls, list) and isinstance(rep_ekls, list):
            if len(log_ekls) == len(rep_ekls):
                # Compare mean_kl of each epoch.
                ok = True
                for le, re_ in zip(log_ekls, rep_ekls):
                    if isinstance(le, dict) and isinstance(re_, dict):
                        lm = le.get("mean_kl", 0.0)
                        rm = re_.get("mean_kl", 0.0)
                        d = abs(float(lm) - float(rm))
                        s = max(1.0, abs(float(lm)), abs(float(rm)))
                        if d > _FLOAT_TOLERANCE_REL * s + _FLOAT_TOLERANCE_ABS:
                            ok = False
                            break
                fields.append(FieldComparison(
                    "epoch_kl_stats", f"len={len(log_ekls)}", f"len={len(rep_ekls)}",
                    ok, "compared by length + per-epoch mean_kl",
                ))
                if ok:
                    n_passed += 1
                else:
                    n_failed += 1
            else:
                fields.append(FieldComparison(
                    "epoch_kl_stats", f"len={len(log_ekls)}", f"len={len(rep_ekls)}",
                    False, "length mismatch",
                ))
                n_failed += 1
        all_keys.discard("epoch_kl_stats")

    for key in sorted(all_keys):
        if key in _SKIP_FIELDS:
            n_skipped += 1
            continue
        log_val = log_stats.get(key)
        rep_val = replay_stats.get(key)
        cmp = _compare_field(key, log_val, rep_val)
        fields.append(cmp)
        if cmp.passed:
            n_passed += 1
        else:
            n_failed += 1

    verdict = "pass" if n_failed == 0 else "fail"

    return VerificationResult(
        snapshot_dir=snapshot_dir,
        update=update,
        episodes_mode=episodes_mode,
        comparable=True,
        fields=fields,
        n_passed=n_passed,
        n_failed=n_failed,
        n_skipped=n_skipped,
        verdict=verdict,
    )


__all__ = [
    "ReplayResult",
    "VerificationResult",
    "FieldComparison",
    "replay_snapshot",
    "verify_against_log",
]
