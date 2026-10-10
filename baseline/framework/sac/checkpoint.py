"""Atomic ``sac_checkpoint_v1`` bundle persistence.

A full SAC checkpoint is a directory bundle, not a loose ``.pt`` file.
It contains model/trainer state, full replay state, runtime counters/RNG,
experiment state, and a manifest that validates schema and file integrity.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from .replay import SACReplayBuffer


SAC_CHECKPOINT_SCHEMA = "sac_checkpoint_v1"
MANIFEST_NAME = "manifest.json"


class SACCheckpointError(RuntimeError):
    """Raised when a checkpoint bundle is invalid or cannot be resumed."""


@dataclass(frozen=True)
class SACCheckpointBundle:
    """Loaded checkpoint payload.

    ``resume_mode`` is either ``full`` or ``warm_start``.  A warm start
    deliberately omits replay/runtime counters; callers must not present it
    as an exact continuation.
    """

    path: Path
    manifest: Dict[str, Any]
    trainer_state: Any
    replay: Optional[SACReplayBuffer]
    runtime_state: Optional[Dict[str, Any]]
    experiment_state: Dict[str, Any]
    config: Dict[str, Any]
    resume_mode: str


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    )


def _delete_dotted(payload: Dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    cur: Any = payload
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            return
        cur = cur.get(part)
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def config_fingerprint(
    config: Mapping[str, Any],
    *,
    allowed_overrides: Sequence[str] = (),
) -> str:
    """Fingerprint config after removing explicitly allowed resume fields."""
    normalized = json.loads(_canonical_json(config))
    for dotted in allowed_overrides:
        _delete_dotted(normalized, dotted)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def save_checkpoint_bundle(
    path: str | Path,
    *,
    trainer_state: Any,
    replay: SACReplayBuffer,
    runtime_state: Mapping[str, Any],
    experiment_state: Mapping[str, Any],
    config: Mapping[str, Any],
    allowed_overrides: Sequence[str] = (),
) -> Path:
    """Atomically write a complete checkpoint bundle."""
    path = Path(path)
    if path.exists():
        raise SACCheckpointError(f"checkpoint path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=path.name + ".", dir=path.parent))
    try:
        trainer_path = tmp_dir / "trainer.pt"
        replay_path = tmp_dir / "replay.pt"
        runtime_path = tmp_dir / "runtime.pt"
        experiment_path = tmp_dir / "experiment.json"
        config_path = tmp_dir / "config.json"

        torch.save(trainer_state, trainer_path)
        replay.save(replay_path)
        torch.save(dict(runtime_state), runtime_path)
        _write_json(experiment_path, experiment_state)
        _write_json(config_path, config)

        files = {}
        for artifact in sorted(tmp_dir.iterdir()):
            if artifact.name == MANIFEST_NAME:
                continue
            files[artifact.name] = {
                "sha256": _sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }

        manifest = {
            "schema": SAC_CHECKPOINT_SCHEMA,
            "algorithm": "sac",
            "replay_schema": replay.state_dict()["schema"],
            "config_fingerprint": config_fingerprint(
                config, allowed_overrides=allowed_overrides,
            ),
            "allowed_overrides": tuple(allowed_overrides),
            "files": files,
        }
        _write_json(tmp_dir / MANIFEST_NAME, manifest)
        tmp_dir.replace(path)
        return path
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _load_manifest(path: Path) -> Dict[str, Any]:
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.exists():
        raise SACCheckpointError(f"checkpoint manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SAC_CHECKPOINT_SCHEMA:
        raise SACCheckpointError(
            f"unsupported checkpoint schema {manifest.get('schema')!r}"
        )
    if manifest.get("algorithm") != "sac":
        raise SACCheckpointError("checkpoint is not a SAC checkpoint")
    return manifest


def _verify_files(path: Path, manifest: Mapping[str, Any]) -> None:
    expected = dict(manifest.get("files") or {})
    required = {"trainer.pt", "replay.pt", "runtime.pt", "experiment.json", "config.json"}
    missing = sorted(required - set(expected))
    if missing:
        raise SACCheckpointError(f"checkpoint manifest missing files {missing}")
    for name, info in expected.items():
        artifact = path / name
        if not artifact.exists():
            raise SACCheckpointError(f"checkpoint artifact missing: {name}")
        if int(info["bytes"]) != artifact.stat().st_size:
            raise SACCheckpointError(f"checkpoint artifact size mismatch: {name}")
        if str(info["sha256"]) != _sha256_file(artifact):
            raise SACCheckpointError(f"checkpoint artifact hash mismatch: {name}")


def load_checkpoint_bundle(
    path: str | Path,
    *,
    expected_config: Optional[Mapping[str, Any]] = None,
    allowed_overrides: Optional[Sequence[str]] = None,
    config_lock: bool = False,
    warm_start: bool = False,
) -> SACCheckpointBundle:
    """Load and validate a checkpoint bundle.

    ``expected_config`` is compared after removing ``allowed_overrides``.
    ``config_lock=True`` is strict mode: overrides are not allowed and an
    expected config is required.
    """
    path = Path(path)
    if not path.is_dir():
        raise SACCheckpointError(f"checkpoint bundle is not a directory: {path}")
    manifest = _load_manifest(path)
    _verify_files(path, manifest)

    if config_lock and expected_config is None:
        raise SACCheckpointError("config_lock resume requires expected_config")
    overrides = tuple(allowed_overrides or ())
    if config_lock:
        overrides = ()
    elif allowed_overrides is None:
        overrides = tuple(manifest.get("allowed_overrides") or ())

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if expected_config is not None:
        actual_fp = str(manifest.get("config_fingerprint"))
        expected_fp = config_fingerprint(
            expected_config, allowed_overrides=overrides,
        )
        if actual_fp != expected_fp:
            raise SACCheckpointError(
                "checkpoint config does not match expected config "
                f"(allowed_overrides={overrides})"
            )

    trainer_state = torch.load(
        path / "trainer.pt", map_location="cpu", weights_only=False,
    )
    experiment_state = json.loads(
        (path / "experiment.json").read_text(encoding="utf-8")
    )

    if warm_start:
        return SACCheckpointBundle(
            path=path,
            manifest=manifest,
            trainer_state=trainer_state,
            replay=None,
            runtime_state=None,
            experiment_state=experiment_state,
            config=config,
            resume_mode="warm_start",
        )

    replay = SACReplayBuffer.load(path / "replay.pt")
    runtime_state = torch.load(
        path / "runtime.pt", map_location="cpu", weights_only=False,
    )
    return SACCheckpointBundle(
        path=path,
        manifest=manifest,
        trainer_state=trainer_state,
        replay=replay,
        runtime_state=runtime_state,
        experiment_state=experiment_state,
        config=config,
        resume_mode="full",
    )


def load_model_only(path: str | Path) -> Any:
    """Load a standalone model payload as an explicit warm-start boundary."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        "resume_mode": "warm_start",
        "model_payload": payload,
        "replay": None,
        "runtime_state": None,
    }


_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint_s\d+$")


def prune_checkpoints(ckpt_dir: str | Path, *, keep_last: int) -> None:
    """Bounded latest-N retention for checkpoint bundle directories.

    ``keep_last <= 0`` disables pruning.  Only directories named exactly
    ``checkpoint_s<digits>`` are eligible: the dotted tmp dirs used by
    :func:`save_checkpoint_bundle`'s atomic write are skipped so a prune
    can never corrupt an in-flight save.  A ``.pinned`` marker file
    exempts a checkpoint (e.g. one reserved for held-out evaluation).
    """
    root = Path(ckpt_dir)
    if int(keep_last) <= 0 or not root.exists():
        return
    candidates = [
        p for p in root.iterdir()
        if p.is_dir()
        and _CHECKPOINT_DIR_RE.match(p.name)
        and not (p / ".pinned").exists()
    ]
    candidates.sort(key=lambda p: p.name)
    for old in candidates[:-int(keep_last)]:
        shutil.rmtree(old)


__all__ = [
    "SAC_CHECKPOINT_SCHEMA",
    "SACCheckpointBundle",
    "SACCheckpointError",
    "config_fingerprint",
    "load_checkpoint_bundle",
    "load_model_only",
    "prune_checkpoints",
    "save_checkpoint_bundle",
]
