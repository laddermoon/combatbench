"""ReplaySimulator — replay recordings through the ``IDataAccessor`` contract.

A :class:`ReplaySimulator` consumes the on-disk layout produced by
:class:`envs.framework.recorder.BaseFrameRecorder` and surfaces it as a
regular :class:`envs.framework.backend.BaseSimulator`. This lets **any**
observer / plugin / training code that talks to the framework through
``IDataAccessor`` run against recorded episodes with **no code changes** — a
backend-agnostic replay path.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------

Basic — replay every episode in an output directory one by one::

    from envs.framework import EnvRuntime, ReplaySimulator

    sim = ReplaySimulator("/path/to/recording_root")
    # Each EnvRuntime.step must advance exactly one recorded frame, so the
    # phy_steps_per_action passed to EnvRuntime MUST match the recorder's
    # stride (default stride=1 ⇒ phy_steps_per_action=1). See "Best practices"
    # below for why.
    runtime = EnvRuntime(
        simulator=sim,
        observer_plugins={...},   # normal observers/plugins
        plugins=[...],
        phy_steps_per_action=1,
    )

    while sim.has_next_episode():
        runtime.reset()
        while runtime.is_episode_active and sim.has_next_step():
            runtime.step(np.zeros(21), np.zeros(21))   # actions IGNORED;
                                                        # replay is read-only

Jump to a specific episode::

    runtime.reset(options={"episode": 7})

Replay a single episode directory::

    sim = ReplaySimulator("/path/to/recording_root/episode_00007")
    runtime.reset()

--------------------------------------------------------------------------
Best practices
--------------------------------------------------------------------------

1. **Match stride**. If the recording was made with ``stride=N``, call
   ``EnvRuntime.step`` with ``phy_steps_per_action=N`` (and the underlying
   ReplaySimulator still advances **one** recorded frame per physical step).
   Mismatched stride → observers see non-contiguous timestamps.

2. **Do not mutate**. ``set_core_state`` / ``set_action`` /
   ``apply_external_force`` all raise :class:`ReplayReadOnlyError`. Replay is
   fundamentally a tape; you cannot "change history". If you need to branch
   off a recorded state, read it via the accessor and feed it to a LIVE
   backend instead.

3. **Actions are decorative**. The ``action`` you pass to ``EnvRuntime.step``
   is silently ignored; ``get_action()`` always returns the RECORDED action.
   This is intentional — it lets you run the same training code path on
   replays to inspect what the learner was doing, without accidentally
   diverging from the recording.

4. **Array dtype note**. The on-disk JSON format loses dtype information
   (everything is stored as nested lists of Python floats). On load this
   class rehydrates numeric lists into ``float32`` ndarrays (matching the
   live simulator convention). If your code needs a different dtype, cast
   at the read site.

5. **Images are lazy**. PNGs are decoded only when
   ``get_broadcastview_image()`` is actually called, and cached per step.
   If you don't need them, the replay cost is zero.

6. **Manifest version**. This simulator accepts ``manifest_version >= 2``
   (the version introduced when the recorder started emitting
   ``derived_state`` / ``sensor_data`` / ``action`` / ``static.json``).
   Older recordings can't be replayed end-to-end; re-record them.

7. **Not thread-safe**. Each ReplaySimulator keeps a cursor; share across
   threads only behind a lock, or instantiate one per thread.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .backend import BaseSimulator


class ReplayError(RuntimeError):
    """Base class for replay-related failures."""


class ReplayReadOnlyError(ReplayError):
    """Raised when caller attempts to mutate a ReplaySimulator."""


class ReplayExhaustedError(ReplayError):
    """Raised when caller asks for a frame beyond the recorded episode."""


# ---------------------------------------------------------------------------
# JSON → ndarray rehydration
# ---------------------------------------------------------------------------
def _rehydrate(value: Any) -> Any:
    """Convert nested lists of numbers back into ``float32`` ndarrays.

    Strategy:
    * ``dict`` → recurse per value.
    * homogeneous numeric list (int/float/bool, any depth of uniformly
      nested numeric lists) → ``np.asarray(..., dtype=float32)``.
    * any other list (mixed types, dicts inside, ragged shapes) → recurse
      element-wise and return a plain Python list.

    This matches how ``BaseFrameRecorder._json_sanitize`` serialized the
    data: ndarrays become ``.tolist()`` on the way out, and we reverse that
    on the way in whenever the shape is unambiguous.
    """
    if isinstance(value, dict):
        return {key: _rehydrate(sub) for key, sub in value.items()}
    if isinstance(value, list):
        if _is_uniform_numeric(value):
            try:
                return np.asarray(value, dtype=np.float32)
            except (ValueError, TypeError):
                pass
        # Fallback: recurse element-wise, preserving list-ness.
        return [_rehydrate(element) for element in value]
    return value


def _is_uniform_numeric(value: Any) -> bool:
    """Return True iff ``value`` is a (possibly nested) rectangular list of
    numeric scalars. Empty lists count as non-numeric (ambiguous shape)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, list):
        if not value:
            return False
        # All children must themselves be uniform numeric AND have the same
        # "shape" (i.e. same length recursively) for np.asarray to succeed.
        if not all(_is_uniform_numeric(child) for child in value):
            return False
        if isinstance(value[0], list):
            first_len = len(value[0])
            return all(isinstance(child, list) and len(child) == first_len
                       for child in value)
        return True
    return False


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def _is_episode_dir(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").exists()


def _discover_episode_dirs(root: Path) -> List[Path]:
    """Return episode_* subdirs under ``root`` in ascending episode_index."""
    candidates = sorted(
        child for child in root.glob("episode_*") if _is_episode_dir(child)
    )
    return candidates


# ---------------------------------------------------------------------------
# ReplaySimulator
# ---------------------------------------------------------------------------
class ReplaySimulator(BaseSimulator):
    """Serves a recording through the :class:`BaseSimulator` contract.

    See module docstring for usage + best practices.

    Parameters
    ----------
    source: path to either an output directory (containing ``index.json``
        and one or more ``episode_*/`` children) or a single episode
        directory (containing ``manifest.json``).
    rehydrate_arrays: rebuild ndarrays from JSON lists (default ``True``).
        Set to ``False`` to keep raw Python lists — faster, but observer
        code that does ``np.asarray`` on access will need to cope.
    strict_manifest_version: reject recordings whose ``manifest_version``
        is below :attr:`MIN_MANIFEST_VERSION` (default ``True``).
    """

    MIN_MANIFEST_VERSION = 2

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        source: Path | str,
        *,
        rehydrate_arrays: bool = True,
        strict_manifest_version: bool = True,
    ) -> None:
        self._source_path = Path(source).expanduser().resolve()
        self._rehydrate_arrays = bool(rehydrate_arrays)
        self._strict_manifest_version = bool(strict_manifest_version)

        if _is_episode_dir(self._source_path):
            # Single-episode mode
            self._episode_dirs: List[Path] = [self._source_path]
        else:
            self._episode_dirs = _discover_episode_dirs(self._source_path)
            if not self._episode_dirs:
                raise ReplayError(
                    f"No episode_* directories found under {self._source_path}. "
                    f"Point ReplaySimulator at either an output_dir containing "
                    f"episode_* folders, or a single episode directory."
                )

        # Index of the episode currently loaded (-1 before first reset).
        self._episode_cursor: int = -1
        # Step cursor INTO the current episode's recorded frames.
        self._step_cursor: int = -1
        # Episode-level state (populated on ``reset``).
        self._manifest: Optional[Dict[str, Any]] = None
        self._static_payload: Dict[str, Any] = {}
        self._recorded_frames: List[Dict[str, Any]] = []  # per-step payload (lazy)
        self._current_frame: Dict[str, Any] = {}
        self._image_cache: Dict[int, np.ndarray] = {}
        self._is_closed = False

    # ------------------------------------------------------------------
    # Introspection helpers (not part of BaseSimulator)
    # ------------------------------------------------------------------
    @property
    def source_path(self) -> Path:
        return self._source_path

    @property
    def num_episodes(self) -> int:
        return len(self._episode_dirs)

    @property
    def episode_index(self) -> int:
        """Index of the currently loaded episode (-1 before first reset)."""
        return self._episode_cursor

    @property
    def step_index(self) -> int:
        """0-based index of the frame most recently exposed through the
        accessor. -1 before the first ``reset``; stays pinned at the last
        frame after the episode is exhausted."""
        return self._step_cursor

    def has_next_episode(self) -> bool:
        return self._episode_cursor + 1 < self.num_episodes

    def has_next_step(self) -> bool:
        return self._step_cursor + 1 < len(self._recorded_frames)

    # ------------------------------------------------------------------
    # BaseSimulator lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """Load the next episode (or the one requested via
        ``options={"episode": N}``)."""
        if self._is_closed:
            raise ReplayError("ReplaySimulator is closed.")

        if options and "episode" in options:
            target_index = int(options["episode"])
            if not 0 <= target_index < self.num_episodes:
                raise ReplayError(
                    f"Requested episode index {target_index} outside "
                    f"[0, {self.num_episodes})."
                )
        else:
            target_index = self._episode_cursor + 1
            if target_index >= self.num_episodes:
                raise ReplayExhaustedError(
                    "No more recorded episodes. Call reset with "
                    "options={'episode': N} to jump back."
                )

        self._load_episode(target_index)

    def physical_step(self) -> None:
        """Advance exactly one recorded frame."""
        if self._is_closed:
            raise ReplayError("ReplaySimulator is closed.")
        if self._episode_cursor < 0:
            raise ReplayError("physical_step called before reset().")
        if not self.has_next_step():
            raise ReplayExhaustedError(
                f"Replay episode {self._episode_cursor} exhausted after "
                f"{len(self._recorded_frames)} frames."
            )
        self._step_cursor += 1
        self._current_frame = self._recorded_frames[self._step_cursor]

    def get_physical_frequency(self) -> float:
        freq = self._static_payload.get("physical_frequency")
        if freq is None:
            raise ReplayError(
                "Recording did not include physical_frequency. Re-record "
                "with save_static_data=True / save_accessor_state=True."
            )
        return float(freq)

    def close(self) -> None:
        self._is_closed = True
        self._recorded_frames.clear()
        self._image_cache.clear()
        self._current_frame = {}

    # ------------------------------------------------------------------
    # IDataAccessor
    # ------------------------------------------------------------------
    def get_static_data(self) -> Dict[str, Any]:
        return self._static_payload.get("static_data", {})

    def get_core_state(self) -> Dict[str, Any]:
        return self._current_frame.get("core_state", {})

    def get_derived_state(self) -> Dict[str, Any]:
        return self._current_frame.get("derived_state", {})

    def get_sensor_data(self) -> Dict[str, Any]:
        return self._current_frame.get("sensor_data", {})

    def get_action(self) -> Dict[str, Any]:
        return self._current_frame.get("action", {})

    def get_observation(self) -> Dict[str, Any]:
        return self._current_frame.get("observation", {})

    def get_broadcastview_image(self) -> np.ndarray:
        """Decode and cache the PNG associated with the current frame.

        Returns a uint8 RGB ndarray. If the recording did not contain an
        image for this step (``save_image=False`` during recording), a
        black ``(1, 1, 3)`` placeholder is returned so downstream code
        that unconditionally reads an image does not crash; callers that
        care should check ``has_image_for_current_step()`` first.
        """
        step = self._step_cursor
        if step in self._image_cache:
            return self._image_cache[step].copy()

        image_name = self._current_frame.get("__image_name__")
        if image_name is None:
            placeholder = np.zeros((1, 1, 3), dtype=np.uint8)
            self._image_cache[step] = placeholder
            return placeholder.copy()

        assert self._current_episode_dir is not None
        image_path = self._current_episode_dir / image_name
        import imageio.v2 as imageio  # lazy import, mirror recorder
        image = np.asarray(imageio.imread(str(image_path)))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.astype(np.uint8)
        self._image_cache[step] = image
        return image.copy()

    def has_image_for_current_step(self) -> bool:
        return self._current_frame.get("__image_name__") is not None

    # ------------------------------------------------------------------
    # IDataMutator — all blocked
    # ------------------------------------------------------------------
    def set_core_state(self, state: Dict[str, Any]) -> None:
        raise ReplayReadOnlyError(
            "ReplaySimulator.set_core_state: replay is read-only. "
            "See module docstring best-practice #2."
        )

    def set_action(self, action: Dict[str, Any]) -> None:
        # ``EnvRuntime.step`` calls this as its first move. Silently
        # accepting the call (and ignoring the action) is part of the
        # documented contract: the stored action is whatever was
        # recorded, not whatever the caller just tried to set. Raising
        # here would make replay incompatible with plain EnvRuntime.step.
        return None

    def apply_external_force(self, *args: Any, **kwargs: Any) -> None:
        raise ReplayReadOnlyError(
            "ReplaySimulator.apply_external_force: replay is read-only."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @property
    def _current_episode_dir(self) -> Optional[Path]:
        if 0 <= self._episode_cursor < self.num_episodes:
            return self._episode_dirs[self._episode_cursor]
        return None

    def _load_episode(self, episode_index: int) -> None:
        episode_dir = self._episode_dirs[episode_index]
        manifest_path = episode_dir / "manifest.json"
        with open(manifest_path, "r") as manifest_file:
            manifest = json.load(manifest_file)

        manifest_version = int(manifest.get("manifest_version", 1))
        if self._strict_manifest_version and manifest_version < self.MIN_MANIFEST_VERSION:
            raise ReplayError(
                f"{manifest_path} has manifest_version={manifest_version} "
                f"but ReplaySimulator requires >= {self.MIN_MANIFEST_VERSION}. "
                f"Re-record the episode with a newer recorder, or disable "
                f"strict_manifest_version at your own risk."
            )

        # Load static payload (optional per recorder config).
        static_name = manifest.get("static")
        if static_name:
            with open(episode_dir / static_name, "r") as static_file:
                raw_static = json.load(static_file)
            self._static_payload = _rehydrate(raw_static) if self._rehydrate_arrays else raw_static
        else:
            self._static_payload = {}

        # Load per-step frames eagerly. Episodes are bounded (seconds of
        # data), so eager loading avoids per-step disk IO during replay.
        # If this ever becomes a memory concern, switch to lazy loading
        # keyed on ``_step_cursor``.
        frames: List[Dict[str, Any]] = []
        for step_entry in manifest.get("steps", []):
            data_name = step_entry.get("data")
            image_name = step_entry.get("image")
            if data_name is not None:
                with open(episode_dir / data_name, "r") as data_file:
                    raw_payload = json.load(data_file)
                payload = _rehydrate(raw_payload) if self._rehydrate_arrays else raw_payload
            else:
                # Image-only recording (no JSON); synthesize an empty frame
                # with the episode/physics step numbers from the manifest.
                payload = {
                    "episode_step": int(step_entry.get("step", len(frames))),
                    "physics_step": int(step_entry.get("physics_step", 0)),
                }
            # Tuck image name into the frame so get_broadcastview_image
            # can look it up without re-reading the manifest.
            payload["__image_name__"] = image_name
            frames.append(payload)

        self._episode_cursor = episode_index
        self._step_cursor = -1
        self._manifest = manifest
        self._recorded_frames = frames
        self._current_frame = {}
        self._image_cache.clear()

        # Advance to the first recorded frame so that the accessor readout
        # right after ``reset`` reflects the episode's initial state (the
        # frame that the recorder captured in ``on_pre_episode``).
        if frames:
            self._step_cursor = 0
            self._current_frame = frames[0]
