"""Post-action recording abstractions.

A ``PostActionRecorder`` is a side-effect-only observer invoked by
``EnvRuntime`` at episode boundaries and after every action step. Unlike
``BasePlugin`` instances, recorders:

* Always run **after** the observer dispatcher has refreshed observer outputs,
  so they see consistent per-step observer snapshots.
* Receive a pre-assembled mapping of observer outputs alongside the read-only
  ``SimContext``.
* Must be pure side effects: they can write files, accumulate statistics, or
  push to external systems, but MUST NOT mutate simulation state.

``BaseFrameRecorder`` is a batteries-included default that emits a
**standard, visualization-agnostic** on-disk format. It records raw data only
(images + JSON); visualization is handled by a separate tool
(:mod:`envs.framework.recorder_viewer`).

Directory layout produced by ``BaseFrameRecorder``::

    <output_dir>/
        index.json                     # list of episodes, updated per episode
        episode_00000/
            manifest.json              # list of recorded steps in this episode
            static.json                # episode-invariant accessor data (see below)
            step_00000.png             # raw broadcast-view image (no overlay)
            step_00000.json            # per-step accessor snapshot (see below)
            step_00001.png
            step_00001.json
            ...
        episode_00001/
            ...

Replay contract
---------------
The data persisted here is designed to be sufficient to **replay every
:class:`envs.framework.backend.IDataAccessor` read** deterministically:

    accessor method                     | on-disk source
    ------------------------------------|-------------------------------------
    get_static_data()                   | <episode>/static.json
    get_physical_frequency()            | <episode>/static.json["physical_frequency"]
    get_core_state()                    | <episode>/step_<N>.json["core_state"]
    get_derived_state()                 | <episode>/step_<N>.json["derived_state"]
    get_sensor_data()                   | <episode>/step_<N>.json["sensor_data"]
    get_action()                        | <episode>/step_<N>.json["action"]
    get_broadcastview_image()           | <episode>/step_<N>.png

A future ``ReplaySimulator`` (not part of this module) can wrap these
files and serve them through the accessor interface so observer /
recorder / training code can replay episodes without a live physics
backend. This recorder's job is only to make the data **complete**.

Per-step JSON schema (``step_*.json``, manifest_version=2)::

    {
      "episode_step":     int,
      "physics_step":     int,
      "observer_outputs": { <observer_name>: <any json-safe value>, ... },  # optional
      "core_state":       { ... },                                          # optional
      "derived_state":    { ... },                                          # optional
      "sensor_data":      { ... },                                          # optional
      "action":           { <agent_id>: <array>, ... }                      # optional
    }

Per-episode JSON schema (``static.json``)::

    {
      "static_data":         <accessor.get_static_data()>,
      "physical_frequency":  float
    }

Which fields are emitted is controlled by the ``save_*`` flags; see
``BaseFrameRecorder.__init__``.

Size caveat
-----------
``derived_state`` is rich (per-body / per-joint dicts + contact lists). JSON
expands ndarrays verbosely so single-step payloads for humanoid21 can reach
~1-2 MB. If this becomes a bottleneck the recommended follow-up is to switch
array payloads to a binary sidecar (``step_<N>.npz``) while keeping this
JSON as a metadata manifest; that change is backward-compatible because the
schema only gains new keys.
"""
from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .context import ReadOnlySimContext

# ``imageio`` is imported lazily inside ``BaseFrameRecorder._write_image`` so
# that framework users who never touch image recording (pure training / CI)
# do not pay the import cost or require the dependency.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_uint8_rgb_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[..., None], 3, axis=2)
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)
    elif image_array.ndim == 3 and image_array.shape[2] >= 3:
        image_array = image_array[..., :3]
    else:
        raise ValueError(f"Unsupported broadcast image shape: {image_array.shape}")
    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating):
            image_array = np.clip(image_array, 0.0, 255.0)
        else:
            image_array = np.clip(image_array.astype(np.float64), 0.0, 255.0)
        image_array = image_array.astype(np.uint8)
    return np.ascontiguousarray(image_array)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(element) for element in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


# ---------------------------------------------------------------------------
# Abstraction + base implementation
# ---------------------------------------------------------------------------
class PostActionRecorder(ABC):
    """Runs at episode boundaries and after every action step.

    Override any subset of ``on_pre_episode`` / ``on_post_action_step`` /
    ``on_post_episode``. ``observer_outputs`` is a snapshot of the EnvRuntime
    observer-plugin outputs at the time the hook fires.
    """

    def on_pre_episode(self, ctx: ReadOnlySimContext, observer_outputs: Mapping[str, Any]) -> None:
        pass

    def on_post_action_step(self, ctx: ReadOnlySimContext, observer_outputs: Mapping[str, Any]) -> None:
        pass

    def on_post_episode(self, ctx: ReadOnlySimContext, observer_outputs: Mapping[str, Any]) -> None:
        pass

    def on_attach(self) -> None:
        pass

    def on_detach(self) -> None:
        pass


class BaseFrameRecorder(PostActionRecorder):
    """Standard raw-data recorder.

    Emits a general-purpose on-disk layout (image + JSON + per-episode manifest
    + root index) that any downstream tool - such as the bundled
    ``recorder_viewer`` web UI - can consume. **No visualization work is done
    here**; overlays and plots belong to the viewer.

    Parameters
    ----------
    output_dir: root directory; created on first write.
    stride: record only every ``stride`` action steps (default 1 = every step).
    save_image: write broadcast image as PNG (default True).
    save_observer_outputs: include ``observer_outputs`` in per-step JSON (default True).
    save_core_state: include ``core_state`` in per-step JSON (default True).
    save_derived_state: include ``derived_state`` in per-step JSON (default True).
    save_sensor_data: include ``sensor_data`` in per-step JSON (default True).
    save_action: include ``action`` in per-step JSON (default True).
    save_static_data: write ``static.json`` once per episode (default True).
    save_accessor_state: convenience override. When not ``None`` this value is
        applied to ``save_core_state`` / ``save_derived_state`` /
        ``save_sensor_data`` / ``save_action`` / ``save_static_data`` at once.
        Use ``save_accessor_state=True`` to get full replay-grade recordings,
        or ``save_accessor_state=False`` to get an image-/observer-only dump.
    image_extension: extension used for image files (default ``"png"``).
    quiet: print a line per saved file if False.
    """

    # manifest_version 2: per-step JSON carries derived_state / sensor_data /
    # action; episode dir may contain ``static.json``.
    MANIFEST_VERSION = 2

    def __init__(
        self,
        output_dir: Path | str,
        stride: int = 1,
        save_image: bool = True,
        save_observer_outputs: bool = True,
        save_core_state: bool = True,
        save_derived_state: bool = True,
        save_sensor_data: bool = True,
        save_action: bool = True,
        save_static_data: bool = True,
        save_accessor_state: Optional[bool] = None,
        image_extension: str = "png",
        quiet: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.stride = max(1, int(stride))
        self.save_image = bool(save_image)
        self.save_observer_outputs = bool(save_observer_outputs)
        # ``save_accessor_state`` is a bulk override; it trumps the individual
        # accessor flags when explicitly set. This is the recommended knob for
        # callers who only care about "replay-grade ON/OFF".
        if save_accessor_state is not None:
            bulk = bool(save_accessor_state)
            save_core_state = bulk
            save_derived_state = bulk
            save_sensor_data = bulk
            save_action = bulk
            save_static_data = bulk
        self.save_core_state = bool(save_core_state)
        self.save_derived_state = bool(save_derived_state)
        self.save_sensor_data = bool(save_sensor_data)
        self.save_action = bool(save_action)
        self.save_static_data = bool(save_static_data)
        self.image_extension = image_extension.lstrip(".")
        self.quiet = bool(quiet)

        self._episode_index: int = -1
        self._current_episode_dir: Optional[Path] = None
        self._current_manifest_steps: list[dict[str, Any]] = []
        self._saved_image_paths: list[Path] = []
        self._saved_data_paths: list[Path] = []

    # ------------------------------------------------------------------
    # Public inspection
    # ------------------------------------------------------------------
    @property
    def current_episode_dir(self) -> Optional[Path]:
        return self._current_episode_dir

    def get_saved_image_paths(self) -> list[Path]:
        return list(self._saved_image_paths)

    def get_saved_data_paths(self) -> list[Path]:
        return list(self._saved_data_paths)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: ReadOnlySimContext, observer_outputs: Mapping[str, Any]) -> None:
        self._episode_index += 1
        self._current_episode_dir = self.output_dir / f"episode_{self._episode_index:05d}"
        self._current_manifest_steps = []
        self._saved_image_paths = []
        self._saved_data_paths = []
        self._static_file_name: Optional[str] = None
        # Capture episode-invariant accessor data once, BEFORE the first
        # per-step snapshot, so ``static.json`` is always present if any
        # per-step data is also present.
        if self.save_static_data:
            self._current_episode_dir.mkdir(parents=True, exist_ok=True)
            self._write_static_data(ctx, self._current_episode_dir / "static.json")
            self._static_file_name = "static.json"
        self._record_step(ctx, observer_outputs)

    def on_post_action_step(self, ctx: ReadOnlySimContext, observer_outputs: Mapping[str, Any]) -> None:
        self._record_step(ctx, observer_outputs)

    def on_post_episode(self, ctx: ReadOnlySimContext, observer_outputs: Mapping[str, Any]) -> None:
        if self._current_episode_dir is None:
            return
        self._write_episode_manifest()
        self._write_root_index()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _any_payload_enabled(self) -> bool:
        return (
            self.save_image
            or self.save_observer_outputs
            or self.save_core_state
            or self.save_derived_state
            or self.save_sensor_data
            or self.save_action
        )

    def _any_step_json_enabled(self) -> bool:
        return (
            self.save_observer_outputs
            or self.save_core_state
            or self.save_derived_state
            or self.save_sensor_data
            or self.save_action
        )

    def _record_step(self, ctx: ReadOnlySimContext, observer_outputs: Mapping[str, Any]) -> None:
        if not self._any_payload_enabled():
            return
        step_index = int(ctx.episode_step)
        if step_index % self.stride != 0:
            return
        assert self._current_episode_dir is not None  # set in on_pre_episode
        self._current_episode_dir.mkdir(parents=True, exist_ok=True)

        manifest_entry: dict[str, Any] = {
            "step": step_index,
            "physics_step": int(ctx.physics_step),
        }

        if self.save_image:
            image_name = f"step_{step_index:05d}.{self.image_extension}"
            image_path = self._current_episode_dir / image_name
            self._write_image(ctx, image_path)
            self._saved_image_paths.append(image_path)
            manifest_entry["image"] = image_name

        if self._any_step_json_enabled():
            data_name = f"step_{step_index:05d}.json"
            data_path = self._current_episode_dir / data_name
            self._write_step_data(ctx, observer_outputs, data_path)
            self._saved_data_paths.append(data_path)
            manifest_entry["data"] = data_name

        self._current_manifest_steps.append(manifest_entry)

    def _write_image(self, ctx: ReadOnlySimContext, image_path: Path) -> None:
        import imageio.v2 as imageio  # lazy import; see module docstring
        image = _ensure_uint8_rgb_image(ctx.accessor.get_broadcastview_image())
        imageio.imwrite(str(image_path), image)
        if not self.quiet:
            print(f"[frame_recorder] saved image: {image_path}", flush=True)

    def _write_step_data(
        self,
        ctx: ReadOnlySimContext,
        observer_outputs: Mapping[str, Any],
        data_path: Path,
    ) -> None:
        payload: dict[str, Any] = {
            "episode_step": int(ctx.episode_step),
            "physics_step": int(ctx.physics_step),
        }
        if self.save_observer_outputs:
            payload["observer_outputs"] = _json_sanitize(dict(observer_outputs))
        # Each accessor read is guarded + defensive: a bug in one read must
        # not take down the whole recording. We embed ``__error__`` so the
        # replay tooling can surface the problem.
        if self.save_core_state:
            payload["core_state"] = self._safe_accessor_call(
                ctx.accessor.get_core_state
            )
        if self.save_derived_state:
            payload["derived_state"] = self._safe_accessor_call(
                ctx.accessor.get_derived_state
            )
        if self.save_sensor_data:
            payload["sensor_data"] = self._safe_accessor_call(
                ctx.accessor.get_sensor_data
            )
        if self.save_action:
            payload["action"] = self._safe_accessor_call(
                ctx.accessor.get_action
            )
        with open(data_path, "w") as data_file:
            json.dump(payload, data_file, indent=2, ensure_ascii=False)
        if not self.quiet:
            print(f"[frame_recorder] saved data: {data_path}", flush=True)

    def _write_static_data(self, ctx: ReadOnlySimContext, static_path: Path) -> None:
        payload: dict[str, Any] = {
            "static_data": self._safe_accessor_call(ctx.accessor.get_static_data),
            "physical_frequency": self._safe_accessor_call(
                ctx.accessor.get_physical_frequency
            ),
        }
        with open(static_path, "w") as static_file:
            json.dump(payload, static_file, indent=2, ensure_ascii=False)
        if not self.quiet:
            print(f"[frame_recorder] saved static: {static_path}", flush=True)

    @staticmethod
    def _safe_accessor_call(fn) -> Any:
        try:
            return _json_sanitize(fn())
        except Exception as exc:  # pragma: no cover - defensive
            return {"__error__": repr(exc)}

    def _write_episode_manifest(self) -> None:
        assert self._current_episode_dir is not None
        manifest_path = self._current_episode_dir / "manifest.json"
        manifest = {
            "manifest_version": self.MANIFEST_VERSION,
            "episode_index": int(self._episode_index),
            "num_steps": len(self._current_manifest_steps),
            "steps": list(self._current_manifest_steps),
            "static": self._static_file_name,
        }
        with open(manifest_path, "w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
        if not self.quiet:
            print(f"[frame_recorder] wrote manifest: {manifest_path}", flush=True)

    def _write_root_index(self) -> None:
        """Maintain ``<output_dir>/index.json`` as the union of episode_* dirs.

        Scans the filesystem (rather than only tracking this recorder's own
        episodes) so multiple recorders or reruns end up with a coherent
        index.

        Concurrency caveat
        ------------------
        This rewrites ``index.json`` non-atomically (plain ``open(..., 'w')``).
        Writing from multiple concurrent recorders targeting the **same**
        ``output_dir`` - e.g. two worker processes sharing a directory -
        can produce a corrupted or racy index. The recommended pattern is
        one ``output_dir`` per recorder (e.g. ``debug/seed_0001/``). Inside a
        single process multiple recorders pointing at the same dir is safe
        because ``on_post_episode`` hooks run sequentially per runtime.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for episode_dir in sorted(self.output_dir.glob("episode_*")):
            if not episode_dir.is_dir():
                continue
            manifest_path = episode_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, "r") as manifest_file:
                    manifest_data = json.load(manifest_file)
            except (OSError, json.JSONDecodeError):
                continue
            entries.append(
                {
                    "episode_index": int(manifest_data.get("episode_index", -1)),
                    "dir": episode_dir.name,
                    "num_steps": int(manifest_data.get("num_steps", 0)),
                }
            )
        entries.sort(key=lambda entry: entry["episode_index"])
        index_path = self.output_dir / "index.json"
        index_payload = {
            "manifest_version": self.MANIFEST_VERSION,
            "episodes": entries,
        }
        with open(index_path, "w") as index_file:
            json.dump(index_payload, index_file, indent=2, ensure_ascii=False)
