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

To actually replay these files through the accessor interface, use
:class:`envs.framework.replay.ReplaySimulator`. It implements
``BaseSimulator`` on top of this on-disk layout, so observer / plugin /
training code can run unmodified against recordings. This recorder's
job is only to make the data **complete**; consumption is separate.

Per-step JSON schema (``step_*.json``, manifest_version=2)::

    {
      "episode_step":     int,
      "physics_step":     int,
      "observer_outputs": { <observer_name>: <any json-safe value>, ... },  # optional
      "core_state":       { ... },                                          # optional
      "derived_state":    { ... },                                          # optional
      "sensor_data":      { ... },                                          # optional
      "action":           { <agent_id>: <array>, ... },                     # optional
      "action_extras":    { <agent_id>: { ... } | null, ... }               # optional
    }

``action_extras`` is the side-channel payload the policy emits alongside
the action (e.g. ``log_prob`` / ``value`` / sampling info). Recorded as
``None`` per-agent when the caller did not supply extras for that agent.
Not part of any accessor read — it is policy-side state and is only
meaningful for training-time recordings; ignored by
:class:`envs.framework.replay.ReplaySimulator`.

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

    Temporal semantics of ``on_post_action_step``
    ----------------------------------------------
    The hook fires *after* the physics step has completed. Every parameter
    reflects a snapshot taken at that exact moment, with one deliberate
    exception:

    * ``observation`` — the observation that **produced** the action
      (i.e. the pre-action observation ``obs_t``).  It is captured
      *before* :meth:`EnvRuntime.step` advances the simulator, because
      it is the state the policy saw when it selected ``action``.
    * ``action`` — the action that was just applied (``action_t``).
    * ``observer_outputs`` — observer-plugin outputs refreshed **after**
      the step. They reflect the new post-action state.
    * ``ctx`` — the read-only context also reflects the post-action state
      (``episode_step`` / ``physics_step`` have been incremented,
      ``termination_proposals`` / ``is_terminated`` are up-to-date).

    In RL-transition terms each call represents one ``(s_t, a_t, s'_{t+1})``
    tuple where ``s_t`` lives in ``observation`` and the post-action world
    lives in ``observer_outputs`` / ``ctx``.
    """

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        pass

    def on_post_action_step(
        self,
        ctx: ReadOnlySimContext,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        """Hook fired after every action step.

        ``observation`` is the **pre-action** observation — the state that
        produced ``action``.  ``observer_outputs`` and ``ctx`` are the
        **post-action** state.

        ``action_extras`` is a per-agent bundle
        ``{"robot_a": <extras_a or None>, "robot_b": <extras_b or None>}``
        forwarded by :meth:`EnvRuntime.step` (see its docstring). It carries
        the side-channel payload produced by the policy alongside the action
        — typically ``log_prob`` / ``value`` / sample info for RL trainers,
        or ``None`` for scripted / inference-only callers that didn't pass
        extras. ``None`` (the parameter default) means "no extras at all
        this step", which is also what you get when an older caller invokes
        :meth:`EnvRuntime.step` without the new extra args.
        """
        pass

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        pass

    def on_attach(self) -> None:
        pass

    def on_detach(self) -> None:
        pass


def _snapshot(value: Any) -> Any:
    """Cheap deep-ish copy that preserves ndarrays as ndarrays.

    Unlike :func:`copy.deepcopy` this is allocation-light: ndarrays are
    copied once via ``np.array(..., copy=True)``, plain dicts/lists/tuples
    are recursed into, everything else is kept by reference (immutables
    like ``int``/``float``/``str``/``None`` are safe; mutable user objects
    would be aliased — observers in this codebase return either ndarrays
    or plain containers so that's fine in practice). The goal is just to
    decouple the recorder's stored frame from any in-place mutation an
    observer might do on its next ``on_post_action_step``.
    """
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_snapshot(element) for element in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(element) for element in value)
    return value


class EpisodeBufferRecorder(PostActionRecorder):
    """In-memory recorder that buffers raw per-step data for one episode.

    Captures, for every step the runtime emits a recorder hook on:

    * ``episode_step`` / ``physics_step``
    * full ``observer_outputs`` snapshot (every observer registered on
      the runtime, exactly as the runtime publishes it)
    * ``action`` (the action passed through from the runtime; present
      on every frame because all frames originate from
      ``on_post_action_step``)
    * ``action_extras`` (the per-agent policy side-channel forwarded by
      :meth:`EnvRuntime.step`; ``None`` when the caller did not supply
      extras)
    * ``termination_proposals`` (tuple) and ``is_terminated`` (bool) —
      lifted directly from the read-only ctx; the reading discipline
      mirrors what attached plugins see at the same hook

    Scope
    -----
    This recorder produces **raw, semantics-free** data. It is intended
    as the substrate from which RL trainers, debuggers, or visualizers
    build whatever they need — but it does **not** itself apply any
    rollout convention (no reward alignment, no advantage / GAE, no
    obs_t vs obs_t+1 staggering, no done-masking, no reset-boundary
    handling beyond "frames are grouped per episode"). Consumers that
    want rollout-shaped tensors must do that mapping themselves.

    No on-disk side effects. Episode data lives on the recorder until
    cleared by the next :meth:`on_pre_episode` (i.e. only the most
    recent episode is retained). Use :meth:`get_episode_data` to read
    it.

    Every frame originates from ``on_post_action_step``. Each frame
    stores the **pre-action** observation (``obs_t``) alongside the
    action that was taken. The terminal step's frame carries the
    populated ``termination_proposals`` so consumers can detect episode
    end without reading the runtime back.

    Parameters
    ----------
    snapshot_arrays : bool
        If True (default), copy ndarrays in observer outputs / actions
        / extras so subsequent observer ``on_post_action_step`` calls cannot
        mutate already-buffered frames. Set False to save the copy cost
        when the consumer is read-only and the episode is short-lived.
    """

    def __init__(self, snapshot_arrays: bool = True) -> None:
        self._snapshot_arrays = bool(snapshot_arrays)
        self._frames: list[dict[str, Any]] = []
        self._episode_index: int = -1
        self._base_seed: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_episode_data(self) -> dict[str, Any]:
        """Return the most recent (or in-progress) episode's data.

        Shape::

            {
                "episode_index": int,         # -1 before the first episode
                "base_seed":     int | None,  # ctx.base_seed at episode start
                "num_frames":    int,
                "frames": [
                    {
                        "episode_step":          int,
                        "physics_step":          int,
                        "observer_outputs":      {<name>: <value>, ...},
                        "action":                {<agent_id>: <ndarray>} | None,
                        "action_extras":         {<agent_id>: <dict | None>} | None,
                        "termination_proposals": tuple[str, ...],
                        "is_terminated":         bool,
                    },
                    ...
                ],
            }

        The returned structure shares (or copies, depending on
        ``snapshot_arrays``) buffers with the recorder; the recorder
        itself will overwrite them on the next ``on_pre_episode``. If
        you need to keep data past the next episode boundary, copy it
        out yourself or build a list of snapshots across episodes.
        """
        return {
            "episode_index": self._episode_index,
            "base_seed": self._base_seed,
            "num_frames": len(self._frames),
            "frames": list(self._frames),
        }

    def get_frames(self) -> list[dict[str, Any]]:
        """Convenience: just the frame list (a shallow-copied list)."""
        return list(self._frames)

    def clear(self) -> None:
        """Drop the buffered episode without waiting for the next reset."""
        self._frames = []

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        """Reset internal state for a new episode — no frame is recorded."""
        self._frames = []
        self._episode_index += 1
        self._base_seed = ctx.base_seed

    def on_post_action_step(
        self,
        ctx: ReadOnlySimContext,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        self._frames.append(
            self._build_frame(ctx, observation=observation, action=action,
                              observer_outputs=observer_outputs, action_extras=action_extras)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_frame(
        self,
        ctx: ReadOnlySimContext,
        observation: Optional[Mapping[str, Any]],
        action: Optional[Mapping[str, Any]],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]],
    ) -> dict[str, Any]:
        snapshot = _snapshot if self._snapshot_arrays else (lambda value: value)
        return {
            "episode_step": int(ctx.episode_step),
            "physics_step": int(ctx.physics_step),
            "observation": snapshot(dict(observation)) if observation is not None else None,
            "action": snapshot(dict(action)) if action is not None else None,
            "observer_outputs": snapshot(dict(observer_outputs)),
            "action_extras": (
                snapshot({agent_id: extras for agent_id, extras in action_extras.items()})
                if action_extras is not None else None
            ),
            "termination_proposals": tuple(ctx.termination_proposals),
            # NOTE: ctx.is_terminated includes TIMEOUT; for RL semantics use
            # termination_proposals directly.  This field is kept for
            # backward-compatible log consumers only.
            "is_terminated": bool(ctx.is_terminated),
        }


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
    save_action_extras: include per-agent policy extras (``log_prob`` /
        ``value`` / etc., forwarded by :meth:`EnvRuntime.step`) in
        per-step JSON (default True). Emitted only on ``on_post_action_step``
        steps; on the ``on_pre_episode`` snapshot the field is absent
        because no action has been taken yet.
    save_observation: include the per-agent observation dict in per-step
        JSON (default True). The stored observation is the **pre-action**
        observation — the state that produced the action in the same
        frame — matching the ``PostActionRecorder`` temporal semantics.
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
        save_action_extras: bool = True,
        save_observation: bool = True,
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
            # ``action_extras`` and ``observation`` are logically part of
            # "full training-grade recording" — flip them together with
            # the accessor bulk.
            save_action_extras = bulk
            save_observation = bulk
        self.save_core_state = bool(save_core_state)
        self.save_derived_state = bool(save_derived_state)
        self.save_sensor_data = bool(save_sensor_data)
        self.save_action = bool(save_action)
        self.save_action_extras = bool(save_action_extras)
        self.save_observation = bool(save_observation)
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
    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._episode_index += 1
        self._current_episode_dir = self.output_dir / f"episode_{self._episode_index:05d}"
        self._current_manifest_steps = []
        self._saved_image_paths = []
        self._saved_data_paths = []
        self._static_file_name: Optional[str] = None
        # Capture the resolved base_seed for this episode so the manifest
        # can persist it; replay re-derives every sub-seed from this value.
        # See envs/framework/SEED.md.
        self._current_base_seed: Optional[int] = ctx.base_seed
        # Capture episode-invariant accessor data once, BEFORE the first
        # per-step snapshot, so ``static.json`` is always present if any
        # per-step data is also present.
        if self.save_static_data:
            self._current_episode_dir.mkdir(parents=True, exist_ok=True)
            self._write_static_data(ctx, self._current_episode_dir / "static.json")
            self._static_file_name = "static.json"
        self._record_step(ctx, {})

    def on_post_action_step(
        self,
        ctx: ReadOnlySimContext,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        self._record_step(ctx, observer_outputs, action_extras=action_extras, observation=observation)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
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
            or self.save_action_extras
        )

    def _any_step_json_enabled(self) -> bool:
        return (
            self.save_observer_outputs
            or self.save_core_state
            or self.save_derived_state
            or self.save_sensor_data
            or self.save_action
            or self.save_action_extras
            or self.save_observation
        )

    def _record_step(
        self,
        ctx: ReadOnlySimContext,
        observer_outputs: Mapping[str, Any],
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
        observation: Optional[Mapping[str, Any]] = None,
    ) -> None:
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
            self._write_step_data(ctx, observer_outputs, data_path, action_extras=action_extras, observation=observation)
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
        action_extras: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
        observation: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "episode_step": int(ctx.episode_step),
            "physics_step": int(ctx.physics_step),
        }
        if self.save_observer_outputs:
            payload["observer_outputs"] = _json_sanitize(dict(observer_outputs))
        # ``observation`` is the pre-action observation passed by the runtime.
        # On the ``on_pre_episode`` snapshot ``observation`` is ``None`` and
        # we fall back to reading the accessor directly (the initial state).
        if self.save_observation:
            if observation is not None:
                payload["observation"] = _json_sanitize(dict(observation))
            else:
                payload["observation"] = self._safe_accessor_call(
                    ctx.accessor.get_observation
                )
        # action_extras is None on the on_pre_episode snapshot (no action
        # has happened yet) and on steps from callers that don't supply
        # extras; in both cases we simply omit the key.
        if self.save_action_extras and action_extras is not None:
            payload["action_extras"] = _json_sanitize(
                {agent_id: extras for agent_id, extras in action_extras.items()}
            )
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
            # Resolved base seed for deterministic replay; re-derive every
            # sub-seed via envs/framework/SEED.md rules. ``None`` when the
            # episode was driven outside an EpisodeRunner (raw EnvRuntime
            # test harness, ad-hoc script).
            "base_seed": (
                int(self._current_base_seed)
                if self._current_base_seed is not None
                else None
            ),
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
