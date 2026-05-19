"""PostActionRecorder lifecycle + BaseFrameRecorder output format.

Covers hook ordering, observer-outputs freshness, on-disk schema (index.json
/ manifest.json / step_*.json / step_*.png), and detach semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from envs.framework.context import ReadOnlySimContext
from envs.framework.env_runtime import EnvRuntime
from envs.framework.recorder import BaseFrameRecorder, PostActionRecorder
from envs.framework.observer_plugin import BaseObserverPlugin


class _TracingRecorder(PostActionRecorder):
    def __init__(self):
        self.events: list[tuple[str, int, dict]] = []
        self.attach_count = 0
        self.detach_count = 0

    def on_attach(self):
        self.attach_count += 1

    def on_detach(self):
        self.detach_count += 1

    def _record(self, hook, ctx, observer_outputs):
        self.events.append((hook, int(ctx.episode_step), dict(observer_outputs)))

    def on_pre_episode(self, ctx):
        self._record("on_pre_episode", ctx, {})

    def on_post_action_step(self, ctx, observation, action, observer_outputs, action_extras=None):
        self._record("on_post_action_step", ctx, observer_outputs)

    def on_post_episode(self, ctx):
        self._record("on_post_episode", ctx, {})


class _StepValueObserver(BaseObserverPlugin):
    """Returns the current ctx.episode_step so we can verify freshness."""
    def __init__(self):
        self._value = -1

    def _refresh(self, ctx):
        self._value = int(ctx.episode_step)

    def on_pre_episode(self, ctx):
        self._refresh(ctx)

    def on_post_action_step(self, ctx):
        self._refresh(ctx)

    def on_post_episode(self, ctx):
        self._refresh(ctx)

    def get_output(self):
        return self._value


class TestRecorderLifecycle:
    def test_attach_detach_counts(self, mock_simulator):
        recorder = _TracingRecorder()
        runtime = EnvRuntime(simulator=mock_simulator, recorders=[recorder])
        assert recorder.attach_count == 1
        runtime.detach_recorder(recorder)
        assert recorder.detach_count == 1

    def test_close_detaches_recorders(self, mock_simulator):
        recorder = _TracingRecorder()
        runtime = EnvRuntime(simulator=mock_simulator, recorders=[recorder])
        runtime.close()
        assert recorder.detach_count == 1

    def test_hook_ordering_and_counts(self, mock_simulator):
        recorder = _TracingRecorder()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[recorder],
            max_steps=3,
        )
        runtime.reset()
        while runtime.is_episode_active:
            runtime.step(np.zeros(21), np.zeros(21))

        hook_sequence = [event[0] for event in recorder.events]
        assert hook_sequence[0] == "on_pre_episode"
        assert hook_sequence[-1] == "on_post_episode"
        # three action steps between pre- and post-episode
        assert hook_sequence.count("on_post_action_step") == 3

    def test_observer_output_is_fresh(self, mock_simulator):
        """Recorders run AFTER observer dispatcher → must see current-step values."""
        observer = _StepValueObserver()
        recorder = _TracingRecorder()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"step": observer},
            recorders=[recorder],
            max_steps=2,
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        runtime.step(np.zeros(21), np.zeros(21))

        post_events = [event for event in recorder.events if event[0] == "on_post_action_step"]
        steps_seen = [event[2]["step"] for event in post_events]
        # The observer's output at on_post_action_step must equal the current
        # ctx.episode_step (i.e. already refreshed, not stale from last step).
        assert steps_seen == [1, 2]


class TestBaseFrameRecorderOutput:
    def test_writes_expected_files(self, mock_simulator, tmp_path: Path):
        recorder = BaseFrameRecorder(
            output_dir=tmp_path,
            stride=1,
            save_image=True,
            save_observer_outputs=True,
            save_accessor_state=True,
        )
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"step": _StepValueObserver()},
            recorders=[recorder],
            max_steps=2,
        )
        runtime.reset()
        while runtime.is_episode_active:
            runtime.step(np.zeros(21), np.zeros(21))

        # index.json at root
        index_path = tmp_path / "index.json"
        assert index_path.exists()
        with open(index_path) as index_file:
            index_data = json.load(index_file)
        assert index_data["episodes"][0]["dir"] == "episode_00000"
        assert index_data["episodes"][0]["num_steps"] == 3  # pre + 2 steps

        # episode dir
        episode_dir = tmp_path / "episode_00000"
        assert (episode_dir / "manifest.json").exists()
        png_files = sorted(episode_dir.glob("step_*.png"))
        json_files = sorted(episode_dir.glob("step_*.json"))
        manifest_file = episode_dir / "manifest.json"

        # manifest excluded from step_*.json glob because name is literal
        assert manifest_file not in json_files
        # manifest.json is NOT step_*.json; glob confirms
        assert len(png_files) == 3
        assert len(json_files) == 3

        # per-step payload has the full accessor surface
        with open(json_files[0]) as payload_file:
            payload = json.load(payload_file)
        for key in (
            "observer_outputs", "core_state", "derived_state",
            "sensor_data", "action", "observation",
        ):
            assert key in payload, f"step payload missing {key!r}"
        # First file is the pre-episode snapshot (observer_outputs empty).
        # Post-action steps carry the actual observer data.
        with open(json_files[1]) as payload_file:
            payload = json.load(payload_file)
        assert "step" in payload["observer_outputs"]

    def test_static_json_written_and_covers_accessor(
        self, mock_simulator, tmp_path: Path
    ):
        """``static.json`` captures ``get_static_data`` + ``get_physical_frequency``."""
        recorder = BaseFrameRecorder(output_dir=tmp_path, save_accessor_state=True)
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[recorder],
            max_steps=1,
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        episode_dir = tmp_path / "episode_00000"
        static_path = episode_dir / "static.json"
        assert static_path.exists()
        with open(static_path) as static_file:
            static_payload = json.load(static_file)
        assert "static_data" in static_payload
        assert "physical_frequency" in static_payload
        # MockSimulator advertises dt=0.002 → 500 Hz
        assert static_payload["physical_frequency"] == pytest.approx(500.0)
        assert static_payload["static_data"]["dt"] == pytest.approx(0.002)

        # manifest points at static.json
        with open(episode_dir / "manifest.json") as manifest_file:
            manifest = json.load(manifest_file)
        assert manifest["manifest_version"] == BaseFrameRecorder.MANIFEST_VERSION
        assert manifest["static"] == "static.json"

    def test_save_accessor_state_false_skips_all_accessor_payload(
        self, mock_simulator, tmp_path: Path
    ):
        """Bulk override ``save_accessor_state=False`` disables static.json and
        every accessor-derived per-step section. Image + observer_outputs are
        independent flags and keep working."""
        recorder = BaseFrameRecorder(
            output_dir=tmp_path,
            save_accessor_state=False,
            save_image=False,
            save_observer_outputs=True,
        )
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"step": _StepValueObserver()},
            recorders=[recorder],
            max_steps=1,
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        episode_dir = tmp_path / "episode_00000"
        assert not (episode_dir / "static.json").exists()
        step_json_files = sorted(episode_dir.glob("step_*.json"))
        assert len(step_json_files) >= 1
        with open(step_json_files[0]) as payload_file:
            payload = json.load(payload_file)
        assert "observer_outputs" in payload
        for key in ("core_state", "derived_state", "sensor_data", "action"):
            assert key not in payload, f"{key!r} should not be written"

    def test_per_flag_selectivity(self, mock_simulator, tmp_path: Path):
        """Individual ``save_*`` flags control their section independently."""
        recorder = BaseFrameRecorder(
            output_dir=tmp_path,
            save_image=False,
            save_observer_outputs=False,
            save_core_state=True,
            save_derived_state=False,
            save_sensor_data=True,
            save_action=False,
            save_static_data=False,
        )
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[recorder],
            max_steps=1,
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        episode_dir = tmp_path / "episode_00000"
        assert not (episode_dir / "static.json").exists()
        with open(sorted(episode_dir.glob("step_*.json"))[0]) as payload_file:
            payload = json.load(payload_file)
        assert set(payload.keys()) == {
            "episode_step", "physics_step", "core_state", "sensor_data",
            "observation",
        }

    def test_save_image_false_skips_png(self, mock_simulator, tmp_path: Path):
        recorder = BaseFrameRecorder(
            output_dir=tmp_path, stride=1, save_image=False
        )
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[recorder],
            max_steps=1,
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))

        episode_dir = tmp_path / "episode_00000"
        assert len(list(episode_dir.glob("step_*.png"))) == 0
        assert len(list(episode_dir.glob("step_*.json"))) >= 1

    def test_stride_filters_steps(self, mock_simulator, tmp_path: Path):
        recorder = BaseFrameRecorder(output_dir=tmp_path, stride=2)
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[recorder],
            max_steps=4,
        )
        runtime.reset()
        while runtime.is_episode_active:
            runtime.step(np.zeros(21), np.zeros(21))

        episode_dir = tmp_path / "episode_00000"
        png_files = sorted(episode_dir.glob("step_*.png"))
        # steps 0, 2, 4 recorded → 3 files
        step_indices = [int(path.stem.split("_")[1]) for path in png_files]
        assert step_indices == [0, 2, 4]
