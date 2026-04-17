from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import imageio.v2 as imageio
import numpy as np

from .context import ReadOnlySimContext, SimContext
from .plugin import BasePlugin


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


def _format_debug_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(np.asarray(value), precision=4, suppress_small=True, max_line_width=120)
    if isinstance(value, (list, tuple)):
        return np.array2string(np.asarray(value), precision=4, suppress_small=True, max_line_width=120)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _normalize_debug_text_lines(text_payload: Any) -> list[str]:
    if text_payload is None:
        return []
    if isinstance(text_payload, str):
        return [text_payload]
    if isinstance(text_payload, dict):
        return [f"{key}: {_format_debug_value(value)}" for key, value in text_payload.items()]
    if isinstance(text_payload, Sequence):
        return [str(line) for line in text_payload]
    return [str(text_payload)]


def _render_debug_text_panel(
    image: np.ndarray,
    lines: Sequence[str],
    background_color: Sequence[int] = (18, 18, 18),
    text_color: Sequence[int] = (240, 240, 240),
) -> np.ndarray:
    image_array = _ensure_uint8_rgb_image(image)
    rendered_lines = [str(line) for line in lines if str(line)]
    if not rendered_lines:
        return image_array
    try:
        import cv2
    except ImportError:
        return image_array
    font_scale = float(np.clip(image_array.shape[1] / 1400.0, 0.45, 0.8))
    thickness = 1 if image_array.shape[1] < 1200 else 2
    baseline_line_height = int(round(24 * font_scale)) + 10
    panel_height = max(40, 10 + baseline_line_height * len(rendered_lines))
    canvas = np.empty((image_array.shape[0] + panel_height, image_array.shape[1], 3), dtype=np.uint8)
    canvas[:panel_height] = np.asarray(background_color, dtype=np.uint8)
    canvas[panel_height:] = image_array
    origin_y = 8 + int(round(18 * font_scale))
    for line_index, line in enumerate(rendered_lines):
        y = origin_y + line_index * baseline_line_height
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            tuple(int(component) for component in text_color),
            thickness,
            cv2.LINE_AA,
        )
    return canvas


class BaseRuntimeUnit(ABC):
    def process_data(self, ctx: ReadOnlySimContext) -> None:
        return None

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self.process_data(ctx)

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        self.process_data(ctx)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self.process_data(ctx)

    def on_manual_refresh(self, ctx: ReadOnlySimContext) -> None:
        self.on_post_step(ctx)

    @abstractmethod
    def get_output(self) -> Any:
        pass


class BaseObserverPlugin(BaseRuntimeUnit, ABC):
    def save_debug_image(
        self,
        ctx: ReadOnlySimContext,
        output_dir: Path | str,
        step_index: int,
        quiet: bool = True,
    ) -> Path:
        image = _ensure_uint8_rgb_image(ctx.accessor.get_broadcastview_image())
        text_lines = _normalize_debug_text_lines(self.get_output())
        rendered_image = _render_debug_text_panel(image, text_lines)
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        image_path = resolved_output_dir / f"step_{int(step_index):05d}.png"
        imageio.imwrite(str(image_path), rendered_image)
        if not quiet:
            print(f"Saved observer debug image: {image_path}", flush=True)
        return image_path


class _ObserverDispatcherPlugin(BasePlugin):
    def __init__(self):
        self.observer_plugins: Dict[str, Optional[BaseObserverPlugin]] = {}
        self._last_process_token: Optional[Tuple[str, int, int, Tuple[str, ...], bool]] = None

    @property
    def name(self) -> str:
        return "observer_dispatcher"

    @property
    def priority(self) -> int:
        return -1_000_000

    @property
    def require_mutator(self) -> bool:
        return False

    def set_observer_plugin(self, name: str, observer_plugin: Optional[BaseObserverPlugin]) -> None:
        self.observer_plugins[name] = observer_plugin
        self._last_process_token = None

    def remove_observer_plugin(self, name: str) -> None:
        self.observer_plugins.pop(name, None)
        self._last_process_token = None

    def get_output(self, name: str) -> Any:
        observer_plugin = self.observer_plugins.get(name)
        return observer_plugin.get_output() if observer_plugin is not None else None

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._process_ctx(ctx, trigger_name="on_reset")

    def on_pre_action_step(self, ctx: SimContext) -> None:
        return None

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_action_step(self, ctx: SimContext) -> None:
        self._process_ctx(ctx, trigger_name="on_post_step")

    def on_post_episode(self, ctx: SimContext) -> None:
        self._process_ctx(ctx, trigger_name="on_post_episode")

    def on_attach(self) -> None:
        self._last_process_token = None

    def on_detach(self) -> None:
        self._last_process_token = None

    def refresh(self, ctx: SimContext, force: bool = False) -> None:
        self._process_ctx(ctx, trigger_name="on_manual_refresh", force=force)

    def _process_ctx(self, ctx: SimContext, trigger_name: str, force: bool = False) -> None:
        readonly_ctx = ReadOnlySimContext.from_sim_context(ctx)
        process_token = (
            trigger_name,
            readonly_ctx.episode_step,
            readonly_ctx.physics_step,
            readonly_ctx.termination_proposals,
            readonly_ctx.is_terminated,
        )
        if not force and process_token == self._last_process_token:
            return
        self._last_process_token = process_token
        for runtime_unit in self._iter_runtime_units():
            getattr(runtime_unit, trigger_name)(readonly_ctx)

    def _iter_runtime_units(self):
        seen: Set[int] = set()
        for runtime_unit in list(self.observer_plugins.values()):
            if runtime_unit is None:
                continue
            unit_id = id(runtime_unit)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            yield runtime_unit
