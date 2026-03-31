from typing import List
from pathlib import Path
import numpy as np

from .context import SimContext, TerminationReason
from .plugin import BasePlugin

class TimeoutPlugin(BasePlugin):
    """
    超时终止插件。
    当 episode 步骤达到 max_steps 时，提出超时终止请求。
    """
    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "timeout"

    def on_post_action_step(self, ctx: SimContext) -> None:
        if ctx.episode_step >= self.max_steps:
            ctx.request_termination(TerminationReason.TIMEOUT)


class VideoRecorderPlugin(BasePlugin):
    videosave_path: str|None = None
    
    @classmethod
    def set_videosave_path(cls, path: str) -> None:
        cls.videosave_path = path

    """
    视频录制插件。
    在物理步按照指定的 fps 采样图像，并在 episode 结束时保存视频。
    """
    def __init__(self, fps: int = 30, output_path: str = "video.mp4"):
        self.fps = fps
        # videosave_path 优先级高于 output_path, for external override
        self.output_path = Path(output_path) if self.videosave_path is None else Path(self.videosave_path)
        self._interval = 1
        self._frames: List[np.ndarray] = []

    @property
    def name(self) -> str:
        return "video_recorder"

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._frames.clear()
        freq = ctx.accessor.get_physical_frequency()
        self._interval = max(1, int(round(freq / self.fps)))
        
        # 录制初始帧
        frame = ctx.accessor.get_broadcastview_image()
        if frame is not None:
            self._frames.append(frame.copy())

    def on_post_phy_step(self, ctx: SimContext) -> None:
        if ctx.physics_step % self._interval == 0:
            frame = ctx.accessor.get_broadcastview_image()
            if frame is not None:
                self._frames.append(frame.copy())

    def on_post_episode(self, ctx: SimContext) -> None:
        if len(self._frames) == 0:
            return
            
        try:
            import cv2
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            height, width = self._frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            
            for frame in self._frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                
            writer.release()
            print(f"Video saved to {self.output_path} ({len(self._frames)} frames, {self.fps} FPS)")
        except ImportError:
            print("Warning: opencv-python not installed, cannot save video")
        except Exception as e:
            print(f"Error saving video: {e}")
