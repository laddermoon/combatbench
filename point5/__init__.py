"""Point5 business language (v2: torso-facing local frame).

See DESIGN.md. This package must not import mujoco / envs.* / baseline.*.
"""
from .contracts import (
    CLIP_FORMAT,
    FRAMES_FORMAT,
    GROUND_Z,
    MIN_FOOT_SEP_XY,
    N_POINTS,
    PLANT_HEIGHT_TOL,
    POINT_INDEX,
    POINT_NAMES,
    P5Clip,
    Point5Error,
    P5,
    STATE_FORMAT,
    TaskValidationError,
    Transfer,
    ankle_line_dir,
    is_degenerate,
    to_local,
    twist_deg,
)
# trajectory module still on the old Keyframe model — re-enable when ported
# from .trajectory import (FrameSample, frames_payload, keyframe_frames,
#                          keyframe_world, sample_clip)

__all__ = [
    "CLIP_FORMAT",
    "FRAMES_FORMAT",
    "GROUND_Z",
    "MIN_FOOT_SEP_XY",
    "N_POINTS",
    "P5Clip",
    "PLANT_HEIGHT_TOL",
    "POINT_INDEX",
    "POINT_NAMES",
    "Point5Error",
    "P5",
    "STATE_FORMAT",
    "TaskValidationError",
    "Transfer",
    "ankle_line_dir",
    "is_degenerate",
    "to_local",
    "twist_deg",
]
