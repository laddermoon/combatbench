"""Humanoid28 CoreClip — simulator-bound motion clip format (radians).

GL backend must be decided BEFORE the first mujoco import — osmesa is
the working backend on this box (egl crashes).
"""
import os as _os
_os.environ.setdefault('MUJOCO_GL', 'osmesa')
from .core_clip import (CONTROL_DT, CONTROL_FPS, CORE_FIELDS, N_DOF,
                        PHY_DT, CoreClip, META_REQUIRED, ROBOT_ID,
                        SPEC_VERSION, record_lineage)
