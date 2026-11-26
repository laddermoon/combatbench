"""Humanoid21 CoreClip — simulator-bound motion clip format.

See CORECLIP_SPEC.md for the format contract.
"""
import os as _os
# GL backend must be decided BEFORE the first mujoco import (the
# .check -> simulator chain sets egl, which is broken on this box).
_os.environ.setdefault('MUJOCO_GL', 'osmesa')
from .core_clip import (CONTROL_DT, CORE_FIELDS, CoreClip, META_REQUIRED,
                        SPEC_VERSION)
from .check import GroundReport, ground_report
from .fix import fix_ground_penetration
