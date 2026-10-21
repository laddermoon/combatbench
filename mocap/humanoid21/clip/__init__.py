"""Humanoid21 CoreClip — simulator-bound motion clip format.

See CORECLIP_SPEC.md for the format contract.
"""
from .core_clip import (CONTROL_DT, CORE_FIELDS, CoreClip, META_REQUIRED,
                        SPEC_VERSION)
from .check import GroundReport, ground_report
from .fix import fix_ground_penetration
