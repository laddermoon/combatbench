"""P5-tracking ablation: sp5_axial (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Axial(P5TrackingBase):
    name = "sp5_axial"
    _DIST_MODE="axial"


EXPERIMENT_CLASS = Sp5Axial
