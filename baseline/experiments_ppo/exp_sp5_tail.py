"""P5-tracking ablation: sp5_tail (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Tail(P5TrackingBase):
    name = "sp5_tail"
    _DIST_MODE="tail"


EXPERIMENT_CLASS = Sp5Tail
