"""P5-tracking ablation: sp5_hand3 (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Hand3(P5TrackingBase):
    name = "sp5_hand3"
    _AW_STRIKE_HAND=3.0


EXPERIMENT_CLASS = Sp5Hand3
