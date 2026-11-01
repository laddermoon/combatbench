"""P5-tracking ablation: sp5_phasebal (phase weights + balance potential)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Phasebal(P5TrackingBase):
    name = "sp5_phasebal"
    _AW_PHASE = {"foot_left": (3.0, 1.5), "foot_right": (3.0, 1.5),
                 "head": (1.0, 0.75)}
    _AW_STRIKE_HAND = (0.75, 6.0)
    _FALL_MODE = "potential"


EXPERIMENT_CLASS = Sp5Phasebal
