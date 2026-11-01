"""P5-tracking ablation: sp5_bal (continuous balance potential, no fall)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Bal(P5TrackingBase):
    name = "sp5_bal"
    _FALL_MODE = "potential"


EXPERIMENT_CLASS = Sp5Bal
