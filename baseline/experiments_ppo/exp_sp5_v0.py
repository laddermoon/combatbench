"""P5-tracking ablation: sp5_v0 (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5V0(P5TrackingBase):
    name = "sp5_v0"


EXPERIMENT_CLASS = Sp5V0
