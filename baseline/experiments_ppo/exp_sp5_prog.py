"""P5-tracking ablation: sp5_prog (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Prog(P5TrackingBase):
    name = "sp5_prog"
    _ANCHOR_MODE="progress"


EXPERIMENT_CLASS = Sp5Prog
