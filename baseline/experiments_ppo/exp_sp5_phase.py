"""P5-tracking ablation: sp5_phase (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Phase(P5TrackingBase):
    name = "sp5_phase"
    _AW_PHASE={"foot_left":(3.0,1.5),"foot_right":(3.0,1.5),"head":(1.0,0.75)}
    _AW_STRIKE_HAND=(0.75,6.0)


EXPERIMENT_CLASS = Sp5Phase
