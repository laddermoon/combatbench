"""P5-tracking ablation: sp5_phaseyp (+torso_yaw +balance potential)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Phaseyp(P5TrackingBase):
    name = "sp5_phaseyp"
    _EXTRAS = ("torso_yaw",)
    _AW_PHASE={"foot_left":(3.0,1.5),"foot_right":(3.0,1.5),"head":(1.0,0.75)}
    _AW_STRIKE_HAND=(0.75,6.0)
    _FALL_MODE = "potential"


EXPERIMENT_CLASS = Sp5Phaseyp
