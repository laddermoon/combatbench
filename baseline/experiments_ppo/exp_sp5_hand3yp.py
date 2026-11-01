"""P5-tracking ablation: sp5_hand3yp (+torso_yaw +balance potential)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Hand3yp(P5TrackingBase):
    name = "sp5_hand3yp"
    _EXTRAS = ("torso_yaw",)
    _AW_STRIKE_HAND=3.0
    _FALL_MODE = "potential"


EXPERIMENT_CLASS = Sp5Hand3yp
