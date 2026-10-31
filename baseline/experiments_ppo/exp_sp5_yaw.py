"""P5-tracking ablation: sp5_yaw (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Yaw(P5TrackingBase):
    name = "sp5_yaw"
    _EXTRAS=("torso_yaw",)


EXPERIMENT_CLASS = Sp5Yaw
