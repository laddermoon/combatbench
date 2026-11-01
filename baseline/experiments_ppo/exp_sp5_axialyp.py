"""P5-tracking ablation: sp5_axialyp (+torso_yaw +balance potential)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Axialyp(P5TrackingBase):
    name = "sp5_axialyp"
    _EXTRAS = ("torso_yaw",)
    _DIST_MODE="axial"
    _FALL_MODE = "potential"


EXPERIMENT_CLASS = Sp5Axialyp
