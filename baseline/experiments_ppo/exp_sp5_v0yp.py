"""P5-tracking ablation: sp5_v0yp (+torso_yaw +balance potential)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5V0yp(P5TrackingBase):
    name = "sp5_v0yp"
    _EXTRAS = ("torso_yaw",)
    _FALL_MODE = "potential"


EXPERIMENT_CLASS = Sp5V0yp
