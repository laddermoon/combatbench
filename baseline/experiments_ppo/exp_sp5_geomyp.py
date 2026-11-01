"""P5-tracking ablation: sp5_geomyp (+torso_yaw +balance potential)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Geomyp(P5TrackingBase):
    name = "sp5_geomyp"
    _EXTRAS = ("torso_yaw", "geom")
    _EXTRAS=("torso_yaw","geom")
    _FALL_MODE = "potential"


EXPERIMENT_CLASS = Sp5Geomyp
