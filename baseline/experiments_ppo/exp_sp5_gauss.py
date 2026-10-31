"""P5-tracking ablation: sp5_gauss (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Gauss(P5TrackingBase):
    name = "sp5_gauss"
    _DIST_MODE="gauss"


EXPERIMENT_CLASS = Sp5Gauss
