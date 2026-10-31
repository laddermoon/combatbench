"""P5-tracking ablation: sp5_geom (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Geom(P5TrackingBase):
    name = "sp5_geom"
    _EXTRAS=("geom",)


EXPERIMENT_CLASS = Sp5Geom
