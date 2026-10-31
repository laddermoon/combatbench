"""P5-tracking ablation: sp5_sparse (see p5_tracking_base.py)."""
from baseline.humanoid21.p5_tracking_base import P5TrackingBase


class Sp5Sparse(P5TrackingBase):
    name = "sp5_sparse"
    _ANCHOR_MODE="keyframe"


EXPERIMENT_CLASS = Sp5Sparse
