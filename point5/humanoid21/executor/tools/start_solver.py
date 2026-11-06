"""Tool 8: Observation -> the start SupportLandMark.

The current state is itself a SupportLandMark — the first element of any
algorithm-1 output sequence.  Extraction:

    com  = tool1 com_estimate(FK-decoded local P5)
    vel  = obs[46:49] root linear velocity rotated to local frame
    feet = obs[52:54] contact force ratios; cop ~= foot xy when
           loaded; foot_pos always the decoded foot xy
    torso_yaw = None (v1)
"""
from __future__ import annotations

import numpy as np

from .com_solver import com_estimate
from ..contracts import FootSupport, SupportLandMark

_CONTACT_EPS = 0.05        # force_ratio below this = airborne


def extract_landmark(observation: np.ndarray,
                     decoder=None) -> SupportLandMark:
    """96-dim policy obs -> start SupportLandMark."""
    obs = np.asarray(observation, np.float64)
    if decoder is None:
        from .obs_decoder import ObsDecoder
        decoder = ObsDecoder("robot_a")
    local_p5, R_loc = decoder.decode_p5_local(obs)

    fr = obs[52:54]
    feet = {}
    for side, fi, k in (("right", 3, 0), ("left", 4, 1)):
        feet[side] = FootSupport(
            force_ratio=float(fr[k]),
            pos=local_p5[fi, :2].copy())
    return SupportLandMark(com=com_estimate(local_p5), feet=feet)
