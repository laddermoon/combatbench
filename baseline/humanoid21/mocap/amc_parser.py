"""Parse CMU AMC (Acclaim Motion Capture) format.

AMC files contain per-frame motion data. Each frame starts with a frame number,
followed by lines of "bonename val1 val2 ...". Values correspond to the DOF
order defined in the ASF for that bone.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def parse_amc(path: str) -> List[Dict[str, np.ndarray]]:
    """Parse an AMC file into a list of frames.

    Returns:
        List of frames, each frame is a dict mapping bone_name → np.ndarray of DOF values.
        Root values are stored under key 'root' as [TX, TY, TZ, RX, RY, RZ].
    """
    frames: List[Dict[str, np.ndarray]] = []
    current_frame: Dict[str, np.ndarray] = {}

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip comments and directives
        if line.startswith("#") or line.startswith(":") or not line:
            i += 1
            continue

        # Frame number
        if line.isdigit():
            if current_frame:
                frames.append(current_frame)
            current_frame = {}
            i += 1
            continue

        # Bone data line
        parts = line.split()
        bone_name = parts[0]
        values = [float(x) for x in parts[1:]]
        current_frame[bone_name] = np.array(values, dtype=np.float64)
        i += 1

    if current_frame:
        frames.append(current_frame)

    return frames
