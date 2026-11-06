"""
Algorithm 1 (bootstrip): (Observation, P5Clip) -> SupportState seq.
"""
import numpy as np

class SupportPlanner:
    """Algorithm 1 (bootstrip): (Observation, P5Clip) -> SupportPlan.
    """
    def plan(self, observation: np.ndarray,
             clip) -> "SupportPlan":
        """(observation, p5 clip) -> SupportPlan (统一首帧 P5 系).

        f-chain: f12(obs)->(I1,I2) 当前支撑+动量，
        f3(clip)->I3 逐帧目标支撑状态，f4(I1,I2,I3)->R 锚点序列。
        """
        from .algo1_functions import f12, f3, f4
        from .contracts import SupportPlan
        i1, i2 = f12(observation)
        return f4(i1, i2, f3(clip))
