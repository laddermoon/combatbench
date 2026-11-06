"""
Algorithm 1 (bootstrip): (Observation, P5Clip) -> SupportState seq.
"""
import numpy as np

class SupportPlanner:
    """Algorithm 1 (bootstrip): (Observation, P5Clip) -> SupportState seq.
    """
    def plan(self, observation: np.ndarray,
             clip) -> list:
        """(observation, p5 clip) -> ordered SupportState seq.

        f-chain: f1(obs)->I1 当前支撑状态，f2(obs)->I2 当前质心动量，
        f3(clip)->I3 逐帧目标支撑状态，f4(I1,I2,I3)->R 锚点序列。
        """
        from .algo1_functions import f1, f2, f3, f4
        return f4(f1(observation), f2(observation),
                  f3(clip)).landmarks
