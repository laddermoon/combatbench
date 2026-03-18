import numpy as np

class CombatPolicy:
    def __init__(self, observation_space, action_space):
        """
        初始化随机策略。
        """
        self.action_dim = action_space.shape[0]

    def act(self, obs, info=None):
        """
        随机产生动作。
        """
        return np.random.uniform(-0.1, 0.1, self.action_dim)
        
    def reset(self):
        pass
