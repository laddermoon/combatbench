import abc
import numpy as np

class BaseRobot(abc.ABC):
    """
    Abstract base class for all combat robots.
    Provides standard interface that CombatGymEnv expects.
    """
    
    @property
    @abc.abstractmethod
    def ACTION_DIM(self) -> int:
        pass

    def __init__(self, physics_engine, position, orientation, robot_id="robot", color=(0.8, 0.2, 0.2)):
        self.physics = physics_engine
        self.model = physics_engine.model
        self.data = physics_engine.data
        self.robot_id = robot_id
        self.color = color

    @abc.abstractmethod
    def apply_action(self, action: np.ndarray):
        """Apply control action to the robot's actuators."""
        pass

    @abc.abstractmethod
    def get_observation(self, opponent_robot=None) -> np.ndarray:
        """Get the full observation vector for this robot."""
        pass

    @abc.abstractmethod
    def get_position(self) -> np.ndarray:
        """Get the root position [x, y, z] of the robot."""
        pass

    @abc.abstractmethod
    def reset(self, position, orientation):
        """Reset the robot to a specific position and orientation."""
        pass

    @abc.abstractmethod
    def get_visual_observation(self, camera_name: str) -> np.ndarray:
        """
        Get the visual observation from a specific camera on the robot.
        Future support for pure vision-based RL.
        """
        pass
