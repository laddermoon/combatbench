# Policy Submission Guide

Welcome to the CombatBench robotic combat challenge! This guide will show you how to submit your control policy to the arena platform.

## 1. Policy Interface Specification

The policy you submit must be encapsulated as a Python class that conforms to the following interface. The platform will instantiate your class and call it to compute actions.

```python
import numpy as np

class CombatPolicy:
    def __init__(self, observation_space, action_space):
        """
        Initialize your policy.
        
        Args:
            observation_space: The observation space definition
            action_space: The action space definition
        """
        self.action_dim = action_space.shape[0]

    def act(self, obs, info=None):
        """
        Calculate the action based on the current environment observation.
        
        Args:
            obs: np.ndarray, the current frame observation vector (details in docs/OBSERVATION.md)
            info: dict, optional additional info
            
        Returns:
            action: np.ndarray, action vector (desired joint torques or positions)
        """
        # Example: Random policy
        return np.random.uniform(-0.1, 0.1, self.action_dim)
        
    def reset(self):
        """
        Called when an episode ends or the environment is reset.
        Useful for cleaning up internal state (e.g., RNN hidden states).
        """
        pass
```

## 2. Dependency Limitations

To ensure fairness and system security, your code must adhere to the following limitations:

- **Allowed Libraries:** Only pre-installed platform libraries like `numpy`, `torch`, `scipy` are allowed by default. If your policy requires special packages, include a `requirements.txt` with your submission.
- **File Size Limit:** The total size of model weight files (.pt, .pth, .onnx) should not exceed **1GB**.
- **Inference Time Limit:** Your `act` function must return an action within **10 milliseconds** (100Hz frequency). Timeouts will result in a penalized zero-action or automatic round forfeiture.

## 3. Submission Workflow

1. Save your policy class as `policy.py`.
2. Place your model weights (if any) in the same directory, e.g., `model.pt`.
3. If there are other auxiliary files (such as configs), place them in the directory as well.
4. Use the provided submission tool to pack and submit the entire directory.

Directory structure example:
```
my_submission/
├── policy.py       # MUST contain the CombatPolicy class
├── model.pt        # Your pre-trained weights
└── config.json     # Custom configuration file
```

## 4. Using the Submission Tool (CLI)

We provide a command-line tool to help you verify and pack your submission:

```bash
# Verify if the policy interface conforms to the specification
python utils/submit_tool.py verify my_submission/

# Pack the submission
python utils/submit_tool.py pack my_submission/
```

This will generate a `submission.zip` file.

## 5. Proceeding to the Web Platform

Once you have the `submission.zip`, head over to our official combat arena platform:

**🔗 [https://arena.combatbench.com](https://arena.combatbench.com) (Example link)**

Log in to your account, navigate to the "Submit" page, and upload the generated ZIP file to participate in the ladder matchmaking.
