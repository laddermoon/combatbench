from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np
import sys
import torch

from envs.framework.policy import Policy

# Standalone weakened policy wrapper that can be loaded in parallel worker processes.
# It wraps the exported u03275 policy and adds tunable Gaussian noise to the actions.

BASE_POLICY_DIR = Path("/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_004703/policy_exports/u03275")

# Ensure the exported policy directory is in sys.path so we can import 'policy'
if str(BASE_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_POLICY_DIR))

try:
    from policy import ExportedMLPPolicy
except ImportError:
    # Fallback to importing from file directly if sys.path fails in child process
    import importlib.util
    spec = importlib.util.spec_from_file_location("policy", BASE_POLICY_DIR / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["policy"] = module
    spec.loader.exec_module(module)
    from policy import ExportedMLPPolicy


class WeakenedExportedMLPPolicy(ExportedMLPPolicy):
    """Weakened version of the u03275 recovery policy.
    Adds tunable Gaussian noise in action space to create a conservative safety buffer.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stochastic: bool = False,
        noise_std: float = 0.08,
        **kwargs: Any,
    ):
        if model_path is None:
            model_path = str(BASE_POLICY_DIR / "model.pt")
        super().__init__(model_path=model_path, stochastic=stochastic, **kwargs)
        self.noise_std = float(noise_std)

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        action, extra = super().act(observation, want_extra=want_extra)
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action.astype(np.float32), extra
