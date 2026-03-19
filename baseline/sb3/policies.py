from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .normalization import ObservationNormalizer


DEFAULT_TARGET_HEIGHT = 1.282



def load_training_metadata(model_path: str | Path) -> dict:
    model_file = Path(model_path)
    candidate_paths = [
        model_file.parent / "run_config.json",
        model_file.parent.parent / "run_config.json",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


class SB3CombatPolicy:
    def __init__(
        self,
        model_path: str | Path,
        deterministic: bool = True,
        device: str = "auto",
        target_height: float | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.deterministic = deterministic
        self.metadata = load_training_metadata(model_path)
        effective_target_height = target_height
        if effective_target_height is None:
            effective_target_height = self.metadata.get("target_height", DEFAULT_TARGET_HEIGHT)
        self.normalizer = ObservationNormalizer(target_height=float(effective_target_height))
        self.model = PPO.load(self.model_path, device=device)

    def act(self, obs, info=None):
        normalized_obs = self.normalizer.normalize(np.asarray(obs, dtype=np.float32))
        action, _ = self.model.predict(normalized_obs, deterministic=self.deterministic)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def reset(self):
        return None
