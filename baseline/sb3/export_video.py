from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parents[2] / "run_policy_video.py"
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    result = subprocess.run([sys.executable, str(script_path), *sys.argv[1:]], env=env, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
