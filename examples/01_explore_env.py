"""Example 01 — 认识这个环境 (Stage 0).

面向 (Audience) : 刚拿到仓库的开发者。
阶段 (Stage)    : 在写任何策略前，先搞清楚 Env 的 I/O 边界。
学到 (Takeaway) :
  - humanoid21 的 action/observation 维度和物理频率。
  - 框架里"observer 输出一个 dict"的约定是怎么走的。
  - 一个 reset+step 的最短路径长什么样。

产物 (Outputs)  : examples/out/01_explore_env/sample_obs.json
运行 (Run)      : python examples/01_explore_env.py
"""
from __future__ import annotations

import json

import numpy as np

from _common import build_humanoid21_runtime, example_out_dir


def _summarize(name: str, arr: np.ndarray) -> dict:
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "first5": arr.flatten()[:5].tolist(),
    }


def main() -> None:
    print("=" * 70)
    print("Example 01 — 认识 humanoid21 Env")
    print("=" * 70)

    control_frequency = 20
    match_duration = 1.0
    runtime = build_humanoid21_runtime(
        match_duration=match_duration,
        control_frequency=control_frequency,
    )

    # 1) Static sizing info — helpful for designing any actor network.
    phys_freq = runtime.simulator.get_physical_frequency()
    # phy_steps_per_action lives on the internal ``_core``; the public
    # contract for policies is: "you step at ``control_frequency`` Hz and
    # the sim does N=phys/control physics ticks per step for you".
    phy_steps_per_action = runtime._core.phy_steps_per_action
    max_steps = int(match_duration * control_frequency)
    print("\n[Sizing]")
    print(f"  physics_frequency   : {phys_freq:.1f} Hz")
    print(f"  control_frequency   : {control_frequency} Hz")
    print(f"  phy_steps_per_action: {phy_steps_per_action}")
    print(f"  max_steps           : {max_steps}")
    print(f"  action_space        : {runtime.action_space}")
    print(f"  observation_space   : {runtime.observation_space}")

    # 2) Reset and peek at observer outputs. The whole framework's convention
    #    is: ``runtime.get_observer_output(name)`` returns whatever that
    #    observer's ``get_output()`` decided to produce. For humanoid21 the
    #    default ``robot_{a,b}_obs`` observers return a flat 96-D np.ndarray.
    runtime.reset(seed=0)
    obs_a = runtime.get_observer_output("robot_a_obs")
    obs_b = runtime.get_observer_output("robot_b_obs")

    print("\n[After reset — observer outputs]")
    print(f"  robot_a_obs : ndarray{obs_a.shape} dtype={obs_a.dtype}")
    print(f"  robot_b_obs : ndarray{obs_b.shape} dtype={obs_b.dtype}")

    # 3) Step once with zero action to see the runtime's shared info.
    zero_action = np.zeros(21, dtype=np.float32)
    runtime.step(zero_action, zero_action)
    info = runtime.get_shared_info()

    print("\n[After one zero-action step — shared_info]")
    # humanoid21's shared_info_builder publishes combat-relevant fields.
    for key in ("health", "damage_taken", "winner", "termination_reasons"):
        if key in info:
            print(f"  {key:<22}: {info[key]}")

    # 4) Dump a minimal, JSON-safe sample of observation + info so the user
    #    can eyeball the structure without launching Python.
    sample = {
        "sizing": {
            "physics_frequency": phys_freq,
            "control_frequency": control_frequency,
            "phy_steps_per_action": phy_steps_per_action,
            "max_steps": max_steps,
            "action_dim_per_robot": 21,
            "obs_dim_per_robot": int(obs_a.shape[0]),
        },
        "observation_summary": {
            "robot_a_obs": _summarize("robot_a_obs", obs_a),
            "robot_b_obs": _summarize("robot_b_obs", obs_b),
        },
        "shared_info_after_one_step": {
            k: info[k] for k in ("health", "damage_taken") if k in info
        },
    }

    out_path = example_out_dir("01_explore_env") / "sample_obs.json"
    out_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
    print(f"\nWrote sample → {out_path.relative_to(out_path.parents[2])}")
    print("\nDone. 下一步建议：阅读 examples/02_scripted_baseline.py 写一个自己的 Policy。")


if __name__ == "__main__":
    main()
