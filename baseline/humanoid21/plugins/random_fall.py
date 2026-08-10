import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from envs.framework import BasePlugin
from envs.framework.context import SimContext


class RandomFallenStatePlugin(BasePlugin):
    """随机摔倒状态初始化插件。

    在 ``on_pre_episode`` 时，通过内部仿真实例让指定机器人从站立姿态
    施加微小扰动后自然倒下，然后将摔倒后的核心状态写回真实环境。

    工作流程：
    1. 读取真实环境的当前 core state 作为内部 sim 的初始状态。
    2. 给目标机器人设置站立姿态 + 微小噪声扰动（而非完全随机动作）。
    3. 非目标机器人每隔 ``reset_interval`` 物理步重置回初始状态，
       防止其倒下后干扰目标机器人的摔倒轨迹。
    4. 循环执行物理步，直到目标机器人高度低于阈值或达到 ``max_phy_steps``。
    5. 取摔倒后的 core state 写回真实环境。
    """

    def __init__(
        self,
        target_robots: str | Sequence[str] = "robot_a",
        max_phy_steps: int = 1000,
        height_threshold: float = 0.3,
        reset_interval: int = 50,
        noise_scale: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robots: 要初始化的机器人，``"robot_a"``、``"robot_b"`` 或 ``"both"``。
            max_phy_steps: 内部仿真最多跑多少物理步。
            height_threshold: 目标机器人 root 高度低于此值时提前终止 (m)。
            reset_interval: 每隔多少物理步重置非目标机器人回初始状态。
            noise_scale: 站立姿态上叠加的噪声幅度 (0~1)。
            random_seed: 随机种子。
        """
        if isinstance(target_robots, str):
            if target_robots == "both":
                self._target_set = {"robot_a", "robot_b"}
            else:
                self._target_set = {target_robots}
        else:
            self._target_set = set(target_robots)

        for rid in self._target_set:
            if rid not in ("robot_a", "robot_b"):
                raise ValueError(f"Invalid target_robot: {rid}")

        self.max_phy_steps = int(max_phy_steps)
        self.height_threshold = float(height_threshold)
        self.reset_interval = int(reset_interval)
        self.noise_scale = float(noise_scale)
        self._rng = np.random.RandomState(random_seed)

        self._internal_sim: Optional[Any] = None

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return "random_fallen_state"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robots": sorted(self._target_set) if len(self._target_set) > 1 else list(self._target_set)[0],
            "max_phy_steps": self.max_phy_steps,
            "height_threshold": self.height_threshold,
            "reset_interval": self.reset_interval,
            "noise_scale": self.noise_scale,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RandomFallenStatePlugin":
        return cls(**config)

    def _ensure_internal_sim(self) -> Any:
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
        return self._internal_sim

    def on_pre_episode(self, ctx: SimContext) -> None:
        sim = self._ensure_internal_sim()

        _debug = os.environ.get("COMBATBENCH_FALL_DEBUG", "0") == "1"
        _debug_dir = Path(os.environ.get("COMBATBENCH_FALL_DEBUG_DIR", "/tmp/fall_debug"))

        def _debug_img(tag: str, step_num: int = -1):
            if not _debug:
                return
            _debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                img = sim.get_broadcastview_image()
                from PIL import Image
                core = sim.get_core_state()
                h = float(core["robot_a"]["root_pos"][2])
                fname = f"{tag}_s{step_num:04d}_h{h:.3f}.png" if step_num >= 0 else f"{tag}.png"
                Image.fromarray(img).save(str(_debug_dir / fname))
                print(f"[fall_debug] saved {fname} (height={h:.4f})", flush=True)
            except Exception as e:
                print(f"[fall_debug] render failed: {e}", flush=True)

        # 1. 读取真实环境的当前 core state
        real_state = ctx.accessor.get_core_state()

        # 2. 初始化内部 sim（reset 到站姿，然后写入真实状态）
        sim.reset()
        sim.set_core_state(real_state)
        _debug_img("00_initial")

        # 3. 保存非目标机器人的初始状态（用于定期重置）
        non_target_state = {
            rid: {k: v.copy() for k, v in state.items()}
            for rid, state in real_state.items()
            if rid not in self._target_set
        }

        # 4. 给目标机器人设置站立姿态 + 微小噪声扰动
        #    站立姿态对应 action=0（关节归一化位置接近 0），
        #    叠加小幅高斯噪声让机器人自然失去平衡而倒下。
        random_action = {}
        for rid in ("robot_a", "robot_b"):
            if rid in self._target_set:
                base = real_state[rid].get(
                    "joint_pos_norm",
                    np.zeros(21, dtype=np.float32),
                )
                noise = self._rng.normal(
                    0.0, self.noise_scale, size=(21,)
                ).astype(np.float32)
                random_action[rid] = np.clip(base + noise, -1.0, 1.0)
            else:
                random_action[rid] = real_state[rid].get(
                    "joint_pos_norm",
                    np.zeros(21, dtype=np.float32),
                )
        sim.set_action(random_action)
        _debug_img("01_after_set_action")

        # 5. 循环物理步
        _debug_milestones = {1, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 1500, 2000, 2500}
        step = -1
        min_height = float("inf")
        for step in range(self.max_phy_steps):
            sim.physical_step()

            # 定期重置非目标机器人
            if non_target_state and (step + 1) % self.reset_interval == 0:
                sim.set_core_state(non_target_state)

            # 检查目标机器人高度是否低于阈值
            core = sim.get_core_state()
            min_height = min(
                (float(core[rid]["root_pos"][2]) for rid in self._target_set if rid in core),
                default=float("inf"),
            )

            if _debug and (step + 1) in _debug_milestones:
                _debug_img("step", step + 1)

            if min_height < self.height_threshold:
                break

        if _debug:
            _debug_img("99_final", step + 1)
            print(f"[fall_debug] total_steps={step+1} final_height={min_height:.4f}", flush=True)

        # 6. 取最终 core state 写回真实环境
        fallen_state = sim.get_core_state()
        result_state = {}
        for rid in self._target_set:
            if rid in fallen_state:
                result_state[rid] = fallen_state[rid]

        ctx.mutator.set_core_state(result_state)

        # 记录 metrics
        for rid in self._target_set:
            if rid in fallen_state:
                ctx.metrics[f"{rid}_fallen_init_steps"] = step + 1
                ctx.metrics[f"{rid}_fallen_init_height"] = float(fallen_state[rid]["root_pos"][2])
                ctx.metrics[f"{rid}_fallen_init_height_threshold"] = min_height < self.height_threshold
