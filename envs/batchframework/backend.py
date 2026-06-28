"""Batch backend contracts for vectorized simulation.

设计原则：
1. 接口与旧 framework 的 backend.py 保持同构，只是数据多了一个 batch dim (B,)。
2. MJX/JAX 的复杂性完全封装在具体实现（如 MjxSimulator）内部，
   用户和插件写者只需要面对 numpy (B, ...) 数组。
3. 终止检测和 reset 逻辑由外层（BatchRuntime / runner）控制，
   simulator 只负责物理推进和数据读写。
4. 不暴露 JAX 概念——所有接口的输入输出都是 numpy array 或 Python 容器。

与旧 backend.py 的对应关系：
    IDataAccessor       → IBatchDataAccessor
    IDataMutator        → IBatchDataMutator
    BaseSimulator       → BaseBatchSimulator

核心区别：
    旧: get_core_state() → {"robot_a": {"root_pos": (3,), ...}}
    新: get_core_state() → {"robot_a": {"root_pos": (B, 3), ...}}
    旧: set_action({"robot_a": (21,)})
    新: set_action({"robot_a": (B, 21)})
    旧: reset(seed=42)
    新: reset(seeds=(B,) array)  或 reset() 用内部 seed
    旧: set_core_state(state) — 全量设置
    新: set_core_state(state, env_ids) — 支持部分重置（只改某些 env）
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Read-only data accessor (batched)
# ---------------------------------------------------------------------------

class IBatchDataAccessor(ABC):
    """批量数据访问器契约（只读）。

    所有返回的数组第一维都是 batch dim (B,)。

    性能约束：
        每次 get_* 调用都会触发一次 host←device 数据传输（JAX→numpy）。
        调用者应尽量在每个 action step 结束后集中读取一次，
        避免在每个物理步后都调用。需要多个物理步的中间状态时，
        应使用 physical_step(n_steps, keep_history=True) 一次性收集，
        再通过 get_core_state(history=True) / get_derived_state(history=True) 读取。
    """

    @abstractmethod
    def get_batch_size(self) -> int:
        """获取 batch 维度大小（环境数量）。"""
        pass

    @abstractmethod
    def get_static_data(self) -> Dict[str, Any]:
        """获取静态属性（不随 batch / step 变化）。

        返回结构与旧 framework 相同，不含 batch dim：
        - per-robot 字段（dof_names, body_names, joint_limits 等）
        - 全局字段（dt, ground_geom_name 等）

        这些是模型固有的，所有 env 共享同一份。
        """
        pass

    @abstractmethod
    def get_core_state(self, history: bool = False) -> Dict[str, Any]:
        """获取核心状态（批量）。

        Args:
            history: False（默认）返回当前最新状态，数组 shape: (B, ...)。
                True 返回上次 physical_step(keep_history=True) 收集的
                逐步历史，数组 shape: (B, n_steps, ...)。
                读取历史不会清空缓冲区——缓冲区在下一次 physical_step
                调用时自动清空。
                如果上次 physical_step 未启用 keep_history，
                则返回空 dict。

        Returns:
            与旧 framework 结构相同，但所有数组第一维是 batch dim。
            history=False 时 shape: (B, ...)。
            history=True 时 shape: (B, n_steps, ...)。
            例如 (history=False):
            {
                "robot_a": {
                    "root_pos": (B, 3),
                    "root_rot": (B, 4),
                    "root_vel_local": (B, 3),
                    "root_angular_vel_local": (B, 3),
                    "joint_pos_norm": (B, 21),
                    "joint_vel_norm": (B, 21),
                },
                "robot_b": { ... },
            }

        Note:
            每次调用触发一次 host←device 传输。建议每个 action step
            结束后调用一次，不要在每个物理步后调用。
        """
        pass

    @abstractmethod
    def get_derived_state(
        self,
        fields: Optional[Sequence[str]] = None,
        history: bool = False,
    ) -> Dict[str, Any]:
        """获取派生状态（批量）。

        Args:
            fields: 需要的字段列表。None 表示返回全部。
                与旧 framework 一致：'torso_distance', 'contacts',
                'robot_a', 'robot_b'。
            history: False（默认）返回当前最新状态，数组 shape: (B, ...)。
                True 返回上次 physical_step(keep_history=True) 收集的
                逐步历史，数组 shape: (B, n_steps, ...)。
                读取历史不会清空缓冲区——缓冲区在下一次 physical_step
                调用时自动清空。
                如果上次 physical_step 未启用 keep_history，
                则返回空 dict。

        Returns:
            所有数组第一维是 batch dim。
            history=False 时 shape: (B, ...)。
            history=True 时 shape: (B, n_steps, ...)。
            例如 (history=False):
            {
                "torso_distance": (B, 1),
                "robot_a": {
                    "root_state": (B, ...),
                    "body_xpos": {"torso": (B, 3), ...},
                    ...
                },
            }

        Note:
            contacts 字段在 batch 下为 padding 后的定长数组，
            配合 contact_count 字段表示有效接触数。

            每次调用触发一次 host←device 传输。建议每个 action step
            结束后调用一次，不要在每个物理步后调用。
        """
        pass

    @abstractmethod
    def get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据（批量）。"""
        pass

    @abstractmethod
    def get_action(self) -> Dict[str, Any]:
        """获取当前正在执行的动作（批量）。

        Returns:
            {"robot_a": (B, action_dim), "robot_b": (B, action_dim)}
        """
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """获取观测向量（批量）。

        Returns:
            {"robot_a": (B, obs_dim), "robot_b": (B, obs_dim)}
            或与旧 framework 一致的嵌套结构，但所有数组带 batch dim。
        """
        pass

    def get_broadcastview_image(self, env_ids: Optional[Sequence[int]] = None) -> Any:
        """获取广播视角图像（可选实现）。

        Args:
            env_ids: 需要渲染的环境索引列表。None 表示渲染第一个环境。
                渲染是昂贵的 CPU/GPU 操作，batch 模式下通常只渲染少量环境。

        Note:
            默认返回 None。需要渲染的 simulator 子类选择性实现。
        """
        return None


# ---------------------------------------------------------------------------
# Write-only data mutator (batched)
# ---------------------------------------------------------------------------

class IBatchDataMutator(ABC):
    """批量数据操作器契约（可写）。

    所有输入数组的第一维都是 batch dim (B,)。
    """

    @abstractmethod
    def set_action(self, action: Dict[str, Any]) -> None:
        """设置动作（批量）。

        设置后在后续的 physical_step / step_n 中持续有效，
        直到下一次 set_action 覆盖。

        Args:
            action: {"robot_a": (B, action_dim), "robot_b": (B, action_dim)}

        Note:
            每个 action step 调用一次即可，physical_step 内部会
            自动使用当前已设置的 action。不需要每个物理步重复设置。
        """
        pass

    @abstractmethod
    def set_core_state(
        self,
        state: Dict[str, Any],
        env_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """设置核心状态（批量，支持部分重置）。

        外层通过此方法实现 env 重置：当某个 env 终止后，
        外层构造一个初始状态（两个机器人站立对峙），
        通过 set_core_state(state, env_ids=[3]) 只重置第 3 个 env。

        Args:
            state: 与 get_core_state() 返回结构相同。
                当 env_ids is None 时，state 的所有数组第一维必须是 B，
                表示全量设置所有 env 的状态。
                当 env_ids 指定时，state 的数组第一维必须等于 len(env_ids)，
                表示只设置这些 env 的状态，其余 env 不受影响。
            env_ids: 需要设置的 env 索引列表。None 表示全量设置所有 env。
                例如 [3, 17, 842] 表示只重置这三个 env。

        Note:
            此方法代价较高（触发 host→device 传输 + 状态覆盖）。
            典型用法是 episode 结束时重置已终止的 env，
            不要在物理步级别频繁调用。
        """
        pass

    def apply_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: Optional[np.ndarray] = None,
        robot_id: str = "robot_a",
    ) -> None:
        """对指定 body 施加持续外力（批量，可选实现）。

        语义：设置后该外力在后续所有 physical_step / step_n 中持续有效，
        直到再次调用本方法覆盖或清除。不需要每个物理步重复设置。

        典型用法：在 action step 开始时设置一次（如风力、推力扰动），
        然后调用 step_n(10)，10 个物理步内该力持续作用。
        下一个 action step 如果需要新的力，再调用本方法覆盖即可。

        要清除外力，调用 apply_external_force(body_name, force=np.zeros((B, 3)))。

        Args:
            body_name: body 名称
            force: (B, 3) 力向量，持续作用直到覆盖或清除
            torque: (B, 3) 可选力矩向量，同样持续有效
            robot_id: 机器人 ID
        """
        pass


# ---------------------------------------------------------------------------
# Batch simulator (combines accessor + mutator + lifecycle)
# ---------------------------------------------------------------------------

class BaseBatchSimulator(IBatchDataAccessor, IBatchDataMutator):
    """批量物理仿真器的抽象契约。

    与旧 BaseSimulator 的对应：
        reset(seed)          → reset(seeds) 或 reset()
        physical_step()      → physical_step(n_steps, keep_history)
        get_physical_frequency() → get_physical_frequency()  (不变)

    新增：
        batch_size 属性
        physical_step 支持 n_steps 多步推进和 keep_history 历史收集
        get_core_state / get_derived_state 支持 history=True 读取历史

    生命周期控制：
        simulator 不负责终止检测和 auto-reset。
        外层（BatchRuntime / runner）负责：
        1. 每步通过 get_core_state() / get_derived_state() 读取状态
        2. 在 Python 层判定哪些 env 需要终止
        3. 通过 set_core_state(state, env_ids=[...]) 重置已终止的 env

    实现者的责任：
        1. physical_step() 内部完成所有 env 的物理推进（GPU 向量化）
        2. 所有 get_* 方法返回 numpy array（从 JAX array 转换到 host）
        3. 所有 set_* 方法接受 numpy array（转换到 device）
        4. set_core_state(state, env_ids) 只修改指定 env 的状态
        5. history 缓冲区在每次 physical_step 调用时自动清空
    """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """batch 维度大小（环境数量）。"""
        pass

    @abstractmethod
    def reset(
        self,
        seeds: Optional[np.ndarray] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """重置所有环境。

        Args:
            seeds: (B,) int array。每个 env 的随机种子。
                None 时使用内部默认种子。
            options: 可选的 per-episode 参数（初始距离、姿态等）。
                可以是标量（所有 env 共享）或 (B,) array（per-env 不同）。
        """
        pass

    @abstractmethod
    def physical_step(
        self,
        n_steps: int = 1,
        keep_history: bool = False,
    ) -> None:
        """执行物理仿真步（所有 env 同时推进）。

        n_steps=1 时为单步推进。n_steps>1 时内部应使用 jax.lax.scan
        把 n 步编译成一个 XLA 循环，中间不回 Python，不需要 host-device
        同步——这是批量仿真器的关键性能优化，将 round-trip 次数从 N 降到 1。

        不包含终止检测或 auto-reset 逻辑——那些由外层负责。

        Args:
            n_steps: 连续推进的物理步数，默认 1。
            keep_history: 是否在 GPU 内部收集每步状态快照。
                False（默认）：只保留最终状态，零额外显存。
                True：收集每步快照存入内部缓冲区，
                    事后通过 get_history() 按需读取。
                    GPU 计算量不变，额外代价仅为显存占用
                    （n_steps × 每步快照 × B）。

        典型用法：
            # 单步
            sim.physical_step()

            # 多步，不需要历史
            sim.physical_step(10)

            # 多步，需要历史
            sim.physical_step(10, keep_history=True)
            core = sim.get_core_state()                        # 读最终状态 (B, ...)
            hist = sim.get_core_state(history=True)             # 读历史 (B, 10, ...)
            # 也可以读派生状态的历史
            dhist = sim.get_derived_state(['contacts'], history=True)
        """
        # 默认实现：逐步调用（兼容非 MJX 后端，子类应覆盖以使用 scan）
        if keep_history:
            self._history_buffer = []
            for _ in range(n_steps):
                self._single_step()
                self._history_buffer.append(self.get_core_state())
        else:
            for _ in range(n_steps):
                self._single_step()

    @abstractmethod
    def get_physical_frequency(self) -> float:
        """获取物理仿真的运行频率（Hz）。"""
        pass

    def close(self) -> None:
        """释放资源。"""
        pass
