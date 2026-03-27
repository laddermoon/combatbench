"""
OpenSimulator 抽象类定义

所有仿真环境都实现这个抽象类，提供统一的接口用于：
- 执行仿真步进
- 访问和修改仿真状态数据
- 获取传感器观测数据
- 获取广播视角图像

这个抽象类使得外部可以灵活地实现多种功能：
- 观测数据采集
- 状态扰动
- 数据记录
- Reward计算
- 碰撞检测
- 约束执行

状态分类：
- 核心状态 (Core State): 可读可写，定义物理世界的最小完备状态集
- 衍生状态 (Derived State): 只读，由物理引擎通过正向运动学和动力学计算得出
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class OpenSimulator(ABC):
    """
    开放式仿真器抽象接口

    提供对底层物理仿真的完全访问能力，包括：
    1. 动作控制：set_action
    2. 物理步进：physical_step
    3. 数据获取：get_sensors, get_static_data, get_core_state, get_derived_state, get_broadcastview_image
    4. 状态修改：set_core_state

    设计理念：
    - 仿真器不包含业务逻辑，只负责物理仿真和数据管理
    - 业务逻辑（如Reward计算、约束执行、终止判定）由外部插件/回调实现
    - 状态修改需要正确处理物理引擎内部缓存，避免不一致

    状态分类说明：

    一、核心状态 (Core State / 最小状态集) - 可读可写
    -----------------------------------------------
    这是"可以用来控制机器人的状态"，在物理引擎中称为：
    - 广义坐标 (Generalized Coordinates, q)
    - 广义速度 (Generalized Velocities, q̇)

    包含内容：
    - 机器人的关节角度/位置
    - 关节角速度
    - 浮动基座（如机器人躯干）在世界坐标系下的 3D 位置与四元数姿态
    - 线速度和角速度

    特性：
    - 最小完备性：拥有所有的 q 和 q̇，就完全确定了当前时刻仿真世界的物理快照
    - 可修改性：这是唯一可以在仿真中显式干预和"重置"或"修改"的数据

    二、衍生状态 (Derived State / 运动学与动力学输出) - 只读
    --------------------------------------------------------
    物理引擎在获取了核心状态（q 和 q̇）以及控制输入后，
    通过复杂的正向运动学和正向动力学计算出来的结果。

    包含内容：
    - 交互与受力：接触点坐标、接触法向力与摩擦力、关节约束力
    - 运动学衍生：末端执行器笛卡尔空间坐标、雅可比矩阵、质心位置
    - 传感器数据：RGB-D 图像、IMU 读数、力矩传感器读数等

    特性：
    - 只读不可写：绝不能直接"修改"接触力或碰撞状态
    - 计算代价高昂：这些数据是物理引擎经过大量矩阵运算得出的
    """

    @abstractmethod
    def get_physical_frequency(self) -> float:
        """
        获取物理仿真频率

        Returns:
            物理仿真的频率（Hz），例如：500Hz 对应 timestep=0.002s
        """
        pass

    @abstractmethod
    def set_action(self, action: Dict[str, Any]) -> None:
        """
        接收动作指令并设置到仿真器

        Args:
            action: 动作指令字典，格式示例：
                {
                    'robot_a': np.ndarray,  # shape=(21,), 机器人A的动作
                    'robot_b': np.ndarray,  # shape=(21,), 机器人B的动作
                }

        注意：
        - 动作在下一个 physical_step 时生效
        - 动作值应该在有效范围内 [-1, 1]
        - 具体的动作格式由具体实现定义
        """
        pass

    @abstractmethod
    def physical_step(self) -> None:
        """
        执行一次物理仿真步进

        功能：
        1. 根据当前 set_action 设置的控制指令计算力矩/力
        2. 执行碰撞检测
        3. 数值积分更新位置和速度
        4. 更新所有内部缓存（正向运动学、雅可比矩阵等）

        注意：
        - 每次调用推进一个物理时间步（如 0.002秒）
        - 调用后核心状态和衍生状态会原地更新
        """
        pass

    @abstractmethod
    def get_sensor_data(self) -> Dict[str, Any]:
        """
        获取传感器数据

        Returns:
            传感器数据字典，包含但不限于：
            {

            }

        注意：
        - 传感器数据是只读的观测值
        - 每次调用返回当前状态的传感器数据
        - 这些值由物理引擎根据核心状态计算得出
        """
        pass

    @abstractmethod
    def get_static_data(self) -> Dict[str, Any]:
        """
        获取静态数据

        静态数据指在整个仿真过程中不变的配置和结构数据。

        Returns:
            静态数据字典，包含但不限于：
            {
                'robots': {
                    'robot_a': {
                        'model_type': str,
                        'dof': int,
                        'joint_names': List[str],
                        'actuator_names': List[str],
                        'body_names': List[str],
                        'geom_names': List[str],
                        'keypoint_bodies': Dict[str, str],  # 如 {'head': 'torso'}
                        'initial_position': np.ndarray,
                        'initial_orientation': np.ndarray,
                    },
                    'robot_b': {...}
                },
                'scene': {
                    'arena_type': str,
                    'arena_size': Tuple[float, float, float],
                    'floor_height': float,
                    'gravity': np.ndarray,
                    'timestep': float,
                },
                'physics': {
                    'solver': str,
                    'iterations': int,
                    'integrator': str,
                },
                'cameras': {
                    'broadcast': {
                        'name': str,
                        'position': np.ndarray,
                        'orientation': np.ndarray,
                        'fovy': float,
                        'resolution': Tuple[int, int],
                    },
                    ...
                }
            }

        注意：
        - 静态数据在仿真期间不应被修改
        - 可以缓存以避免重复计算
        """
        pass

    @abstractmethod
    def get_core_state(self) -> Dict[str, Any]:
        """
        获取核心状态（可读可写）

        核心状态是定义物理世界的最小完备状态集：
        - 广义坐标 (Generalized Coordinates, q)
        - 广义速度 (Generalized Velocities, q̇)

        Returns:
            核心状态字典，包含但不限于：
            {
                'time': float,  # 当前仿真时间
                'robots': {
                    'robot_a': {
                        # 浮动基座状态（7维：位置3 + 四元数4）
                        'root_position': np.ndarray,  # (3,) 世界坐标系位置
                        'root_orientation': np.ndarray,  # (4,) 四元数 [w,x,y,z]

                        # 浮动基座速度（6维：线速度3 + 角速度3）
                        'root_linear_velocity': np.ndarray,  # (3,)
                        'root_angular_velocity': np.ndarray,  # (3,)

                        # 关节状态
                        'joint_positions': np.ndarray,  # (21,) 关节角度
                        'joint_velocities': np.ndarray,  # (21,) 关节角速度
                    },
                    'robot_b': {...}
                }
            }

        注意：
        - 返回的核心状态数据是深拷贝，避免外部修改
        - 这些状态是唯一可以"读取和修改"的状态
        - 拥有这些状态就完全确定了当前时刻的物理快照
        """
        pass

    @abstractmethod
    def set_core_state(self, state: Dict[str, Any]) -> None:
        """
        设置核心状态（可读可写）

        直接设置仿真的核心状态，用于：
        - 重置到特定状态
        - 从历史状态恢复
        - 实现特定的测试场景
        - 域随机化（Domain Randomization）

        Args:
            state: 核心状态字典，格式与 get_core_state 返回值相同

        重要：
        修改核心状态后必须正确处理物理引擎内部缓存：
        1. 运动学缓存：正向运动学结果（body位置、朝向等）
        2. 动力学缓存：雅可比矩阵、惯性矩阵
        3. 碰撞检测缓存：Bounding Box、碰撞对等

        典型实现（以MuJoCo为例）：
        ```python
        def set_core_state(self, state):
            # 1. 设置广义坐标 q 和广义速度 q̇
            self.data.qpos[:] = state['qpos']
            self.data.qvel[:] = state['qvel']
            self.data.time = state['time']

            # 2. 清除并重新计算运动学缓存
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_camLight(self.model, self.data)
            mujoco.mj_tendon(self.model, self.data)
            mujoco.mj_transmission(self.model, self.data)

            # 3. 重新计算碰撞检测缓存
            mujoco.mj_collision(self.model, self.data)

            # 4. 重新计算动力学缓存（如果需要）
            mujoco.mj_factorM(self.model, self.data)
        ```

        注意：
        - 状态修改要保证一致性，避免非物理状态
        - 修改后缓存更新顺序很重要（先运动学，后碰撞，最后动力学）
        - 衍生状态（如接触力）会在下一个物理步重新计算
        """
        pass

    @abstractmethod
    def get_derived_state(self) -> Dict[str, Any]:
        """
        获取衍生状态（只读）

        衍生状态是物理引擎根据核心状态计算得出的结果，包括：
        - 正向运动学结果（末端位置、姿态等）
        - 正向动力学结果（接触力、约束力等）
        - 传感器读数

        Returns:
            衍生状态字典，包含但不限于：
            {
                'contacts': [
                    {
                        'geom_a': str,
                        'geom_b': str,
                        'body_a': str,
                        'body_b': str,
                        'position': np.ndarray,  # (3,) 接触点位置
                        'normal': np.ndarray,  # (3,) 接触法向
                        'force': float,  # 接触力大小
                        'friction': float,  # 摩擦力
                        'distance': float,  # 穿透深度
                    },
                    ...
                ],
                'robots': {
                    'robot_a': {
                        # 运动学衍生
                        'keypoint_positions': Dict[str, np.ndarray],  # {'head': (3,), ...}
                        'keypoint_velocities': Dict[str, np.ndarray],
                        'jacobian': Dict[str, np.ndarray],  # 各关节的雅可比矩阵
                        'com_position': np.ndarray,  # 质心位置

                        # 动力学衍生
                        'contact_forces': Dict[str, np.ndarray],  # 各接触点的力
                        'joint_constraint_forces': np.ndarray,  # 关节约束力
                    },
                    'robot_b': {...}
                }
            }

        注意：
        - 衍生状态是只读的，不能直接修改
        - 这些值是物理引擎经过复杂计算得出的
        - 如果想改变衍生状态，必须通过修改核心状态实现
        - 例如：想要产生100N接触力，只能通过调整核心状态（位置/速度）让物理引擎计算出这个力
        """
        pass

    @abstractmethod
    def get_broadcastview_image(self) -> np.ndarray:
        """
        获取当前状态下广播视角的观测图片（属于衍生状态，只读）

        Returns:
            图像数组，格式：
            - shape: (height, width, 3) 或 (height, width, 4)
            - dtype: np.uint8
            - channel: RGB 或 RGBA

        注意：
        - 每次调用渲染当前状态
        - 如果使用离线渲染，需要指定renderer
        - 图像分辨率由静态数据中的camera配置决定
        """
        pass

