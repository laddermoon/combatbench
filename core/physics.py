"""
物理引擎封装

提供 MuJoCo 物理仿真的基础功能。
"""

import mujoco
import numpy as np


class PhysicsEngine:
    """
    MuJoCo 物理引擎封装

    负责管理物理仿真的生命周期和配置。
    """

    def __init__(self, gui=True, dt=0.002, arena_xml=None):
        """
        初始化物理引擎

        Args:
            gui: 是否显示 GUI
            dt: 物理步长时间 (秒)，默认 0.002 (500Hz)
            arena_xml: 八角笼 XML 文件路径
        """
        self.dt = dt
        self.gui = gui
        self.arena_loaded = False

        # 从 XML 加载模型
        if arena_xml:
            self.model = mujoco.MjSpec.from_file(arena_xml).compile()
            self.arena_loaded = True
            print(f"PhysicsEngine: Arena loaded from {arena_xml}")
        else:
            # 创建默认地面
            spec = mujoco.MjSpec()
            spec.worldbody.add('geom', type='plane', size=[20, 20, 0.1],
                               rgba=[0.8, 0.9, 0.8, 1])
            spec.opt.timestep = dt
            spec.opt.iterations = 50
            spec.opt.solver = 'PGS'
            spec.opt.integrator = 'RK4'
            self.model = spec.compile()
            print("PhysicsEngine: Using default ground plane")

        # 创建数据对象
        self.data = mujoco.MjData(self.model)

        # GUI 模式
        if gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data
            )
            print("PhysicsEngine: GUI viewer launched")
        else:
            self.viewer = None

        print(f"PhysicsEngine: Initialized with timestep={dt}")

    def step(self):
        """
        执行一步物理仿真
        """
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def reset(self):
        """
        重置物理引擎状态

        注意：这不会重置已加载的物体，只是重置物理状态
        """
        # MuJoCo 需要重新创建 MjData 来完全重置
        self.data = mujoco.MjData(self.model)

    def close(self):
        """
        关闭物理引擎
        """
        if self.viewer:
            del self.viewer
            self.viewer = None
            print("PhysicsEngine: Viewer closed")

    def get_contact_points(self, body_a, body_b):
        """
        获取两个 body 之间的接触点

        Args:
            body_a: 第一个 body 的索引
            body_b: 第二个 body 的索引

        Returns:
            contacts: 接触点列表，每个元素包含 (geom1, geom2, position, normal, force)
        """
        contacts = []

        # 遍历所有接触点
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # 获取 geom 对应的 body
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # 检查是否是目标 body 之间的接触
            if (body1 == body_a and body2 == body_b) or \
               (body1 == body_b and body2 == body_a):
                # 获取接触位置
                contact_pos = self.data.contact[i].pos
                contact_normal = self.data.contact[i].frame

                # 获取接触力 (简化版本，实际需要更复杂的计算)
                # MuJoCo 的接触力计算比较复杂，这里返回位置信息
                contacts.append({
                    'geom_a': geom1,
                    'geom_b': geom2,
                    'body_a': body1,
                    'body_b': body2,
                    'position': contact_pos,
                    'normal': contact_normal,
                })

        return contacts
