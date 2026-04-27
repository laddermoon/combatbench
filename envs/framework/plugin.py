from typing import Any, Dict, List, Optional
from .context import SimContext

class BasePlugin:
    """
    仿真扩展插件（Simulation Plugin）。
    
    职责：
    1. 在仿真引擎的特定生命周期节点注入自定义逻辑。
    2. 通过 ctx.accessor 读取数据。
    3. 通过 ctx.mutator 修改数据（如果当前生命周期被允许，否则为 None）。
    """
    
    @property
    def name(self) -> str:
        return "unnamed_plugin"
        
    @property
    def priority(self) -> int:
        return 0

    @property
    def require_mutator(self) -> bool:
        """
        是否申请数据操作权限。
        如果为 False，即使处于允许修改的生命周期（如 on_pre_phy_step），
        传给此插件的 ctx.mutator 依然会是 None。遵循最小权限原则。
        """
        return False

    # ==========================================
    # 随机性钩子 (Randomness Hook)
    # ==========================================

    def set_episode_seed(self, seed: int) -> None:
        """[时机]: Episode 开始前、on_pre_episode 之前，由 EpisodeRunner 调用。
        [职责]: 持有 RNG 的 plugin 在此立即重建自己的 RNG。

        默认 no-op：不消费随机性的 plugin 无须重写。实现方请在此方法内
        直接 ``self._rng = np.random.RandomState(int(seed))`` （或等价写法），
        不要推迟到 on_pre_episode，以保证 set_episode_seed 是唯一的 RNG
        重建入口。详见 ``SEED.md``。
        """
        pass

    # ==========================================
    # 生命周期钩子 (Lifecycle Hooks)
    # ==========================================

    def on_pre_episode(self, ctx: SimContext) -> None:
        """
        [时机]: 在一个新的 Episode 开始前，环境刚刚被重置时。
        [职责]: 初始化状态（Resetter）、清空历史统计。
        [权限]: 读写（ctx.mutator 可用）。
        """
        pass

    def on_pre_action_step(self, ctx: SimContext) -> None:
        """
        [时机]: 接收到外部动作（Action），但还未拆分并下发给物理引擎之前。
        [职责]: 控制模式映射（Control Modes）、动作空间映射、动作限幅。
        [权限]: 读写（ctx.mutator 可用，可修改动作）。
        """
        pass

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """
        [时机]: 在每一次极细粒度的物理仿真步（Physical Step）执行之前。
        [职责]: 注入外部扰动（Disturbances）。
        [权限]: 读写（ctx.mutator 可用，可修改外部力）。
        """
        pass

    def on_post_phy_step(self, ctx: SimContext) -> None:
        """
        [时机]: 在每一次极细粒度的物理仿真步执行之后，且状态已更新。
        [职责]: 刚性状态约束（Constraints），高频数据收集。
        [权限]: 读写（ctx.mutator 可用，可强行覆盖状态实现投影）。
        """
        pass

    def on_post_action_step(self, ctx: SimContext) -> None:
        """
        [时机]: 一个 Action Step 对应的所有物理步都执行完毕之后。
        [职责]: 指标聚合（Metrics）、犯规判定（Terminations）、计算奖励。
        [权限]: 只读（ctx.mutator 为 None）。
        """
        pass

    def on_post_episode(self, ctx: SimContext) -> None:
        """
        [时机]: Episode 确定终止之后。
        [职责]: 整局维度的日志汇总、数据上报。
        [权限]: 只读（ctx.mutator 为 None）。
        """
        pass

    # ==========================================
    # 管理钩子 (Management Hooks)
    # ==========================================
    #
    # 设计意图
    # --------
    # ``on_pre_episode`` / ``on_post_episode`` 绑定的是 **episode** 生命
    # 周期（每局都触发）。``on_attach`` / ``on_detach`` 绑定的是 **runtime
    # 挂载** 生命周期（跨所有 episode 只触发一次），用来管理"跟随 plugin
    # 实例而非 episode"的一次性资源与缓存。
    #
    # 为什么不能用 ``__init__`` / ``__del__`` 替代：
    #   * ``__init__`` 在用户代码里执行，那时 plugin 还没挂到 runtime 上，
    #     无法感知 runtime 的生命周期（没有办法在 ``runtime.close()`` 时
    #     得到通知）；
    #   * ``__del__`` 在 Python 里时机不确定（可能永远不触发），不适合
    #     释放文件句柄 / socket / GPU context 这类有副作用的资源。
    #
    # 框架调度点（不要依赖其它调用时机）：
    #   * ``EnvRuntime.attach_plugin(plugin)`` / ``attach_recorder``
    #     → ``plugin.on_attach()``
    #   * ``EnvRuntime.detach_plugin(plugin)`` / ``detach_recorder``
    #     → ``plugin.on_detach()``
    #   * ``EnvRuntime.close()`` 内部 ``clear()`` 逐个 detach →
    #     所有在册 plugin 的 ``on_detach`` 都会被保证调用一次（优雅关闭
    #     契约，见 tests/test_edge_cases.py::test_close_clears_plugins）。

    def on_attach(self) -> None:
        """
        [时机]: plugin 被挂到 runtime 时（``EnvRuntime.attach_plugin``），
                同一个 plugin 实例在一个 runtime 上只触发一次；若被 detach
                后再 attach，会再次触发。
        [职责]: 一次性资源的申请与缓存初始化——
                * 打开日志 / 视频写入文件句柄；
                * 建立 socket、连接远程服务；
                * 分配 GPU context、预编译 kernel；
                * 清空/初始化跨 episode 复用的缓存（例如
                  ``_ObserverDispatcherPlugin`` 在此清除去重 token）。
        [权限]: 此时还没有 ctx，不要访问仿真状态；需要读取初始状态请放到
                ``on_pre_episode``。
        [默认实现]: no-op。没有一次性资源需求的 plugin 无须重写。
        """
        pass

    def on_detach(self) -> None:
        """
        [时机]: plugin 从 runtime 脱离时（``EnvRuntime.detach_plugin``），
                或 ``runtime.close()`` 统一清理时（此时所有 plugin 的
                ``on_detach`` 都会被保证调用，契约见
                ``test_edge_cases.py::test_close_clears_plugins``）。
                与 ``on_attach`` 一一对应；在一个 runtime 上不会出现没
                attach 过就 detach 的情况。
        [职责]: 释放 ``on_attach`` 申请的资源——
                * flush 并关闭视频 / 日志文件；
                * 关闭 socket、断开远程连接；
                * 释放 GPU 显存、清空持久缓存。
        [权限]: 此时 ctx 已不可用，且 episode 可能已经终止或从未开始；
                不要访问仿真状态。
        [默认实现]: no-op。
        """
        pass
