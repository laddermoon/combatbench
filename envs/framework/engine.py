from typing import Any, Dict, List, Optional
import warnings

from .backend import BaseSimulator
from .context import SimContext
from .plugin import BasePlugin

class PluginManager:
    """内部组件：按优先级调度插件。"""
    def __init__(self):
        self._plugins: List[BasePlugin] = []

    def attach(self, plugin: BasePlugin) -> None:
        if plugin in self._plugins:
            return
        self._plugins.append(plugin)
        self._plugins.sort(key=lambda p: p.priority, reverse=True)
        plugin.on_attach()

    def detach(self, plugin: BasePlugin) -> None:
        if plugin in self._plugins:
            self._plugins.remove(plugin)
            plugin.on_detach()

    def clear(self) -> None:
        for plugin in list(self._plugins):
            self.detach(plugin)

    def invoke(self, hook_name: str, ctx: SimContext, allow_mutator: bool = False) -> None:
        """
        触发所有插件的指定生命周期方法。
        引擎可以通过 allow_mutator 决定当前生命周期是否在宏观上允许数据修改。
        同时结合 plugin.require_mutator 进行双重检查，实现最小权限原则。
        """
        for plugin in self._plugins:
            try:
                method = getattr(plugin, hook_name, None)
                if method:
                    # 细粒度权限控制：仅当生命周期允许且插件主动申请时才授予 Mutator
                    if allow_mutator and plugin.require_mutator:
                        ctx._grant_mutator()
                    else:
                        ctx._revoke_mutator()
                        
                    method(ctx)
            except Exception as e:
                warnings.warn(f"Plugin '{plugin.name}' failed at {hook_name}: {e}")
        
        # 兜底：循环结束后回收操作权限
        ctx._revoke_mutator()

class SimEngine:
    """仿真核心驱动引擎。"""
    def __init__(
        self, 
        simulator: BaseSimulator, 
        phy_steps_per_action: int = 1,
    ):
        self.simulator = simulator
        self.phy_steps_per_action = phy_steps_per_action
        
        self.ctx = SimContext(simulator)
        self.plugin_manager = PluginManager()
        self._is_episode_active = False

    def attach_plugin(self, plugin: BasePlugin) -> None:
        self.plugin_manager.attach(plugin)

    def detach_plugin(self, plugin: BasePlugin) -> None:
        self.plugin_manager.detach(plugin)

    @property
    def is_episode_active(self) -> bool:
        return self._is_episode_active

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        self.ctx.clear_episode_state()
        self._is_episode_active = True
        
        self.simulator.reset(seed=seed, options=options)
        
        self.plugin_manager.invoke('on_pre_episode', self.ctx, allow_mutator=True)

        if self.ctx.is_terminated:
            self._handle_termination()
            return

    def step(self, action: Dict[str, Any]) -> None:
        if not self._is_episode_active:
            return

        # 临时赋予 mutator 权限以接收初始动作
        self.ctx._grant_mutator()
        self.ctx.mutator.set_action(action)
        self.ctx._revoke_mutator()
        
        # pre_action_step 允许修改动作
        self.plugin_manager.invoke('on_pre_action_step', self.ctx, allow_mutator=True)
        
        if self._check_and_handle_termination(): return

        for _ in range(self.phy_steps_per_action):
            # pre_phy_step 允许施加扰动力
            self.plugin_manager.invoke('on_pre_phy_step', self.ctx, allow_mutator=True)
            if self._check_and_handle_termination(): return
            
            # --- 核心物理推进 ---
            self.simulator.physical_step()
            self.ctx.physics_step += 1

            # post_phy_step 允许强制拉回状态
            self.plugin_manager.invoke('on_post_phy_step', self.ctx, allow_mutator=True)
            if self._check_and_handle_termination(): return

        self.ctx.episode_step += 1

        # post_action_step 为只读，强制屏蔽 mutator (allow_mutator 默认为 False)
        self.plugin_manager.invoke('on_post_action_step', self.ctx, allow_mutator=False)
        self._check_and_handle_termination()

    def _check_and_handle_termination(self) -> bool:
        if self.ctx.is_terminated:
            self._handle_termination()
            return True
        return False

    def _handle_termination(self) -> None:
        self._is_episode_active = False
        # post_episode 也是只读
        self.plugin_manager.invoke('on_post_episode', self.ctx)

    def close(self) -> None:
        self.plugin_manager.clear()
        self._is_episode_active = False
        self.simulator.close()
