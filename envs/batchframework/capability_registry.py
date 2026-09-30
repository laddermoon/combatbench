"""能力注册表——旧插件/observer 类 → 加速路径处置方式（M3 W4）。

四种状态（对应 ROADMAP M3 放行条件的显式声明）：

- ``NATIVE``      有设备原生实现（factory 产出 BaseDevicePlugin/Observer）
- ``COMPAT``      经 HostBatchCompatAdapter/LegacyPluginAdapter 执行
- ``HOST_SLOW``   同上但显式慢路径（per-env 循环/逐子步语义近似）
- ``UNSUPPORTED`` 已知不可用（须人工转换）

未注册的类 = ``pending`` —— blueprint 解析时**启动即失败**，
不做猜测映射，不静默回退。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class Capability(Enum):
    NATIVE = "native"           # 设备原生实现
    COMPAT = "compat"           # host 兼容层（每 hook 至多一次物化）
    HOST_SLOW = "host_slow"     # host 兼容层，逐 env/逐子步近似
    PENDING = "pending"         # 已登记待转换——接入时按 UNSUPPORTED 拒绝
    UNSUPPORTED = "unsupported"  # 已知不可用，须人工转换


@dataclass
class CapabilityEntry:
    capability: Capability
    factory: Optional[Callable[..., Any]] = None   # (config, **ctx) → plugin
    adapter: Optional[str] = None                  # "batch" | "legacy" | "observer"
    note: str = ""


# ---------------------------------------------------------------------------
# 注册表：key = "<module>:<ClassName>"（blueprint 的 cls 字段原文）
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, CapabilityEntry] = {
    # --- 内置 ---
    "envs.batchframework.device_runtime:DeviceTimeoutPlugin": CapabilityEntry(
        Capability.NATIVE,
        factory=lambda cfg, **kw: _mk_timeout(cfg),
        note="per-env timeout，设备原生"),
    # --- standup 目标实验（M4 转换前占位，标注待转换） ---
    "envs.humanoid21.disturbance_plugins:RandomFallenStatePlugin":
        CapabilityEntry(Capability.PENDING,
                        note="M4 原生转换目标；暂不可经 compat 复用"
                             "（逐 env 随机姿态注入语义不等价）"),
    "baseline.humanoid21.rewards.standup_4stage:Standup4StageRewarder":
        CapabilityEntry(Capability.UNSUPPORTED,
                        note="M4 原生 observer 转换目标"),
    "baseline.humanoid21.plugins.standup_termination:StandupTerminationPlugin":
        CapabilityEntry(Capability.UNSUPPORTED, note="M4 原生转换目标"),
}


def _mk_timeout(cfg):
    from .device_runtime import DeviceTimeoutPlugin
    return DeviceTimeoutPlugin(max_steps=int(cfg.get("max_steps", 200)))


def register(cls_path: str, entry: CapabilityEntry) -> None:
    REGISTRY[cls_path] = entry


def lookup(cls_path: str) -> CapabilityEntry:
    """查表。未注册 → 返回 UNSUPPORTED pending 条目（由调用方决定拒绝）。"""
    return REGISTRY.get(cls_path, CapabilityEntry(
        Capability.UNSUPPORTED,
        note="unregistered — pending conversion; add explicit entry"))


def resolve_plugin(cls_path: str, config: Dict[str, Any], sim,
                   stats=None) -> Any:
    """按注册表把一个 blueprint 插件实例化并接入。

    - NATIVE: factory(config) → BaseDevicePlugin
    - COMPAT: factory 产出旧插件 → HostBatchCompatAdapter
    - HOST_SLOW: LegacyPluginAdapter（调用方须 allow_host_slow）
    - UNSUPPORTED / 未注册: 抛 ValueError（启动即失败，不猜测映射）
    """
    entry = lookup(cls_path)
    if entry.capability is Capability.NATIVE:
        if entry.factory is None:
            raise ValueError(f"{cls_path}: registered NATIVE without factory")
        return entry.factory(config)
    if entry.capability in (Capability.UNSUPPORTED, Capability.PENDING):
        raise ValueError(
            f"plugin '{cls_path}' is {entry.capability.value} on the device "
            f"path: {entry.note}")
    # COMPAT / HOST_SLOW
    if entry.factory is None:
        raise ValueError(f"{cls_path}: compat entry without factory")
    inner = entry.factory(config)
    if entry.capability is Capability.COMPAT:
        from .host_compat import HostBatchCompatAdapter
        return HostBatchCompatAdapter(inner, sim, stats=stats)
    from .host_compat import LegacyPluginAdapter
    return LegacyPluginAdapter(inner, sim)
