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
    # {config_key: 处置说明}——blueprint config 里**不在**单元构造签名
    # 中的键必须在此显式解释（如 "由 from_blueprint 消费"/"无设备语义，
    # 已确认忽略"），否则 migration_audit 判定为 unknown 处置失败。
    config_notes: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# 注册表：key = "<module>:<ClassName>"（blueprint 的 cls 字段原文）
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, CapabilityEntry] = {
    # --- 内置 ---
    "envs.batchframework.device_runtime:DeviceTimeoutPlugin": CapabilityEntry(
        Capability.NATIVE,
        factory=lambda cfg, **kw: _mk_timeout(cfg),
        note="per-env timeout，设备原生"),
    # --- simulator 条目（device 绑定见 binding_registry；config_notes
    #     供 migration_audit 处置 + 绑定的未消费键检查） ---
    "envs.humanoid21.simulator:Humanoid21Simulator": CapabilityEntry(
        Capability.NATIVE,
        note="device binding: humanoid21-warp（binding_registry）",
        config_notes={
            "debug_torque": "CPU-only 调试打印，设备端无语义——忽略",
        }),
    # --- step 目标实验（端到端步态迁移） ---
    "baseline.humanoid21.end2end.gait_clock_simulator"
    ":GaitClockSimulator": CapabilityEntry(
        Capability.NATIVE,
        note="WarpGaitClockSimulator + GaitClockObsBuilder"
             "（binding humanoid21-gait-warp）；obs 96→99 附加"
             "cmd_L/cmd_R/wprog 步态时钟，帧索引 = episode_steps",
        config_notes={
            "initial_distance": "绑定消费——转发进 warp sim 构造"
                                "（CPU 经 **kwargs 入 Humanoid21Simulator）",
            "initial_pose_a": "同上——绑定消费",
            "initial_pose_b": "同上——绑定消费",
            "debug_torque": "CPU-only 调试打印，设备端无语义——忽略",
        }),
    "baseline.humanoid21.end2end.foot_state_observer"
    ":FootStateObserver": CapabilityEntry(
        Capability.NATIVE,
        factory=lambda cfg, **kw: _mk_foot_state(cfg, **kw),
        note="DeviceFootStateObserver；STANDING_FOOT_Z/端点几何引用 "
             "CPU 模块单一来源，接触判定与 _detect_contact 同式"),
    # --- standup 目标实验（M4 已转换，见 M4_RESULTS.md） ---
    "envs.humanoid21.disturbance_plugins:RandomFallenStatePlugin":
        CapabilityEntry(Capability.NATIVE,
                        factory=lambda cfg, **kw: _mk_fallen(cfg, **kw),
                        note="DeviceFallenResetPlugin；摔倒分布统计等价"
                             "（fp32 并行 rollout，验收见 M4_RESULTS §3）"),
    "baseline.humanoid21.rewards.standing_balance_4stage"
    ":StandingBalance4StageRewarder":
        CapabilityEntry(Capability.NATIVE,
                        factory=lambda cfg, **kw: _mk_standup_rewarder(
                            cfg, **kw),
                        note="DeviceStandup4StageRewarder observer；"
                             "常量引用 CPU 模块单一来源"),
    "baseline.humanoid21.plugins.standup_termination:StandupTerminationPlugin":
        CapabilityEntry(Capability.UNSUPPORTED,
                        note="不在 standup_4stage_dense_v2 蓝图内；"
                             "需要时显式做原生转换"),
    # --- basic_balance 目标实验（E5 迁移，验收见 E5_PLAN §放行） ---
    "baseline.humanoid21.plugins.imbalance_termination"
    ":DualImbalanceTerminationPlugin":
        CapabilityEntry(Capability.NATIVE,
                        factory=lambda cfg, **kw: _mk_dual_imbalance(
                            cfg, **kw),
                        note="DeviceDualImbalancePlugin；逐 agent 终止，"
                             "接触判定逐字段对齐（fp32 近似）"),
    "baseline.humanoid21.rewards.cross_support"
    ":CrossSupportBalanceRewarder":
        CapabilityEntry(Capability.NATIVE,
                        factory=lambda cfg, **kw: _mk_cross_support(
                            cfg, **kw),
                        note="DeviceCrossSupportObserver；状态机张量化"),
    "baseline.humanoid21.rewards.posture_reward:PostureRewarder":
        CapabilityEntry(Capability.NATIVE,
                        factory=lambda cfg, **kw: _mk_posture(cfg, **kw),
                        note="DevicePostureObserver；"
                             "STANDING_JOINT_POS 引用 CPU 常量"),
    "baseline.humanoid21.plugins.height_phi_observer:HeightPhiObserver":
        CapabilityEntry(Capability.NATIVE,
                        factory=lambda cfg, **kw: _mk_height_phi(
                            cfg, **kw),
                        note="DeviceHeightPhiObserver"),
    # --- E7-W0 计量探针（device_examples 模板5；仅 benchmark 蓝图引用） ---
    "envs.batchframework.device_examples:SubstepProbePlugin":
        CapabilityEntry(Capability.NATIVE,
                        factory=lambda cfg, **kw: _mk_substep_probe(
                            cfg, **kw),
                        note="空子步 hook——测子步驱动路径固定开销"),
}


def _mk_foot_state(cfg, sim=None, **_kw):
    from .device_step import DeviceFootStateObserver
    if sim is None:
        raise ValueError("DeviceFootStateObserver factory requires sim=")
    aid = cfg.get("agent_id", "robot_a")
    params = {k: v for k, v in cfg.items() if k != "agent_id"}
    return DeviceFootStateObserver.from_sim(
        sim, 0 if aid == "robot_a" else 1, **params)


def _mk_substep_probe(cfg, **_kw):
    from .device_examples import SubstepProbePlugin
    return SubstepProbePlugin()


def _mk_timeout(cfg, **_kw):
    from .device_runtime import DeviceTimeoutPlugin
    return DeviceTimeoutPlugin(max_steps=int(cfg.get("max_steps", 200)))


def _mk_fallen(cfg, sim=None, **_kw):
    """需要共享 batch sim（bind_shared_sim）——由 resolve_plugin 传入。"""
    from .device_standup import DeviceFallenResetPlugin
    from .warp_simulator import WarpHumanoid21Simulator
    if sim is None:
        raise ValueError("DeviceFallenResetPlugin factory requires sim=")
    return DeviceFallenResetPlugin(
        sim_factory=lambda b: WarpHumanoid21Simulator(batch_size=b),
        target_robots=cfg.get("target_robots", ["robot_a", "robot_b"]),
        max_phy_steps=int(cfg.get("max_phy_steps", 1000)),
        height_threshold=float(cfg.get("height_threshold", 0.3)),
        reset_interval=int(cfg.get("reset_interval", 5)),
    ).bind_shared_sim(sim)


def _mk_standup_rewarder(cfg, sim=None, **_kw):
    """observer 版 NATIVE factory——经 resolve_observer 实例化。"""
    from .device_standup import DeviceStandup4StageRewarder
    if sim is None:
        raise ValueError("DeviceStandup4StageRewarder factory requires sim=")
    aid = cfg.get("agent_id", "robot_a")
    return DeviceStandup4StageRewarder.from_sim(
        sim, 0 if aid == "robot_a" else 1)


def _mk_dual_imbalance(cfg, sim=None, **_kw):
    from .device_balance import DeviceDualImbalancePlugin
    if sim is None:
        raise ValueError("DeviceDualImbalancePlugin factory requires sim=")
    return DeviceDualImbalancePlugin.from_sim(
        sim,
        force_threshold=float(cfg.get("force_threshold", 1.0)),
        tolerance=int(cfg.get("tolerance", 1)),
        min_height=float(cfg.get("min_height", 0.0)))


def _mk_cross_support(cfg, sim=None, **_kw):
    from .device_balance import DeviceCrossSupportObserver
    if sim is None:
        raise ValueError("DeviceCrossSupportObserver factory requires sim=")
    aid = cfg.get("agent_id", "robot_a")
    params = {k: v for k, v in cfg.items() if k != "agent_id"}
    return DeviceCrossSupportObserver.from_sim(
        sim, 0 if aid == "robot_a" else 1, **params)


def _mk_posture(cfg, sim=None, **_kw):
    from .device_balance import DevicePostureObserver
    if sim is None:
        raise ValueError("DevicePostureObserver factory requires sim=")
    aid = cfg.get("agent_id", "robot_a")
    return DevicePostureObserver.from_sim(sim, 0 if aid == "robot_a" else 1)


def _mk_height_phi(cfg, sim=None, **_kw):
    from .device_balance import DeviceHeightPhiObserver
    if sim is None:
        raise ValueError("DeviceHeightPhiObserver factory requires sim=")
    aid = cfg.get("agent_id", "robot_a")
    return DeviceHeightPhiObserver.from_sim(
        sim, 0 if aid == "robot_a" else 1,
        standing_height=float(cfg.get("standing_height", 1.28)))


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
        return entry.factory(config, sim=sim)
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


def resolve_observer(cls_path: str, config: Dict[str, Any], sim) -> Any:
    """实例化 blueprint observer（NATIVE factory → BaseDeviceObserver）。

    observer 无 compat 路径——复用旧 BaseObserverPlugin 走
    host_compat.LegacyObserverAdapter，需要时显式构造。
    """
    entry = lookup(cls_path)
    if entry.capability is not Capability.NATIVE or entry.factory is None:
        raise ValueError(
            f"observer '{cls_path}' is {entry.capability.value} on the "
            f"device path: {entry.note}")
    return entry.factory(config, sim=sim)
