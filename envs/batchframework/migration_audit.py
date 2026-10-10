"""迁移审计——把 blueprint 解析成单元级能力/配置处置表（E5 W0）。

对应 discuss.md D14 的两条硬要求：

1. **单元级能力表**：simulator/plugin/observer 各自的能力状态
   （native/compat/host_slow/pending/unsupported）+ 原因。
2. **配置处置表**：blueprint 里每个 config 键必须有明确处置——
   ``consumed``（构造签名消费）/ ``declared``（注册条目的
   ``config_notes`` 显式解释的非签名键）/ ``unknown``（无解释 →
   审计失败）。未使用的配置项**不允许静默忽略**。

审计不实例化任何单元（不依赖 GPU/CUDA）；CPU 单元的读取面经
``inspect.getsource`` 静态扫描 ``ctx.accessor.*`` 调用得出——是
提示性信息（供迁移排期），不是能力证明。

用法::

    rep = audit_blueprint(env_bp, source="basic_balance_v2_phi_dual_env.yaml")
    assert not rep.unknown_keys()          # 全部 config 键有处置
    print(rep.to_markdown())
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from envs.framework.blueprint import ClassSpec, EnvBlueprint

from . import capability_registry as _capreg


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class ConfigDisposition:
    """单个 config 键的处置结论。"""

    key: str
    disposition: str          # "consumed" | "declared" | "unknown"
    note: str = ""


@dataclass
class UnitAudit:
    """blueprint 中一个单元（plugin/observer/simulator）的审计结果。"""

    role: str                 # "simulator" | "plugin" | "observer"
    name: str                 # observer 为 blueprint 键名，其余为 cls 短名
    cls: str
    capability: str           # Capability.value 或 "unbound"(simulator)
    note: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    config_disposition: List[ConfigDisposition] = field(default_factory=list)
    cpu_reads: List[str] = field(default_factory=list)     # accessor.* 调用
    cpu_hooks: List[str] = field(default_factory=list)     # 覆写的 on_* 钩子
    unit_hash: str = ""                                     # cls+config+src 指纹
    device_cls: str = ""                                   # 注册条目声明的设备实现类

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role, "name": self.name, "cls": self.cls,
            "capability": self.capability, "note": self.note,
            "config": self.config,
            "config_disposition": [
                {"key": d.key, "disposition": d.disposition, "note": d.note}
                for d in self.config_disposition],
            "cpu_reads": list(self.cpu_reads),
            "cpu_hooks": list(self.cpu_hooks),
            "unit_hash": self.unit_hash,
            "device_cls": self.device_cls,
        }


@dataclass
class AuditReport:
    """一个 blueprint 的完整审计结果。"""

    source: str               # blueprint 来源（文件名/实验名）
    source_bp_hash: str       # blueprint 整体指纹（manifest 失效追踪用）
    runtime_fields: Dict[str, Any] = field(default_factory=dict)
    units: List[UnitAudit] = field(default_factory=list)

    def unit(self, cls: str) -> Optional[UnitAudit]:
        for u in self.units:
            if u.cls == cls:
                return u
        return None

    def unknown_keys(self) -> List[Tuple[str, str]]:
        """返回 [(unit_name, key)]——所有无处置解释的 config 键。"""
        out: List[Tuple[str, str]] = []
        for u in self.units:
            for d in u.config_disposition:
                if d.disposition == "unknown":
                    out.append((u.name, d.key))
        return out

    def blocked_units(self) -> List[UnitAudit]:
        """设备路径上会启动即拒绝的单元。"""
        return [u for u in self.units
                if u.capability in ("unsupported", "pending", "unbound")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_bp_hash": self.source_bp_hash,
            "runtime_fields": self.runtime_fields,
            "units": [u.to_dict() for u in self.units],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True,
                          ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [f"# 迁移审计：{self.source}",
                 f"- source_bp_hash: `{self.source_bp_hash}`",
                 f"- runtime: `{self.runtime_fields}`", "",
                 "| 单元 | 角色 | 能力 | 未知键 | 备注 |",
                 "|---|---|---|---|---|"]
        for u in self.units:
            unk = [d.key for d in u.config_disposition
                   if d.disposition == "unknown"]
            lines.append(
                f"| `{u.name}` | {u.role} | **{u.capability}** | "
                f"{','.join(unk) or '-'} | {u.note} |")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 审计实现
# ---------------------------------------------------------------------------

_HOOK_RE = re.compile(r"def (on_\w+)\(")
_ACCESSOR_RE = re.compile(r"accessor\.(get_\w+|observe_\w+)")


def _import_cls(cls_path: str):
    module, _, name = cls_path.partition(":")
    try:
        import importlib
        mod = importlib.import_module(module)
        return getattr(mod, name)
    except Exception:
        return None


def _init_params(cls_path: str) -> Optional[set]:
    """CPU 类 ``__init__`` 的显式参数名集（构造签名消费键）。"""
    cls = _import_cls(cls_path)
    if cls is None:
        return None
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    return {p for p in sig.parameters
            if p not in ("self", "args", "kwargs")
            and sig.parameters[p].kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD)}


def _static_surface(cls_path: str) -> Tuple[List[str], List[str]]:
    """静态扫描 CPU 单元源码：(accessor 调用, 覆写的 hook)。"""
    cls = _import_cls(cls_path)
    if cls is None:
        return [], []
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return [], []
    reads = sorted(set(_ACCESSOR_RE.findall(src)))
    hooks = sorted(set(_HOOK_RE.findall(src)))
    return reads, hooks


def _src_fingerprint(cls_path: str) -> Optional[str]:
    """cls 定义所在**模块文件**的 sha256（16 hex）；解析不出 → ``None``。

    指纹对象是文件而非 ``inspect.getsource(cls)``——模块级常量
    （如 foot_state 的 STANDING_FOOT_Z、公式阈值）在类源码之外，
    文件粒度才能覆盖。已知残留：跨模块依赖（基类、他模块 import
    的常量/函数）不纳入——那需要依赖闭包，超出现阶段粒度。
    """
    cls = _import_cls(cls_path)
    if cls is None:
        return None
    try:
        f = inspect.getsourcefile(cls)
    except TypeError:
        return None
    if not f:
        return None
    try:
        return hashlib.sha256(
            Path(f).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _unit_hash(cls: str, config: Dict[str, Any],
               device_cls: Optional[str] = None) -> str:
    """单元证据指纹 = {cls 名, config, CPU 模块文件, 设备实现模块文件}。

    任一侧漂移 → 旧证据 stale（discuss D14：源任务、目标实现变化
    均使验证失效）。源码解析失败的一侧记空串——`REGISTRY` 全
    NATIVE 条目的可解析性由测试锁定（指纹不静默空转）。
    """
    payload = json.dumps(
        {"cls": cls, "config": config,
         "cpu_src": _src_fingerprint(cls) or "",
         "dev_src": (_src_fingerprint(device_cls) or "")
         if device_cls else ""},
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _disposition(cls_path: str, config: Dict[str, Any],
                 entry: _capreg.CapabilityEntry) -> List[ConfigDisposition]:
    """对照构造签名 + 注册条目 config_notes 处置每个 config 键。"""
    params = _init_params(cls_path)
    notes = getattr(entry, "config_notes", None) or {}
    out: List[ConfigDisposition] = []
    for key in sorted(config):
        if key in notes:
            out.append(ConfigDisposition(key, "declared", notes[key]))
        elif params is not None and key in params:
            out.append(ConfigDisposition(key, "consumed",
                                         "constructor parameter"))
        elif params is None:
            out.append(ConfigDisposition(key, "unknown",
                                         "class signature unavailable"))
        else:
            out.append(ConfigDisposition(key, "unknown",
                                         "not in constructor signature"))
    return out


def _audit_unit(role: str, name: str, spec: ClassSpec) -> UnitAudit:
    entry = _capreg.lookup(spec.cls)
    reads, hooks = _static_surface(spec.cls)
    return UnitAudit(
        role=role, name=name, cls=spec.cls,
        capability=entry.capability.value, note=entry.note,
        config=dict(spec.config),
        config_disposition=_disposition(spec.cls, spec.config, entry),
        cpu_reads=reads, cpu_hooks=hooks,
        unit_hash=_unit_hash(spec.cls, spec.config,
                             getattr(entry, "device_cls", None)),
        device_cls=getattr(entry, "device_cls", None) or "")


def audit_blueprint(env_bp: EnvBlueprint, source: str = "") -> AuditReport:
    """对 EnvBlueprint 做单元级迁移审计（不实例化、无 GPU 依赖）。

    simulator 行用 ``binding_registry`` 判定（解析失败 = unbound），
    plugin/observer 用 ``capability_registry``。
    """
    rep = AuditReport(
        source=source,
        source_bp_hash=_unit_hash("blueprint", env_bp.to_dict()),
        runtime_fields={
            "phy_steps_per_action": env_bp.phy_steps_per_action,
            "max_steps": env_bp.max_steps,
            "strict": env_bp.strict,
        })

    # simulator：能力 = binding_registry 是否可解析
    sim = env_bp.simulator
    try:
        from .binding_registry import resolve_binding
        binding = resolve_binding(sim.cls)
        sim_cap, sim_note = "native", f"binding→{type(binding).__name__}"
    except Exception as exc:
        sim_cap, sim_note = "unbound", str(exc)
    sim_entry = _capreg.lookup(sim.cls)
    rep.units.append(UnitAudit(
        role="simulator", name=sim.cls.rpartition(":")[2], cls=sim.cls,
        capability=sim_cap, note=sim_note,
        config=dict(sim.config),
        config_disposition=_disposition(sim.cls, sim.config, sim_entry),
        unit_hash=_unit_hash(
            sim.cls, sim.config,
            getattr(sim_entry, "device_cls", None)),
        device_cls=getattr(sim_entry, "device_cls", None) or ""))

    for spec in env_bp.plugins:
        rep.units.append(_audit_unit(
            "plugin", spec.cls.rpartition(":")[2], spec))
    for name, spec in env_bp.observer_plugins.items():
        rep.units.append(_audit_unit("observer", name, spec))
    return rep


def audit_yaml(path, source: str = "") -> AuditReport:
    """从 parameterized blueprint yaml 直接审计。"""
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    env_bp = ParameterizedEnvBlueprint.load(path).materialize()
    return audit_blueprint(env_bp, source=source or str(path))
