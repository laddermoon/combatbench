"""迁移 manifest——单元迁移的证据记录与失效追踪（E5 W0）。

对应 discuss.md D14："源任务、模型、配置、目标实现、依赖版本、
验证用例与结果形成可追踪 manifest；来源变化使相关验证失效。"

模型：

- ``MigrationManifest`` 由 ``AuditReport`` 派生，每单元带
  ``unit_hash``（cls+config 指纹）与 ``evidence`` 列表；
- 证据条目记录验证等级（``unit_replay`` / ``wave_contract`` /
  ``e2e_collect`` / ``train_smoke``）、输入指纹与时间；
- ``validate_freshness(env_bp)`` 逐单元重算指纹——source
  cls/config 变了 → 该单元标 ``stale``，其 evidence 全部视为
  失效（不删除，便于追溯）。

落盘位置约定：``envs/batchframework/migration_manifests/<name>.json``。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from envs.framework.blueprint import EnvBlueprint

from .migration_audit import AuditReport, _unit_hash

MANIFEST_DIR = Path(__file__).resolve().parent / "migration_manifests"

# 证据等级（由弱到强）
EVIDENCE_LEVELS = (
    "unit_replay",     # 同输入状态回放：CPU 单元 vs 设备单元逐字段对照
    "wave_contract",   # FakeBackend/real backend 波契约测试通过
    "e2e_collect",     # 设备 collect 产出契约合法 Episode
    "train_smoke",     # 训练接入冒烟（PPO 消费 device Episode）
)


@dataclass
class Evidence:
    level: str                  # EVIDENCE_LEVELS 之一
    passed: bool
    input_hash: str = ""        # 验证输入指纹（测试向量/episode 集）
    detail: str = ""
    pass_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "passed": self.passed,
                "input_hash": self.input_hash, "detail": self.detail,
                "pass_ts": self.pass_ts}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        return cls(level=d["level"], passed=bool(d["passed"]),
                   input_hash=d.get("input_hash", ""),
                   detail=d.get("detail", ""),
                   pass_ts=float(d.get("pass_ts", 0.0)))


@dataclass
class UnitManifest:
    name: str
    cls: str
    capability: str
    role: str = "plugin"                       # simulator/plugin/observer
    device_cls: str = ""                       # 设备实现类（NATIVE 时填）
    config_disposition: List[Dict[str, Any]] = field(default_factory=list)
    unit_hash: str = ""
    stale: bool = False
    evidence: List[Evidence] = field(default_factory=list)

    def add_evidence(self, level: str, passed: bool,
                     input_hash: str = "", detail: str = "") -> Evidence:
        assert level in EVIDENCE_LEVELS, f"unknown evidence level {level}"
        ev = Evidence(level=level, passed=passed, input_hash=input_hash,
                      detail=detail, pass_ts=time.time())
        self.evidence.append(ev)
        return ev

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "cls": self.cls,
                "capability": self.capability, "role": self.role,
                "device_cls": self.device_cls,
                "config_disposition": self.config_disposition,
                "unit_hash": self.unit_hash, "stale": self.stale,
                "evidence": [e.to_dict() for e in self.evidence]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnitManifest":
        return cls(name=d["name"], cls=d["cls"],
                   capability=d["capability"], role=d.get("role", "plugin"),
                   device_cls=d.get("device_cls", ""),
                   config_disposition=d.get("config_disposition", []),
                   unit_hash=d.get("unit_hash", ""),
                   stale=bool(d.get("stale", False)),
                   evidence=[Evidence.from_dict(e)
                             for e in d.get("evidence", [])])


@dataclass
class MigrationManifest:
    experiment_name: str
    blueprint_name: str
    source_bp_hash: str
    runtime_fields: Dict[str, Any] = field(default_factory=dict)
    units: List[UnitManifest] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_audit(cls, report: AuditReport,
                   experiment_name: str = "",
                   blueprint_name: str = "") -> "MigrationManifest":
        units = [UnitManifest(
            name=u.name, cls=u.cls, capability=u.capability, role=u.role,
            config_disposition=[{"key": d.key,
                                 "disposition": d.disposition,
                                 "note": d.note}
                                for d in u.config_disposition],
            unit_hash=u.unit_hash) for u in report.units]
        return cls(experiment_name=experiment_name,
                   blueprint_name=blueprint_name or report.source,
                   source_bp_hash=report.source_bp_hash,
                   runtime_fields=report.runtime_fields, units=units)

    def unit(self, name: str) -> UnitManifest:
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(name)

    # ------------------------------------------------------------------
    # 失效追踪
    # ------------------------------------------------------------------
    def validate_freshness(self, env_bp: EnvBlueprint) -> List[str]:
        """逐单元重算 cls+config 指纹；变化者标 ``stale``，返回名单。

        blueprint 整体漂移（runtime 字段/单元增减）也会反映出来：
        找不到匹配 cls 的旧单元标 stale，新增单元以
        ``"untracked:<cls>"`` 形式返回。
        """
        # 单元身份：observer 按 blueprint 键名（同 cls 可多实例、
        # config 不同）；plugin/simulator 按 cls（至多一实例语义上
        # 唯一，同名多实例由装配期名字唯一校验挡掉）。
        live_by_name = {n: spec
                        for n, spec in env_bp.observer_plugins.items()}
        live_by_cls = {spec.cls: spec for spec in env_bp.plugins}
        live_by_cls[env_bp.simulator.cls] = env_bp.simulator
        stale: List[str] = []
        for u in self.units:
            spec = (live_by_name.get(u.name) if u.role == "observer"
                    else live_by_cls.get(u.cls))
            if spec is None:
                u.stale = True
                stale.append(u.name)
                continue
            if _unit_hash(spec.cls, spec.config) != u.unit_hash:
                u.stale = True
                stale.append(u.name)
        return stale

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"experiment_name": self.experiment_name,
                "blueprint_name": self.blueprint_name,
                "source_bp_hash": self.source_bp_hash,
                "runtime_fields": self.runtime_fields,
                "units": [u.to_dict() for u in self.units]}

    def save(self, path=None) -> Path:
        path = Path(path) if path else (
            MANIFEST_DIR / f"{self.experiment_name or self.blueprint_name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2,
                                   sort_keys=True, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path) -> "MigrationManifest":
        d = json.loads(Path(path).read_text())
        return cls(experiment_name=d["experiment_name"],
                   blueprint_name=d["blueprint_name"],
                   source_bp_hash=d["source_bp_hash"],
                   runtime_fields=d.get("runtime_fields", {}),
                   units=[UnitManifest.from_dict(u)
                          for u in d.get("units", [])])


# ---------------------------------------------------------------------------
# 运行时查询（E6-W4）：collect/resume 路径的过期证据检测
# ---------------------------------------------------------------------------
def check_manifest(m: MigrationManifest, env_bp) -> Dict[str, Any]:
    """manifest × live blueprint → 新鲜度判定。

    ``blueprint_drift``：整体指纹变了但单元全部新鲜——如 max_steps
    变化（runtime 字段，不属单元证据范围），证据仍然有效但如实标注。
    ``stale_units``：单元 cls+config 漂移——过期证据，不可沿用。
    """
    from .migration_audit import _unit_hash
    bp_hash = _unit_hash("blueprint", env_bp.to_dict())
    stale = m.validate_freshness(env_bp)
    return {"experiment": m.experiment_name,
            "blueprint": m.blueprint_name,
            "source_bp_hash": m.source_bp_hash,
            "blueprint_drift": bp_hash != m.source_bp_hash,
            "stale_units": stale,
            "fresh": not stale}


def find_manifest_for(env_bp) -> Optional[MigrationManifest]:
    """在 MANIFEST_DIR 中找适用于该 blueprint 的 manifest。

    判定：manifest 记录的全部单元都能在 live blueprint 中解析到
    （simulator cls 相同 + observer 名/plugin cls 均在）——不匹配
    即视为该 manifest 与此 blueprint 无关。无匹配返回 None。
    """
    for p in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            m = MigrationManifest.load(p)
        except Exception:
            continue
        live_by_name = dict(env_bp.observer_plugins)
        live_cls = {s.cls for s in env_bp.plugins}
        live_cls.add(env_bp.simulator.cls)
        ok = True
        for u in m.units:
            if u.role == "observer":
                ok &= u.name in live_by_name
            else:
                ok &= u.cls in live_cls
        if ok and m.units:
            return m
    return None
