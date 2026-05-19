"""EnvBlueprint: serializable specification of an :class:`EnvRuntime`.

Purpose
-------
A blueprint captures **what defines a simulation environment** —
- the :class:`BaseSimulator` class and its construction config,
- the ordered list of :class:`BasePlugin` instances (excluding debug-only
  ones such as :class:`VideoRecorderPlugin`),
- the named :class:`BaseObserverPlugin` map,
- runtime-level knobs (``phy_steps_per_action``, ``max_steps``, ``strict``).

It deliberately **does not** capture recorders, video plugins, or any
plugin marked ``BLUEPRINT_EXCLUDE = True``: those are debug / IO concerns
attached on top of an environment, not part of its identity.

The blueprint can be serialized as YAML (preferred, human-readable) or
JSON (fallback when PyYAML is missing) and round-tripped back into an
``EnvRuntime`` via :meth:`EnvBlueprint.build`. The restored runtime is a
fresh, fully constructed object — once handed back to the user there is
no lingering restriction; they may attach any plugin via the standard
``EnvRuntime`` API.

Component protocol
------------------
Each serializable class **may** opt in by implementing two methods::

    class MyPlugin(BasePlugin):
        def to_blueprint(self) -> dict:
            return {"foo": self.foo}

        @classmethod
        def from_blueprint(cls, config: dict) -> "MyPlugin":
            return cls(**config)

If absent, the defaults are:

* ``to_blueprint`` -> ``{}`` (works for stateless plugins / observers).
* ``from_blueprint`` -> ``cls(**config)``.

A class that should never be serialized (e.g. video / debug plugins) sets
``BLUEPRINT_EXCLUDE = True`` at class scope; ``EnvRuntime.to_blueprint``
silently filters such instances out.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

from .backend import BaseSimulator
from .plugin import BasePlugin
from .recorder import PostActionRecorder
from .observer_plugin import BaseObserverPlugin

BLUEPRINT_VERSION = 1


# ---------------------------------------------------------------------------
# ClassSpec helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassSpec:
    """Serializable handle to a Python class plus its construction config.

    ``cls`` is encoded as ``"package.module:QualName"``. ``config`` is the
    payload returned by ``instance.to_blueprint()`` (or ``{}`` when the
    component does not implement the protocol).
    """

    cls: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"cls": self.cls, "config": dict(self.config)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassSpec":
        if "cls" not in data:
            raise ValueError(f"ClassSpec entry missing 'cls': {data!r}")
        return cls(cls=str(data["cls"]), config=dict(data.get("config") or {}))


def _qualified_name(target: Type[Any]) -> str:
    return f"{target.__module__}:{target.__qualname__}"


def _resolve_class(qualified: str) -> Type[Any]:
    """Resolve ``"module:QualName"`` (or ``"module.QualName"`` for compat)."""
    if ":" in qualified:
        module_path, qual_name = qualified.split(":", 1)
    else:  # tolerate dotted form
        module_path, _, qual_name = qualified.rpartition(".")
        if not module_path:
            raise ValueError(f"Cannot resolve class spec {qualified!r}")
    module = importlib.import_module(module_path)
    target: Any = module
    for part in qual_name.split("."):
        target = getattr(target, part)
    if not isinstance(target, type):
        raise TypeError(f"{qualified!r} did not resolve to a class")
    return target


def _is_blueprint_excluded(component: Any) -> bool:
    return bool(getattr(component, "BLUEPRINT_EXCLUDE", False))


def _capture_config(component: Any) -> Dict[str, Any]:
    """Extract the construction config of a component (default: ``{}``)."""
    method = getattr(component, "to_blueprint", None)
    if method is None:
        return {}
    payload = method()
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{type(component).__name__}.to_blueprint() must return a mapping, "
            f"got {type(payload).__name__}"
        )
    return dict(payload)


def _instantiate(spec: ClassSpec) -> Any:
    """Reconstruct a component from a :class:`ClassSpec`.

    Calls ``cls.from_blueprint(config)`` if defined, else ``cls(**config)``.
    """
    target_cls = _resolve_class(spec.cls)
    builder = getattr(target_cls, "from_blueprint", None)
    if callable(builder):
        instance = builder(dict(spec.config))
    else:
        instance = target_cls(**spec.config)
    return instance


# ---------------------------------------------------------------------------
# EnvBlueprint
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvBlueprint:
    """Serializable specification of an :class:`EnvRuntime`.

    See module docstring for the full design. Use :meth:`from_runtime` /
    :meth:`build` for the round-trip, and :meth:`save` / :meth:`load` for
    on-disk persistence.
    """

    simulator: ClassSpec
    plugins: Tuple[ClassSpec, ...] = ()
    observer_plugins: Dict[str, ClassSpec] = field(default_factory=dict)
    phy_steps_per_action: int = 1
    max_steps: Optional[int] = None
    strict: bool = True
    version: int = BLUEPRINT_VERSION

    # ------------------------------------------------------------------
    # Construction from a live runtime
    # ------------------------------------------------------------------
    @classmethod
    def from_runtime(cls, runtime: "EnvRuntime") -> "EnvBlueprint":  # noqa: F821
        """Snapshot a runtime into a blueprint.

        Filters out:

        * The internal observer dispatcher.
        * The auto-attached :class:`TimeoutPlugin` (re-derived from
          ``max_steps``).
        * Any plugin with ``BLUEPRINT_EXCLUDE = True``.
        """
        # Local imports to avoid a hard cycle with env_runtime.py.
        from .common_plugins import TimeoutPlugin
        from .runtime_plugin import _ObserverDispatcherPlugin

        simulator = runtime.simulator
        sim_spec = ClassSpec(
            cls=_qualified_name(type(simulator)),
            config=_capture_config(simulator),
        )

        plugin_specs: list[ClassSpec] = []
        max_steps_from_plugin: Optional[int] = None
        for plugin in runtime.plugins:
            if isinstance(plugin, _ObserverDispatcherPlugin):
                continue
            if isinstance(plugin, TimeoutPlugin):
                # Capture so we can round-trip ``max_steps``; do not
                # serialize as a regular plugin.
                max_steps_from_plugin = int(plugin.max_steps)
                continue
            if _is_blueprint_excluded(plugin):
                continue
            plugin_specs.append(
                ClassSpec(
                    cls=_qualified_name(type(plugin)),
                    config=_capture_config(plugin),
                )
            )

        observer_specs: Dict[str, ClassSpec] = {}
        for name, observer in runtime.observer_plugins.items():
            if observer is None:
                continue
            if _is_blueprint_excluded(observer):
                continue
            observer_specs[name] = ClassSpec(
                cls=_qualified_name(type(observer)),
                config=_capture_config(observer),
            )

        return cls(
            simulator=sim_spec,
            plugins=tuple(plugin_specs),
            observer_plugins=observer_specs,
            phy_steps_per_action=int(runtime._core.phy_steps_per_action),
            max_steps=max_steps_from_plugin,
            strict=bool(runtime._strict),
            version=BLUEPRINT_VERSION,
        )

    # ------------------------------------------------------------------
    # Build a fresh runtime
    # ------------------------------------------------------------------
    def build(
        self,
        recorders: Sequence[PostActionRecorder] = (),
        debug_plugins: Sequence[BasePlugin] = (),
    ) -> "EnvRuntime":  # noqa: F821
        """Instantiate a fresh :class:`EnvRuntime` from this blueprint.

        Parameters
        ----------
        recorders:
            Optional :class:`PostActionRecorder` instances to attach after
            construction. Recorders are intentionally not part of a
            blueprint, so they are passed in here.
        debug_plugins:
            Optional plugins to attach in addition to the blueprint-defined
            ones. Each must declare ``BLUEPRINT_EXCLUDE = True`` (e.g.
            :class:`VideoRecorderPlugin`); supplying any other plugin
            raises :class:`ValueError`. The returned runtime can still be
            modified freely afterwards via the standard ``EnvRuntime`` API
            — this restriction lives only at ``build`` time.
        """
        # Local import to avoid a hard cycle with env_runtime.py.
        from .env_runtime import EnvRuntime

        for plugin in debug_plugins:
            if not _is_blueprint_excluded(plugin):
                raise ValueError(
                    f"build(debug_plugins=...) only accepts plugins with "
                    f"BLUEPRINT_EXCLUDE = True; "
                    f"{type(plugin).__name__} is not marked. Attach it on "
                    f"the returned runtime via attach_plugin() instead."
                )

        simulator = _instantiate(self.simulator)
        if not isinstance(simulator, BaseSimulator):
            raise TypeError(
                f"Simulator class {self.simulator.cls!r} did not produce a "
                f"BaseSimulator instance"
            )

        plugins: list[BasePlugin] = []
        for spec in self.plugins:
            instance = _instantiate(spec)
            if not isinstance(instance, BasePlugin):
                raise TypeError(
                    f"Plugin class {spec.cls!r} did not produce a BasePlugin"
                )
            plugins.append(instance)
        plugins.extend(debug_plugins)

        observer_plugins: Dict[str, BaseObserverPlugin] = {}
        for name, spec in self.observer_plugins.items():
            instance = _instantiate(spec)
            if not isinstance(instance, BaseObserverPlugin):
                raise TypeError(
                    f"Observer class {spec.cls!r} did not produce a "
                    f"BaseObserverPlugin"
                )
            observer_plugins[name] = instance

        runtime = EnvRuntime(
            simulator=simulator,
            observer_plugins=observer_plugins,
            plugins=plugins,
            recorders=list(recorders),
            phy_steps_per_action=self.phy_steps_per_action,
            max_steps=self.max_steps,
            strict=self.strict,
        )
        return runtime

    # ------------------------------------------------------------------
    # Dict / YAML / JSON I/O
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "runtime": {
                "phy_steps_per_action": int(self.phy_steps_per_action),
                "max_steps": self.max_steps,
                "strict": bool(self.strict),
            },
            "simulator": self.simulator.to_dict(),
            "plugins": [spec.to_dict() for spec in self.plugins],
            "observer_plugins": {
                name: spec.to_dict() for name, spec in self.observer_plugins.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvBlueprint":
        version = int(data.get("version", BLUEPRINT_VERSION))
        if version != BLUEPRINT_VERSION:
            raise ValueError(
                f"Unsupported blueprint version {version}; "
                f"expected {BLUEPRINT_VERSION}."
            )
        runtime_section = dict(data.get("runtime") or {})
        sim_section = data.get("simulator")
        if sim_section is None:
            raise ValueError("Blueprint missing 'simulator' section")
        plugin_section = list(data.get("plugins") or [])
        observer_section = dict(data.get("observer_plugins") or {})
        return cls(
            simulator=ClassSpec.from_dict(sim_section),
            plugins=tuple(ClassSpec.from_dict(item) for item in plugin_section),
            observer_plugins={
                name: ClassSpec.from_dict(item)
                for name, item in observer_section.items()
            },
            phy_steps_per_action=int(runtime_section.get("phy_steps_per_action", 1)),
            max_steps=runtime_section.get("max_steps"),
            strict=bool(runtime_section.get("strict", True)),
            version=version,
        )

    def to_yaml(self) -> str:
        try:
            import yaml  # type: ignore

            return yaml.safe_dump(
                self.to_dict(), sort_keys=False, allow_unicode=True
            )
        except ImportError:
            # Fallback to JSON: still human-readable, no extra dep.
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_yaml(cls, text: str) -> "EnvBlueprint":
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError("Blueprint document must be a mapping at top level")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EnvBlueprint":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "BLUEPRINT_VERSION",
    "ClassSpec",
    "EnvBlueprint",
]
