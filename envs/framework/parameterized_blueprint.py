"""ParameterizedEnvBlueprint: an :class:`EnvBlueprint` template with named
parameters.

Motivation
----------
:class:`EnvBlueprint` is a fully concrete spec — every config value is
baked in. In practice, the same environment recipe is often run with a
handful of varying knobs (initial distance, match duration, damage
scale, ...). ``ParameterizedEnvBlueprint`` lets you declare those knobs
once, with optional default values, and substitute them at build time::

    pb = ParameterizedEnvBlueprint.load("curriculum.yaml")
    bp = pb.materialize(initial_distance=2.5)   # other params use defaults
    runtime = bp.build()

YAML layout
-----------
Identical to :class:`EnvBlueprint`'s YAML with one extra top-level
``parameters`` section, and ``${name}`` placeholders allowed anywhere
inside config dict / list / scalar values::

    version: 1
    parameters:
      initial_distance:
        default: 2.0
        description: "Spawn distance between the two robots (meters)."
      match_duration_seconds: {}        # required, no default
    runtime:
      phy_steps_per_action: 25
      max_steps: "${max_steps}"
    simulator:
      cls: "envs.humanoid21.simulator:MujocoCombatSimulator"
      config:
        initial_distance: "${initial_distance}"

Substitution rules
------------------
* A string of the form ``"${name}"`` (exact match) is replaced by the
  raw parameter value, preserving its type. This is how you pass
  numbers, lists, dicts, booleans through.
* A string with embedded ``${name}`` references (e.g.
  ``"path/${run_id}/checkpoint.pt"``) is treated as a template and
  substituted via ``str.format``-style replacement; each referenced
  parameter is stringified.
* Substitution recurses into dicts and lists. Other Python scalars
  (int / float / bool / None) pass through unchanged.
* Referencing an unknown parameter raises :class:`KeyError` at
  ``materialize`` time. Passing an unknown override likewise raises.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .blueprint import BLUEPRINT_VERSION, EnvBlueprint

# Sentinel for "no default value provided".
_MISSING = object()

# Matches a string consisting solely of a single ``${name}`` reference.
_FULL_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
# Matches every ``${name}`` occurrence inside a string for inline substitution.
_INLINE_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameter:
    """A named knob with an optional default value and description.

    ``default`` uses a private sentinel to distinguish "no default" from
    "default is None"; query via :pyattr:`has_default`.
    """

    name: str
    default: Any = _MISSING
    description: str = ""

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.has_default:
            out["default"] = self.default
        if self.description:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, name: str, data: Optional[Mapping[str, Any]]) -> "Parameter":
        data = data or {}
        if not isinstance(data, Mapping):
            raise TypeError(
                f"Parameter {name!r} entry must be a mapping, got {type(data).__name__}"
            )
        if "default" in data:
            return cls(
                name=name,
                default=data["default"],
                description=str(data.get("description", "")),
            )
        return cls(name=name, description=str(data.get("description", "")))


# ---------------------------------------------------------------------------
# Substitution helpers
# ---------------------------------------------------------------------------
def _substitute(node: Any, values: Mapping[str, Any]) -> Any:
    """Recursively replace ``${name}`` placeholders inside ``node``."""
    if isinstance(node, str):
        full = _FULL_REF_RE.match(node)
        if full is not None:
            key = full.group(1)
            if key not in values:
                raise KeyError(
                    f"Blueprint references undeclared parameter ${{{key}}}"
                )
            return values[key]
        # Inline substitution — produces a string.
        def _replace(match: "re.Match[str]") -> str:
            key = match.group(1)
            if key not in values:
                raise KeyError(
                    f"Blueprint references undeclared parameter ${{{key}}}"
                )
            return str(values[key])

        return _INLINE_REF_RE.sub(_replace, node)
    if isinstance(node, Mapping):
        return {k: _substitute(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, values) for v in node]
    if isinstance(node, tuple):
        return tuple(_substitute(v, values) for v in node)
    return node


# ---------------------------------------------------------------------------
# ParameterizedEnvBlueprint
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterizedEnvBlueprint:
    """An :class:`EnvBlueprint` template parameterized by named knobs.

    The ``template`` field stores the full blueprint document (the same
    shape :meth:`EnvBlueprint.to_dict` produces) **with** ``${name}``
    placeholders left intact. :meth:`materialize` substitutes them and
    returns a concrete :class:`EnvBlueprint`.

    Construct directly via :meth:`from_dict` / :meth:`from_yaml` /
    :meth:`load`, or programmatically::

        pb = ParameterizedEnvBlueprint(
            template={...},     # EnvBlueprint.to_dict()-shaped, with ${...}
            parameters=(Parameter("initial_distance", default=2.0),),
        )
    """

    template: Dict[str, Any]
    parameters: Tuple[Parameter, ...] = ()
    version: int = BLUEPRINT_VERSION

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------
    def materialize(self, **overrides: Any) -> EnvBlueprint:
        """Substitute parameter values and return a concrete blueprint.

        Parameters listed with a default may be omitted; required
        parameters (no default) must be supplied via ``overrides``.
        Passing an unknown keyword raises :class:`ValueError`.
        """
        known = {p.name: p for p in self.parameters}
        unknown = set(overrides).difference(known)
        if unknown:
            raise ValueError(
                f"Unknown parameter override(s): {sorted(unknown)}; "
                f"declared parameters are {sorted(known)}"
            )
        values: Dict[str, Any] = {}
        missing: list[str] = []
        for name, param in known.items():
            if name in overrides:
                values[name] = overrides[name]
            elif param.has_default:
                values[name] = param.default
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                f"Missing required parameter value(s): {sorted(missing)}"
            )
        resolved = _substitute(self.template, values)
        if not isinstance(resolved, Mapping):
            raise TypeError(
                "ParameterizedEnvBlueprint.template must be a mapping at top level"
            )
        return EnvBlueprint.from_dict(resolved)

    def build(self, *, recorders=(), debug_plugins=(), **overrides: Any):
        """Convenience: ``materialize(**overrides).build(...)`` in one step."""
        return self.materialize(**overrides).build(
            recorders=recorders, debug_plugins=debug_plugins,
        )

    # ------------------------------------------------------------------
    # Dict / YAML I/O
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        # Place ``parameters`` after ``version`` and before the rest of
        # the template for readability when written as YAML.
        out: Dict[str, Any] = {"version": int(self.version)}
        if self.parameters:
            out["parameters"] = {p.name: p.to_dict() for p in self.parameters}
        # Copy the template body, dropping any duplicate ``version`` key
        # (the outer one is authoritative).
        for key, value in self.template.items():
            if key in ("version", "parameters"):
                continue
            out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterizedEnvBlueprint":
        if not isinstance(data, Mapping):
            raise TypeError("Document must be a mapping at top level")
        version = int(data.get("version", BLUEPRINT_VERSION))
        if version != BLUEPRINT_VERSION:
            raise ValueError(
                f"Unsupported blueprint version {version}; "
                f"expected {BLUEPRINT_VERSION}."
            )
        params_section = data.get("parameters") or {}
        if not isinstance(params_section, Mapping):
            raise TypeError("'parameters' section must be a mapping")
        parameters = tuple(
            Parameter.from_dict(name, entry)
            for name, entry in params_section.items()
        )
        template = {
            key: value
            for key, value in data.items()
            if key not in ("version", "parameters")
        }
        # Always carry the version inside the template too so the
        # materialized EnvBlueprint.from_dict sees the expected key.
        template["version"] = version
        return cls(template=template, parameters=parameters, version=version)

    def to_yaml(self) -> str:
        try:
            import yaml  # type: ignore

            return yaml.safe_dump(
                self.to_dict(), sort_keys=False, allow_unicode=True
            )
        except ImportError:
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_yaml(cls, text: str) -> "ParameterizedEnvBlueprint":
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError("Document must be a mapping at top level")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ParameterizedEnvBlueprint":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "Parameter",
    "ParameterizedEnvBlueprint",
]
