"""Canonical Policy ABC for the combatbench framework.

This module is the **single source of truth** for what counts as a
"policy" in this project. Anything plugged into :class:`EpisodeRunner`,
:class:`RoundRunner`, or :class:`ParallelRunner` must subclass
:class:`Policy` defined here.

Contract
--------
Required:
    ``act(observation, want_extra: bool = False) -> (action, extra)``
        Synchronous. Always returns a 2-tuple ``(action, extra)``. When
        ``want_extra`` is False the policy may return ``extra=None`` and
        skip any work needed only for that payload. Returning ``None``
        for ``action`` is NOT allowed — return an explicit action.

Optional (runners detect via ``hasattr``):
    ``reset(seed: Optional[int] = None) -> None``
        Called once per episode before the first ``act``. ``seed`` is a
        deterministic per-policy child seed derived from the runner's
        ``base_seed`` via :class:`numpy.random.SeedSequence`. Policies
        that hold their own RNGs SHOULD use it for reproducibility, but
        the framework does not enforce this — see "Determinism" below.
        Default: no-op.
    ``close() -> None``
        Release resources. Runners never call this automatically; caller
        owns policy lifecycle. :meth:`EpisodeRunner.close` invokes it as
        a convenience.

Observation / action / extra types
----------------------------------
The framework places **no constraints** on the Python types flowing in
and out of :meth:`Policy.act`:

* ``observation`` is whatever the bound observer plugin's
  ``get_output()`` returned — a dict, a numpy array, a custom
  dataclass, anything. The policy and the observer agree on the schema;
  the runner is just a pipe.
* ``action`` is whatever the simulator's ``BaseSimulator.step`` accepts
  for that agent. Some sims want ``np.ndarray(float32)`` joint torques,
  others want dicts of high-level commands. Match the sim; the
  framework does not coerce.
* ``extra`` is fully policy-defined. Typical contents include log-prob
  / value estimates for on-policy RL, attention maps for debugging, or
  raw policy-network outputs. May be ``None`` when ``want_extra`` is
  False or when the policy has nothing to report.

Determinism vs. stochasticity
-----------------------------
Whether a policy is deterministic or stochastic — and, if stochastic,
how it is seeded — is **the policy's own responsibility**, managed
inside ``__init__`` (and optionally re-seeded inside :meth:`reset`).
The framework does not introspect or alter this. Concretely:

* A deterministic policy ignores ``reset(seed=...)`` (or accepts it and
  no-ops). Same observation in -> same action out.
* A stochastic policy owns its RNG (``np.random.Generator``, a torch
  ``Generator``, etc.), constructs it in ``__init__``, and re-seeds in
  :meth:`reset` when the runner-provided ``seed`` is not ``None``.
* Hybrid policies (e.g. exploration noise toggled by an ``eval`` flag)
  expose that toggle via constructor args — there is no framework-level
  ``eval()`` switch.

No ``__init__`` contract
------------------------
This ABC intentionally does **not** define ``__init__``. Subclasses are
free to design their constructors however they want (load checkpoints,
take hyperparameters, wire RNGs, decide deterministic-vs-stochastic
behaviour — whatever). The :func:`load_policy` loader just calls
``cls(**kwargs)`` with parsed query-string arguments; subclasses that
want to participate should accept ``**kwargs`` so unknown parameters
don't crash construction.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


__all__ = [
    "Policy",
    "PolicyBlueprint",
    "PolicyParameter",
    "ParameterizedPolicyBlueprint",
]


class Policy(ABC):
    """Abstract base class for all combatbench policies. See module docstring."""

    @abstractmethod
    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[Any, Any | None]:
        """Compute an action for the given observation.

        Parameters
        ----------
        observation:
            Whatever the simulator's ``get_observation()``
            returned. The framework imposes no type constraint; the
            policy and the observer agree on the schema.
        want_extra:
            If True the runner wants the optional ``extra`` payload
            (e.g. log-prob / value estimates for on-policy RL). When
            False the policy may return ``extra=None`` and skip any
            work needed only for that payload.

        Returns
        -------
        action:
            Whatever the simulator's ``step`` accepts for this agent.
            Type/shape are simulator-defined; the framework does not
            coerce. Must not be ``None``.
        extra:
            Policy-defined auxiliary payload, or ``None``. Common
            choices: a dict of log-prob / value / entropy tensors for
            on-policy RL trainers.

        Stochasticity is the policy's responsibility — see the module
        docstring's "Determinism vs. stochasticity" section.
        """
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        """Per-episode reset hook. Default: no-op.

        Override when the policy holds RNG or recurrent state. ``seed``
        is a deterministic per-policy child seed derived from the
        runner's ``base_seed``; stochastic policies SHOULD reseed their
        internal RNG with it so rollouts stay reproducible. Determini-
        stic policies can ignore it. The framework neither inspects nor
        enforces what the policy does with this value.
        """
        return None

    def to_blueprint(self, dest_path: Optional[str] = None) -> "PolicyBlueprint":
        """Export this policy instance to a deployable :class:`PolicyBlueprint`.

        Optional hook. Policies that support self-export (e.g. those with
        bundled weights) should override this to write their source /
        checkpoint assets into ``dest_path`` and return a
        :class:`PolicyBlueprint` that can rebuild a functionally equivalent
        policy via :meth:`PolicyBlueprint.build`.

        Parameters
        ----------
        dest_path:
            Directory to save source code and model weights into. When
            ``None`` a temporary directory is created automatically.

        Returns
        -------
        PolicyBlueprint
            A blueprint that points to the exported assets and carries the
            correct constructor kwargs.

        Raises
        ------
        NotImplementedError
            Default implementation — the policy does not support
            blueprint export.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_blueprint()."
        )


# ---------------------------------------------------------------------------
# PolicyBlueprint
# ---------------------------------------------------------------------------
# A lightweight serializable handle to a Policy class plus its constructor
# kwargs. Mirrors the design of :class:`envs.framework.blueprint.EnvBlueprint`
# but is intentionally minimal: it stores ONLY the Python entry point of the
# Policy subclass (``"package.module:QualName"``) and the kwargs to pass to
# its ``__init__``. It does NOT bundle source code, weights, or anything else
# — those are the policy implementation's own concern (e.g. via a
# ``model_path`` kwarg).
POLICY_BLUEPRINT_VERSION = 1

# Sentinel for "no default value provided" (PolicyParameter).
_POLICY_PARAM_MISSING = object()

# Matches a string consisting solely of a single ``${name}`` reference.
_POLICY_FULL_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
# Matches every ``${name}`` occurrence inside a string for inline substitution.
_POLICY_INLINE_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Magic variable for blueprint-relative path resolution.  When a YAML file
# is loaded via :meth:`PolicyBlueprint.load` (or
# :meth:`ParameterizedPolicyBlueprint.load`), every ``${DIR}`` occurrence in
# the raw text is replaced with the YAML file's parent directory (absolute).
# This lets users reference co-uploaded assets without knowing the final
# extraction path:
#
#   cls: "file:${DIR}/policy.py:MyPolicy"
#   config:
#     model_path: "${DIR}/model.pt"
_DIR_VAR = "${DIR}"


def _substitute_dir(raw_text: str, dir_path: Path) -> str:
    """Replace ``${DIR}`` with the stringified absolute *dir_path*."""
    return raw_text.replace(_DIR_VAR, str(dir_path.resolve()))


def _resolve_policy_class(entry: str) -> type:
    """Resolve a class from a string descriptor.

    Supports two formats:
      1. ``"package.module:QualName"`` - standard Python import path.
      2. ``"file:/path/to/file.py:QualName"`` - load from a standalone Python file.
    """
    if ":" not in entry:
        raise ValueError(
            f"PolicyBlueprint.cls must be 'package.module:QualName' or "
            f"'file:/path/to/file.py:QualName', got {entry!r}"
        )

    # Handle file: prefix for standalone policy files (no repo deps).
    if entry.startswith("file:"):
        file_path, qualname = entry[5:].split(":", 1)
        file_path = Path(file_path.strip()).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Policy file not found: {file_path}")
        module_name = f"_policy_{file_path.stem}_{file_path.parent.as_posix().replace('/', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module_path, qualname = entry.split(":", 1)
        module = importlib.import_module(module_path.strip())

    obj: Any = module
    for part in qualname.strip().split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"Resolved entry {entry!r} is not a class: {type(obj).__name__}")
    return obj


@dataclass(frozen=True)
class PolicyBlueprint:
    """Serializable handle to a :class:`Policy` subclass plus its init kwargs.

    ``cls`` is encoded as ``"package.module:QualName"`` (matching the
    convention used by :class:`envs.framework.blueprint.ClassSpec`).
    ``config`` is the dict of keyword arguments forwarded to the
    policy's ``__init__``. Anything the policy needs to load weights /
    pick a device / tweak behaviour is expressed via ``config`` — the
    blueprint does NOT capture source code or model state itself.

    Use :meth:`build` to instantiate a live :class:`Policy`. Use
    :meth:`save` / :meth:`load` (YAML, JSON fallback) for persistence.
    """

    cls: str
    config: Dict[str, Any] = field(default_factory=dict)
    version: int = POLICY_BLUEPRINT_VERSION

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, **overrides: Any) -> "Policy":
        """Instantiate the policy.

        ``overrides`` are merged on top of ``self.config`` (overrides win)
        and forwarded to the policy's ``__init__``. The result is checked
        to be a :class:`Policy` subclass instance; subclasses that accept
        ``**kwargs`` will silently absorb keys they do not recognize, by
        the loader's existing convention.
        """
        cls = _resolve_policy_class(self.cls)
        if not issubclass(cls, Policy):
            raise TypeError(
                f"{self.cls} resolves to {cls.__name__}, which does not "
                f"subclass envs.framework.policy.Policy"
            )
        kwargs: Dict[str, Any] = {**self.config, **overrides}
        instance = cls(**kwargs)
        if not isinstance(instance, Policy):
            raise TypeError(
                f"{self.cls}(**kwargs) produced {type(instance).__name__}, "
                f"which is not a Policy instance"
            )
        return instance

    # ------------------------------------------------------------------
    # Dict / YAML I/O
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "cls": str(self.cls),
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyBlueprint":
        if not isinstance(data, Mapping):
            raise TypeError("PolicyBlueprint document must be a mapping at top level")
        version = int(data.get("version", POLICY_BLUEPRINT_VERSION))
        if version != POLICY_BLUEPRINT_VERSION:
            raise ValueError(
                f"Unsupported policy blueprint version {version}; "
                f"expected {POLICY_BLUEPRINT_VERSION}."
            )
        if "cls" not in data:
            raise ValueError("PolicyBlueprint document missing required 'cls' key")
        return cls(
            cls=str(data["cls"]),
            config=dict(data.get("config") or {}),
            version=version,
        )

    def to_yaml(self) -> str:
        try:
            import yaml  # type: ignore

            return yaml.safe_dump(
                self.to_dict(), sort_keys=False, allow_unicode=True
            )
        except ImportError:
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_yaml(cls, text: str) -> "PolicyBlueprint":
        """Load a policy blueprint from YAML/JSON text.

        Auto-detects parameterized documents: if the top-level document
        contains a ``parameters`` section (i.e. it was produced by
        :class:`ParameterizedPolicyBlueprint`), it is materialized using
        each parameter's **default value**. Any parameter without a
        default raises :class:`ValueError` — supply overrides via
        :meth:`ParameterizedPolicyBlueprint.materialize` instead.
        """
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError("PolicyBlueprint document must be a mapping at top level")
        if data.get("parameters"):
            return ParameterizedPolicyBlueprint.from_dict(data).materialize()
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PolicyBlueprint":
        """Load a policy blueprint from disk (YAML or JSON).

        ``${DIR}`` in the raw text is replaced with the YAML file's parent
        directory, so blueprint-relative paths like
        ``cls: "file:${DIR}/policy.py:MyPolicy"`` work without the user
        knowing the final extraction path.

        See :meth:`from_yaml` for parameterized-document handling.
        """
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        raw = _substitute_dir(raw, p.parent)
        return cls.from_yaml(raw)


# ---------------------------------------------------------------------------
# PolicyParameter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyParameter:
    """A named knob for :class:`ParameterizedPolicyBlueprint`.

    ``default`` uses a private sentinel to distinguish "no default" from
    "default is None"; query via :pyattr:`has_default`.
    """

    name: str
    default: Any = _POLICY_PARAM_MISSING
    description: str = ""

    @property
    def has_default(self) -> bool:
        return self.default is not _POLICY_PARAM_MISSING

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.has_default:
            out["default"] = self.default
        if self.description:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(
        cls, name: str, data: Optional[Mapping[str, Any]]
    ) -> "PolicyParameter":
        data = data or {}
        if not isinstance(data, Mapping):
            raise TypeError(
                f"PolicyParameter {name!r} entry must be a mapping, "
                f"got {type(data).__name__}"
            )
        if "default" in data:
            return cls(
                name=name,
                default=data["default"],
                description=str(data.get("description", "")),
            )
        return cls(name=name, description=str(data.get("description", "")))


def _policy_substitute(node: Any, values: Mapping[str, Any]) -> Any:
    """Recursively replace ``${name}`` placeholders inside ``node``.

    Mirrors :func:`envs.framework.parameterized_blueprint._substitute`;
    duplicated here to keep ``policy.py`` free of cross-module imports.
    """
    if isinstance(node, str):
        full = _POLICY_FULL_REF_RE.match(node)
        if full is not None:
            key = full.group(1)
            if key not in values:
                raise KeyError(
                    f"PolicyBlueprint references undeclared parameter ${{{key}}}"
                )
            return values[key]

        def _replace(match: "re.Match[str]") -> str:
            key = match.group(1)
            if key not in values:
                raise KeyError(
                    f"PolicyBlueprint references undeclared parameter ${{{key}}}"
                )
            return str(values[key])

        return _POLICY_INLINE_REF_RE.sub(_replace, node)
    if isinstance(node, Mapping):
        return {k: _policy_substitute(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_policy_substitute(v, values) for v in node]
    if isinstance(node, tuple):
        return tuple(_policy_substitute(v, values) for v in node)
    return node


# ---------------------------------------------------------------------------
# ParameterizedPolicyBlueprint
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterizedPolicyBlueprint:
    """A :class:`PolicyBlueprint` template parameterized by named knobs.

    The ``template`` field stores a :meth:`PolicyBlueprint.to_dict`-shaped
    document **with** ``${name}`` placeholders left intact (typically
    inside ``config`` values). :meth:`materialize` substitutes them and
    returns a concrete :class:`PolicyBlueprint`.

    YAML layout
    -----------
    ::

        version: 1
        parameters:
          model_path:
            description: "Path to the trained checkpoint."
          device:
            default: "cpu"
        cls: "policy.humanoid21.standing.policy:StandingCombatPolicy"
        config:
          model_path: "${model_path}"
          device: "${device}"

    Substitution rules match :class:`ParameterizedEnvBlueprint`:
    a string of the exact form ``"${name}"`` is replaced by the raw
    parameter value (preserving its type); strings with embedded
    ``${name}`` references are inline-substituted via stringification.
    """

    template: Dict[str, Any]
    parameters: Tuple[PolicyParameter, ...] = ()
    version: int = POLICY_BLUEPRINT_VERSION

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------
    def materialize(self, **overrides: Any) -> PolicyBlueprint:
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
        resolved = _policy_substitute(self.template, values)
        if not isinstance(resolved, Mapping):
            raise TypeError(
                "ParameterizedPolicyBlueprint.template must be a mapping at top level"
            )
        return PolicyBlueprint.from_dict(resolved)

    def build(self, **overrides: Any) -> "Policy":
        """Convenience: ``materialize(**overrides).build()`` in one step."""
        return self.materialize(**overrides).build()

    # ------------------------------------------------------------------
    # Dict / YAML I/O
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"version": int(self.version)}
        if self.parameters:
            out["parameters"] = {p.name: p.to_dict() for p in self.parameters}
        for key, value in self.template.items():
            if key in ("version", "parameters"):
                continue
            out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterizedPolicyBlueprint":
        if not isinstance(data, Mapping):
            raise TypeError("Document must be a mapping at top level")
        version = int(data.get("version", POLICY_BLUEPRINT_VERSION))
        if version != POLICY_BLUEPRINT_VERSION:
            raise ValueError(
                f"Unsupported policy blueprint version {version}; "
                f"expected {POLICY_BLUEPRINT_VERSION}."
            )
        params_section = data.get("parameters") or {}
        if not isinstance(params_section, Mapping):
            raise TypeError("'parameters' section must be a mapping")
        parameters = tuple(
            PolicyParameter.from_dict(name, entry)
            for name, entry in params_section.items()
        )
        template = {
            key: value
            for key, value in data.items()
            if key not in ("version", "parameters")
        }
        # Always carry the version inside the template so the materialized
        # PolicyBlueprint.from_dict sees the expected key.
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
    def from_yaml(cls, text: str) -> "ParameterizedPolicyBlueprint":
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
    def load(cls, path: str | Path) -> "ParameterizedPolicyBlueprint":
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        raw = _substitute_dir(raw, p.parent)
        return cls.from_yaml(raw)