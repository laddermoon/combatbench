"""Round runner backed by :class:`EnvBlueprint`.

Builds an :class:`EnvRuntime` from a blueprint, attaches optional video and
recorders, and drives a single episode with two policies.

Public surface:

* ``RoundRunner(blueprint, policy_a, policy_b, video_plugin=None, recorders=())``
* ``runner.run(seed=None, options=None) -> dict`` with keys ``steps`` /
  ``termination_reasons`` / ``seed``.

The runner owns the runtime lifecycle — ``runtime.close()`` is called
automatically when the runner is garbage-collected or explicitly closed.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .blueprint import EnvBlueprint
from .common_plugins import VideoRecorderPlugin
from .episode_runner import EpisodeRunner
from .policy import PolicyBlueprint
from .recorder import PostActionRecorder


class RoundRunner:
    """Run a single combat round from an :class:`EnvBlueprint`.

    Parameters
    ----------
    blueprint:
        Serializable environment specification.
    policy_a, policy_b:
        Policy instances for each robot.
    video_plugin:
        Optional :class:`VideoRecorderPlugin` attached as a debug plugin.
    recorders:
        Optional :class:`PostActionRecorder` instances attached to the runtime.
    """

    def __init__(
        self,
        blueprint: EnvBlueprint,
        policy_a: Any,
        policy_b: Any,
        video_plugin: Optional[VideoRecorderPlugin] = None,
        recorders: Sequence[PostActionRecorder] = (),
    ) -> None:
        self._blueprint = blueprint
        debug_plugins = [video_plugin] if video_plugin is not None else []
        self._runtime = blueprint.build(
            recorders=list(recorders),
            debug_plugins=debug_plugins,
        )
        self._runner = EpisodeRunner(
            runtime=self._runtime,
            policy_a=policy_a,
            policy_b=policy_b,
        )

    @property
    def runtime(self):
        """The live :class:`EnvRuntime` built from the blueprint."""
        return self._runtime

    def run(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        want_extras: bool = False,
        initial_health_a: Optional[float] = None,
        initial_health_b: Optional[float] = None,
        score_log_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one round and return a result dict.

        Parameters
        ----------
        seed:
            Episode seed ( forwarded to MuJoCo reset).
        options:
            Forwarded to :meth:`EnvRuntime.reset` and visible to plugins /
            observers via ``ctx.episode_options``.
        want_extras:
            Whether each ``policy.act`` is called with ``want_extra=True``.
        initial_health_a, initial_health_b:
            Starting HP for each robot.  When provided they are merged into
            ``options`` so that :class:`CombatScoringPlugin` picks them up
            via ``ctx.episode_options``.  ``None`` means the plugin default
            (typically 100) is used.
        score_log_file:
            Path to append a per-substep combat score audit log.  Merged
            into ``options`` under ``"score_log_file"`` so the plugin picks
            it up via ``ctx.episode_options``.  ``None`` disables logging.

        Returns a dict with ``steps``, ``termination_reasons``, ``seed``,
        ``health_a``, ``health_b``.
        """
        if initial_health_a is not None or initial_health_b is not None:
            options = dict(options) if options else {}
            if initial_health_a is not None:
                options["initial_health_a"] = float(initial_health_a)
            if initial_health_b is not None:
                options["initial_health_b"] = float(initial_health_b)

        if score_log_file is not None:
            options = dict(options) if options else {}
            options["score_log_file"] = score_log_file

        self._runner.run_episode(seed=seed, options=options, want_extras=want_extras)
        ctx = self._runtime.ctx
        return {
            "steps": int(ctx.episode_step),
            "termination_reasons": list(ctx.termination_proposals),
            "seed": int(ctx.base_seed) if ctx.base_seed is not None else None,
            "health_a": float(ctx.metrics.get("health_a", 0.0)),
            "health_b": float(ctx.metrics.get("health_b", 0.0)),
        }

    def close(self) -> None:
        """Close the underlying runtime. Idempotent."""
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _coerce_cli_value(raw: str) -> Any:
    text = str(raw)
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        import json
        return json.loads(text)
    except Exception:
        return text


def _parse_spec(spec: str) -> tuple[str, str, Dict[str, Any]]:
    module_and_class, sep, query = str(spec).partition("?")
    if ":" not in module_and_class:
        raise ValueError(
            f"Invalid spec: {spec!r}. Expected format module.path:ClassName?key=value"
        )
    module_path, class_name = module_and_class.split(":", 1)
    module_path = module_path.strip()
    class_name = class_name.strip()
    if not module_path or not class_name:
        raise ValueError(
            f"Invalid spec: {spec!r}. module.path and ClassName must be non-empty."
        )
    kwargs: Dict[str, Any] = {}
    if sep and query:
        from urllib.parse import parse_qsl
        for key, value in parse_qsl(query, keep_blank_values=True):
            k = key.strip()
            if not k:
                continue
            kwargs[k] = _coerce_cli_value(value)
    return module_path, class_name, kwargs


def _load_from_spec(spec: str) -> Any:
    import importlib
    module_path, class_name, kwargs = _parse_spec(spec)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Class {class_name!r} not found in module {module_path!r}"
        )
    if not callable(cls):
        raise TypeError(f"Target {module_path}:{class_name} is not callable")
    return cls(**kwargs)


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run a combat round from an EnvBlueprint.",
    )
    parser.add_argument(
        "--env-blueprint", type=str, required=True,
        dest="blueprint",
        help="Path to the environment blueprint JSON or YAML file.",
    )
    parser.add_argument(
        "--policy-a-blueprint", type=str, required=True,
        dest="policy_a",
        help="Path to the policy A blueprint JSON or YAML file.",
    )
    parser.add_argument(
        "--policy-b-blueprint", type=str, required=True,
        dest="policy_b",
        help="Path to the policy B blueprint JSON or YAML file.",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to save video (e.g., match.mp4).",
    )
    parser.add_argument(
        "--recorder", action="append", default=[],
        metavar="SPEC",
        help=(
            "Inject post-action recorder (PostActionRecorder subclass), repeatable. "
            "Format: module.path:ClassName?key=value"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Episode seed (default: random).",
    )
    parser.add_argument(
        "--health-a", type=float, default=None,
        help="Initial HP for robot A (default: plugin default, usually 100).",
    )
    parser.add_argument(
        "--health-b", type=float, default=None,
        help="Initial HP for robot B (default: plugin default, usually 100).",
    )
    parser.add_argument(
        "--want-extras", action="store_true",
        help="Request side-channel payloads from policies (log-prob / value estimates, etc.).",
    )
    parser.add_argument(
        "--score-log-file", type=str, default=None,
        help="Append a per-substep combat score audit log to this file.",
    )
    args = parser.parse_args()

    blueprint = EnvBlueprint.load(args.blueprint)

    policy_a = PolicyBlueprint.load(args.policy_a).build()
    policy_b = PolicyBlueprint.load(args.policy_b).build()

    video_plugin = None
    if args.video:
        video_plugin = VideoRecorderPlugin(fps=30, output_path=args.video)

    recorders = [_load_from_spec(spec) for spec in args.recorder]

    with RoundRunner(
        blueprint=blueprint,
        policy_a=policy_a,
        policy_b=policy_b,
        video_plugin=video_plugin,
        recorders=recorders,
    ) as runner:
        result = runner.run(
            seed=args.seed,
            want_extras=args.want_extras,
            initial_health_a=args.health_a,
            initial_health_b=args.health_b,
            score_log_file=args.score_log_file,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
