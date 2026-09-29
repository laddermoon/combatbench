from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shlex
import subprocess

import numpy as np


SCHEMA_VERSION = 1
KINDS = {"logic": "V2", "physics": "V1", "action_sequence": "V1", "policy_eval": "V3"}


class Unsupported(Exception):
    pass


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"unsupported array dtype: {value.dtype}")
        return {"__array__": value.tolist(), "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if "__array__" in value:
            if set(value) != {"__array__", "dtype", "shape"}:
                raise ValueError("invalid array envelope")
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "biuf":
                raise ValueError("unsupported array dtype")
            array = np.asarray(value["__array__"], dtype=dtype)
            shape = tuple(value["shape"])
            if array.size == 0 and math.prod(shape) == 0:
                array = array.reshape(shape)
            if array.shape != shape:
                raise ValueError("array shape mismatch")
            return array
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def digest(value):
    payload = json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def save(path, value, overwrite=False):
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(encode(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_fixtures(adapter, cases, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    files = {}
    for case in cases:
        name = f"{case['id']}.json"
        bundle = capture(case, adapter)
        save(outdir / name, bundle, overwrite=True)
        files[name] = digest(bundle)
    index = {"schema_version": SCHEMA_VERSION, "format": "fixture-index", "files": files}
    save(outdir / "index.json", index, overwrite=True)
    return index


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return decode(json.load(stream))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    packages = {}
    for name in ("numpy", "scipy", "mujoco", "mujoco-mjx", "jax", "jaxlib", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def provenance(adapter):
    files = {str(Path(p).resolve()): file_hash(p) for p in adapter.dependencies}
    if not files:
        raise ValueError("adapter must declare source/model/config dependencies")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                         capture_output=True, text=True, check=True).stdout.strip()
    return {"backend": adapter.backend, "version": adapter.version,
            "git_commit": git, "files": files, "environment": environment()}


def check_case(case):
    required = {"id", "kind", "level", "seed", "episode", "frame", "input"}
    if set(case) != required:
        raise ValueError(f"case fields must be {sorted(required)}")
    if case["kind"] not in KINDS or case["level"] != KINDS[case["kind"]]:
        raise ValueError("invalid case kind/validation level")
    if not isinstance(case["id"], str) or not case["id"] or not isinstance(case["input"], dict):
        raise ValueError("case needs a nonempty id and input object")
    for key in ("seed", "episode", "frame"):
        if type(case[key]) is not int or case[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    digest(case)


def check_tolerance(tolerance):
    if set(tolerance) != {"atol", "rtol"}:
        raise ValueError("tolerance requires explicit atol and rtol")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
           for v in tolerance.values()):
        raise ValueError("tolerances must be finite and nonnegative")


def compare(expected, actual, tolerance, field_tolerances=None):
    check_tolerance(tolerance)
    overrides = {} if field_tolerances is None else field_tolerances
    for field, rule in overrides.items():
        if not isinstance(field, str) or not field.startswith("output."):
            raise ValueError("field tolerance must use an output.* path")
        check_tolerance(rule)
    failures = []
    used = set()

    def rule_for(path):
        matches = [key for key in overrides if path == key or path.startswith(key + ".")
                   or path.startswith(key + "[") or path.startswith(key + "(")]
        used.update(matches)
        return overrides[max(matches, key=len)] if matches else tolerance

    def fail(path, reason, a, b):
        failures.append({"field": path, "reason": reason, "expected": repr(a),
                         "actual": repr(b), "tolerance": dict(rule_for(path))})

    def visit(a, b, path):
        tolerance = rule_for(path)
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray):
                fail(path, "type", "ndarray", type(b).__name__)
            elif a.shape != b.shape or a.dtype != b.dtype:
                fail(path, "shape/dtype", (a.shape, str(a.dtype)), (b.shape, str(b.dtype)))
            else:
                finite = np.isfinite(a) & np.isfinite(b)
                if a.dtype.kind == "f":
                    equal = np.isclose(a, b, atol=tolerance["atol"], rtol=tolerance["rtol"])
                else:
                    equal = a == b
                for index in np.argwhere(~(finite & equal)):
                    idx = tuple(index)
                    fail(path + str(idx), "value/nonfinite", a[idx].item(), b[idx].item())
            return
        if type(a) is not type(b):
            fail(path, "type", type(a).__name__, type(b).__name__)
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                child = f"{path}.{key}"
                if key not in a:
                    fail(child, "unexpected-field", "absent", b[key])
                elif key not in b:
                    fail(child, "missing-field", a[key], "absent")
                else:
                    visit(a[key], b[key], child)
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                fail(path, "length", len(a), len(b))
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f"{path}[{i}]")
        elif isinstance(a, float):
            if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > tolerance["atol"] + tolerance["rtol"] * abs(a):
                fail(path, "value/nonfinite", a, b)
        elif a != b:
            fail(path, "value", a, b)

    visit(expected, actual, "output")
    if set(overrides) - used:
        raise ValueError(f"unused tolerance paths: {sorted(set(overrides) - used)}")
    return failures


def capture(case, adapter):
    check_case(case)
    source = provenance(adapter)
    expected = adapter.execute(copy.deepcopy(case))
    if not isinstance(expected, dict) or not expected:
        raise ValueError("reference must return a nonempty output object")
    if compare(expected, expected, {"atol": 0.0, "rtol": 0.0}):
        raise ValueError("nonfinite reference output")
    return {"schema_version": SCHEMA_VERSION, "case": copy.deepcopy(case),
            "source": source, "validator": {str(Path(__file__).resolve()): file_hash(__file__)},
            "tolerance": {"atol": 0.0, "rtol": 0.0}, "field_tolerances": {}, "expected": expected}


def replay(bundle, adapter, approved_digest):
    case = bundle["case"]
    check_case(case)
    check_tolerance(bundle["tolerance"])
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported fixture schema")
    report = {"schema_version": SCHEMA_VERSION, "scope": "single-case", "status": "not-run",
              "case_id": case["id"], "level": case["level"], "seed": case["seed"],
              "episode": case["episode"], "frame": case["frame"], "fixture_digest": digest(bundle),
              "approved_digest": approved_digest, "source": bundle["source"], "target": None,
              "tolerance": bundle["tolerance"], "failures": []}

    def stop(status, field, expected, actual):
        report["status"] = status
        report["failures"].append({"field": field, "reason": status,
                                   "expected": str(expected), "actual": str(actual),
                                   "tolerance": bundle["tolerance"]})
        return report

    if digest(bundle) != approved_digest:
        return stop("stale", "fixture_digest", approved_digest, digest(bundle))
    for path, sha in {**bundle["source"]["files"], **bundle["validator"]}.items():
        current = file_hash(path) if Path(path).is_file() else "missing"
        if current != sha:
            return stop("stale", path, sha, current)
    if environment() != bundle["source"]["environment"]:
        return stop("stale", "environment", bundle["source"]["environment"], environment())
    if adapter is None:
        return report
    report["target"] = provenance(adapter)
    try:
        actual = adapter.execute(copy.deepcopy(case))
        report["failures"] = compare(bundle["expected"], actual, bundle["tolerance"], bundle["field_tolerances"])
    except Unsupported as exc:
        return stop("unsupported", "capability", case["kind"], str(exc))
    except Exception as exc:
        return stop("fail", "execution", "successful execution", f"{type(exc).__name__}: {exc}")
    report["status"] = "fail" if report["failures"] else "pass"
    return report


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture")
    collect.add_argument("output", type=Path)
    collect.add_argument("--case", choices=("reward", "physics", "trajectory"), required=True)
    collect.add_argument("--index", type=int, default=0)
    collect.add_argument("--steps", type=int, choices=(1, 25), default=1)
    build = commands.add_parser("make-fixtures")
    build.add_argument("dir", type=Path)
    run = commands.add_parser("replay")
    run.add_argument("fixture", type=Path)
    approved = run.add_mutually_exclusive_group()
    approved.add_argument("--digest")
    approved.add_argument("--index", type=Path,
                          help="fixture index.json; digest is looked up by fixture filename")
    run.add_argument("--candidate", help="trusted Python module exporting Adapter()")
    run.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "make-fixtures":
        from .validation_cpu import CPUAdapter, all_cases
        index = write_fixtures(CPUAdapter(), all_cases(), args.dir)
        print(json.dumps({"status": "pass", "fixtures": len(index["files"]),
                          "index": str((args.dir / "index.json").resolve())}))
        return 0
    if args.command == "capture":
        from .validation_cpu import CPUAdapter, physics_case, reward_cases, trajectory_case
        if args.case == "reward":
            cases = reward_cases()
            if not 0 <= args.index < len(cases):
                parser.error(f"reward index must be in [0, {len(cases) - 1}]")
            case = cases[args.index]
        elif args.case == "physics":
            case = physics_case(args.steps)
        else:
            case = trajectory_case()
        bundle = capture(case, CPUAdapter())
        save(args.output, bundle)
        print(json.dumps({"fixture": str(args.output.resolve()), "digest": digest(bundle)}))
        return 0
    if args.index:
        index = load(args.index)
        if index.get("format") != "fixture-index" or not isinstance(index.get("files"), dict):
            parser.error("index must be a fixture-index produced by make-fixtures")
        digest_value = index["files"].get(args.fixture.name)
        if digest_value is None:
            parser.error(f"{args.fixture.name} is not registered in {args.index}")
    else:
        if not args.digest:
            parser.error("replay requires --digest or --index")
        digest_value = args.digest
    adapter = importlib.import_module(args.candidate).Adapter() if args.candidate else None
    bundle = load(args.fixture)
    report = replay(bundle, adapter, digest_value)
    command = ["python3", "-m", "envs.batchframework.validation", "replay",
               str(args.fixture.resolve()), "--digest", digest_value]
    if args.candidate:
        command += ["--candidate", args.candidate]
    report["replay"] = shlex.join(command)
    if args.output:
        save(args.output, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    import sys

    # `python -m` 将本文件加载为 __main__；候选适配器经 `from .validation import
    # Unsupported` 拿到的是规范命名的模块实例。若不建立别名，两边 Unsupported 是
    # 不同类，replay 无法把它识别为 unsupported（会误判为 fail）。
    sys.modules.setdefault("envs.batchframework.validation", sys.modules["__main__"])
    raise SystemExit(main())
