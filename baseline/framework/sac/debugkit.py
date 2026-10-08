"""SAC L2 dump capture, inspection, and recomputation.

``sac_dump_v1`` captures the exact sampled batch and the model/optimizer state
surrounding one ``critic_tick``.  The minimum supported evidence level is
``recompute``: Bellman targets, critic losses, actor decomposition, and the
alpha loss can be recomputed from the dump without replay storage still being
live.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from .clocks import SACClockState
from .experiment import SACParams, SACRewardChannel
from .networks import MultiHeadQCritic
from .s01_actor import S01Actor


SAC_DUMP_SCHEMA = "sac_dump_v1"


class SACDumpError(RuntimeError):
    """Raised when a SAC dump is malformed or cannot be recomputed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _trainer_spec(
    actor: S01Actor,
    critic: MultiHeadQCritic,
    channels: Sequence[SACRewardChannel],
    sp: SACParams,
    grad_clip_norm: float,
    critic_lr: float,
) -> Dict[str, Any]:
    return {
        "actor": {
            "kind": "s01_shared_sigma",
            "obs_dim": int(actor.obs_dim),
            "action_dim": int(actor.action_dim),
            "hidden_dim": int(actor.hidden_dim),
            "log_std_min": float(actor.log_std_min),
            "log_std_max": float(actor.log_std_max),
            "init_log_std": float(actor.init_log_std),
        },
        "critic": {
            "kind": "multi_head_q_trunk_heads",
            "obs_dim": int(critic.obs_dim),
            "action_dim": int(critic.action_dim),
            "hidden_dim": int(critic.hidden_dim),
            "groups": {
                key: list(group.channel_names)
                for key, group in critic.groups.items()
            },
        },
        "channels": [asdict(ch) for ch in channels],
        "sac_params": asdict(sp),
        "grad_clip_norm": float(grad_clip_norm),
        "critic_learning_rate": float(critic_lr),
    }


def begin_critic_tick_dump(
    *,
    dump_root: Path,
    critic_tick: int,
    clocks: SACClockState,
    batch: Mapping[str, Any],
    trainer_pre_state: Mapping[str, Any],
    hypothesis: str = "",
) -> Path:
    """Atomically prepare and persist the pre-update half of one L2 dump."""
    dump_root = Path(dump_root)
    dump_root.mkdir(parents=True, exist_ok=True)
    final_dir = dump_root / f"critic_tick_{int(critic_tick):08d}"
    if final_dir.exists():
        raise SACDumpError(f"dump already exists for critic_tick={critic_tick}: {final_dir}")
    tmp = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}_", dir=str(dump_root)))
    try:
        _write_json(tmp / "request.json", {
            "schema_version": SAC_DUMP_SCHEMA,
            "kind": "critic_tick",
            "critic_tick": int(critic_tick),
            "hypothesis": str(hypothesis),
            "created_unix": time.time(),
        })
        _write_json(tmp / "clocks.json", clocks.snapshot())
        torch.save(dict(batch), tmp / "batch.pt")
        torch.save(dict(trainer_pre_state), tmp / "trainer_pre.pt")
        return tmp
    except Exception:
        _write_json(tmp / "failed.json", {"stage": "begin"})
        raise


def finish_critic_tick_dump(
    tmp_dir: Path,
    *,
    forward_capture: Mapping[str, Any],
    trainer_post_state: Mapping[str, Any],
    update_stats: Mapping[str, Any],
    actor: S01Actor,
    critic: MultiHeadQCritic,
    channels: Sequence[SACRewardChannel],
    sp: SACParams,
    grad_clip_norm: float,
    critic_lr: float,
    keep_last: int = 8,
) -> Path:
    """Complete a dump begun by :func:`begin_critic_tick_dump`."""
    tmp = Path(tmp_dir)
    request = json.loads((tmp / "request.json").read_text(encoding="utf-8"))
    try:
        torch.save(dict(forward_capture), tmp / "forward.pt")
        torch.save(dict(trainer_post_state), tmp / "trainer_post.pt")
        _write_json(tmp / "spec.json", _trainer_spec(
            actor, critic, channels, sp, grad_clip_norm, critic_lr,
        ))
        _write_json(tmp / "analysis.json", {
            "evidence_level": "recompute",
            "update_stats": dict(update_stats),
        })
        files = {}
        for path in sorted(tmp.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        _write_json(tmp / "manifest.json", {
            "schema_version": SAC_DUMP_SCHEMA,
            "kind": "critic_tick",
            "critic_tick": int(request["critic_tick"]),
            "evidence_level": "recompute",
            "files": files,
        })
        final = tmp.with_name(tmp.name.lstrip(".").rsplit("_", 1)[0])
        os.replace(tmp, final)
        prune_critic_tick_dumps(final.parent, keep_last=keep_last)
        return final
    except Exception:
        _write_json(tmp / "failed.json", {"stage": "finish"})
        raise


def fail_critic_tick_dump(
    tmp_dir: Path,
    *,
    error: BaseException,
) -> Path:
    """Preserve an explicit failure artifact instead of hiding capture errors."""
    tmp = Path(tmp_dir)
    _write_json(tmp / "failed.json", {
        "schema_version": SAC_DUMP_SCHEMA,
        "error_type": type(error).__name__,
        "error": str(error),
    })
    failed = tmp.with_name(tmp.name.lstrip(".") + "_failed")
    if not failed.exists():
        os.replace(tmp, failed)
        return failed
    return tmp


def prune_critic_tick_dumps(dump_root: Path, *, keep_last: int = 8) -> None:
    """Apply the approved bounded latest-N retention for unpinned dumps."""
    root = Path(dump_root)
    if int(keep_last) <= 0 or not root.exists():
        return
    dumps = [
        p for p in root.iterdir()
        if p.is_dir()
        and p.name.startswith("critic_tick_")
        and not p.name.endswith("_failed")
        and not (p / ".pinned").exists()
    ]
    dumps.sort(key=lambda p: p.name)
    for old in dumps[:-int(keep_last)]:
        import shutil
        shutil.rmtree(old)


def load_dump(dump_dir: Path) -> Dict[str, Any]:
    root = Path(dump_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SACDumpError(f"missing dump manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SAC_DUMP_SCHEMA:
        raise SACDumpError(
            f"unsupported dump schema {manifest.get('schema_version')!r}"
        )
    for name, entry in manifest.get("files", {}).items():
        path = root / name
        if not path.exists():
            raise SACDumpError(f"dump file missing: {name}")
        if _sha256(path) != entry.get("sha256"):
            raise SACDumpError(f"dump file hash mismatch: {name}")
    payload: Dict[str, Any] = {"manifest": manifest}
    for name in ("request", "clocks", "spec", "analysis"):
        path = root / f"{name}.json"
        payload[name] = json.loads(path.read_text(encoding="utf-8"))
    for name in ("batch", "forward", "trainer_pre", "trainer_post"):
        payload[name] = torch.load(
            root / f"{name}.pt", map_location="cpu", weights_only=False,
        )
    return payload


def _build_actor(spec: Mapping[str, Any], state: Mapping[str, Any]) -> S01Actor:
    actor_spec = spec["actor"]
    if actor_spec.get("kind") != "s01_shared_sigma":
        raise SACDumpError(f"unsupported actor kind {actor_spec.get('kind')!r}")
    actor = S01Actor(
        obs_dim=int(actor_spec["obs_dim"]),
        action_dim=int(actor_spec["action_dim"]),
        hidden_dim=int(actor_spec["hidden_dim"]),
        log_std_min=float(actor_spec["log_std_min"]),
        log_std_max=float(actor_spec["log_std_max"]),
        init_log_std=float(actor_spec["init_log_std"]),
    )
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()
    return actor


def _build_critic(
    spec: Mapping[str, Any],
    channels: Sequence[SACRewardChannel],
    state: Mapping[str, Any],
) -> MultiHeadQCritic:
    critic_spec = spec["critic"]
    critic = MultiHeadQCritic(
        obs_dim=int(critic_spec["obs_dim"]),
        action_dim=int(critic_spec["action_dim"]),
        channels=tuple(channels),
        hidden_dim=int(critic_spec["hidden_dim"]),
        layer_norm=bool(spec["sac_params"]["q_layer_norm"]),
        critic_lr=float(spec["critic_learning_rate"]),
        device=torch.device("cpu"),
    )
    critic.load_state_dict(state["critic_state_dict"])
    return critic


def recompute_dump(
    dump_dir: Path,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> Dict[str, Any]:
    """Recompute the supported SAC losses from one ``critic_tick`` dump."""
    dump = load_dump(dump_dir)
    spec = dump["spec"]
    batch = dump["batch"]
    forward = dump["forward"]
    pre = dump["trainer_pre"]
    post = dump["trainer_post"]
    reference = dump["analysis"]["update_stats"]
    sp = SACParams(**spec["sac_params"])
    channels = tuple(SACRewardChannel(**item) for item in spec["channels"])

    actor = _build_actor(spec, pre)
    pre_critic = _build_critic(spec, channels, pre)
    post_critic = _build_critic(spec, channels, post)
    del actor  # sampling outputs are frozen in forward.pt for exactness

    with torch.no_grad():
        obs = batch["obs"].float()
        actions = batch["actions"].float()
        rewards = batch["rewards"].float()
        channel_valid = batch["channel_valid"].bool()
        bootstrap = batch["bootstrap"].float()
        actor_weight = batch["actor_weight"].float()
        sample_weight = batch["sample_weight"].float()
        sample_weight = sample_weight / (sample_weight.mean() + 1e-8)
        next_log_probs = forward["next_log_probs"].float()
        new_actions = forward["new_actions"].float()
        new_log_probs = forward["new_log_probs"].float()
        alpha = pre["log_alpha"].float().exp()
        log_alpha = pre["log_alpha"].float()

        recomputed: Dict[str, float] = {"alpha": float(alpha.item())}
        critic_loss = torch.zeros(())
        q_next_by_channel: Dict[str, torch.Tensor] = {}
        for c, ch in enumerate(channels):
            q1_next = pre_critic.q1_target_forward(
                batch["next_obs"].float(), forward["next_actions"].float(), ch.name,
            )
            q2_next = pre_critic.q2_target_forward(
                batch["next_obs"].float(), forward["next_actions"].float(), ch.name,
            )
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_next_by_channel[ch.name] = q_next
            target = rewards[:, c] * float(sp.reward_scale) + ch.gamma * bootstrap * q_next
            q1_pred = pre_critic.q1_forward(obs, actions, ch.name)
            q2_pred = pre_critic.q2_forward(obs, actions, ch.name)
            mask = channel_valid[:, c].float() * sample_weight
            mask_sum = mask.sum()
            q1_loss = (mask * (q1_pred - target).pow(2)).sum() / mask_sum
            q2_loss = (mask * (q2_pred - target).pow(2)).sum() / mask_sum
            critic_loss += q1_loss + q2_loss
            recomputed[f"q1_loss_{ch.name}"] = float(q1_loss.item())
            recomputed[f"q2_loss_{ch.name}"] = float(q2_loss.item())
            recomputed[f"q1_mean_{ch.name}"] = float(q1_pred.mean().item())
            recomputed[f"q2_mean_{ch.name}"] = float(q2_pred.mean().item())
            recomputed[f"td_abs_mean_{ch.name}"] = float(
                (q1_pred - target).abs().mean().item()
            )
            recomputed[f"actor_weight_mean_{ch.name}"] = float(
                actor_weight[:, c].mean().item()
            )

        q1_all = post_critic.q1_forward_all(obs, new_actions)
        q2_all = post_critic.q2_forward_all(obs, new_actions)
        weighted_q = torch.zeros_like(new_log_probs)
        for c, ch in enumerate(channels):
            weighted_q += actor_weight[:, c] * torch.min(
                q1_all[ch.name], q2_all[ch.name],
            )
        actor_loss = (sample_weight * (alpha * new_log_probs - weighted_q)).mean()
        target_entropy = (
            -float(spec["actor"]["action_dim"])
            if sp.target_entropy is None else float(sp.target_entropy)
        )
        alpha_loss = -(log_alpha * (new_log_probs + target_entropy)).mean()

        recomputed.update({
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "log_prob_mean": float(new_log_probs.mean().item()),
            "entropy_proxy_mean": float((-new_log_probs).mean().item()),
        })

    compared: Dict[str, Dict[str, float]] = {}
    max_abs = 0.0
    passed = True
    for key, expected in reference.items():
        if key not in recomputed:
            continue
        actual = float(recomputed[key])
        expected_f = float(expected)
        diff = abs(actual - expected_f)
        max_abs = max(max_abs, diff)
        ok = bool(np.isclose(actual, expected_f, atol=atol, rtol=rtol))
        passed = passed and ok
        compared[key] = {
            "reference": expected_f,
            "recomputed": actual,
            "abs_diff": diff,
            "ok": ok,
        }
    return {
        "dump_dir": str(Path(dump_dir)),
        "schema_version": SAC_DUMP_SCHEMA,
        "evidence_level": "recompute",
        "passed": bool(passed),
        "max_abs_diff": float(max_abs),
        "compared": compared,
        "recomputed": recomputed,
    }


def summarize_dump(dump_dir: Path) -> Dict[str, Any]:
    dump = load_dump(dump_dir)
    batch = dump["batch"]
    return {
        "dump_dir": str(Path(dump_dir)),
        "schema_version": dump["manifest"]["schema_version"],
        "kind": dump["manifest"]["kind"],
        "critic_tick": dump["manifest"]["critic_tick"],
        "clocks": dump["clocks"],
        "batch_size": len(batch["sample_ids"]),
        "sample_ids": [int(v) for v in batch["sample_ids"].tolist()],
        "source_keys": list(batch["source_keys"]),
        "update_stats": dump["analysis"]["update_stats"],
    }


def find_sample(dump_dir: Path, *, sample_id: int) -> Dict[str, Any]:
    dump = load_dump(dump_dir)
    batch = dump["batch"]
    matches = np.nonzero(batch["sample_ids"].cpu().numpy() == int(sample_id))[0]
    if len(matches) != 1:
        raise SACDumpError(
            f"sample_id={sample_id} occurs {len(matches)} times in dump"
        )
    i = int(matches[0])
    return {
        "sample_id": int(sample_id),
        "index": i,
        "source_key": str(batch["source_keys"][i]),
        "metadata": dump["batch"].get("metadata", [{}] * len(matches))[i],
        "obs": batch["obs"][i],
        "action": batch["actions"][i],
        "next_obs": batch["next_obs"][i],
        "rewards": batch["rewards"][i],
        "terminated": bool(batch["terminated"][i]),
        "truncated": bool(batch["truncated"][i]),
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect SAC sac_dump_v1 bundles")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "recompute"):
        p = sub.add_parser(name)
        p.add_argument("dump", type=Path)
    find = sub.add_parser("find-sample")
    find.add_argument("dump", type=Path)
    find.add_argument("sample_id", type=int)
    args = parser.parse_args(argv)

    if args.command == "summary":
        out = summarize_dump(args.dump)
    elif args.command == "recompute":
        out = recompute_dump(args.dump)
    else:
        out = find_sample(args.dump, sample_id=args.sample_id)
    print(json.dumps(_jsonable(out), indent=2, sort_keys=True))
    return 0 if args.command != "recompute" or out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "SAC_DUMP_SCHEMA",
    "SACDumpError",
    "begin_critic_tick_dump",
    "fail_critic_tick_dump",
    "finish_critic_tick_dump",
    "find_sample",
    "load_dump",
    "prune_critic_tick_dumps",
    "recompute_dump",
    "summarize_dump",
]
