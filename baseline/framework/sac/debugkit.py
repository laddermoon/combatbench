"""SAC L2 dump capture, inspection, and recomputation.

``sac_dump_v3`` captures the exact sampled batch and the model/optimizer state
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
import shutil
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
from .tn_actor import TNActor
from .trainer import (
    _RegularizerArgs,
    _forward_all_candidates,
    _state_regularizer,
)


SAC_DUMP_SCHEMA = "sac_dump_v3"
SUPPORTED_DUMP_SCHEMAS = frozenset({"sac_dump_v2", "sac_dump_v3"})


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


def _batch_consistency(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Row-level contract violations in a sampled batch (v3 dumps)."""
    def _t(name: str) -> Optional[torch.Tensor]:
        v = batch.get(name)
        return v.detach().cpu() if isinstance(v, torch.Tensor) else None

    terminated = _t("terminated")
    truncated = _t("truncated")
    bootstrap = _t("bootstrap")
    out: Dict[str, Any] = {"checked": False, "violations": {}}
    if terminated is None or bootstrap is None:
        return out
    violations: Dict[str, int] = {}
    term_bs = int(((terminated.bool()) & (bootstrap > 0)).sum().item())
    if term_bs:
        violations["terminated_with_bootstrap"] = term_bs
    if truncated is not None:
        trunc_no_bs = int(
            ((truncated.bool()) & (bootstrap <= 0)).sum().item()
        )
        if trunc_no_bs:
            violations["truncated_without_bootstrap"] = trunc_no_bs
    out["checked"] = True
    out["violations"] = violations
    out["rows"] = int(terminated.numel())
    out["terminated_rows"] = int(terminated.sum().item())
    out["truncated_rows"] = (
        int(truncated.sum().item()) if truncated is not None else 0
    )
    return out


def _trainer_spec(
    actor: S01Actor,
    critic: MultiHeadQCritic,
    channels: Sequence[SACRewardChannel],
    sp: SACParams,
    grad_clip_norm: float,
    critic_lr: float,
    slice_contract: Optional[Mapping[str, Any]] = None,
    behavior_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if hasattr(actor, "state_payload"):
        payload = actor.state_payload()
        arch = str(payload.get("policy_arch", ""))
        if arch.startswith("tn_"):
            actor_spec = {
                "kind": "tn_actor",
                "policy_arch": arch,
                "arch": str(payload["arch"]),
                "obs_dim": int(payload["obs_dim"]),
                "action_dim": int(payload["action_dim"]),
                "hidden_dim": int(payload["hidden_dim"]),
                "num_components": int(payload["num_components"]),
                "sigma_min": float(payload["sigma_min"]),
                "sigma_max": float(payload["sigma_max"]),
                "init_std": float(payload["init_std"]),
                "explore_alpha": float(payload["explore_alpha"]),
                "component_init_noise": float(
                    payload["component_init_noise"]
                ),
            }
        else:
            actor_spec = {
                "kind": "s01_shared_sigma",
                "obs_dim": int(actor.obs_dim),
                "action_dim": int(actor.action_dim),
                "hidden_dim": int(actor.hidden_dim),
                "log_std_min": float(actor.log_std_min),
                "log_std_max": float(actor.log_std_max),
                "init_log_std": float(actor.init_log_std),
            }
    else:
        raise SACDumpError(
            f"actor {type(actor).__name__} has no state_payload; "
            "cannot write a reconstructable dump spec"
        )
    return {
        "actor": actor_spec,
        "critic": {
            "kind": "independent_channel_twin_q",
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
        # Self-describing slice contract: the formula constants needed to
        # recompute rewards/actor_weight from task_facts/reward_features.
        # None on legacy dumps — verifiers fall back to class defaults
        # and mark the check's contract_source accordingly.
        "slice_contract": (
            dict(slice_contract) if slice_contract is not None else None
        ),
        # WP3: which policy produced this batch — enables replay-back
        # verification and episode reconstruction long after the export
        # dirs are pruned.  ``model_path`` inside the blueprint config is
        # rewritten to the dump-local snapshot by the caller.
        "behavior_policy": (
            dict(behavior_policy) if behavior_policy is not None else None
        ),
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
    replay_stats: Optional[Mapping[str, Any]] = None,
    slice_contract: Optional[Mapping[str, Any]] = None,
    behavior_policy_bp: Optional[Mapping[str, Any]] = None,
    behavior_policy_dir: Optional[Path] = None,
) -> Path:
    """Complete a dump begun by :func:`begin_critic_tick_dump`.

    ``behavior_policy_bp``/``behavior_policy_dir`` snapshot the behavior
    policy that collected this batch's newest transitions — model files
    are copied into ``<dump>/behavior_policy/`` so replay-back and
    episode reconstruction keep working after ``policy_exports/`` is
    pruned.
    """
    tmp = Path(tmp_dir)
    request = json.loads((tmp / "request.json").read_text(encoding="utf-8"))
    behavior_policy: Optional[Dict[str, Any]] = None
    if behavior_policy_bp is not None:
        bp_dict = dict(behavior_policy_bp)
        cfg = dict(bp_dict.get("config") or {})
        snap_dir = tmp / "behavior_policy"
        src_dir = (
            Path(behavior_policy_dir) if behavior_policy_dir is not None
            else None
        )
        model_src = src_dir / "model.pt" if src_dir is not None else None
        snapshot_dir: Optional[str] = None
        if model_src is not None and model_src.exists():
            snap_dir.mkdir(exist_ok=True)
            shutil.copy2(model_src, snap_dir / "model.pt")
            if (src_dir / "policy_blueprint.yaml").exists():
                shutil.copy2(
                    src_dir / "policy_blueprint.yaml",
                    snap_dir / "policy_blueprint.yaml",
                )
            snapshot_dir = "behavior_policy"
        # model_path in cfg keeps pointing at the ORIGINAL export dir
        # (provenance); consumers resolve the dump-local copy via
        # snapshot_dir first, falling back to the original path.
        bp_dict["config"] = cfg
        behavior_policy = {
            "blueprint": bp_dict,
            "snapshot_dir": snapshot_dir,
            "source_dir": str(src_dir) if src_dir is not None else None,
        }
    try:
        torch.save(dict(forward_capture), tmp / "forward.pt")
        torch.save(dict(trainer_post_state), tmp / "trainer_post.pt")
        _write_json(tmp / "spec.json", _trainer_spec(
            actor, critic, channels, sp, grad_clip_norm, critic_lr,
            slice_contract=slice_contract,
            behavior_policy=behavior_policy,
        ))
        analysis_payload: Dict[str, Any] = {
            "evidence_level": "recompute",
            "update_stats": dict(update_stats),
            "consistency": _batch_consistency(
                torch.load(
                    tmp / "batch.pt", map_location="cpu",
                    weights_only=False,
                )
            ),
        }
        if replay_stats is not None:
            analysis_payload["replay_stats"] = _jsonable(dict(replay_stats))
        _write_json(tmp / "analysis.json", analysis_payload)
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
        final = tmp.parent / f"critic_tick_{int(request['critic_tick']):08d}"
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
    if manifest.get("schema_version") not in SUPPORTED_DUMP_SCHEMAS:
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


def _build_actor(spec: Mapping[str, Any], state: Mapping[str, Any]):
    actor_spec = spec["actor"]
    kind = actor_spec.get("kind")
    if kind == "s01_shared_sigma":
        actor = S01Actor(
            obs_dim=int(actor_spec["obs_dim"]),
            action_dim=int(actor_spec["action_dim"]),
            hidden_dim=int(actor_spec["hidden_dim"]),
            log_std_min=float(actor_spec["log_std_min"]),
            log_std_max=float(actor_spec["log_std_max"]),
            init_log_std=float(actor_spec["init_log_std"]),
        )
    elif kind == "tn_actor":
        actor = TNActor(
            obs_dim=int(actor_spec["obs_dim"]),
            action_dim=int(actor_spec["action_dim"]),
            arch=str(actor_spec["arch"]),
            hidden_dim=int(actor_spec["hidden_dim"]),
            n_components=int(actor_spec["num_components"]),
            sigma_min=float(actor_spec["sigma_min"]),
            sigma_max=float(actor_spec["sigma_max"]),
            init_std=float(actor_spec["init_std"]),
            explore_alpha=float(actor_spec["explore_alpha"]),
            component_init_noise=float(actor_spec["component_init_noise"]),
        )
    else:
        raise SACDumpError(f"unsupported actor kind {kind!r}")
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
    """Recompute the supported SAC losses from one ``critic_tick`` dump.

    Full coverage (R2-W35): both the Shannon path and the U-regularizer
    paths (``u_bonus``/``u_floor``) are recomputed — the per-state U term
    comes from the same ``_state_regularizer`` the trainer uses, the
    candidates are frozen via captured ``u_next``/``u_actor``, so the
    recomputed values are exact replicas of the recorded update.
    ``coverage.unsupported_keys`` is retained for forward compatibility
    (future modes would land there instead of being silently skipped).
    """
    dump = load_dump(dump_dir)
    spec = dump["spec"]
    batch = dump["batch"]
    forward = dump["forward"]
    pre = dump["trainer_pre"]
    post = dump["trainer_post"]
    reference = dump["analysis"]["update_stats"]
    sp = SACParams(**spec["sac_params"])
    channels = tuple(SACRewardChannel(**item) for item in spec["channels"])
    mode = sp.regularizer_mode
    is_shannon = mode == "shannon"
    supported_modes = {"shannon", "u_bonus", "u_floor"}
    if mode not in supported_modes:
        raise SACDumpError(
            f"regularizer_mode {mode!r} is not recomputable"
        )

    actor = _build_actor(spec, pre)
    pre_critic = _build_critic(spec, channels, pre)
    post_critic = _build_critic(spec, channels, post)

    with torch.no_grad():
        obs = batch["obs"].float()
        next_obs = batch["next_obs"].float()
        actions = batch["actions"].float()
        rewards = batch["rewards"].float()
        channel_valid = batch["channel_valid"].bool()
        bootstrap = batch["bootstrap"].float()
        actor_weight = batch["actor_weight"].float()
        actor_weight_next = batch["actor_weight_next"].float()
        sample_weight = batch["sample_weight"].float()
        alpha = pre["log_alpha"].float().exp()
        log_alpha = pre["log_alpha"].float()
        C = len(channels)

        # P4-TRAIN-2: enumerated candidates are frozen via u_next/u_actor;
        # recompute regenerates actions/logp/weights from rebuilt actor.
        u_next = forward["u_next"].float()
        u_actor = forward["u_actor"].float()
        next_actions, next_log_probs, integ_w_next = actor.expectation_samples(
            next_obs, u_next,
        )
        new_actions, new_log_probs, integ_w = actor.expectation_samples(
            obs, u_actor,
        )
        recomputed: Dict[str, float] = {"alpha": float(alpha.item())}
        recomputed["action_replay_max_diff"] = float(
            max(
                (next_actions - forward["next_actions"].float()).abs().max(),
                (new_actions - forward["new_actions"].float()).abs().max(),
            ).item()
        )
        critic_loss = torch.zeros(())
        q1_next = torch.stack(
            [
                _forward_all_candidates(
                    lambda o, a: pre_critic.q1_target_forward(o, a, ch.name),
                    next_obs, next_actions,
                )
                for ch in channels
            ],
            dim=-1,
        )
        q2_next = torch.stack(
            [
                _forward_all_candidates(
                    lambda o, a: pre_critic.q2_target_forward(o, a, ch.name),
                    next_obs, next_actions,
                )
                for ch in channels
            ],
            dim=-1,
        )
        q_next_pairs = torch.stack((q1_next, q2_next), dim=-1)
        w_next = actor_weight_next[:, None, None, :, None]
        f_next = (q_next_pairs * w_next).sum(dim=3)
        pair_next = f_next.argmin(dim=-1)
        selected_next = q_next_pairs.gather(
            -1, pair_next[..., None, None].expand(-1, -1, -1, C, 1),
        ).squeeze(-1)
        recomputed["target_pair1_frac"] = float(pair_next.float().mean().item())
        # Continuation regularizer B(s'): shannon uses the per-candidate
        # entropy term; U modes add the per-state regularizer once (the
        # same _state_regularizer the trainer calls).
        if is_shannon:
            b_next = -alpha * next_log_probs
            b_state_next = None
        else:
            b_next = torch.zeros_like(next_log_probs)
            b_state_next = _state_regularizer(
                actor, next_obs,
                _RegularizerArgs(
                    regularizer_mode=mode,
                    reg_lambda=sp.reg_lambda,
                    u_floor=sp.u_floor,
                    u_kind=sp.u_kind,
                ),
            )
            recomputed["reg_next_mean"] = float(b_state_next.mean().item())
        for c, ch in enumerate(channels):
            q1_pred = pre_critic.q1_forward(obs, actions, ch.name)
            q2_pred = pre_critic.q2_forward(obs, actions, ch.name)
            mask = channel_valid[:, c].float() * sample_weight
            mask_sum = mask.sum()
            recomputed[f"critic_valid_weight_{ch.name}"] = float(mask_sum.item())
            if mask_sum.item() <= 0:
                recomputed[f"critic_updated_{ch.name}"] = 0.0
                continue
            recomputed[f"critic_updated_{ch.name}"] = 1.0
            recomputed[f"q1_mean_{ch.name}"] = float(q1_pred.mean().item())
            recomputed[f"q2_mean_{ch.name}"] = float(q2_pred.mean().item())
            recomputed[f"actor_weight_mean_{ch.name}"] = float(
                actor_weight[:, c].mean().item()
            )
            recomputed[f"actor_weight_next_mean_{ch.name}"] = float(
                actor_weight_next[:, c].mean().item()
            )
            v_c = (
                integ_w_next
                * (selected_next[..., c] + b_next)
            ).sum(dim=(1, 2))
            if b_state_next is not None:
                v_c = v_c + b_state_next
            target = (
                rewards[:, c] * float(sp.reward_scale)
                + ch.gamma
                * bootstrap
                * v_c
            )
            q1_loss = (mask * (q1_pred - target).pow(2)).sum() / mask_sum
            q2_loss = (mask * (q2_pred - target).pow(2)).sum() / mask_sum
            critic_loss += q1_loss + q2_loss
            recomputed[f"q1_loss_{ch.name}"] = float(q1_loss.item())
            recomputed[f"q2_loss_{ch.name}"] = float(q2_loss.item())
            recomputed[f"td_abs_mean_{ch.name}"] = float(
                (q1_pred - target).abs().mean().item()
            )

        actor_mask = channel_valid.all(dim=1).float() * sample_weight
        actor_den = actor_mask.sum()
        recomputed["actor_valid_weight"] = float(actor_den.item())
        recomputed["actor_valid_count"] = float(
            channel_valid.all(dim=1).sum().item()
        )
        q1_all = torch.stack(
            [
                _forward_all_candidates(
                    lambda o, a: post_critic.q1_forward(o, a, ch.name),
                    obs, new_actions,
                )
                for ch in channels
            ],
            dim=-1,
        )
        q2_all = torch.stack(
            [
                _forward_all_candidates(
                    lambda o, a: post_critic.q2_forward(o, a, ch.name),
                    obs, new_actions,
                )
                for ch in channels
            ],
            dim=-1,
        )
        q_online_pairs = torch.stack((q1_all, q2_all), dim=-1)
        share = torch.tensor(
            [1.0 if ch.actor_weight_share else 0.0 for ch in channels],
            dtype=actor_weight.dtype, device=actor_weight.device,
        )
        w_actor = (
            actor_weight[:, None, None, :, None]
            * share[None, None, None, :, None]
        )
        f_actor = (q_online_pairs * w_actor).sum(dim=3)
        pair_actor = f_actor.argmin(dim=-1)
        recomputed["actor_pair1_frac"] = float(pair_actor.float().mean().item())
        logp_bar = (integ_w * new_log_probs).sum(dim=(1, 2))
        recomputed.update({
            "log_prob_mean": float(logp_bar.mean().item()),
            "entropy_proxy_mean": float((-logp_bar).mean().item()),
        })

        f_selected = f_actor.gather(
            -1, pair_actor[..., None],
        ).squeeze(-1)
        if is_shannon:
            weighted_q = (
                integ_w * (alpha * new_log_probs - f_selected)
            ).sum(dim=(1, 2))
            reg_term = torch.zeros_like(weighted_q)
        else:
            weighted_q = (integ_w * (-f_selected)).sum(dim=(1, 2))
            reg_term = -_state_regularizer(
                actor, obs,
                _RegularizerArgs(
                    regularizer_mode=mode,
                    reg_lambda=sp.reg_lambda,
                    u_floor=sp.u_floor,
                    u_kind=sp.u_kind,
                ),
            )
            recomputed["reg_actor_term_mean"] = float(
                (actor_mask * reg_term).sum().item() / actor_den.item()
            )
        actor_objective = weighted_q + reg_term
        actor_loss = (actor_mask * actor_objective).sum() / actor_den
        # alpha_loss exists only when auto-alpha is on; otherwise the
        # recorded stat is the constant 0.0 (fixed-alpha and U modes).
        if is_shannon and sp.auto_alpha:
            target_entropy = (
                -float(spec["actor"]["action_dim"])
                if sp.target_entropy is None else float(sp.target_entropy)
            )
            alpha_loss = -(
                actor_mask * log_alpha * (logp_bar + target_entropy)
            ).sum() / actor_den
        else:
            alpha_loss = torch.zeros(())
        recomputed.update({
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
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
    unsupported = sorted(
        k for k in reference
        if k not in recomputed and (
            k.startswith(("q1_loss_", "q2_loss_", "td_abs_mean_"))
            or k in {"critic_loss", "actor_loss", "alpha_loss"}
        )
    )
    uncompared = sorted(
        k for k in reference
        if k not in recomputed and k not in unsupported
    )
    return {
        "dump_dir": str(Path(dump_dir)),
        "schema_version": SAC_DUMP_SCHEMA,
        "evidence_level": "recompute",
        "passed": bool(passed),
        "max_abs_diff": float(max_abs),
        "compared": compared,
        "recomputed": recomputed,
        "coverage": {
            "regularizer_mode": mode,
            "status": "full",
            "compared_keys": sorted(compared),
            "unsupported_keys": unsupported,
            "uncompared_keys": uncompared,
            "note": (
                "all regularizer modes are recomputed (R2-W35); "
                "unsupported_keys is retained for forward compatibility"
            ),
        },
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
    metadata = batch.get("metadata")
    if metadata is None:
        metadata_item = {}
    else:
        metadata_item = metadata[i]
    return {
        "sample_id": int(sample_id),
        "index": i,
        "source_key": str(batch["source_keys"][i]),
        "metadata": metadata_item,
        "obs": batch["obs"][i],
        "action": batch["actions"][i],
        "next_obs": batch["next_obs"][i],
        "rewards": batch["rewards"][i],
        "actor_gate": batch["actor_gate"][i],
        "actor_weight": batch["actor_weight"][i],
        "actor_gate_next": batch["actor_gate_next"][i],
        "actor_weight_next": batch["actor_weight_next"][i],
        "terminated": bool(batch["terminated"][i]),
        "truncated": bool(batch["truncated"][i]),
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    from . import analysis

    parser = argparse.ArgumentParser(
        description="Inspect SAC runs and dump bundles (v2/v3)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "recompute", "inspect", "pipeline"):
        p = sub.add_parser(name)
        p.add_argument("dump", type=Path)
    find = sub.add_parser("find-sample")
    find.add_argument("dump", type=Path)
    find.add_argument("sample_id", type=int)
    cat = sub.add_parser("catalog")
    cat.add_argument("--prefix", default=None)
    samp = sub.add_parser("samples")
    samp.add_argument("dump", type=Path)
    samp.add_argument("--sort", default="td_abs")
    samp.add_argument("--channel", default=None)
    samp.add_argument("--limit", type=int, default=50)
    tr = sub.add_parser("trace")
    tr.add_argument("dump", type=Path)
    tr.add_argument("--sample-id", type=int, default=None)
    tr.add_argument("--source-key", default=None)
    at = sub.add_parser(
        "attribute",
        help="provenance × update-contribution joint attribution",
    )
    at.add_argument("dump", type=Path)
    at.add_argument(
        "--group-by",
        default="collection_round",
        choices=[
            "collection_round", "policy_fingerprint",
            "explore_factor", "random_start",
        ],
    )
    dc = sub.add_parser(
        "decomp",
        help="target/actor-objective part decomposition",
    )
    dc.add_argument("dump", type=Path)
    dc.add_argument("--sample-id", type=int, default=None)
    dr = sub.add_parser(
        "drift",
        help="pre/post fixed-input policy+Q drift on the dump batch",
    )
    dr.add_argument("dump", type=Path)
    ga = sub.add_parser(
        "grad-attr",
        help="HEAVY on-demand gradient attribution: rebuilds pre-state "
             "and takes per-group/per-term autograd grads",
    )
    ga.add_argument("dump", type=Path)
    ga.add_argument(
        "--group-by", default="collection_round",
        choices=[
            "collection_round", "policy_fingerprint",
            "explore_factor", "random_start",
        ],
    )
    ser = sub.add_parser("series")
    ser.add_argument("run_dir", type=Path)
    ser.add_argument("--metric", required=True)
    ser.add_argument("--event", default=None)
    runs_p = sub.add_parser("runs")
    runs_p.add_argument("runs_root", type=Path)
    run_p = sub.add_parser("run")
    run_p.add_argument("run_dir", type=Path)
    rep = sub.add_parser("replay")
    rep.add_argument("path", type=Path)
    sc = sub.add_parser(
        "slice-ctx",
        help="rebuild the live frames of one slice from a replay state",
    )
    sc.add_argument("path", type=Path,
                    help="run dir, checkpoint dir, or replay.pt")
    sc.add_argument("--source-key", default=None)
    sc.add_argument("--slice-id", default=None)
    sc.add_argument("--sample-id", type=int, default=None)
    qy = sub.add_parser("query")
    qy.add_argument("artifact", type=Path)
    qy.add_argument("path")
    # ---- contract ledger + diagnostic cuts (WP1) ----
    lc = sub.add_parser(
        "contracts",
        help="run the falsifiable contract ledger for one critic_tick dump",
    )
    lc.add_argument("dump", type=Path)
    lc.add_argument(
        "--check", action="append", default=None,
        help="restrict to check id (C4.1) or cut name (target); repeatable",
    )
    for name, helptext in (
        ("explore", "cut 0: realized exploration eps = a_env - a_det"),
        ("slice-verify", "cut 1: episode→slice reward/gate contract recompute"),
        ("demography", "cut 2: batch vs replay provenance census"),
        ("twins", "cut 3: twin-min / candidate-enumeration audit"),
        ("targets", "cut 4: Bellman target closure + counterfactual recompute"),
        ("fit", "cut 5: critic EV / scatter / TD concentration"),
        ("actor", "cut 6: actor objective gate chain + U(s)/floor activity"),
        ("update-checks", "cut 8: Polyak identity / alpha / layer steps"),
        ("post", "cut 9: post-update per-sample drift on same batch"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("dump", type=Path)
        if name == "targets":
            p.add_argument("--gamma", type=float, default=None)
            p.add_argument("--reward-scale", type=float, default=None)
            p.add_argument("--alpha", type=float, default=None)
            p.add_argument("--u-floor", type=float, default=None)
            p.add_argument("--reg-lambda", type=float, default=None)
    gs = sub.add_parser(
        "gradsig",
        help="cut 7 (HEAVY): exact per-sample gradient projection "
             "g_i·G_hat over the full batch + channel conflict matrix",
    )
    gs.add_argument("dump", type=Path)
    gs.add_argument(
        "--sides", default="actor,critic",
        help="comma list: actor,critic",
    )
    rc2 = sub.add_parser(
        "recon",
        help="cut 0/1/9 anchor (HEAVY): rerun the episode that produced "
             "one batch row, render mp4, verify replayed frames",
    )
    rc2.add_argument("dump", type=Path)
    rc2.add_argument("--sample-id", type=int, default=None)
    rc2.add_argument("--source-key", default=None)
    rc2.add_argument("--fps", type=int, default=30)
    dl = sub.add_parser(
        "delta",
        help="cross-generation delta: replay every checkpoint's det "
             "policy on this dump's batch observations",
    )
    dl.add_argument("dump", type=Path)
    dl.add_argument("--stride", type=int, default=1)
    dl.add_argument("--max-checkpoints", type=int, default=24)
    dq = sub.add_parser(
        "dump",
        help="write dump_request.json so a running SAC loop captures "
             "the next (or a given) critic tick",
    )
    dq.add_argument("run_dir", type=Path)
    dq.add_argument("--hypothesis", default="")
    dq.add_argument("--at-tick", type=int, default=None)
    sv = sub.add_parser("serve", help="read-only HTTP JSON debug server")
    sv.add_argument("root", type=Path)
    sv.add_argument("--port", type=int, default=8766)
    sv.add_argument("--host", default="127.0.0.1")
    vw = sub.add_parser(
        "viewer",
        help="interactive debug viewer (HTTP server + SPA)",
    )
    vw.add_argument("root", type=Path)
    vw.add_argument("--port", type=int, default=8766)
    vw.add_argument("--host", default="127.0.0.1")
    vw.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.command in ("serve", "viewer"):
        from .debugserver import make_server, _root_mode
        server = make_server(args.root, args.port, args.host)
        url = f"http://{args.host}:{args.port}/"
        print(
            f"[sac-debug] serving {args.root} "
            f"({_root_mode(args.root)}) on {url}",
            flush=True,
        )
        if args.command == "viewer" and not args.no_browser:
            try:
                import webbrowser

                webbrowser.open(url)
            except Exception:
                pass
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    if args.command == "summary":
        out = summarize_dump(args.dump)
    elif args.command == "recompute":
        out = recompute_dump(args.dump)
    elif args.command == "inspect":
        out = analysis.dump_inspect(args.dump)
    elif args.command == "pipeline":
        out = analysis.pipeline_summary(args.dump)
    elif args.command == "find-sample":
        out = find_sample(args.dump, sample_id=args.sample_id)
    elif args.command == "catalog":
        out = analysis.metric_catalog(prefix=args.prefix)
    elif args.command == "samples":
        out = analysis.dump_samples(
            args.dump, sort=args.sort, channel=args.channel,
            limit=args.limit,
        )
    elif args.command == "trace":
        out = analysis.dump_trace(
            args.dump, sample_id=args.sample_id,
            source_key=args.source_key,
        )
    elif args.command == "attribute":
        out = analysis.batch_attribution(
            args.dump, group_by=args.group_by,
        )
    elif args.command == "decomp":
        out = analysis.objective_decomp(
            args.dump, sample_id=args.sample_id,
        )
    elif args.command == "drift":
        out = analysis.policy_drift(args.dump)
    elif args.command == "grad-attr":
        out = analysis.grad_attribution(
            args.dump, group_by=args.group_by,
        )
    elif args.command == "series":
        out = analysis.metric_series(
            args.run_dir, key=args.metric, event=args.event,
        )
    elif args.command == "runs":
        from . import runview

        out = runview.list_runs(args.runs_root)
    elif args.command == "run":
        out = analysis.run_summary(args.run_dir)
    elif args.command == "replay":
        out = analysis.replay_report(args.path)
    elif args.command == "slice-ctx":
        out = analysis.slice_context(
            args.path, source_key=args.source_key,
            slice_id=args.slice_id, sample_id=args.sample_id,
        )
    elif args.command == "contracts":
        out = analysis.verify_dump_contracts(
            args.dump, checks=args.check,
        )
    elif args.command == "explore":
        out = analysis.rollout_exploration(args.dump)
    elif args.command == "slice-verify":
        out = analysis.slice_verification(args.dump)
    elif args.command == "demography":
        out = analysis.batch_demography(args.dump)
    elif args.command == "twins":
        out = analysis.candidate_twin_analysis(args.dump)
    elif args.command == "targets":
        out = analysis.recompute_targets(
            args.dump,
            gamma=args.gamma, reward_scale=args.reward_scale,
            alpha=args.alpha, u_floor=args.u_floor,
            reg_lambda=args.reg_lambda,
        )
    elif args.command == "fit":
        out = analysis.critic_fit(args.dump)
    elif args.command == "actor":
        out = analysis.actor_contract(args.dump)
    elif args.command == "update-checks":
        out = analysis.update_checks(args.dump)
    elif args.command == "post":
        out = analysis.post_detail(args.dump)
    elif args.command == "gradsig":
        from . import gradsig as _gradsig
        out = _gradsig.per_sample_gradsig(
            args.dump,
            sides=tuple(s for s in args.sides.split(",") if s),
        )
    elif args.command == "recon":
        from . import episode_recon as _recon
        try:
            out = _recon.reconstruct_episode(
                args.dump, sample_id=args.sample_id,
                source_key=args.source_key, fps=args.fps,
            )
        except KeyError as exc:
            # Expected outcome when artifacts were pruned or provenance
            # is incomplete — report structurally, not as a traceback.
            out = {
                "status": "unavailable",
                "reason": str(exc).strip("'"),
            }
            print(json.dumps(_jsonable(out), indent=2, sort_keys=True))
            return 3
    elif args.command == "delta":
        from . import episode_recon as _recon
        out = _recon.cross_gen_delta(
            args.dump, stride=args.stride,
            max_checkpoints=args.max_checkpoints,
        )
    elif args.command == "dump":
        request = {"hypothesis": args.hypothesis}
        if args.at_tick is not None:
            request["critic_tick"] = int(args.at_tick)
        target = args.run_dir / "dump_request.json"
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(request) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        out = {"request_path": str(target), "request": request}
    else:  # query
        out = analysis.query_payload(
            analysis.load_artifact(args.artifact), args.path,
        )
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
