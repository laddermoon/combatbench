"""Dump capture — write complete update data to disk for review.

Called from the training loop after ``ppo_update`` returns.  All data
comes from the actual training objects — episodes, trajectories, buffer,
ppo_update internals (via dump_callback), and the Job's env/policy/seed
configuration.  No reconstruction, no separate pipeline.

Output layout::

    <run_dir>/dumps/u{N:05d}/
    ├── request.json          # original request (moved by poll)
    ├── manifest.json         # update, timestamp, experiment, hypothesis
    ├── env_blueprint.yaml    # serialized from jobs[0].env_bp
    ├── episode_options.json  # from jobs[0].episode_options
    ├── stochastic_policy/    # self-contained policy with baked-in explore_factor
    │   ├── model.pt          # copied from policy_exports/u{N:05d}/
    │   ├── policy.py         # inner ExportedTruncNormPolicy + ef callable + wrapper
    │   └── policy_blueprint.yaml
    ├── episodes.npz          # per-frame: obs, actions, observer_outputs
    ├── trajectories.npz      # per-channel: reward, actor_weight, floor_weight
    ├── buffer.npz            # obs, actions, log_probs, frame_ids
    ├── gae.npz               # per-channel: values, advantages, returns
    ├── combine.npz           # combined_adv, aw_normed, confidence
    ├── update.npz            # UpdateStats + grad norms
    └── RECORD_GUIDE.md       # exact recorder commands for visual inspection
"""
from __future__ import annotations

import inspect
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from baseline.framework.ppo.experiment import UpdateStats
from baseline.framework.ppo.trainer import PPOBuffer
from baseline.framework.ppo.trajectory import Trajectory
from baseline.framework.rollout.episode import Episode
from baseline.framework.rollout.job import Job


# ---------------------------------------------------------------------------
# Frame ID generation — obs content matching (dump-only, not in production)
# ---------------------------------------------------------------------------

def _make_frame_ids(
    trajectories: List[Trajectory],
    episodes: List[Episode],
) -> np.ndarray:
    """用 obs 内容匹配生成 frame_id 数组。

    trajectory.obs 是 episode.observations[agent_id] 的连续切片，
    所以 obs[0] 可以唯一定位到 (episode_pos, agent_id, t_start)。

    用 hash 索引加速：O(total_frames) 建表 + O(n_trajs) 查表。
    匹配失败的 trajectory 用 ``flat:{i}`` 标记。

    返回 ``(n_total_frames,)`` dtype=object 的字符串数组。
    """
    # 建 hash 索引：obs bytes hash → [(ep_pos, agent_id, t), ...]
    index: Dict[int, List[Tuple[int, str, int]]] = {}
    for ep_pos, episode in enumerate(episodes):
        for agent_id, ep_obs in episode.observations.items():
            obs_arr = np.asarray(ep_obs, dtype=np.float32)
            for t in range(len(obs_arr)):
                h = hash(obs_arr[t].tobytes())
                index.setdefault(h, []).append((ep_pos, agent_id, t))

    n_total = sum(len(t.obs) for t in trajectories)
    ids = np.empty(n_total, dtype=object)
    flat = 0
    for traj in trajectories:
        T = len(traj.obs)
        if T == 0:
            continue
        # 查表匹配
        h = hash(np.asarray(traj.obs[0], dtype=np.float32).tobytes())
        matched = False
        for ep_pos, agent_id, t_start in index.get(h, []):
            ep_obs = np.asarray(
                episodes[ep_pos].observations[agent_id], dtype=np.float32,
            )
            if t_start + T > len(ep_obs):
                continue
            if np.array_equal(traj.obs, ep_obs[t_start:t_start + T]):
                for j in range(T):
                    ids[flat + j] = (
                        f"ep{ep_pos:04d}:{agent_id}:{t_start + j}"
                    )
                matched = True
                break
        if not matched:
            for j in range(T):
                ids[flat + j] = f"flat:{flat + j}"
        flat += T
    return ids


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _flatten_observer_outputs(
    episodes: List[Episode],
) -> Dict[str, np.ndarray]:
    """Flatten per-episode observer_outputs into concatenated arrays.

    Keys: ``observer_outputs.{observer_key}.{field}.{agent_id}``
    (agent_id appended only when the field is per-agent).

    Most observer outputs are already per-agent (keyed by agent_id like
    ``foot_state_a``), so we use the observer key directly.
    """
    out: Dict[str, np.ndarray] = {}
    if not episodes:
        return out
    # Collect all observer keys from the first episode.
    first = episodes[0]
    for obs_key, fields in first.observer_outputs.items():
        if not isinstance(fields, dict):
            # Scalar observer output — stack across episodes.
            out[f"observer_outputs.{obs_key}"] = np.array(
                [ep.observer_outputs.get(obs_key) for ep in episodes],
                dtype=object,
            )
            continue
        for field_name in fields:
            vals = []
            for ep in episodes:
                oo = ep.observer_outputs.get(obs_key, {})
                v = oo.get(field_name) if isinstance(oo, dict) else None
                if v is None:
                    vals.append(None)
                else:
                    vals.append(np.asarray(v))
            # Only stack if all non-None and same shape.
            non_none = [v for v in vals if v is not None]
            if len(non_none) == len(vals) and non_none:
                shapes = set(v.shape for v in non_none)
                if len(shapes) <= 1:
                    out[f"observer_outputs.{obs_key}.{field_name}"] = (
                        np.concatenate(non_none, axis=0)
                    )
                    continue
            # Fallback: store as object array.
            out[f"observer_outputs.{obs_key}.{field_name}"] = np.array(vals, dtype=object)
    return out


def _serialize_episodes(episodes: List[Episode]) -> Dict[str, Any]:
    """Build the episodes.npz payload from actual Episode objects."""
    data: Dict[str, Any] = {}
    data["n_episodes"] = np.array(len(episodes))
    data["episode_indices"] = np.array([ep.episode_index for ep in episodes])
    data["num_frames"] = np.array([ep.num_frames for ep in episodes])
    data["base_seeds"] = np.array([ep.base_seed for ep in episodes])
    # Frame offsets for locating a specific episode's frames in the
    # concatenated arrays.  episode_frame_offsets[i] is the start index
    # of episode i; episode_frame_offsets[i+1] is the end (exclusive).
    frame_offsets = np.zeros(len(episodes) + 1, dtype=np.int64)
    for i, ep in enumerate(episodes):
        frame_offsets[i + 1] = frame_offsets[i] + ep.num_frames
    data["episode_frame_offsets"] = frame_offsets

    # Per-agent arrays (concatenated across episodes).
    agent_ids = set()
    for ep in episodes:
        agent_ids.update(ep.observations.keys())
    for aid in sorted(agent_ids):
        obs_list = [ep.observations[aid] for ep in episodes if aid in ep.observations]
        act_list = [ep.actions[aid] for ep in episodes if aid in ep.actions]
        if obs_list:
            data[f"obs.{aid}"] = np.concatenate(obs_list, axis=0)
        if act_list:
            data[f"actions.{aid}"] = np.concatenate(act_list, axis=0)
        # final_observation (one per episode, not concatenated).
        fin_list = [ep.final_observation[aid] for ep in episodes if aid in ep.final_observation]
        if fin_list:
            data[f"final_obs.{aid}"] = np.stack(fin_list, axis=0)
        # explore_factors
        ef_list = [ep.explore_factors[aid] for ep in episodes if aid in ep.explore_factors]
        if ef_list:
            data[f"explore_factors.{aid}"] = np.concatenate(ef_list, axis=0)

    # Observer outputs (flattened).
    data.update(_flatten_observer_outputs(episodes))

    # Termination records (as JSON string — variable structure).
    term_data = {}
    for i, ep in enumerate(episodes):
        term_data[f"ep{i:04d}"] = {
            aid: [list(r) for r in records]
            for aid, records in ep.agent_termination_proposal_records.items()
        }
    data["_termination_records"] = np.array(json.dumps(term_data, ensure_ascii=False))

    return data


def _serialize_trajectories(
    trajectories: List[Trajectory],
    frame_ids: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Build the trajectories.npz payload from actual Trajectory objects."""
    data: Dict[str, Any] = {}
    data["n_trajectories"] = np.array(len(trajectories))
    data["ep_lengths"] = np.array([len(t.obs) for t in trajectories])

    # Per-channel reward / actor_weight (concatenated).
    if trajectories:
        channel_names = list(trajectories[0].channels.keys())
    else:
        channel_names = []
    data["channel_names"] = np.array(channel_names, dtype=object)

    for ch in channel_names:
        rewards = []
        weights = []
        for t in trajectories:
            if ch in t.channels:
                rewards.append(t.channels[ch].reward)
                aw = t.channels[ch].actor_weight
                if np.isscalar(aw):
                    weights.append(np.full(len(t.obs), aw, dtype=np.float32))
                else:
                    weights.append(np.asarray(aw, dtype=np.float32))
            else:
                rewards.append(np.zeros(len(t.obs), dtype=np.float32))
                weights.append(np.zeros(len(t.obs), dtype=np.float32))
        data[f"reward.{ch}"] = np.concatenate(rewards, axis=0)
        data[f"actor_weight.{ch}"] = np.concatenate(weights, axis=0)
        data[f"is_terminated.{ch}"] = np.array(
            [t.channels[ch].is_terminated if ch in t.channels else False
             for t in trajectories]
        )

    # floor_weight, explore_factor, importance (concatenated).
    fw_list = []
    ef_list = []
    imp_list = []
    for t in trajectories:
        if t.floor_weight is not None:
            fw_list.append(np.asarray(t.floor_weight, dtype=np.float32))
        else:
            fw_list.append(np.ones(len(t.obs), dtype=np.float32))
        if t.explore_factor is not None:
            ef_list.append(np.asarray(t.explore_factor, dtype=np.float32))
        else:
            ef_list.append(np.zeros(len(t.obs), dtype=np.float32))
        imp_list.append(np.array(t.importance, dtype=np.float32))
    data["floor_weight"] = np.concatenate(fw_list, axis=0) if fw_list else np.zeros(0, dtype=np.float32)
    data["explore_factor"] = np.concatenate(ef_list, axis=0) if ef_list else np.zeros(0, dtype=np.float32)
    data["importance"] = np.array(imp_list, dtype=np.float32)

    # Provenance (frame_id for correlation).
    if frame_ids is not None:
        data["frame_id"] = frame_ids
    else:
        n = sum(len(t.obs) for t in trajectories)
        data["frame_id"] = np.array([f"flat:{i}" for i in range(n)], dtype=object)

    return data


def _serialize_buffer(buf: PPOBuffer, frame_ids: np.ndarray) -> Dict[str, Any]:
    """Build the buffer.npz payload from the actual PPOBuffer."""
    data: Dict[str, Any] = {
        "obs": buf.obs,
        "actions": buf.actions,
        "log_probs": buf.log_probs,
        "sample_weights": buf.sample_weights,
        "explore_factor": buf.explore_factor,
        "floor_weight": buf.floor_weight,
        "ep_lengths": np.array(buf.ep_lengths),
        "frame_id": frame_ids,
    }
    return data


def _serialize_stats(stats: UpdateStats) -> Dict[str, Any]:
    """Build the update.npz payload from actual UpdateStats."""
    data: Dict[str, Any] = {
        "approx_kl": np.array(stats.approx_kl),
        "max_kl": np.array(stats.max_kl),
        "early_stop_kl": np.array(stats.early_stop_kl),
        "clip_frac": np.array(stats.clip_frac),
        "ratio_mean": np.array(stats.ratio_mean),
        "ratio_max": np.array(stats.ratio_max),
        "policy_loss": np.array(stats.policy_loss),
        "value_loss": np.array(stats.value_loss),
        "grad_norm_actor": np.array(stats.grad_norm_actor),
        "epochs_done": np.array(stats.epochs_done),
        "actor_epochs_done": np.array(stats.actor_epochs_done),
        "n_batches": np.array(stats.n_batches),
        "n_episodes": np.array(stats.n_episodes),
        "total_steps": np.array(stats.total_steps),
    }
    for key, val in stats.critic_losses.items():
        data[f"critic_loss.{key}"] = np.array(val)
    for key, val in stats.explained_variance.items():
        data[f"ev.{key}"] = np.array(val)
    for key, val in stats.confidence.items():
        data[f"confidence.{key}"] = np.array(val)
    for key, val in stats.adv_mean.items():
        data[f"adv_mean.{key}"] = np.array(val)
    for key, val in stats.adv_std.items():
        data[f"adv_std.{key}"] = np.array(val)
    for key, val in stats.ret_mean.items():
        data[f"ret_mean.{key}"] = np.array(val)
    for key, val in stats.ret_std.items():
        data[f"ret_std.{key}"] = np.array(val)
    for key, val in stats.critic_grad_norms.items():
        data[f"critic_grad_norm.{key}"] = np.array(val)
    return data


# ---------------------------------------------------------------------------
# Stochastic policy export — bake explore_factor into a self-contained policy
# ---------------------------------------------------------------------------

def _serialize_explore_factor(ef_spec) -> str:
    """Serialize an EfSpec (float or callable) to embeddable Python source.

    Returns a function definition ``def _explore_factor(obs, step): ...``
    that can be inlined into the generated ``policy.py``.

    For callables, this captures:
    - The function source via ``inspect.getsource``.
    - Any module-level scalar constants (int/float/str/bool) the function
      references, emitted as literal assignments.
    """
    import builtins

    if isinstance(ef_spec, (int, float)):
        return f"\ndef _explore_factor(obs, step):\n    return {float(ef_spec)!r}\n"

    if not callable(ef_spec):
        raise TypeError(f"explore_factor must be float or callable, got {type(ef_spec).__name__}")

    func = ef_spec
    func_name = func.__name__
    func_source = inspect.getsource(func)
    # Dedent in case the function was nested or indented.
    lines = func_source.splitlines()
    if lines and lines[0].startswith(" "):
        strip = len(lines[0]) - len(lines[0].lstrip())
        lines = [ln[strip:] if len(ln) >= strip else ln for ln in lines]
    func_source = "\n".join(lines)

    # Rename the function to _explore_factor so the wrapper class can
    # call it by a fixed name regardless of the original function name.
    func_source = func_source.replace(
        f"def {func_name}(", "def _explore_factor(", 1,
    )

    # Capture scalar globals referenced by the function.
    code = func.__code__
    globs = getattr(func, "__globals__", {})
    dep_lines: List[str] = []
    seen_names: set = set()
    for name in code.co_names:
        if name in seen_names:
            continue
        seen_names.add(name)
        if name in globs and not hasattr(builtins, name):
            val = globs[name]
            if isinstance(val, (int, float, str, bool)) and not isinstance(val, type):
                dep_lines.append(f"{name} = {val!r}")
    dep_block = "\n".join(dep_lines) if dep_lines else ""

    parts = []
    if dep_block:
        parts.append(dep_block)
    parts.append(func_source)
    return "\n".join(parts) + "\n"


# --- Wrapper class templates (plain strings, not f-strings) ---

_WRAPPER_SAME_EF = '''

# ---------------------------------------------------------------------------
# Stochastic wrapper — act() calls sample() with baked-in explore_factor
# ---------------------------------------------------------------------------

class ExportedExploratoryPolicy(Policy):
    """Stochastic policy with baked-in explore_factor.

    act() delegates to inner.sample(obs, explore_factor=_explore_factor(obs, step)),
    reproducing training rollout behavior for both agent A and agent B.
    """
    def __init__(self, model_path=None, **_):
        self._inner = ExportedTruncNormPolicy(model_path=model_path)
        self._step = 0

    def act(self, observation, *, want_extra=False):
        ef = _explore_factor(observation, self._step)
        self._step += 1
        action, extra = self._inner.sample(
            observation, explore_factor=ef, want_extra=want_extra,
        )
        if extra is not None:
            extra["explore_factor"] = float(ef)
        else:
            extra = {"explore_factor": float(ef)}
        return action, extra

    def reset(self, seed=None):
        self._step = 0
        reset_fn = getattr(self._inner, "reset", None)
        if callable(reset_fn):
            reset_fn(seed)
'''

_WRAPPER_DIFF_EF = '''

# ---------------------------------------------------------------------------
# Stochastic wrapper — act() calls sample() with baked-in explore_factor
# ---------------------------------------------------------------------------

class ExportedExploratoryPolicy(Policy):
    """Stochastic policy with baked-in explore_factor.

    Uses _explore_factor_a for agent A and _explore_factor_b for agent B.
    The agent is selected via the 'agent' config key in the blueprint.
    """
    def __init__(self, model_path=None, agent="a", **_):
        self._inner = ExportedTruncNormPolicy(model_path=model_path)
        self._step = 0
        self._ef_fn = _explore_factor_a if agent == "a" else _explore_factor_b

    def act(self, observation, *, want_extra=False):
        ef = self._ef_fn(observation, self._step)
        self._step += 1
        action, extra = self._inner.sample(
            observation, explore_factor=ef, want_extra=want_extra,
        )
        if extra is not None:
            extra["explore_factor"] = float(ef)
        else:
            extra = {"explore_factor": float(ef)}
        return action, extra

    def reset(self, seed=None):
        self._step = 0
        reset_fn = getattr(self._inner, "reset", None)
        if callable(reset_fn):
            reset_fn(seed)
'''


def _export_stochastic_policy(
    dump_dir: Path,
    run_dir: Path,
    update: int,
    job: Job,
) -> Path:
    """Export a self-contained stochastic policy with baked-in explore_factor.

    Creates ``<dump_dir>/stochastic_policy/`` containing:
    - ``model.pt`` — copied from ``policy_exports/u{N:05d}/model.pt``
    - ``policy.py`` — self-contained: inner ExportedTruncNormPolicy +
      embedded explore_factor callable + ExportedExploratoryPolicy wrapper
    - ``policy_blueprint.yaml`` — points to ExportedExploratoryPolicy

    The wrapper's ``act()`` calls ``inner.sample(obs, explore_factor=ef)``,
    reproducing training rollout behavior.  When ``round_runner`` loads
    this blueprint and calls ``act()``, it gets stochastic sampling with
    the same per-frame explore_factor as training.

    Returns the path to ``policy_blueprint.yaml``.
    """
    export_src = run_dir / "policy_exports" / f"u{update:05d}"
    model_pt = export_src / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(
            f"model.pt not found at {model_pt} — policy export missing"
        )

    policy_dir = dump_dir / "stochastic_policy"
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Copy model.pt
    shutil.copy2(str(model_pt), str(policy_dir / "model.pt"))

    # Read the self-contained export template (inner policy).
    template_path = (
        Path(__file__).resolve().parent.parent
        / "policies" / "_export_template.py"
    )
    inner_code = template_path.read_text(encoding="utf-8")

    # Serialize explore_factor for both agents.
    ef_a_src = _serialize_explore_factor(job.explore_factor_a)
    ef_b_src = _serialize_explore_factor(job.explore_factor_b)

    # Generate wrapper class.
    # If ef_a == ef_b (common case), use one function for both.
    # Otherwise, generate two functions.
    if job.explore_factor_a == job.explore_factor_b:
        ef_block = ef_a_src
        wrapper_class = _WRAPPER_SAME_EF
    else:
        ef_block = (
            ef_a_src.replace("_explore_factor", "_explore_factor_a") + "\n"
            + ef_b_src.replace("_explore_factor", "_explore_factor_b")
        )
        wrapper_class = _WRAPPER_DIFF_EF

    # Combine: inner policy code + explore_factor + wrapper class.
    full_code = inner_code + "\n\n" + ef_block + wrapper_class
    (policy_dir / "policy.py").write_text(full_code, encoding="utf-8")

    # Write policy_blueprint.yaml
    blueprint_path = policy_dir / "policy_blueprint.yaml"
    blueprint_content = {
        "cls": f"file:{policy_dir / 'policy.py'}:ExportedExploratoryPolicy",
        "config": {"model_path": str(policy_dir / "model.pt")},
    }
    blueprint_path.write_text(
        json.dumps(blueprint_content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return blueprint_path


# ---------------------------------------------------------------------------
# RECORD_GUIDE.md generation
# ---------------------------------------------------------------------------

def _render_record_guide(
    dump_dir: Path,
    run_dir: Path,
    update: int,
    job: Job,
    stochastic_policy_path: Optional[Path] = None,
) -> str:
    """Render RECORD_GUIDE.md with exact recorder commands.

    Uses the actual training cross-section: env_blueprint saved from
    jobs[0].env_bp, stochastic policy (with baked-in explore_factor)
    from the dump directory, seed and episode_options from jobs[0].

    When ``stochastic_policy_path`` is provided, the recording command
    uses the stochastic wrapped policy — its ``act()`` calls ``sample()``
    with the same per-frame explore_factor as training, so the recorded
    episode should reproduce the training episode (same seed + same env +
    same policy + same explore_factor = same trajectory).
    """
    env_bp_path = dump_dir / "env_blueprint.yaml"
    options_path = dump_dir / "episode_options.json"
    if stochastic_policy_path is not None:
        policy_export = stochastic_policy_path
        policy_desc = "stochastic wrapped (explore_factor baked in)"
    else:
        policy_export = run_dir / "policy_exports" / f"u{update:05d}" / "policy_blueprint.yaml"
        policy_desc = "raw θ_old (deterministic — no explore_factor)"
    record_output = dump_dir / "record"

    lines = [
        f"# 录制指南 — Update {update}",
        "",
        "## 1. 录制 Episode（在独立终端运行）",
        "",
        "以下命令使用训练时的实际配置（env blueprint、policy、seed、",
        "episode_options 均来自训练截面），录制一局可回放的 episode：",
        "",
        f"> policy: {policy_desc}",
        "",
        "```bash",
        "cd /data1/mono/things/combatbench",
        f"PYTHONPATH=. python3 -m envs.framework.round_runner \\",
        f"  --env-blueprint {env_bp_path} \\",
        f"  --policy-a-blueprint {policy_export} \\",
        f"  --policy-b-blueprint {policy_export} \\",
        f'  --recorder "envs.framework.recorder:BaseFrameRecorder?output_dir={record_output}" \\',
        f"  --seed {job.seed} \\",
    ]
    if job.episode_options:
        lines.append(f"  --options-json {options_path}")
    lines.extend([
        "```",
        "",
        "## 2. 查看录制",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m envs.framework.recorder_viewer {record_output}",
        "```",
        "",
        "浏览器打开 http://localhost:8765/viewer.html ，逐帧查看图片 + observer 输出。",
        "肉眼抽检：foot_state 对不对？contacts 切对没有？height_phi 曲线合不合理？",
        "",
        "## 3. 自动录制 + 校验（推荐）",
        "",
        "用 `debug.py render` 子命令自动录制图片并校验数据一致性：",
        "",
        "```bash",
        f"PYTHONPATH=. python3 baseline/framework/ppo/debug.py render {dump_dir} \\",
        f"  --episode 0",
        "```",
        "",
        "该命令会：",
        "1. 读取 dump 中的 stochastic_policy + env_blueprint + episode seed",
        "2. 用 round_runner + BaseFrameRecorder 生成逐帧 PNG + JSON",
        "3. 自动校验录制的 obs/actions 与 episodes.npz 逐帧一致",
        "4. 写入 association.json 记录图片与 dump 数据的关联",
        "",
        "校验失败时会明确警告，请仔细阅读警告信息。",
        "",
        "## 4. 手动录制（可选）",
        "",
        "上面的手动 round_runner 命令仍然可用，适合需要自定义参数的场景。",
        "手动录制不会自动校验，需要自行对比数据。",
        "",
        "## 5. 录制多个 Episode",
        "",
        "修改 `--episode` 参数（render 子命令）或 `--seed` 参数（手动命令）",
        "录制不同初始条件的 episode。每个 episode 的 seed 记录在 episodes.npz 的 base_seeds 数组中。",
        "",
        "## 数据说明",
        "",
        f"- env_blueprint: 训练 update {update} 实际使用的 env 配置",
        f"- policy: {policy_desc}",
        f"- seed: {job.seed} (训练时该 episode 的实际 seed)",
        f"- episode seeds: 见 episodes.npz 中的 base_seeds 数组（所有 episode 的 seed）",
    ])
    if job.episode_options:
        lines.append(f"- episode_options: {dict(job.episode_options)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main capture function
# ---------------------------------------------------------------------------

def capture_dump(
    run_dir: Path,
    update: int,
    request,  # DumpRequest
    *,
    episodes: List[Episode],
    trajectories: List[Trajectory],
    buf: PPOBuffer,
    stats: UpdateStats,
    jobs: List[Job],
    dump_collector: Dict[str, Dict[str, Any]],
    experiment_name: str,
) -> Path:
    """Write complete update data to ``<run_dir>/dumps/u{update:05d}/``.

    All data comes from the actual training objects — no reconstruction.

    Args:
        run_dir: Training run directory.
        update: Current update number.
        request: Parsed dump request (with hypothesis).
        episodes: Actual rollout episodes for this update.
        trajectories: Actual build_trajectories output.
        buf: Actual PPOBuffer.
        stats: Actual UpdateStats from ppo_update.
        jobs: Actual rollout jobs (for env_bp, seed, episode_options).
        dump_collector: Stage data collected by dump_callback inside ppo_update.
        experiment_name: Experiment name (for manifest).

    Returns:
        Path to the dump directory.
    """
    dump_dir = run_dir / "dumps" / f"u{update:05d}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # --- manifest.json ---
    manifest: Dict[str, Any] = {
        "update": update,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "hypothesis": request.hypothesis,
        "include_full_grad": request.include_full_grad,
        "n_episodes": len(episodes),
        "n_trajectories": len(trajectories),
        "total_steps": int(sum(len(t.obs) for t in trajectories)),
        "has_timeline": "timeline" in dump_collector,
        "has_epoch_frames": "epoch_frames" in dump_collector,
    }
    # Git commit from code_snapshot.json if present.
    snapshot_info = run_dir / "code_snapshot.json"
    if snapshot_info.exists():
        try:
            with open(snapshot_info) as f:
                si = json.load(f)
            manifest["git_commit"] = si.get("commit")
            manifest["git_branch"] = si.get("branch")
        except (json.JSONDecodeError, OSError):
            pass

    (dump_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- env_blueprint.yaml (from jobs[0].env_bp — actual training env) ---
    if jobs:
        job = jobs[0]
        env_bp_path = dump_dir / "env_blueprint.yaml"
        job.env_bp.save(str(env_bp_path))

        # --- episode_options.json (from jobs[0].episode_options) ---
        if job.episode_options:
            (dump_dir / "episode_options.json").write_text(
                json.dumps(dict(job.episode_options), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    else:
        job = None

    # --- frame_ids for correlation (obs content matching, dump-only) ---
    frame_ids = _make_frame_ids(trajectories, episodes)

    # --- episodes.npz ---
    ep_data = _serialize_episodes(episodes)
    np.savez_compressed(dump_dir / "episodes.npz", **ep_data)

    # --- trajectories.npz ---
    traj_data = _serialize_trajectories(trajectories, frame_ids)
    np.savez_compressed(dump_dir / "trajectories.npz", **traj_data)

    # --- buffer.npz ---
    buf_data = _serialize_buffer(buf, frame_ids)
    np.savez_compressed(dump_dir / "buffer.npz", **buf_data)

    # --- gae.npz (from dump_collector) ---
    if "gae" in dump_collector:
        np.savez_compressed(dump_dir / "gae.npz", **dump_collector["gae"])

    # --- combine.npz (from dump_collector) ---
    if "combine" in dump_collector:
        np.savez_compressed(dump_dir / "combine.npz", **dump_collector["combine"])

    # --- update.npz (from stats + dump_collector) ---
    update_data = _serialize_stats(stats)
    if "update" in dump_collector:
        update_data.update(dump_collector["update"])
    np.savez_compressed(dump_dir / "update.npz", **update_data)

    # --- timeline.npz (from dump_collector, Scene 4) ---
    # Per-minibatch training dynamics: KL, clip_frac, ratio, loss, grad
    # across all epochs × minibatches.  Very small (~13 KB).
    if "timeline" in dump_collector:
        np.savez_compressed(
            dump_dir / "timeline.npz",
            **dump_collector["timeline"],
        )

    # --- epoch_frames.npz (from dump_collector, Scene 3) ---
    # Per-epoch full-batch forward pass snapshots: per-frame ratio,
    # clip_mask, new_log_prob, new_value at each epoch's end state.
    # Larger (~20 MB) but only captured when dump_callback is active.
    if "epoch_frames" in dump_collector:
        ef_list = dump_collector["epoch_frames"]["epochs"]
        n_epochs = len(ef_list)
        actor_stopped_epoch = next(
            (e["epoch"] for e in ef_list if e["actor_stopped"]), -1,
        )
        ef_data: Dict[str, Any] = {
            "n_epochs": np.array(n_epochs),
            "actor_stopped_epoch": np.array(actor_stopped_epoch, dtype=np.int64),
        }
        for e in ef_list:
            ep = e["epoch"]
            ef_data[f"ratio.{ep}"] = e["ratio"]
            ef_data[f"clip_mask.{ep}"] = e["clip_mask"]
            ef_data[f"new_log_prob.{ep}"] = e["new_log_prob"]
            for ch, v in e["new_value"].items():
                ef_data[f"new_value.{ep}.{ch}"] = v
        np.savez_compressed(dump_dir / "epoch_frames.npz", **ef_data)

    # --- stochastic policy export (bake explore_factor into policy.py) ---
    stochastic_policy_path: Optional[Path] = None
    if job is not None:
        try:
            stochastic_policy_path = _export_stochastic_policy(
                dump_dir, run_dir, update, job,
            )
            print(
                f"[dump] stochastic policy: {stochastic_policy_path}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[dump] stochastic policy export failed: {e}; "
                f"falling back to raw policy in RECORD_GUIDE.md",
                flush=True,
            )

    # --- RECORD_GUIDE.md ---
    if job is not None:
        guide = _render_record_guide(
            dump_dir, run_dir, update, job,
            stochastic_policy_path=stochastic_policy_path,
        )
        (dump_dir / "RECORD_GUIDE.md").write_text(guide, encoding="utf-8")

    print(
        f"[dump] captured update {update} → {dump_dir} "
        f"(episodes={len(episodes)}, trajs={len(trajectories)}, "
        f"steps={manifest['total_steps']}, "
        f"full_grad={request.include_full_grad})",
        flush=True,
    )
    if job is not None:
        print(
            f"[dump] 录制指南: {dump_dir / 'RECORD_GUIDE.md'}",
            flush=True,
        )

    return dump_dir


__all__ = ["capture_dump"]
