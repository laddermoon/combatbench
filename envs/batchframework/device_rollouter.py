"""DeviceRollouter — 设备端波次同步 episode 采集器（M5 W1）。

与 ``ParallelRollouter`` 同契约：``collect(jobs) -> List[Episode]``，
jobs 同序返回。下游（build_trajectories / PPOBuffer / GAE / dump）
对数据来源无感知。

执行模型（ROADMAP §6"同步定长模式"）：

- 所有 job 必须共享同一 env blueprint 与同一对 policy blueprint
  （M5 范围：standup 类 self-play 实验；混合 bp 显式拒绝）；
- N 个 job 按 ``batch_size`` 分波，每波 B 个 env lockstep 推进；
- 波内 env 若提前终止：该 env 行由 runtime 自动部分 reset 进入下一
  episode——**帧截取到首个 env 终止步**，后续行数据不串台；
- 热路径零 host sync；波末一次批量 ``.cpu()`` 组装 Episode。

策略加载：``job.policy_*_bp`` 的 ``file:`` 导出蓝图 → 重建
``TruncatedNormalPolicy``（weights 来自 policy_exports/uNNNNN，
保持"rollout 消费导出版本"的版本流语义；self-play 双 agent 共享）。

支持边界（不满足即 raise，不静默降级）：

- sampling spec 仅支持标量 ``explore_factor``（per-env 可异值）；
  callable / reference / delta_mix!=0 一律拒绝；
- blueprint 插件/observer 必须在 capability_registry 中为 NATIVE；
- per-agent 早停按 ``post_termination_action="policy"`` 语义处理
  （终止 agent 仍继续采样动作——与 EpisodeRunner 默认一致）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from baseline.framework.ppo.sampling_context import SamplingContext
from baseline.framework.rollout.episode import Episode, blueprint_hash
from baseline.framework.rollout.job import Job
from envs.framework.blueprint import EnvBlueprint

from .capability_registry import resolve_observer, resolve_plugin
from .device_plugin import BaseDevicePlugin, TERM_NAMES
from .device_runtime import BatchRuntime, DeviceTimeoutPlugin

AGENT_IDS = ("robot_a", "robot_b")

#: blueprint simulator cls → 设备后端替换映射的白名单（CPU 参考类）。
_SUPPORTED_SIM_CLS = {
    "envs.humanoid21.simulator:Humanoid21Simulator",
}


# ---------------------------------------------------------------------------
# Wave 内录数据插件
# ---------------------------------------------------------------------------
class _WaveRecorder(BaseDevicePlugin):
    """每步把 observer 输出拷入设备缓冲；终止时捕获 per-agent 记录。

    缓冲是普通属性而非 pstate——plugin pool 行在部分 reset 时清零，
    而录数据必须跨 reset 边界连续。priority 最低——读到的是本步
    dispatcher 已刷新 + timeout 已标记后的完整输出。
    """

    def __init__(self, observer_names: Sequence[str], max_steps: int,
                 batch_size: int, device: torch.device):
        self._names = list(observer_names)
        self._max = int(max_steps)
        self._B = int(batch_size)
        self._dev = device
        self._rt: Optional[BatchRuntime] = None
        self.obs_bufs: Dict[str, Dict[str, torch.Tensor]] = {}
        self.term_records: List[Dict[str, List]] = []
        self.final_obs: Optional[Dict[str, torch.Tensor]] = None
        self.env_term_step: Optional[torch.Tensor] = None
        self._seen_term: Optional[torch.Tensor] = None
        self._t = 0

    @property
    def name(self):
        return "wave_recorder"

    @property
    def priority(self):
        return -10_000  # 最后执行——读到本步完整输出

    def set_runtime(self, rt: BatchRuntime) -> None:
        self._rt = rt

    # --- 生命周期 ---
    def begin_wave(self) -> None:
        self.obs_bufs = {}
        self.term_records = [dict(robot_a=[], robot_b=[])
                             for _ in range(self._B)]
        self._seen_term = torch.zeros(self._B, 2, dtype=torch.bool,
                                      device=self._dev)
        self.env_term_step = torch.full((self._B,), -1, dtype=torch.int32,
                                        device=self._dev)
        # final_obs 兜底初始化为当前 io.obs（未终止 env 用末步 obs）
        self.final_obs = None
        self._t = 0

    def on_post_action_step(self, ctx) -> None:
        t = self._t
        rt = self._rt
        # observer 输出 → (T,B) 缓冲（按输出键惰性分配）
        for name in self._names:
            out = rt.get_observer_output(name)
            if out is None:
                continue
            bufs = self.obs_bufs.setdefault(name, {})
            for k, v in out.items():
                if not torch.is_tensor(v) or v.shape[:1] != (self._B,):
                    continue  # 只收 per-env (B,) 叶——与 CPU observer dict 对齐
                buf = bufs.get(k)
                if buf is None:
                    buf = torch.zeros(self._max, self._B, dtype=v.dtype,
                                      device=self._dev)
                    bufs[k] = buf
                buf[t].copy_(v)

        ep = ctx.episode
        # per-agent 终止首次出现 → (reason, episode_step)；env 已终止
        # 的行（属于下一 episode）不再记录
        new = ep.agent_terminated & ~self._seen_term \
            & (self.env_term_step < 0).unsqueeze(-1)
        self._seen_term |= ep.agent_terminated
        if bool(new.any()):
            ids = torch.nonzero(new, as_tuple=False).cpu()
            steps = ep.episode_steps.cpu()
            reasons = ep.agent_term_reason.cpu()
            for e, a in ids.tolist():
                self.term_records[e][AGENT_IDS[a]].append(
                    (TERM_NAMES.get(int(reasons[e, a]), "custom"),
                     int(steps[e])))

        # env 级终止：捕获 final_obs 行。此刻 io.obs 已是 obs_{t+1}
        # （obs 构建在 step 内早于终止消费），恰是 bootstrap 后继态。
        env_term = ep.terminated_flag | ep.agent_terminated.all(dim=-1)
        first = env_term & (self.env_term_step < 0)
        if bool(first.any()):
            ids = torch.nonzero(first, as_tuple=False).squeeze(-1)
            self.env_term_step[ids] = ep.episode_steps[ids].to(torch.int32)
            if self.final_obs is None:
                self.final_obs = {"robot_a": ctx.io.obs_a.clone(),
                                  "robot_b": ctx.io.obs_b.clone()}
            self.final_obs["robot_a"][ids] = ctx.io.obs_a[ids]
            self.final_obs["robot_b"][ids] = ctx.io.obs_b[ids]
        self._t += 1

    def finalize_wave(self, ctx_io) -> None:
        """波末兜底：未终止 env 的 final_obs = 末步 io.obs。"""
        if self.final_obs is None:
            self.final_obs = {"robot_a": ctx_io.obs_a.clone(),
                              "robot_b": ctx_io.obs_b.clone()}
        else:
            live = self.env_term_step < 0
            if bool(live.any()):
                self.final_obs["robot_a"][live] = ctx_io.obs_a[live]
                self.final_obs["robot_b"][live] = ctx_io.obs_b[live]

    def on_envs_reset(self, ctx) -> None:
        # 行复位后允许该 env 的下一 episode 重新记终止（env_term_step
        # 保留——该行本波只产出一个 episode，复位后的终止不入记录）
        ids = ctx.reset_env_ids
        if ids is not None and len(ids):
            self._seen_term[ids] = False


# ---------------------------------------------------------------------------
# Job 校验 / 策略加载
# ---------------------------------------------------------------------------
def _check_spec(spec_dict: Dict[str, Any], tag: str):
    """返回 (explore_factor, delta_factor, delta_mix) 标量三元组；
    不支持的 spec 直接拒绝。"""
    ef = spec_dict.get("explore_factor", 0.0)
    if callable(ef):
        raise ValueError(
            f"job[{tag}]: callable explore_factor not supported on device "
            f"collector (per-frame host evaluation required)")
    if spec_dict.get("reference") is not None:
        raise ValueError(
            f"job[{tag}]: reference ensemble not supported on device "
            f"collector (M5 scope)")
    if float(spec_dict.get("delta_mix", 0.0)) != 0.0:
        raise ValueError(
            f"job[{tag}]: delta_mix != 0 not supported on device collector")
    return (float(ef), float(spec_dict.get("delta_factor", 0.0)),
            float(spec_dict.get("delta_mix", 0.0)))


def _load_policy(policy_bp_dict: Dict[str, Any], device: str):
    """file: 导出蓝图 → TruncatedNormalPolicy（训练侧类，有批量 API）。"""
    cls = policy_bp_dict.get("cls", "")
    if not cls.startswith("file:"):
        raise ValueError(
            f"device collector requires file: policy export, got {cls!r}")
    path = cls[5:].rsplit(":", 1)[0]
    model_pt = Path(path).with_name("model.pt")
    payload = torch.load(model_pt, map_location="cpu", weights_only=False)
    arch = payload["arch"]
    from baseline.framework.ppo.policies.truncated_normal_mlp import (
        TruncatedNormalPolicy)
    pol = TruncatedNormalPolicy(obs_dim=int(arch["obs_dim"]),
                                action_dim=int(arch["action_dim"]),
                                hidden_dim=int(arch["hidden_dim"]),
                                device=device)
    pol.load_state_dict(payload["state_dict"])
    pol.eval()
    return pol


# ---------------------------------------------------------------------------
# DeviceRollouter
# ---------------------------------------------------------------------------
class DeviceRollouter:
    """波次同步设备采集器。``collect(jobs)`` 与 ParallelRollouter 同契约。"""

    def __init__(self, batch_size: int = 64, device: str = "cuda"):
        self.batch_size = int(batch_size)
        self.device = device
        self._sim = None
        self._rt: Optional[BatchRuntime] = None
        self._env_key: Optional[str] = None
        self._recorder: Optional[_WaveRecorder] = None
        self._policy_cache: Dict[str, Any] = {}
        # 计时分解（W4 消费）——按 collect 调用累计
        self.timing = {"reset": 0.0, "policy": 0.0, "step": 0.0,
                       "assemble": 0.0, "n_waves": 0}

    # ------------------------------------------------------------------
    def _build_runtime(self, env_bp: EnvBlueprint) -> None:
        """按 env_bp 装配 warp runtime（blueprint 变化才重建）。"""
        if env_bp.simulator.cls not in _SUPPORTED_SIM_CLS:
            raise ValueError(
                f"device collector supports {_SUPPORTED_SIM_CLS}, got "
                f"{env_bp.simulator.cls!r}")
        from .warp_simulator import WarpHumanoid21Simulator
        sim = WarpHumanoid21Simulator(batch_size=self.batch_size)
        sim.reset()
        rt = BatchRuntime(
            sim, obs_builder=sim.device_obs_builder(),
            phy_substeps=env_bp.phy_steps_per_action)
        for spec in env_bp.plugins:
            rt.attach(resolve_plugin(spec.cls, spec.config, sim))
        if env_bp.max_steps:
            rt.attach(DeviceTimeoutPlugin(int(env_bp.max_steps)))
        for name, spec in env_bp.observer_plugins.items():
            rt.set_observer(name, resolve_observer(spec.cls, spec.config, sim))
        self._recorder = _WaveRecorder(
            list(env_bp.observer_plugins), int(env_bp.max_steps),
            self.batch_size, torch.device(self.device))
        self._recorder.set_runtime(rt)
        rt.attach(self._recorder)
        self._sim, self._rt = sim, rt
        self._env_key = json.dumps(env_bp.to_dict(), sort_keys=True)

    def _policy(self, bp_dict: Dict[str, Any]):
        key = json.dumps(bp_dict, sort_keys=True)
        if key not in self._policy_cache:
            self._policy_cache[key] = _load_policy(bp_dict, self.device)
        return self._policy_cache[key]

    # ------------------------------------------------------------------
    def collect(self, jobs: Sequence[Job]) -> List[Episode]:
        if not jobs:
            raise ValueError("jobs must not be empty")

        env_bp = jobs[0].env_bp
        env_key = json.dumps(env_bp.to_dict(), sort_keys=True)
        pa_key = json.dumps(jobs[0].policy_a_bp.to_dict(), sort_keys=True)
        pb_key = json.dumps(jobs[0].policy_b_bp.to_dict(), sort_keys=True)
        for i, j in enumerate(jobs):
            if json.dumps(j.env_bp.to_dict(), sort_keys=True) != env_key:
                raise ValueError(
                    "device collector requires uniform env blueprint "
                    "per collect() call (M5 scope)")
            if (json.dumps(j.policy_a_bp.to_dict(), sort_keys=True) != pa_key
                    or json.dumps(j.policy_b_bp.to_dict(), sort_keys=True)
                    != pb_key):
                raise ValueError(
                    "device collector requires uniform policy blueprints "
                    "per collect() call (M5 scope)")

        # per-job per-agent 标量 ef（跨 job 可异值——记录进 extras）
        ef_list = [(
            _check_spec(j.sampling_a.to_dict(), f"{i}/a"),
            _check_spec(j.sampling_b.to_dict(), f"{i}/b"),
            bool(j.stochastic),
        ) for i, j in enumerate(jobs)]

        if self._env_key != env_key:
            self._teardown()
            self._build_runtime(env_bp)
        rt = self._rt
        B, T = self.batch_size, int(env_bp.max_steps or 0)
        if T <= 0:
            raise ValueError(
                "device collector requires env_bp.max_steps "
                "(wave-synchronous fixed horizon)")

        episodes: List[Optional[Episode]] = [None] * len(jobs)
        for w0 in range(0, len(jobs), B):
            wave = jobs[w0:w0 + B]
            self._run_wave(wave, w0, ef_list[w0:w0 + B], env_bp, rt, T,
                           episodes)
        return [e for e in episodes]

    # ------------------------------------------------------------------
    def _run_wave(self, wave, w0, efs, env_bp, rt, T, episodes):
        sim, rec, B = self._sim, self._recorder, self.batch_size
        n = len(wave)
        st = rt.state
        dev = torch.device(self.device)

        # 策略（全波一致——collect 已校验）；self-play 共享 module
        pa = self._policy(wave[0].policy_a_bp.to_dict())
        pb = (pa if wave[0].policy_a_bp.to_dict()
              == wave[0].policy_b_bp.to_dict()
              else self._policy(wave[0].policy_b_bp.to_dict()))
        stochastic = efs[0][2]
        # per-env ef → (B,) ctx 张量（pad 行填末值）
        ef_pad = list(efs) + [efs[-1]] * max(0, B - n)
        ef_a_t = torch.tensor([e[0][0] for e in ef_pad],
                              dtype=torch.float32, device=dev)
        ef_b_t = torch.tensor([e[1][0] for e in ef_pad],
                              dtype=torch.float32, device=dev)
        ctx_a = SamplingContext(explore_factor=ef_a_t)
        ctx_b = SamplingContext(explore_factor=ef_b_t)
        # self-play 合并前向时 obs 是 (2B,·)——ctx 也要合并
        ctx_ab = SamplingContext(
            explore_factor=torch.cat([ef_a_t, ef_b_t]))

        t0 = time.perf_counter()
        seeds = torch.tensor(
            [j.seed for j in wave] + [wave[-1].seed] * (B - n),
            dtype=torch.int64, device=dev)
        # per-env options：白名单键做 (B,) 广播，未知键拒绝；
        # 同一键必须全波都在（混合缺省语义不明确，拒绝而非猜测）
        _OPTION_KEYS = ("initial_distance", "initial_pose_a",
                        "initial_pose_b")
        opt_keys = {k for j in wave for k in (j.episode_options or {})}
        unknown = opt_keys - set(_OPTION_KEYS)
        if unknown:
            raise ValueError(
                f"episode_options keys {sorted(unknown)} not supported "
                f"by device collector")
        options = {}
        for k in opt_keys:
            vals = [(j.episode_options or {}).get(k) for j in wave]
            if any(v is None for v in vals):
                raise ValueError(
                    f"episode_options[{k!r}] must be set for all jobs "
                    f"or none")
            options[k] = np.asarray(vals + [vals[0]] * (B - n))
        rt.reset(seeds=seeds, options=options or None)
        rt.obs_builder.build(st)          # obs_0
        self.timing["reset"] += time.perf_counter() - t0

        obs_buf = {rid: torch.zeros(T, B, 96, device=dev)
                   for rid in AGENT_IDS}
        act_buf = {rid: torch.zeros(T, B, 21, device=dev)
                   for rid in AGENT_IDS}
        lp_buf = {rid: torch.zeros(T, B, device=dev) for rid in AGENT_IDS}
        rec.begin_wave()

        for t in range(T):
            obs_buf["robot_a"][t].copy_(st.io.obs_a)
            obs_buf["robot_b"][t].copy_(st.io.obs_b)
            tp = time.perf_counter()
            with torch.no_grad():
                if stochastic:
                    if pa is pb:
                        both = torch.cat([st.io.obs_a, st.io.obs_b], dim=0)
                        a_all, lp_all = pa.sample_action(both, ctx=ctx_ab)
                        a_a, a_b = a_all[:B], a_all[B:]
                        lp_a, lp_b = lp_all[:B], lp_all[B:]
                    else:
                        a_a, lp_a = pa.sample_action(st.io.obs_a, ctx=ctx_a)
                        a_b, lp_b = pb.sample_action(st.io.obs_b, ctx=ctx_b)
                else:
                    a_a = pa.deterministic_action(st.io.obs_a)
                    a_b = pb.deterministic_action(st.io.obs_b)
                    lp_a = torch.zeros(B, device=dev)
                    lp_b = torch.zeros(B, device=dev)
            self.timing["policy"] += time.perf_counter() - tp
            act_buf["robot_a"][t].copy_(a_a)
            act_buf["robot_b"][t].copy_(a_b)
            lp_buf["robot_a"][t].copy_(lp_a)
            lp_buf["robot_b"][t].copy_(lp_b)
            tp = time.perf_counter()
            rt.step((a_a, a_b))
            self.timing["step"] += time.perf_counter() - tp
        self.timing["n_waves"] += 1

        # --- 波末组装：一次 .cpu() ---
        t0 = time.perf_counter()
        rec.finalize_wave(st.io)
        np_bufs = dict(
            obs={rid: obs_buf[rid].cpu().numpy() for rid in AGENT_IDS},
            act={rid: act_buf[rid].cpu().numpy() for rid in AGENT_IDS},
            lp={rid: lp_buf[rid].cpu().numpy() for rid in AGENT_IDS},
            final_obs={rid: rec.final_obs[rid].cpu().numpy()
                       for rid in AGENT_IDS},
            obs_out={name: {k: v.cpu().numpy() for k, v in bufs.items()}
                     for name, bufs in rec.obs_bufs.items()},
            env_term=rec.env_term_step.cpu().numpy(),
            ef_a=ef_a_t.cpu().numpy(), ef_b=ef_b_t.cpu().numpy(),
        )
        fallen_pool = next(
            (v for k, v in st.plugin.items() if "fallen" in k.lower()),
            None)
        metrics_np = {}
        if fallen_pool:
            for k in ("init_steps", "init_height", "init_hit"):
                if k in fallen_pool:
                    metrics_np[k] = fallen_pool[k].cpu().numpy()

        env_hash = blueprint_hash(env_bp)
        for i, job in enumerate(wave):
            episodes[w0 + i] = self._assemble_episode(
                job, w0 + i, i, efs[i], env_hash, np_bufs,
                rec.term_records[i], metrics_np, T)
        self.timing["assemble"] += time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _assemble_episode(self, job, ep_index, row, ef_spec, env_hash,
                          np_bufs, term_records, metrics_np, T) -> Episode:
        """per-env 帧 dict → Episode.from_buffer_frames（复用 stack 语义）。"""
        term_step = int(np_bufs["env_term"][row])
        t_use = term_step if 0 < term_step <= T else T
        frames = []
        for t in range(t_use):
            frames.append({
                "observation": {rid: np_bufs["obs"][rid][t, row]
                                for rid in AGENT_IDS},
                "action": {rid: np_bufs["act"][rid][t, row]
                           for rid in AGENT_IDS},
                "observer_outputs": {
                    # 标量叶子——与 CPU 一致地走 _try_stack 的 list 分支
                    name: {k: v[t, row].item() for k, v in fields.items()}
                    for name, fields in np_bufs["obs_out"].items()},
                "action_extras": {
                    "robot_a": {
                        "log_prob": float(np_bufs["lp"]["robot_a"][t, row]),
                        "explore_factor": float(np_bufs["ef_a"][row]),
                        # SamplingPolicy.act 的 ctx.record_fields() 等价物
                        "sctx__delta_factor": np.float32(ef_spec[0][1]),
                        "sctx__delta_mix": np.float32(ef_spec[0][2]),
                    },
                    "robot_b": {
                        "log_prob": float(np_bufs["lp"]["robot_b"][t, row]),
                        "explore_factor": float(np_bufs["ef_b"][row]),
                        "sctx__delta_factor": np.float32(ef_spec[1][1]),
                        "sctx__delta_mix": np.float32(ef_spec[1][2]),
                    },
                },
            })
        metrics = {"backend": "warp-fp32"}
        for rid_i, rid in enumerate(AGENT_IDS):
            if "init_steps" in metrics_np:
                metrics[f"{rid}_fallen_init_steps"] = int(
                    metrics_np["init_steps"][row])
            if "init_height" in metrics_np:
                metrics[f"{rid}_fallen_init_height"] = float(
                    metrics_np["init_height"][row, rid_i])
            if "init_hit" in metrics_np:
                metrics[f"{rid}_fallen_init_height_threshold"] = bool(
                    metrics_np["init_hit"][row])
        return Episode.from_buffer_frames(
            frames=frames,
            final_observation={rid: np_bufs["final_obs"][rid][row]
                               for rid in AGENT_IDS},
            base_seed=int(job.seed),
            episode_index=int(ep_index),
            blueprint_hash=env_hash,
            agent_termination_proposal_records={
                rid: tuple(term_records[rid]) for rid in AGENT_IDS},
            episode_options=dict(job.episode_options),
            episode_metrics=metrics,
        )

    # ------------------------------------------------------------------
    def _teardown(self):
        if self._sim is not None:
            close = getattr(self._sim, "close", None)
            if callable(close):
                close()
        self._sim = self._rt = self._recorder = None
        self._env_key = None

    def close(self):
        self._teardown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False
