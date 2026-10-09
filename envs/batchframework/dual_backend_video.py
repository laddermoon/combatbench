"""dual_backend_video — 同策略、同初始位姿的 GPU(MjWarp)/CPU(MuJoCo) 对比视频。

验证用临时工具（不追求速度）。流程：

1. **GPU 侧**：在 ``BatchRuntime`` 上用确定性策略跑一回合，逐
   action step 记录 ``qpos``/``qvel``（B=1）。
2. **CPU 侧**：同一策略 + 同 seed + 同 ``initial_distance`` 在
   ``EnvRuntime`` 跑一回合；reset 后把 core state 覆写为 GPU 侧
   记录到的 post-reset 状态——两端 RNG 流不同，单靠 seed 对不上
   摔倒位姿，覆写保证**严格同初始位姿**对比。
3. **渲染**：GPU 轨迹的 qpos 逐帧灌进 ``Humanoid21Simulator`` +
   ``mj_forward``，复用其 broadcast-view 相机（跟踪/EMA/arena 钳制
   逻辑逐像素一致）；CPU 侧同样逐 action step 调同一渲染口。

输出 ``gpu.mp4`` / ``cpu.mp4`` / ``compare.mp4``（左右拼接）+
``trajectory.npz``。fps 按 action rate（physics_freq /
phy_steps_per_action），两侧等时长。

Usage::

    PYTHONPATH=. python3 -m envs.batchframework.dual_backend_video \
        --policy baseline/runs/<run>/policy_exports/u00990 \
        --seed 12345 --device cuda:2 --out-dir /tmp/dualvid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint

from .binding_registry import resolve_binding
from .capability_registry import resolve_observer, resolve_plugin
from .device_runtime import BatchRuntime, DeviceTimeoutPlugin
from .policy_executor import TruncatedNormalExecutor

# 导出策略目录约定：{dir}/policy.py 里定义 ExportedTruncNormPolicy。
_POLICY_ENTRY = "policy.py"
_POLICY_CLASS = "ExportedTruncNormPolicy"

_DEFAULT_ENV_BP = ("baseline/humanoid21/blueprints/"
                   "standup_4stage_dense_v2_env.yaml")


def _policy_bp(policy_dir: Path) -> PolicyBlueprint:
    """由训练导出的策略目录构造 deployable PolicyBlueprint。"""
    return PolicyBlueprint(
        cls=f"file:{policy_dir / _POLICY_ENTRY}:{_POLICY_CLASS}")


def _build_batch_runtime(env_bp: EnvBlueprint, device: str,
                         batch_size: int = 1) -> BatchRuntime:
    """与 DeviceRollouter._build_runtime 同构的最小装配（无 RecordStore）。"""
    # 与 worker.py 一致：先绑 torch 当前设备，warp 分配跟随它。
    torch.cuda.set_device(torch.device(device).index or 0)
    binding = resolve_binding(env_bp.simulator.cls)
    sim = binding.make_sim(batch_size, device,
                           sim_config=dict(env_bp.simulator.config))
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
    return rt


def gpu_rollout(env_bp: EnvBlueprint, policy_bp: PolicyBlueprint, *,
                seed: int, distance: float, device: str, T: int):
    """GPU 确定性 rollout → (qpos_frames, qvel_frames)，(T+1, n*) 各。"""
    rt = _build_batch_runtime(env_bp, device, batch_size=1)
    executor = TruncatedNormalExecutor(policy_bp.to_dict(), device)
    try:
        rt.reset(
            seeds=torch.tensor([seed], dtype=torch.int64, device=device),
            options={"initial_distance": np.asarray([distance])})
        rt.obs_builder.build(rt.state)
        st = rt.state
        qpos = [st.sim.qpos[0].cpu().numpy().copy()]
        qvel = [st.sim.qvel[0].cpu().numpy().copy()]
        for _ in range(T):
            a_a, a_b, _, _ = executor.act(
                st.io.obs_a, st.io.obs_b, stochastic=False,
                ctx_a=None, ctx_b=None, ctx_ab=None, shared=False)
            rt.step((a_a, a_b))
            qpos.append(st.sim.qpos[0].cpu().numpy().copy())
            qvel.append(st.sim.qvel[0].cpu().numpy().copy())
            if not bool(rt.any_running()):
                break
        return np.stack(qpos), np.stack(qvel)
    finally:
        executor.close()


def cpu_rollout(env_bp: EnvBlueprint, policy_bp: PolicyBlueprint, *,
                seed: int, distance: float,
                init_qpos: np.ndarray, init_qvel: np.ndarray,
                n_frames: int):
    """CPU 确定性 rollout，post-reset 覆写为 GPU 的初始位姿。

    返回逐 action-step 渲染帧列表（首帧 = 覆写后的初始状态）。
    """
    policy = policy_bp.build()
    runtime = env_bp.build()
    runtime.reset(seed=seed, options={"initial_distance": distance})
    sim = runtime._core.simulator  # 临时工具：覆写 post-reset 位姿用
    assert sim.model.nq == init_qpos.shape[-1], \
        f"nq mismatch: cpu={sim.model.nq} gpu={init_qpos.shape[-1]}"
    sim.data.qpos[:] = init_qpos
    sim.data.qvel[:] = init_qvel
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    frames = [sim.get_broadcastview_image()]
    for _ in range(n_frames - 1):
        if not runtime._core.is_episode_active:
            break
        obs_a, obs_b = runtime.get_observation()
        a_a, _ = policy.act(obs_a)
        a_b, _ = policy.act(obs_b)
        runtime.step(a_a, a_b)
        frames.append(sim.get_broadcastview_image())
    runtime.close()
    return frames


def render_gpu_frames(qpos_frames: np.ndarray, qvel_frames: np.ndarray):
    """GPU 轨迹 qpos → MuJoCo broadcast-view 帧（复用同一相机管线）。"""
    import mujoco
    from envs.humanoid21.simulator import Humanoid21Simulator
    sim = Humanoid21Simulator()
    sim.reset()
    frames = []
    for qp, qv in zip(qpos_frames, qvel_frames):
        sim.data.qpos[:] = qp
        sim.data.qvel[:] = qv
        mujoco.mj_forward(sim.model, sim.data)
        frames.append(sim.get_broadcastview_image())
    return frames


def write_mp4(frames, path: Path, fps: int) -> None:
    import cv2
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", required=True, type=Path,
                    help="训练导出的策略目录（含 policy.py/model.pt），"
                         "两 agent 共用（self-play）")
    ap.add_argument("--env-blueprint", type=Path,
                    default=Path(_DEFAULT_ENV_BP))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--fps", type=int, default=20,
                    help="action rate = physics_freq/phy_steps_per_action")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    env_bp = EnvBlueprint.load(args.env_blueprint)
    T = int(env_bp.max_steps or 0)
    if T <= 0:
        sys.exit("env blueprint has no max_steps")
    policy_bp = _policy_bp(args.policy)

    print(f"[gpu] rollout on {args.device} (T={T}, seed={args.seed}, "
          f"distance={args.distance}) ...", flush=True)
    qpos_frames, qvel_frames = gpu_rollout(
        env_bp, policy_bp, seed=args.seed, distance=args.distance,
        device=args.device, T=T)
    n = len(qpos_frames)
    print(f"[gpu] captured {n} frames", flush=True)

    print("[cpu] rollout with same init pose ...", flush=True)
    cpu_frames = cpu_rollout(
        env_bp, policy_bp, seed=args.seed, distance=args.distance,
        init_qpos=qpos_frames[0], init_qvel=qvel_frames[0],
        n_frames=n)
    print(f"[cpu] rendered {len(cpu_frames)} frames", flush=True)

    print("[render] replaying GPU qpos through MuJoCo renderer ...",
          flush=True)
    gpu_frames = render_gpu_frames(qpos_frames, qvel_frames)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "trajectory.npz",
                        qpos=qpos_frames, qvel=qvel_frames)
    write_mp4(gpu_frames, args.out_dir / "gpu.mp4", args.fps)
    write_mp4(cpu_frames, args.out_dir / "cpu.mp4", args.fps)
    m = min(len(gpu_frames), len(cpu_frames))
    side = []
    for i in range(m):
        g, c = gpu_frames[i], cpu_frames[i]
        import cv2
        cv2.putText(g, "GPU/MjWarp", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 80, 80), 3)
        cv2.putText(c, "CPU/MuJoCo", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 255, 80), 3)
        side.append(np.concatenate([g, c], axis=1))
    write_mp4(side, args.out_dir / "compare.mp4", args.fps)
    print(f"[done] {args.out_dir}/gpu.mp4 cpu.mp4 compare.mp4 "
          f"trajectory.npz", flush=True)


if __name__ == "__main__":
    main()
