"""E2E mujoco-warp rollout 性能探针（与 probe_e2e_jax.py 同一问题，不同后端）。

目的：回答"同一复杂度模型 + 等规模策略，NWORLDS 原生批量布局的
mujoco-warp 吞吐能到多少"，对照 mjx-jax 的 ~355-435 env-substeps/s。

与 jax 版相同的复杂度约束：
  - 同一 XML（双 humanoid, nq=56/nv=54/nu=42, condim=3, impratio=10）
  - 同一控制频率（25 物理子步/动作步）与同一 PD 公式
    （torque = kp*(a*scale+ref - qpos) - kd*qvel，clip 到 ctrlrange）
  - 同一策略规模：96→256→256→21 tanh + log_std ≈95K 参数，双机共享
  - 观测：同维 96 提取（自身 48 + 对手 48）

结构差异（warp 的特性）：
  - 批量 = NWORLDS：make_data(nworld=B)，state 字段原生带 leading
    batch 维，mjw.step 一次处理所有 env（非 vmap）
  - PD 用 wp.kernel 写在 warp stream 上（每子步 1 kernel + mjw.step），
    全程不离开设备
  - policy 仍在 JAX：wp.to_jax 零拷贝视图读 qpos/qvel → jit 推理 →
    wp.from_jax 写回动作目标。每动作步一次 host sync（warp 与 jax 是
    两条 stream，正确性要求顺序同步）

用法:
    CUDA_VISIBLE_DEVICES=N python3 -B -u envs/batchframework/probe_e2e_warp.py \
        --batch 128 --steps 60
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--substeps", type=int, default=25)
    ap.add_argument("--nconmax", type=int, default=0, help="0=默认容量")
    args = ap.parse_args()

    import warp as wp
    import mujoco
    import mujoco_warp as mjw
    import jax
    import jax.numpy as jp

    from envs.batchframework.mjx_simulator import MjxHumanoid21Simulator
    from envs.batchframework.probe_e2e_jax import (
        build_policy_params, make_policy_fn, make_obs_fn,
    )
    from envs.humanoid21.simulator import Humanoid21Simulator

    wp.init()
    B = args.batch

    t0 = time.time()
    # 只借 mjx sim 的静态表（不 reset → 不触发 jax 编译）
    tab = MjxHumanoid21Simulator(batch_size=1)
    # 真实 reset 姿态（CPU mujoco，无 jit）
    cpu = Humanoid21Simulator(initial_distance=1.28)
    cpu.reset(seed=0)
    mjm = mujoco.MjModel.from_xml_path(tab.ARENA_XML)
    mjd0 = cpu.data
    print(f"init: {time.time() - t0:.1f}s", flush=True)

    m = mjw.put_model(mjm)
    kw = {"nconmax": args.nconmax} if args.nconmax else {}
    d = mjw.put_data(mjm, mjd0, nworld=B, **kw)
    print(f"put_model+put_data done, contact cap per field: "
          f"{d.contact.dist.shape}", flush=True)

    # --- PD 表（42 维拼接双机） ---
    statics = tab._jax_statics
    dev = "cuda:0"

    def cat_w(key):
        return np.concatenate([np.asarray(statics["robot_a"][key]),
                               np.asarray(statics["robot_b"][key])])

    qpos_idx = wp.array(cat_w("qpos_indices"), dtype=wp.int32, device=dev)
    qvel_idx = wp.array(cat_w("qvel_indices"), dtype=wp.int32, device=dev)
    act_ids = wp.array(cat_w("actuator_ids"), dtype=wp.int32, device=dev)
    gear = wp.array(cat_w("gear"), dtype=wp.float32, device=dev)
    lo = wp.array(cat_w("ctrl_lo"), dtype=wp.float32, device=dev)
    hi = wp.array(cat_w("ctrl_hi"), dtype=wp.float32, device=dev)
    norm_ref = cat_w("norm_ref").astype(np.float32)
    norm_scale = cat_w("norm_scale").astype(np.float32)
    kp = wp.array(np.concatenate([tab.KP, tab.KP]), dtype=wp.float32, device=dev)
    kd = wp.array(np.concatenate([tab.KD, tab.KD]), dtype=wp.float32, device=dev)
    act_target = wp.zeros((B, 42), dtype=wp.float32, device=dev)

    @wp.kernel
    def pd_kernel(qpos: wp.array(dtype=wp.float32, ndim=2),
                  qvel: wp.array(dtype=wp.float32, ndim=2),
                  target: wp.array(dtype=wp.float32, ndim=2),
                  qpos_idx: wp.array(dtype=wp.int32),
                  qvel_idx: wp.array(dtype=wp.int32),
                  act_ids: wp.array(dtype=wp.int32),
                  gear: wp.array(dtype=wp.float32),
                  lo: wp.array(dtype=wp.float32),
                  hi: wp.array(dtype=wp.float32),
                  kp: wp.array(dtype=wp.float32),
                  kd: wp.array(dtype=wp.float32),
                  ctrl: wp.array(dtype=wp.float32, ndim=2)):
        w, i = wp.tid()
        t = (kp[i] * (target[w, i] - qpos[w, qpos_idx[i]])
             - kd[i] * qvel[w, qvel_idx[i]])
        ctrl[w, act_ids[i]] = wp.clamp(t / gear[i], lo[i], hi[i])

    def physics_substep():
        wp.launch(pd_kernel, dim=(B, 42),
                  inputs=[d.qpos, d.qvel, act_target, qpos_idx, qvel_idx,
                          act_ids, gear, lo, hi, kp, kd, d.ctrl], device=dev)
        mjw.step(m, d)

    # --- JAX 侧：观测 + 策略（零拷贝视图） ---
    class _V:  # make_obs_fn 只访问 .qpos/.qvel
        pass

    v = _V()
    v.qpos = wp.to_jax(d.qpos)
    v.qvel = wp.to_jax(d.qvel)
    obs_a_fn, obs_b_fn, _ = make_obs_fn(tab)
    params = build_policy_params(np.random.RandomState(0))
    act_fn = make_policy_fn(params)
    policy_a = jax.jit(lambda k: act_fn(obs_a_fn(v), k))
    policy_b = jax.jit(lambda k: act_fn(obs_b_fn(v), k))
    keys = jax.random.split(jax.random.PRNGKey(0), args.steps)
    target_jax_scale = jp.asarray(norm_scale)
    target_jax_ref = jp.asarray(norm_ref)

    def jax_side(k):
        """一次动作步的 jax 部分：obs→policy→target rad (B,42)。"""
        k1, k2 = jax.random.split(k)
        a = jp.concatenate([policy_a(k1), policy_b(k2)], axis=-1)
        return (a * target_jax_scale + target_jax_ref).astype(jp.float32)

    to_target = jax.jit(jax_side)

    # --- 一个动作步 ---
    def action_step(k):
        wp.synchronize()  # warp 侧物理完成，jax 可读视图
        tgt = to_target(k)
        jax.block_until_ready(tgt)  # jax 算完，warp 可写
        act_target.assign(wp.from_jax(tgt))
        for _ in range(args.substeps):
            physics_substep()

    t0 = time.time()
    action_step(keys[0])
    wp.synchronize()
    print(f"first step (warp kernel 编译+运行): {time.time() - t0:.1f}s",
          flush=True)

    t0 = time.time()
    for k in keys[1:6]:
        action_step(k)
    wp.synchronize()
    print(f"diag 5 steps: {(time.time() - t0) / 5 * 1000:.0f}ms/step",
          flush=True)

    t0 = time.time()
    for k in keys[6:]:
        action_step(k)
    wp.synchronize()
    dt = time.time() - t0
    n_meas = args.steps - 6

    env_substeps = B * n_meas * args.substeps
    transitions = B * n_meas * 2
    print(f"B={B} steps={n_meas} substeps={args.substeps} warp-fp32: "
          f"{dt:.2f}s measured ({dt / n_meas * 1000:.0f}ms/action-step)")
    print(f"  -> {env_substeps / dt:,.0f} env-substeps/s")
    print(f"  -> {B * n_meas / dt:,.0f} env-action-steps/s")
    print(f"  -> {transitions / dt:,.0f} agent-transitions/s")


if __name__ == "__main__":
    main()
