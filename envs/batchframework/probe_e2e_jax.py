"""E2E 纯 JAX rollout 性能探针（M2 性能上限验证，非语义验证）。

目的：回答"MJX 物理 + 等规模策略网络全在 GPU 上跑，吞吐天花板是多少"。
不追求与 CPU 主路径行为一致——只保证复杂度同量级：
  - 同一 XML 模型（双 humanoid + condim=3 + impratio=10 + 圆墙）
  - 同一控制频率（每动作步 25 物理子步）与 PD 控制（复用 simulator 的
    jit 函数，与正式实现同源）
  - 策略与 TruncatedNormalPolicy 等参数量/FLOPs：
    MLP 96→256(tanh)→256(tanh)→21(tanh) + log_std(21) ≈ 95K 参数，
    双机共享权重（与 self-play 一致），采样动作。
  - 观测为同维度 (B,96) 的设备端代理提取（逐元素算子，成本与真实
    观测同级，数量级远小于物理步）。

结构：obs→policy→采样融合为一个 jit（小图，实测 ~2ms/步，可忽略），
物理直接调用 simulator 已编译的 physical_step。Python 循环逐动作步驱动
——每步 3 个 launch，全部驻留设备端无 host 往返。不嵌套 jit：外层
jit 会把 25 子步 scan 内联进全 episode 图，XLA 编译超时（实测 >15min）。

实测结论（B=128/256, fp32, RTX 4090, 接触密集随机动作）:
  ~350-430 env-substeps/s —— 策略开销 <0.1%，瓶颈是 mjx.step 的
  接触求解。站立（接触少）可到 ~4.5K/s，说明吞吐强依赖接触数量。
  详见 envs/batchframework/M2_E2E_JAX_RESULTS.md。

用法:
    CUDA_VISIBLE_DEVICES=N python3 -B envs/batchframework/probe_e2e_jax.py \
        --batch 128 --steps 200
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jp


def build_policy_params(rng: np.random.RandomState):
    """TruncatedNormalPolicy 同构参数 pytree（96→256→256→21 + log_std）。"""
    def w(n_in, n_out):
        return jp.asarray(rng.normal(0, n_in ** -0.5, (n_in, n_out)), dtype=jp.float32)

    return {
        "W1": w(96, 256), "b1": jp.zeros(256, jp.float32),
        "W2": w(256, 256), "b2": jp.zeros(256, jp.float32),
        "W3": w(256, 21), "b3": jp.zeros(21, jp.float32),
        "log_std": jp.full(21, -0.8, jp.float32),
    }


def make_policy_fn(params):
    def act(obs, key):
        h = jp.tanh(obs @ params["W1"] + params["b1"])
        h = jp.tanh(h @ params["W2"] + params["b2"])
        mean = jp.tanh(h @ params["W3"] + params["b3"])
        noise = jax.random.normal(key, mean.shape)
        return jp.clip(mean + jp.exp(params["log_std"]) * noise, -1.0, 1.0)
    return act


def make_obs_fn(sim):
    """设备端 96 维观测代理：与真实观测同为逐状态逐元素提取。

    组成（参考真实 96 维结构：self 48 + opp 26 + task 22）：
      自身 joint_pos_norm(21) + joint_vel_norm(21) + root 位姿速度(6)
      = 48；对手机器人同样 48 → 96。
    """
    idx_a = sim._robots["robot_a"]["qpos_indices"]
    idx_b = sim._robots["robot_b"]["qpos_indices"]
    vid_a = sim._robots["robot_a"]["qvel_indices"]
    vid_b = sim._robots["robot_b"]["qvel_indices"]
    ra_a = sim._robots["robot_a"]["root_qpos_adr"]
    ra_b = sim._robots["robot_b"]["root_qpos_adr"]
    sc_a = jp.asarray(sim._norm_params["robot_a"]["scale"], dtype=jp.float32)
    rf_a = jp.asarray(sim._norm_params["robot_a"]["reference"], dtype=jp.float32)
    sc_b = jp.asarray(sim._norm_params["robot_b"]["scale"], dtype=jp.float32)
    rf_b = jp.asarray(sim._norm_params["robot_b"]["reference"], dtype=jp.float32)

    def _self_feats(d, idx, vid, ra):
        rva = sim._robots["robot_a"]["root_qvel_adr"] if idx is idx_a else \
            sim._robots["robot_b"]["root_qvel_adr"]
        return jp.concatenate([
            (d.qpos[:, idx] - rf_a if idx is idx_a else d.qpos[:, idx] - rf_b) /
            (sc_a if idx is idx_a else sc_b),
            d.qvel[:, vid] / (sc_a if idx is idx_a else sc_b),
            d.qpos[:, ra:ra + 3] - jp.array([0.0, 0.0, 1.282]),
            d.qvel[:, rva:rva + 3] * 0.1,
        ], axis=-1).astype(jp.float32)  # 48

    # 96 维 = 自身 48 + 对手 48（与真实观测 self+opponent 结构同量级）
    def obs_a(d):
        return jp.concatenate([_self_feats(d, idx_a, vid_a, ra_a),
                               _self_feats(d, idx_b, vid_b, ra_b)], axis=-1)

    def obs_b(d):
        return jp.concatenate([_self_feats(d, idx_b, vid_b, ra_b),
                               _self_feats(d, idx_a, vid_a, ra_a)], axis=-1)

    return obs_a, obs_b, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200, help="action steps（episode 长）")
    ap.add_argument("--substeps", type=int, default=25)
    ap.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    args = ap.parse_args()

    from envs.batchframework.mjx_simulator import MjxHumanoid21Simulator

    B = args.batch
    t_init = time.time()
    sim = MjxHumanoid21Simulator(batch_size=B, precision=args.precision)
    print(f"init: {time.time() - t_init:.1f}s", flush=True)
    t_reset = time.time()
    sim.reset()  # 编译 step/scan jit
    print(f"reset+jit: {time.time() - t_reset:.1f}s", flush=True)
    data = sim._mjx_data

    params = build_policy_params(np.random.RandomState(0))
    act_fn = make_policy_fn(params)
    obs_a_fn, obs_b_fn, _ = make_obs_fn(sim)

    # 融合的 obs→policy→采样 jit（小图）；物理用已编译的 physical_step。
    policy_a = jax.jit(lambda d, k: act_fn(obs_a_fn(d), k))
    policy_b = jax.jit(lambda d, k: act_fn(obs_b_fn(d), k))
    keys = jax.random.split(jax.random.PRNGKey(0), args.steps)

    def step_once(d, k):
        # 直接写 _action_jax：set_action 走 np.asarray 会强制 host sync，
        # 这里保持全链路在设备上（policy 输出已 clip 到 [-1,1]）。
        k1, k2 = jax.random.split(k)
        sim._mjx_data = d
        sim._action_jax["robot_a"] = policy_a(d, k1)
        sim._action_jax["robot_b"] = policy_b(d, k2)
        sim.physical_step(n_steps=args.substeps)
        return sim._mjx_data

    t0 = time.time()
    out = step_once(data, keys[0])
    jax.block_until_ready(out)
    compile_s = time.time() - t0
    print(f"first step (jit+run): {compile_s:.1f}s", flush=True)

    # 分段诊断：前 5 步逐步同步，区分 policy 与 physics 耗时
    t_pol = t_phys = 0.0
    for k in keys[1:6]:
        k1, k2 = jax.random.split(k)
        sim._mjx_data = out
        t0 = time.time()
        sim._action_jax["robot_a"] = policy_a(out, k1)
        sim._action_jax["robot_b"] = policy_b(out, k2)
        jax.block_until_ready(sim._action_jax["robot_b"])
        t_pol += time.time() - t0
        t0 = time.time()
        sim.physical_step(n_steps=args.substeps)
        jax.block_until_ready(sim._mjx_data)
        t_phys += time.time() - t0
        out = sim._mjx_data
    print(f"diag 5 steps: policy={t_pol / 5 * 1000:.0f}ms/step "
          f"physics={t_phys / 5 * 1000:.0f}ms/step", flush=True)

    t0 = time.time()
    for k in keys[6:]:
        out = step_once(out, k)
    jax.block_until_ready(out)
    dt = time.time() - t0
    n_meas = args.steps - 6
    dt_per = dt / n_meas

    env_substeps = B * n_meas * args.substeps
    transitions = B * n_meas * 2  # 双 agent 轨迹
    print(f"B={B} steps={n_meas} substeps={args.substeps} "
          f"{args.precision}: {dt:.2f}s measured ({dt_per * 1000:.0f}ms/action-step)")
    print(f"  -> {env_substeps / dt:,.0f} env-substeps/s")
    print(f"  -> {B * n_meas / dt:,.0f} env-action-steps/s")
    print(f"  -> {transitions / dt:,.0f} agent-transitions/s")


if __name__ == "__main__":
    main()
