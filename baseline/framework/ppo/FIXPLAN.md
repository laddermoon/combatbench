# PPO 框架修复指导（Product-Ready 路线）

> **读者**：具备 Python / PyTorch 基础、了解 PPO 基本原理的中级开发者。
> **目标**：把本框架从「内部可用」推进到「可开源、可被外部开发者使用和 critique」。
> **文档性质**：每个条目都是一个可独立完成的工作项，包含「问题是什么 / 根因 / 修复关键点 / 注意事项 / 验收标准」。
>
> 本文档中的所有问题**均已在当前代码上实际复现**，不是代码审查的猜测。复现方式写在各条目的「复现」小节里。

---

## 0. 怎么用这份文档

### 0.1 优先级定义

| 级别 | 含义 | 处理时机 |
|---|---|---|
| **P0** | 静默的正确性错误，或让外部用户在第一分钟就撞墙 | 开源发布前必须全部修完 |
| **P1** | 算法语义偏差、设计耦合、可信度缺口 | 发布前尽量修完，至少要在 README「已知局限」中声明 |
| **P2** | 打磨项：性能、可维护性、文档工程 | 发布后迭代 |

### 0.2 建议的 PR 批次

一次 PR 做一件事，每批都要能独立通过测试再合并。

| 批次 | 包含条目 | 主题 | 预估 |
|---|---|---|---|
| **PR-1** | P0-1, P0-2 | 修 KL 统计口径 + 补空测试 | 半天 |
| **PR-2** | P0-3, P0-4 | 空 buffer 短路 + 策略 device | 2 小时 |
| **PR-3** | P0-5, P0-6 | 策略导出契约（strict + 自包含） | 1~2 天 |
| **PR-4** | P2-1, P2-3 | 打包 + CI + 修 undefined name | 1 天 |
| **PR-5** | P1-4, P1-5 | 文档与代码对齐，示例进 CI | 1 天 |
| **PR-6** | P1-2 | rollout↔训练一致性守卫 | 半天 |
| **PR-7** | P1-1 | EV 语义修正 | 半天（需先做决策） |
| **PR-8** | P2-2 | Gym reference run | 2~3 天 |
| 后续 | P1-3 及以下 | 逐项打磨 | — |

### 0.3 通用工作纪律

1. **先写能复现问题的失败测试，再动手改**。本文档每条都给了复现方法，直接改写成 pytest。
2. **不要在修 bug 的 PR 里顺手重构**。P0 批次只做最小修改，重构留到 P1/P2。
3. 改完跑全量：`cd /data1/mono/things/combatbench && PYTHONPATH=. python3 -m pytest baseline/framework/ppo -q`
   （当前基线：102 passed）
4. **改动涉及训练语义时（P0-1、P1-1、P1-3），必须跑一次 `--smoke` 并人工比对日志**，确认没有把别的东西弄坏。

---

# P0 —— 发布前必须修完

---

## P0-1. early-stop 之后 `UpdateStats.approx_kl` 被错误地报成 0.0

**危害等级：最高。这是一个会让闭环探索调度做出完全相反决策的静默逻辑错误。**

### 问题是什么

当 PPO 因为 KL 超过 `target_kl` 触发 early-stop 后，`UpdateStats.approx_kl` 会变成 `0.0`，而真实的 KL 可能高达 0.5。同时日志会打出三条自相矛盾的告警。

### 复现

```python
# 见分析报告的 probe 脚本；核心是构造一个高 LR、KL 必然超标的 update
stats = ppo_update(..., pp=make_pp_params(target_kl=0.001, update_epochs=4), ...)
```

实测输出：

```
early_stop_kl   = 0.496857   ← KL 真的爆了
approx_kl       = 0.000000   ← on_update() 看到的值，错的
max_kl          = 0.993714
epochs_done     = 4          ← 语义已失效，永远等于 update_epochs
diagnostics:
  [early_stop] epoch=0 mb=1 running_mean_kl=0.4969 > target_kl=0.0010
  [warn] epoch=1 mean_kl=0.000000 too small, policy may be stuck or LR too low   ← 自相矛盾
  [warn] epoch=2 mean_kl=0.000000 too small, policy may be stuck or LR too low
  [warn] epoch=3 mean_kl=0.000000 too small, policy may be stuck or LR too low
```

### 根因

代码路径（`trainer.py`）：

```
L800   actor_stopped = False
L802   for epoch in range(pp.update_epochs):
L804       epoch_kls = []                    # 每个 epoch 重置
L851       if actor_stopped: continue        # ← actor 停更后直接跳过整个 minibatch 体
L872       epoch_kls.append(approx_kl)       # ← 这行在 L851 之后，所以永远不会执行
L972       mean_epoch_kl = mean(epoch_kls) if epoch_kls else 0.0    # ← 空列表 → 0.0
L1004      epoch_kl_stats.append({... "mean_kl": mean_epoch_kl ...})
L1055  final_kl = epoch_kl_stats[-1]["mean_kl"]   # ← 取最后一个 epoch → 0.0
L1081  approx_kl=final_kl
```

这是 **B1 改动（critic 与 actor early-stop 解耦）的回归**。B1 之前，early-stop 会 `break` 出 epoch 循环，`epoch_kl_stats[-1]` 自然就是触发停止的那个 epoch。B1 改成 `continue` 之后，后面还会追加 N 个全零的 epoch 记录，`[-1]` 的语义就坏了。

### 为什么危害巨大

`GUIDE.md` §5.2 推荐的闭环探索调度写法是：

```python
def on_update(self, stats, update):
    self._kl_history.append(stats.approx_kl)

def exploration(self, update):
    recent = self._kl_history[-3:]
    if all(kl < 0.005 for kl in recent):
        return ExplorationSpec(explore_factor=0.5)   # KL 太平 → 加大探索
    elif max(recent) > 0.1:
        return ExplorationSpec(explore_factor=-0.3)  # KL 太大 → 压制探索
```

**KL 炸到 0.5、触发 early-stop 的那些 update，恰恰会被判定为「KL 太平，加大探索」** —— 形成正反馈，把训练推向发散。而且日志里那三行 `policy may be stuck or LR too low` 会把排查方向带偏到「调大 LR」，进一步加剧问题。

### 修复关键点

1. **`approx_kl` 的口径改为「actor 实际执行过的所有 minibatch 的 KL 均值」**，而不是「最后一个 epoch 的均值」。
   最简做法：在 epoch 循环外维护一个 `all_actor_kls: List[float]`，在 L872 处同时 append 到它；最后 `approx_kl = mean(all_actor_kls) if all_actor_kls else 0.0`。

2. **拆分 `epochs_done` 的语义**。它现在是 `len(epoch_kl_stats)`，恒等于 `update_epochs`，已无信息量。
   - 新增 `actor_epochs_done`：actor 实际更新过的 epoch 数
   - `epochs_done` 保留为 critic 的 epoch 数（或直接改名 `critic_epochs_done`，但注意这会破坏 `to_log_dict()` 的键名兼容性，见「注意事项」）

3. **KL 异常告警要跳过 actor 已停更的 epoch**。L983 的 `if mean_epoch_kl < 0.001` 及 L989 起的单调/跳变检测，都应该加 `if not actor_stopped and epoch_kls:` 守卫。

4. **`epoch_kl_stats` 里为 actor 停更的 epoch 打标记**，例如加一个 `"actor_active": False` 字段，让离线分析脚本能区分「KL 真的是 0」和「actor 没跑」。

### 注意事项

- ⚠️ **`to_log_dict()` 的键名是对外契约**。`analyze_training.py` 和任何监控脚本都在解析 `__RAW_STATS__` 里的 `approx_kl` / `epochs_done`。改语义可以，**改键名要同步改下游**，或者用「新增键 + 旧键保留一个版本」的方式过渡。
- ⚠️ 不要顺手把 `continue` 改回 `break`。B1 的设计意图（critic 回归固定 target、不受 trust region 约束、砍早了只损失 value 质量）是**正确的**，要保留。这里修的是统计口径，不是控制流。
- ⚠️ `early_stop_kl` 目前只记录第一次触发时的值，这个行为是合理的，不要动。
- 顺带确认：`max_kl` 的计算 `max(s["max_kl"] for s in epoch_kl_stats)` 因为取 max 且全零 epoch 贡献 0，所以**恰好没错**。但依赖「零不影响 max」是脆弱的，建议也改成基于 `all_actor_kls`。

### 验收标准

新增测试，必须在修复前失败、修复后通过：

```python
def test_approx_kl_reflects_real_kl_after_early_stop():
    """early-stop 触发后，approx_kl 必须仍反映真实 KL，不能是 0。"""
    # 构造高 LR + 紧 target_kl 的 update
    stats = ppo_update(..., pp=make_pp_params(target_kl=0.001, update_epochs=4), ...)
    assert stats.early_stop_kl > 0.0, "前置条件：本测试需要 early-stop 真的触发"
    assert stats.approx_kl > 0.0
    # approx_kl 应与触发值同量级，而不是 0
    assert stats.approx_kl == pytest.approx(stats.early_stop_kl, rel=0.5)
    assert stats.actor_epochs_done < stats.epochs_done

def test_no_stuck_warning_when_actor_stopped():
    """actor 停更的 epoch 不应产生 'policy may be stuck' 告警。"""
    stats = ppo_update(...)  # 同上
    stuck = [d for d in stats.diagnostics if "may be stuck" in d]
    assert stuck == []
```

---

## P0-2. 两个 early-stop 测试是「空测试」，这是 P0-1 能存活的直接原因

### 问题是什么

`tests/test_trainer.py` 里的 `test_kl_early_stop_triggers` 和 `test_kl_early_stop_mid_epoch` **恒为绿灯但什么也没测**。它们提供了虚假的安全感，是 P0-1 这类回归能悄悄上线的根本原因。

### 根因

**测试 1（`test_kl_early_stop_triggers`，L1258）**

docstring 声称 *"target_kl=0.0 forces immediate early stop after first minibatch"*，但 `trainer.py` L952 的守卫是：

```python
if target_kl > 0.0 and epoch_kls:   # target_kl=0.0 → 条件为 False
```

`target_kl=0.0` 语义是**关闭** early-stop，不是「零容忍」。于是 `early_stop_kl` 恒为 0，整个断言块被 `if stats.early_stop_kl > 0.0:` 跳过。

更严重的是，被跳过的断言里有一句：

```python
assert stats.epochs_done <= 1
```

B1 改动之后 `epochs_done` 恒等于 `update_epochs`（=4），**这句如果真的执行会立刻失败**。它是 B1 之前的遗留断言 —— 说明 B1 上线时，本该守护它的测试早就已经失效了。

**测试 2（`test_kl_early_stop_mid_epoch`，L1315）**

```python
last_epoch = stats.epoch_kl_stats[-1]
assert last_epoch["n_minibatches"] < n_batches_per_epoch   # 0 < 8，平凡成立
```

early-stop 后最后一个 epoch 的 `n_minibatches` 必然是 0（actor 没跑），所以断言恒真，**并不能证明「在 epoch 中途停下」**。

### 修复关键点

1. **测试 1**：用一个真实的小 `target_kl`（如 `1e-6`）+ 高 LR 来触发，而不是 `0.0`。断言改为无条件（先 `assert stats.early_stop_kl > 0.0` 作为前置条件校验）。
2. **测试 2**：断言要指向**触发停止的那个 epoch**，而不是 `[-1]`：
   ```python
   triggering = next(e for e in stats.epoch_kl_stats if e["n_minibatches"] > 0
                     and e["mean_kl"] > pp.target_kl)
   assert 0 < triggering["n_minibatches"] < n_batches_per_epoch
   ```
3. **补一个测试专门锁死 `target_kl=0.0` 的语义**：
   ```python
   def test_target_kl_zero_disables_early_stop():
       """target_kl=0.0 表示关闭 early-stop（不是零容忍）。"""
       stats = ppo_update(..., pp=make_pp_params(target_kl=0.0, update_epochs=3), ...)
       assert stats.early_stop_kl == 0.0
       assert stats.actor_epochs_done == 3
   ```
   顺便把这个语义写进 `PPOParams.target_kl` 的 docstring —— 现在它完全没说明。

4. **全量审查同类模式**。用这条命令找出所有「断言被条件包裹」的测试，逐个确认条件在正常路径下会不会成立：
   ```bash
   grep -n "if .*:$" -A3 tests/test_trainer.py | grep -B1 "assert"
   ```

### 注意事项

- ⚠️ **不要把这类测试直接删掉**。它们覆盖的场景（early-stop）是真实且重要的，要修成有效测试。
- ⚠️ 触发 KL 超标的测试**天生不稳定**（依赖随机初始化 + LR）。要用固定 `rng = np.random.default_rng(42)` + `torch.manual_seed()`，并且**把「前置条件是否成立」写成 assert 而不是 if**。这样一旦环境变化导致条件不再成立，测试会失败而不是静默跳过 —— 这正是当前 bug 的教训。
- ⚠️ 本仓库测试普遍用 `print("test_xxx: PASS")` 风格（见 P2-5）。**修 P0-2 时不要顺手改风格**，会让 diff 无法审查。

### 验收标准

- 两个测试改完后，**故意把 `trainer.py` 的 early-stop 逻辑注释掉，它们必须失败**。这是判断测试是否有效的唯一标准。
- P0-1 的修复被这两个测试 + 新增的 `test_approx_kl_reflects_real_kl_after_early_stop` 共同锁死。

---

## P0-3. `build_trajectories` 返回 `[]` 会崩，但文档明确承诺可以

### 问题是什么

`experiment.py` 的 `build_trajectories` docstring 写着 *"Returns an empty list to skip all episodes entirely."*，`PPOBuffer` 也实现了 `is_empty()`，但 `loop.py` 从不检查就直接调用 `ppo_update`。

### 复现

```
buf.is_empty() = True
ppo_update on empty buffer RAISED:
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x0 and 8x16)
```

### 根因

`PPOBuffer` 在空输入时把 `self.obs` 设为 `np.zeros((0,), np.float32)` —— 注意这是 **1 维** 形状 `(0,)`，不是 `(0, obs_dim)`。`ppo_update` L461 直接 `torch.as_tensor(buf.obs)` 然后喂给 `critics[key](obs_t)`，`nn.Linear` 收到 1-D 输入被当成单样本，形状不匹配报错。

`loop.py` L486-511 构造完 `buf` 直接进 `ppo_update`，中间没有任何守卫。

`test_buffer_empty` 只测了 buffer 构造成功，没有测下游的 `ppo_update`。

### 为什么这是真实场景

「本轮 rollout 全部被过滤掉」是完全正常的：
- 课程学习早期，所有 episode 都在第 1 帧就摔倒（`T == 0` → GUIDE 示例里的 `if T == 0: continue`）
- gating 实验里本轮没有任何 episode 进入目标阶段
- 实验自己实现的质量过滤把整批数据丢掉

后果是：一个跑了 3000 轮的后台训练突然崩掉，而且因为 `--background` 模式 stderr 重定向到日志，用户可能几小时后才发现。

### 修复关键点

1. **在 `loop.py` 的主循环里短路**，这是最小、最正确的修法：
   ```python
   buf = PPOBuffer(...)
   if buf.is_empty():
       print(f"[update {u:4d}] [skip] build_trajectories returned no usable frames "
             f"(episodes={len(episodes)}); skipping PPO update", flush=True)
       continue   # 注意：见下方「注意事项」
   ```

2. **同时在 `ppo_update` 入口加防御性早返回**，因为它是公开 API，会被测试和其他调用方直接使用：
   ```python
   if buf.is_empty():
       return UpdateStats(... 全零/空的合法实例 ...)
   ```
   建议给 `UpdateStats` 加一个 `@classmethod empty(reward_keys)` 工厂方法来构造这个零值实例，避免在 `ppo_update` 里手写 20 个字段。

3. **顺手修 `PPOBuffer` 的空形状**：把 `np.zeros((0,), np.float32)` 改成 `np.zeros((0, 0), np.float32)`，让空 buffer 的 `obs` 至少维度正确。这不是修复的必要条件，但能减少下游意外。

### 注意事项

- ⚠️ **`continue` 会跳过本轮的 eval、日志、checkpoint**。这可能不是你想要的 —— 如果连续 100 轮都空，用户会看不到任何输出，以为进程挂了。建议：
  - 短路后仍然打印 `[update N] [skip]` 行（上面已包含）
  - 考虑用一个 `consecutive_empty` 计数器，超过阈值（比如 10）就 raise，因为这几乎一定是实验代码写错了而不是正常现象
- ⚠️ 空 update 时**不要**调用 `experiment.on_update(stats, u)` 传一个全零的 `UpdateStats` —— 实验的 KL 历史会被污染（又是一次「0.0 被误读为 KL 太平」）。要么跳过 `on_update`，要么给 `UpdateStats` 加一个 `is_empty: bool = False` 标记让实验能识别。
- ⚠️ 修完记得同步 `experiment.py` 的 docstring：明确「返回空列表会跳过本轮 PPO 更新，但仍会正常记录日志」。

### 验收标准

```python
def test_ppo_update_empty_buffer_returns_empty_stats():
    buf = PPOBuffer([], actor, torch.device("cpu"), ("r_a",))
    stats = ppo_update(...)   # 修复前 RuntimeError
    assert stats.total_steps == 0
    assert stats.approx_kl == 0.0

def test_loop_survives_empty_trajectories():
    """build_trajectories 返回 [] 时训练循环不崩。"""
    # 用一个 build_trajectories 恒返回 [] 的 fake experiment 跑 2 轮
```

---

## P0-4. `TruncatedNormalPolicy.to(device)` 之后 `self.device` 不更新，`act()` 崩

### 问题是什么

`self.device` 是 `__init__` 里存的普通 Python 属性，`nn.Module.to()` 只搬参数和 buffer，**不会改它**。而 `GUIDE.md` §4 的示例代码正是 `return bp.build().to(device)`。

### 复现

```
param device=cuda:0  self.device=cpu
act() RAISED: RuntimeError: Expected all tensors to be on the same device,
              but found at least two devices, cuda:0 and cpu!
```

### 根因

`policies/truncated_normal_mlp.py` L169-181：

```python
def __init__(self, obs_dim, action_dim, hidden_dim, device="cpu"):
    ...
    self.device = torch.device(device)   # ← 快照，之后再也不同步
```

L363 / L381 的 `act()` / `sample()` 用它建 tensor：

```python
obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
```

参数在 cuda、输入在 cpu → `addmm` 报错。

### 修复关键点

1. **删掉 `self.device` 属性，改为运行时查询参数所在设备**：
   ```python
   @property
   def device(self) -> torch.device:
       return next(self.parameters()).device
   ```
   这是 PyTorch 生态里的标准惯用法，对 `.to()` / `.cuda()` / `DataParallel` 全部自动正确。

2. **保留 `__init__` 的 `device=` 形参以兼容现有调用**，但改成构造末尾 `self.to(device)`，而不是存快照：
   ```python
   def __init__(self, obs_dim, action_dim, hidden_dim, device="cpu"):
       super().__init__()
       ...
       self.to(torch.device(device))
   ```

3. **检查所有其他策略实现是否有同样的 pattern**（`policies/todo/` 下的 8 个策略族），因为它们迟早会被启用：
   ```bash
   grep -rn "self.device" policies/
   ```

### 注意事项

- ⚠️ **`@property device` 与 `nn.Module` 不冲突**（`nn.Module` 本身没有 `device` 属性），但要确认没有别处在做 `policy.device = ...` 赋值 —— property 没有 setter 会直接报错。先 grep 确认。
- ⚠️ 对**无参数**的 module，`next(self.parameters())` 会抛 `StopIteration`。本策略一定有参数，但如果要做成通用工具函数，要处理这个边界。
- ⚠️ 这个 bug 之所以一直没暴露，是因为实际训练路径里 actor 是 `build_actor(device)` 内部构造好就在正确设备上，而 `act()` 只在 eval / 导出后的子进程里调用（那里是 CPU）。**修完要专门测 cuda 路径**，不能只跑 CPU 测试。

### 验收标准

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_policy_act_after_to_cuda():
    p = TruncatedNormalPolicy(8, 3, 16).to("cuda")
    assert p.device.type == "cuda"
    a, _ = p.act(np.zeros(8, dtype=np.float32))   # 修复前 RuntimeError
    assert a.shape == (3,)

def test_policy_device_follows_to():
    p = TruncatedNormalPolicy(8, 3, 16)
    assert p.device.type == "cpu"
```

---

## P0-5. 导出策略用 `load_state_dict(strict=False)`，会静默产出随机权重的策略

**这对一个 benchmark 项目是致命的静默失败。**

### 问题是什么

`policies/truncated_normal_mlp.py` L79：

```python
self._policy.load_state_dict(payload["state_dict"], strict=False)
```

`strict=False` 会**同时忽略**「checkpoint 里多出来的键」和「模型需要但 checkpoint 里缺失的键」。缺失的层保持随机初始化，然后被拿去做 rollout 和评测，全程不报错。

### 复现

```python
# 把 net.0.weight 从 model.pt 里删掉，再 bp.build()
missing w0 -> [ 0.215  0.220 -0.174]   ← 导出策略输出
true        -> [-0.071  0.556 -0.393]   ← 真实策略输出
```

`build()` 成功返回，没有任何 warning。

> 注：如果只是把权重**置零**而不是删除，用 `obs=zeros` 测不出差异（第一层输出只剩 bias）。测这类问题一定要用**非零输入**。这是个容易自己骗自己的坑。

### 为什么危害巨大

任何一次层重命名、结构调整、`hidden_dim` 变更，都会得到一个「能跑、不报错、但完全是随机初始化」的策略。后果：
- 训练日志一切正常（训练用的是主进程的 actor，是对的）
- rollout 收集到的是随机策略的数据 → importance ratio 全错 → PPO 收敛到垃圾
- 或者：导出的比赛策略是随机的，但评测分数「看起来只是差一点」

### 修复关键点

1. **改成 `strict=True`**。这是核心，一行改动。

2. **在 `model.pt` payload 里加版本与完整性信息**：
   ```python
   payload = {
       "format_version": 1,
       "policy_class": "TruncatedNormalPolicy",
       "arch": {"obs_dim": ..., "action_dim": ..., "hidden_dim": ...},
       "state_dict": {...},
       "state_dict_keys": sorted(sd.keys()),      # 便于给出好的错误信息
   }
   ```
   加载时先校验 `format_version` 和 `policy_class`，不匹配就抛带明确指引的异常。

3. **移除 `**_ignored: Any` 这个静默吞参数的签名**（L69）。它会让「blueprint 里写错参数名」变成静默无效。改成显式参数 + 未知参数报错。

4. **加载后做一次前向一致性自检**（可选但强烈推荐）：导出时记录一个固定输入的输出指纹，加载后重算并比对。见 P1-2，两者可以合并实现。

### 注意事项

- ⚠️ **改成 `strict=True` 会让存量的 81 个已导出 artifact 立刻报错**（它们本来就已经坏了，见 P0-6）。这是**好事** —— 从静默错误变成显式错误。但要：
  - 在 CHANGELOG 里明确写「破坏性变更」
  - 提供一个 `scripts/migrate_policy_export.py` 或至少一份升级说明
- ⚠️ 不要用 `strict=False` + 手动检查 `missing_keys` 的折中方案。`load_state_dict` 返回的 `IncompatibleKeys` 很容易被忽略，`strict=True` 直接抛异常更可靠。
- ⚠️ 如果确实存在「有意的部分加载」需求（比如 fine-tune 时只加载 backbone），应该是**一个显式的、单独命名的 API**（`load_backbone_only()`），而不是默认路径上的 `strict=False`。

### 验收标准

```python
def test_export_rejects_missing_keys():
    pol = TruncatedNormalPolicy(8, 3, 16)
    d = tempfile.mkdtemp(); bp = pol.to_blueprint(dest_path=d)
    p = torch.load(f"{d}/model.pt"); sd = dict(p["state_dict"])
    sd.pop("net.0.weight")
    p["state_dict"] = sd; torch.save(p, f"{d}/model.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        bp.build()

def test_export_roundtrip_exact():
    """导出再加载，对非零输入必须逐位一致。"""
    pol = TruncatedNormalPolicy(8, 3, 16)
    bp = pol.to_blueprint(dest_path=tempfile.mkdtemp())
    obs = np.random.randn(8).astype(np.float32)   # 关键：非零输入
    np.testing.assert_allclose(bp.build().act(obs)[0], pol.act(obs)[0], rtol=0, atol=0)
```

---

## P0-6. 导出的策略不是自包含的，81 个历史 artifact 已经全部失效

### 问题是什么

`to_blueprint()` 生成的 `policy.py` 是**一段写在字符串字面量里的源码**（`_build_export_policy_code()`，L33-119），而这段源码 `from baseline.framework.ppo.policies.truncated_normal_mlp import TruncatedNormalPolicy` —— 即**导出物反向依赖仓库当前状态**。而它的 docstring 却宣称 "standalone"。

### 复现

```bash
$ grep -rl "ppo.policies.tanh_gaussian_mlp" policy/**/policy.py | wc -l
81
$ PYTHONPATH=. python3 -c "import baseline.framework.ppo.policies.tanh_gaussian_mlp"
ModuleNotFoundError: No module named 'baseline.framework.ppo.policies.tanh_gaussian_mlp'
```

`tanh_gaussian_mlp.py` 被挪进了 `policies/todo/`，于是 **81 个已导出的策略产物永久失效**。

### 为什么这对本项目不可接受

CombatBench 的定位是「外部开发者提交策略参赛」的 benchmark。策略 artifact 必须满足：
- **时间上稳定**：半年前导出的策略今天还能跑
- **空间上可移植**：不需要用户有仓库、不依赖 `PYTHONPATH`
- **不受重构影响**：仓库内部重命名不能让历史产物失效

当前设计三条全不满足。而且这不是理论风险 —— 已经发生了，81 次。

### 修复关键点

这是本文档里唯一需要**设计决策**的 P0，有两条路：

**方案 A（推荐）：内联最小推理代码**

导出的 `policy.py` 包含完整的、不依赖仓库的推理实现。
- 优点：真正自包含，符合 benchmark 语义
- 代价：推理代码在两处（训练用的 + 导出的），有漂移风险
- **必须配一个 parity 测试**：随机初始化一个策略，导出，对同一批随机输入比对两份实现的输出逐位相等。这个测试是方案 A 的生命线。

**方案 B：稳定的 loader 契约**

定义一个独立的、承诺向后兼容的 `combatbench.policy_format` 小包（不依赖 `baseline.*`），导出物只依赖它。
- 优点：推理代码单一来源
- 代价：需要真的维护版本兼容（`format_version` 分发到不同 loader）
- 适合长期，但工作量更大

**无论选哪个，都要做的事：**

1. **停止用字符串字面量做 codegen**。把导出模板放到一个真实的 `.py` 模板文件里（`policies/_export_template.py`），用 `importlib.resources` 读取。这样模板本身能被 lint、被类型检查、被测试。
2. **`format_version` 字段**（与 P0-5 共用）。
3. **导出目录里放一份 `MANIFEST.json`**：记录导出时间、git commit、policy class、arch、格式版本。artifact 出问题时能追溯。
4. **写一个 artifact 兼容性测试**：仓库里存一两个「黄金 artifact」（小模型，几十 KB），CI 每次都加载它们并比对输出。这样任何破坏历史产物的改动都会被立刻拦下。

### 注意事项

- ⚠️ **存量的 81 个 artifact 要单独决策**：是写迁移脚本救回来（把 `tanh_gaussian_mlp.py` 从 `todo/` 挪回并保持路径稳定），还是标记为废弃。如果这些是历史比赛结果或 baseline 基线，**必须救回来**，否则你失去了所有历史可比性。
- ⚠️ 不要把 `torch.save` 的 pickle 当成稳定格式。`weights_only=False` 的 `torch.load` 既有安全问题（反序列化任意对象）也有兼容问题。payload 里只放**纯 tensor + 纯 Python 标量/字符串**，不要放自定义类实例。
- ⚠️ `loop.py` L199 的 `torch.load(..., weights_only=False)`（checkpoint 路径）同理需要审视 —— 它加载的是自己写的 checkpoint 所以风险低，但外部提交的 artifact 绝不能用 `weights_only=False`。
- ⚠️ `policies/todo/` 目录本身是这次事故的根源。开源前必须处理：要么正式支持，要么彻底移出仓库（放到分支或 `experimental/` 并明确不承诺稳定性）。**不要让公开的导出路径指向一个叫 `todo` 的目录。**

### 验收标准

```python
def test_export_is_self_contained(tmp_path, monkeypatch):
    """导出物在没有仓库的环境下也能加载。"""
    pol = TruncatedNormalPolicy(8, 3, 16)
    bp = pol.to_blueprint(dest_path=str(tmp_path))
    obs = np.random.randn(8).astype(np.float32)
    expected = pol.act(obs)[0]
    # 在子进程里以空 PYTHONPATH 加载并推理
    # （方案 A 应通过；方案 B 只依赖 policy_format 包）

def test_golden_artifact_still_loads():
    """仓库内置的黄金 artifact 必须永远可加载且输出不变。"""
```

---

# P1 —— 算法与设计层面

---

## P1-1. `explained_variance` 其实不是 explained variance，而 confidence 加权完全建立在它之上

### 问题是什么

`trainer.py` L581-591 计算的量被命名为 `explained_variance`，注释说它 *"Measures how well the critic predicts actual returns"*。但由于 GAE 恒等式 `rets_all = advs_all + values_all`，有：

```
ret - V  ≡  A        （恒等，不是近似）
EV = 1 - Var(ret - V)/Var(ret)  ≡  1 - Var(A)/Var(R)
```

这**不是**「critic 预测真实回报的能力」，而是「GAE advantage 相对 bootstrapped return 的方差占比」—— 一个**与 λ 强耦合的自一致性统计量**，只在 λ=1 时才退化为教科书意义的 explained variance。

### 复现

固定 reward 序列、已知真值 V、人为加噪：

```
critic noise=0.5:  λ=0 → EV=+0.685   λ=0.9 → +0.797   λ=1 → +0.783   真实EV(vs MC)=+0.783
critic noise=2.0:  λ=0 → EV=-0.741   λ=0.9 → -3.372   λ=1 → -3.076   真实EV(vs MC)=-3.076
```

λ=1 时与真值吻合（必然，因为此时 R 就是 MC return）；λ<1 时存在系统偏差，**且偏差方向随 critic 质量翻转**（noise=0.5 时偏低，noise=2.0 时偏高）。

### 为什么这直接击中框架的核心卖点

`GUIDE.md` §5.3 **鼓励**不同 channel 用不同 λ：

```python
RewardChannel("r_ko",      gamma=0.99, gae_lambda=0.98)   # 稀疏终端：高 λ
RewardChannel("r_balance", gamma=0.99, gae_lambda=0.90)   # 密集 shaping：低 λ
```

然后 L624-632 用 EV 派生的 `confidence = sqrt(clip(EV,0,1))` 去**跨 channel 加权 advantage**。也就是说：**在拿两个尺度不同、偏差方向不同的量做直接比较**，并用比较结果决定各 channel 对 policy gradient 的影响力。这个链条的每一环单独看都合理，合起来不成立。

### 修复关键点

按代价从低到高，三个层次，**至少要做第 1 层**：

**第 1 层（必做，半小时）：正名 + 修注释**

把这个量改名为反映其真实含义的名字，例如 `td_ratio` 或 `gae_self_consistency`，并把注释改对：

```python
# gae_self_consistency = 1 - Var(A)/Var(R).
# 注意：由于 R = A + V（GAE 恒等式），这个量衡量的是 advantage 相对
# bootstrapped return 的方差占比，而**不是** critic 对真实回报的解释力。
# 它与 gae_lambda 耦合：只有 λ=1 时才等于教科书意义的 explained variance。
# 因此它不能在 λ 不同的 channel 之间直接比较。
```

⚠️ 改名要同步 `UpdateStats.explained_variance` 字段、`to_log_dict()` 的 `ev_*` 键、`loop.py` 的日志行、以及 `analyze_training.py`。考虑保留 `ev_*` 键名一个版本做过渡。

**第 2 层（推荐）：算一个真正的 EV**

框架里已经有现成的 `compute_returns_to_go`（目前是死代码，见 P1-6）。用它算 MC return 作为独立 target：

```python
mc_ret = compute_returns_to_go(rewards, last_value=last_value, gamma=gamma)
true_ev = 1 - np.var(mc_ret - V) / np.var(mc_ret)
```

代价：多一次 O(T) 循环（可向量化，见 P1-9）。收益：得到一个 λ 无关、可跨 channel 比较、语义正确的量。**confidence 应该改用这个。**

⚠️ MC return 在长 episode + 高 γ 下方差很大，EV 会偏噪声。可以做窗口平滑（跨 update 的 EMA），这本身也能让 confidence 更稳定。

**第 3 层（可选）：重新审视 confidence 是否该存在**

`--no-confidence` 开关已经存在，说明团队自己也不确定。建议：
- 做一次 A/B（同 seed、开/关 confidence），把结果写进文档
- 如果收益不明确，考虑默认关闭、标注为 experimental。**一个开源框架里，默认开启的、语义不清的算法魔改，是最容易被 critique 的地方。**

### 注意事项

- ⚠️ **改 confidence 会改变训练动力学**。修完必须跑 smoke + 至少一个短程真实实验，和修改前的曲线对比。不要在同一个 PR 里同时改 EV 定义和 confidence 公式。
- ⚠️ `clip(EV, 0, 1) ** 0.5` 里的开方是「阻尼」，注释解释得很好（EV=0.5 → conf=0.71），这个设计本身合理，不用动。要改的是**喂给它的输入**。
- ⚠️ 注意 L586 的 `var_y < 1e-8 → ev_val = 0.0`。语义是「return 无方差 → 无从判断 → 不信任」。这个 fallback 合理，但要意识到它会让「reward 恒为常数」的 channel 的 confidence 恒为 0 → 该 channel 永远不影响 actor。这可能是意料之外的（用户会疑惑「我的 channel 为什么不起作用」）。已有 cold-start 告警覆盖了部分场景，可以再加一条针对这个情况的诊断。

### 验收标准

```python
def test_ev_equals_textbook_ev_at_lambda_one():
    """λ=1 时框架的 EV 必须等于对 MC return 的 EV。"""

def test_ev_is_lambda_invariant_after_fix():
    """（第 2 层修完后）同一个 critic，不同 λ 下 EV 应基本一致。"""
    # 修复前：λ=0 → 0.685, λ=0.9 → 0.797（差 0.11）
    # 修复后：两者应在 1e-3 内
```

---

## P1-2. 没有任何 rollout↔训练的一致性守卫

### 问题是什么

框架**丢弃**了 rollout 时算出的 log_prob（`ExploratoryPolicy` 明明把它放进了 `extra["log_prob"]`），改用 `PPOBuffer` 在主进程 GPU 上用活 actor 重算。

理论上等价（同一个 θ_old），但实践上这条等价性跨越了：

```
主进程 GPU 活 actor  →  to_blueprint() 序列化  →  model.pt  →
worker 子进程反序列化  →  CPU float32 推理  →  action
                                                   ↓
                          主进程 GPU 活 actor 重算 log_prob
```

任何一处失配 —— P0-5 的 `strict=False`、导出漏字段、`**_ignored` 吞参数、CPU/GPU 数值差异、`_explore_scale` 在两边实现不一致 —— 都会让 importance ratio 从 1 偏移。**而 PPO 会照常「收敛」到一个错的目标上，全程没有任何报错。**

### 为什么必须加守卫

这是本框架架构下**最危险的一类 bug**：
- 它不会崩，不会报错，只会让训练效果变差
- 排查时人会先怀疑 reward、超参、环境，最后才想到 log_prob
- P0-5 和 P0-6 已经证明这条链路真的会断

而这个守卫**几乎免费**：第一个 minibatch 的 ratio 在数学上必须严格等于 1（θ 还没更新过）。

### 修复关键点

1. **最简版（强烈推荐，5 行）**：在 `ppo_update` 的第一个 epoch、第一个 minibatch 处断言：
   ```python
   if epoch == 0 and mb_idx == 0:
       with torch.no_grad():
           max_dev = (ratio - 1.0).abs().max().item()
       if max_dev > 1e-4:
           diagnostics.append(
               f"  [warn] first-minibatch |ratio-1| = {max_dev:.2e} (expected ~0). "
               f"log_prob recomputation disagrees with theta_old — check policy "
               f"export/import fidelity and explore_factor threading."
           )
   ```
   ⚠️ **先用 warning 而不是 raise**。上线后观察一两周，确认阈值合理、没有误报，再考虑升级为异常。float32 下 1e-4 是个合理起点，但要实测校准。

2. **更强版：直接比对 rollout 的 log_prob**。`ExploratoryPolicy` 已经把 log_prob 放进 `extra` 了，让实验可选地把它填进 `Trajectory`（新增 `rollout_log_prob: Optional[np.ndarray]` 字段），buffer 里比对：
   ```python
   if traj.rollout_log_prob is not None:
       dev = np.abs(lp_seg - traj.rollout_log_prob).max()
   ```
   这个版本能定位得更准（区分「导出失真」和「explore_factor 串错」）。但需要改 `Trajectory` 契约，作为第二步做。

3. **配套：导出 parity 测试**（与 P0-5/P0-6 的验收标准合并实现）。

### 注意事项

- ⚠️ **不要**为了「让 ratio 精确等于 1」而改成用 rollout 的 log_prob 做 `old_lp`。当前的重算设计是**对的**：它保证 `old_lp` 和 `new_lp` 出自同一份实现、同一套数值路径，避免了「两个实现的系统性偏差被当成策略变化」这个更麻烦的问题。守卫的目的是**检测**失配，不是绕过它。
- ⚠️ 第一个 minibatch 的 ratio 不会是**精确** 1.0：buffer 的批量调用是全 batch 一次前向，minibatch 是子集前向，浮点累加顺序不同会有 ~1e-6 级别差异。阈值要留余量。
- ⚠️ 如果 `explore_factor` 在 rollout 和训练侧的语义不一致（比如 `ExploratoryPolicy` 传的是 `float`，`evaluate_actions` 收的是 `(B,)` tensor，两边 `_explore_scale` 实现漂移），这个守卫会捕获它 —— 这正是它最大的价值。

### 验收标准

```python
def test_first_minibatch_ratio_is_one():
    """θ 未更新时，第一个 minibatch 的 ratio 必须≈1。"""
    stats = ppo_update(...)
    assert not any("first-minibatch" in d for d in stats.diagnostics)

def test_ratio_guard_fires_on_corrupted_old_logp():
    """人为污染 buf.log_probs，守卫必须报警。"""
    buf.log_probs += 0.5
    stats = ppo_update(...)
    assert any("first-minibatch" in d for d in stats.diagnostics)
```

---

## P1-3. `combined_adv` 合并后不再归一化，有效学习率随 critic 质量漂移

### 问题是什么

L1 归一化解决了 `actor_weight` 的尺度问题（这是本框架的亮点），但 `confidence ∈ [0,1]` 直接乘在外面，**没有任何补偿**：

```python
combined_adv += aw_normed * conf * normed    # aw_normed 的 L1 和为 1，conf 却是 [0,1] 的缩放
```

后果：
- 训练早期所有 critic 的 EV≈0 → confidence≈0 → `combined_adv`≈0 → **actor 梯度≈0**
- critic 学好后 confidence→1 → 梯度突增

也就是 actor 的**有效 step size 在训练过程中被 critic 质量隐式调制了一个数量级**，而且和 `learning_rate` 纠缠在一起。

### 为什么这是个问题

1. `use_confidence` 开/关会显著改变最优 LR。用户按开启时调好的 LR 去关掉 confidence 跑对照实验，得到的是「LR 也变了」的混淆结果 —— 这让任何 A/B 实验都不可信。
2. 文档里**完全没有提示**这个 coupling。
3. 已有的 cold-start 告警（L641-659）只覆盖了「全部 channel confidence=0」的极端情况，对「confidence=0.3 导致梯度缩小 3 倍」这种常态无感知。

### 修复关键点

这里**先不要急着改公式**，按顺序做：

1. **第一步：把它变可见**（必做，低风险）。在 `UpdateStats` 里加：
   ```python
   combined_adv_std: float      # combined_adv 在 aw≠0 帧上的 std
   combined_adv_abs_mean: float
   ```
   并在 `loop.py` 的 `[PPO Opt]` 行打出来。**没有这个指标，你无法判断第二步该怎么做。**

2. **第二步：决策**。三个选项，各有代价，需要实验数据支撑：
   - **(a) 保持现状 + 文档化**：在 README 明确写「confidence 会调制有效学习率，这是有意的 —— critic 不可信时自动减小步长」。这是一个**可以辩护的设计**，只要说清楚。
   - **(b) 合并后再全局归一化**：`combined_adv /= (combined_adv[mask].std() + eps)`。好处是 step size 稳定；坏处是**丢掉了「所有 critic 都不可信时应该少走」这个信号**，也让 cold-start 保护失效。
   - **(c) 折中**：保留 confidence 的相对加权作用，但归一化其整体幅度：先用 `conf_c / Σ_c conf_c·|aw_c|` 做归一，再乘一个显式的、可配置的全局系数。

   我倾向 **(a)** —— 现有行为其实有道理，问题只在于它是隐式的、没文档、且和 LR 混在一起。

3. **第三步：写清楚**。无论选哪个，在 `PPOParams` / `CommonParams.learning_rate` 的 docstring 里说明 confidence 与 LR 的关系。

### 注意事项

- ⚠️ **这条不要和 P1-1 放同一个 PR**。P1-1 改的是 confidence 的输入，本条改的是它的输出用法。两个一起改，实验结果无法归因。
- ⚠️ 选 (b) 或 (c) 时，注意 `_normalize_adv` 已经在 per-channel 层面做过 z-score。再加一层全局归一化会让「advantage 的绝对幅度」这个信息彻底消失 —— 对某些依赖 advantage 幅度的诊断（比如判断 reward 设计是否有信号）是损失。
- ⚠️ 修改后**必须跑 A/B**：同 seed、同实验，对比修改前后的 `combined_adv_std` 曲线和最终 eval 指标。

### 验收标准

- `__RAW_STATS__` 里能看到 `combined_adv_std` 随 update 的变化曲线
- README 有一节明确说明 confidence 与有效学习率的关系
- 若改了公式：附一份 A/B 实验报告（至少 2 个 seed）

---

## P1-4. `ExplorationSpec` 的代码与两份文档完全不一致

### 问题是什么

`ExplorationSpec` 的实际字段只有两个：

```
$ python3 -c "import dataclasses; ...; print([f.name for f in dataclasses.fields(ExplorationSpec)])"
['uncertainty_floor', 'uncertainty_coef']

$ ExplorationSpec(explore_factor=0.0, uncertainty_floor=0.3)
TypeError: ExplorationSpec.__init__() got an unexpected keyword argument 'explore_factor'

$ hasattr(ExplorationSpec, 'resolve')
False
```

但文档到处都在用 `explore_factor` 和 `resolve()`：

| 位置 | 问题 |
|---|---|
| `GUIDE.md` §3.4 | 示例 `ExplorationSpec(explore_factor=0.0, ...)` → TypeError |
| `GUIDE.md` §4 最小示例 | 同上，**照抄 GUIDE 写实验的人第一分钟就撞墙** |
| `GUIDE.md` §5.2 | 探索调度示例全部 TypeError |
| `DESIGN_unified_exploration_control.md` §4 | 给出的 dataclass 定义和 `resolve()` 方法都不存在 |
| `DESIGN` §2.2 数据流图 | `resolve() → build_jobs(explore_factor=ef)` 这条链路已不成立 |
| `DESIGN` §3.4 | 声称 `uncertainty_coef` 默认联动 `0.01 × max(ef,0)`，代码里硬编码 `0.0` |
| `experiment.py` L164-193 | `ExplorationSpec` 自己的 docstring 还在讲 `explore_factor`，而 **同一文件** L639-661 的 `exploration()` docstring 又明确说它不在 spec 里 —— 自相矛盾 |

### 根因

这是 `explore_factor` 从 `ExplorationSpec` 迁移到 `Job.explore_factor_a/b` 时的**半途重构**：代码改完了，文档只改了一处（`exploration()` 的 docstring），其他全部留在旧世界。

### 附带影响

`loop.py` 的 `save_run_config` 依赖 `dataclasses.asdict(initial_spec)` 来保证可复现性（L113-129 的注释专门解释了这一点）。字段一变，`config.json` 里的 `initial_exploration` 就**静默丢失了探索配置** —— 而这正是那段注释想要防止的事情。

### 修复关键点

1. **先做决策：`explore_factor` 到底属于哪里？**
   - 当前实现（在 `build_jobs` 里、写进 `Job`）的**优点**是支持 per-job / per-agent / per-frame 的不同值，比单个 spec 字段表达力强。
   - **缺点**是「探索意图」被分散到了两个地方（`exploration()` 管 uncertainty，`build_jobs()` 管 explore_factor），破坏了 DESIGN 文档里「统一探索控制」这个卖点，而且实验必须自己在 `build_jobs` 里读内部状态。
   - **建议**：保留当前实现，但在 `ExplorationSpec` 里加回一个**可选的** `explore_factor: Optional[float]`，作为「默认值/建议值」传给 `build_jobs`，实验可以覆盖。这样既保留表达力，又恢复了单一入口和 `config.json` 的完整性。

2. **不管选哪条，把三份文档改成一致**，并且**以代码为准**：
   - `experiment.py` 的 `ExplorationSpec` docstring
   - `GUIDE.md` §3.4 / §4 / §5.2
   - `DESIGN_unified_exploration_control.md` §2.2 / §4 / §3.4

3. **`uncertainty_coef` 的默认值要么实现，要么删掉文档里的说法**。当前 `exploration()` docstring 说默认 0.0（正确），DESIGN §3.4 说联动 explore_factor（未实现）。二者必须统一。

4. **给 `ExplorationSpec` 加字段校验**（`__post_init__`）：`uncertainty_floor` 必须在 [0,1]，`uncertainty_coef >= 0`。当前完全没有校验，用户传 `uncertainty_floor=30`（误以为是百分数）会静默产生巨大的 hinge loss。

### 注意事项

- ⚠️ **文档修复的根本解法是让示例可执行**，见 P1-5。手工同步文档必然再次漂移。
- ⚠️ 改 `ExplorationSpec` 字段会影响 `config.json` 的 schema。如果有工具在解析历史 `config.json`，要考虑兼容。
- ⚠️ `trainer.py` L445-449 用 `exploration.uncertainty_floor or 0.0` 处理 None。注意 **`or` 对 `0.0` 也会取右值** —— 这里恰好无害（0.0 or 0.0 = 0.0），但这是个危险的 pattern，建议改成显式 `if ... is None`。审查代码里其他 `or` 兜底的地方是否有同样问题。

### 验收标准

- `GUIDE.md` 和两份 DESIGN 里的**每一段** `ExplorationSpec` 相关代码都能直接跑通
- `python3 -c "from ...; ExplorationSpec(uncertainty_floor=30)"` 抛出带明确说明的 `ValueError`
- P1-5 的文档示例测试覆盖 `exploration()` 的用法

---

## P1-5. `GUIDE.md` 的示例代码跑不起来

### 问题是什么

除了 P1-4 的 `ExplorationSpec` 之外，`GUIDE.md` §4 那个「完整的最小示例」还有：

| 问题 | 实际情况 |
|---|---|
| `from baseline.framework.ppo.policies import CriticMLP` | `CriticMLP` 在 `baseline/framework/critic_mlp.py`，不在 `ppo.policies` 里 |
| 示例代码用了 `Job(...)` | import 块里没有 `Job`（应为 `from baseline.framework.rollout.job import Job`） |
| `from baseline.framework.rollout import extract_per_step_scalar` | 需核实是否真的从这里导出 |

一个宣称「读完本指南，你就能写出一个完整的 PPO 实验并跑起来」的文档，示例连 import 都过不了。

### 修复关键点

**不要手工修文档 —— 手工同步的文档必然再次漂移。** 正确做法：

1. **把最小示例抽成真实文件**：`examples/exp_minimal.py`，它是一个能被 import、能被 `--experiment` 注册的真实实验。

2. **给它配一个 smoke 测试**（进 CI）：
   ```python
   def test_minimal_example_runs_2_updates(tmp_path):
       from examples.exp_minimal import EXPERIMENT_CLASS
       train_ppo(EXPERIMENT_CLASS(), run_dir=tmp_path, ...)  # 2 updates
   ```
   ⚠️ 这需要示例能在**没有 MuJoCo 环境**的情况下跑（用 fake rollouter），或者接受 CI 依赖 MuJoCo。前者更好，见 P2-2。

3. **`GUIDE.md` 里改成引用而非复制**：
   ```markdown
   完整可运行的最小示例见 [`examples/exp_minimal.py`](examples/exp_minimal.py)。
   下面逐段讲解其中的关键部分。
   ```
   或者用 literalinclude 式的构建步骤把文件内容注入文档。**关键是让「文档里的代码」和「被测试的代码」是同一份。**

4. **顺手核实 GUIDE 里其他所有可执行断言**：
   - §4.2 的运行命令 `PYTHONPATH=. python3 baseline/framework/train.py --experiment my_experiment --algo ppo --smoke` 是否真的可用
   - §6 的 `extract_per_step_field` / `extract_per_step_scalar` 签名是否匹配
   - §4.1 的方法清单里每个方法名、是否 abstract、调用时机是否与 `experiment.py` 一致

### 注意事项

- ⚠️ 示例的**质量标准比普通代码更高**。它是外部开发者读到的第一段代码，会被直接复制。示例里的每一个选择（`gae_lambda=0.95`、`actor_weight=3.0`、`clip_eps=0.2`）都会被当成推荐值继承下去。要么给出理由，要么标注「这里的数值仅为演示」。
- ⚠️ `GUIDE.md` §7「调试建议」那 4 条非常有价值（尤其「`is_terminated` 设错是最常见的 bug」），保留并扩充。可以把本文档发现的问题也补进去。

### 验收标准

- CI 里有一个 job 执行 `examples/exp_minimal.py` 的 2-update smoke run 并通过
- `GUIDE.md` 里不再有任何未被测试覆盖的完整代码块

---

## P1-6. `algos/ppo.py` 是第二套、且与实际使用的实现不一致的 PPO

### 问题是什么

`algos/ppo.py`（`ppo_loss` + `PPOLossOutput`）整个文件、以及 `compute_grpo_advantages`、`compute_returns_to_go`，**全仓库零引用**：

```bash
$ grep -rn "ppo_loss\|compute_grpo_advantages\|compute_returns_to_go" --include=*.py . \
    | grep -v "framework/ppo/algos"
（无输出）
```

而 `trainer.py` 是内联手写的 surrogate。两者行为**不一致**：

| | `algos/ppo.py::ppo_loss`（死代码） | `trainer.py`（实际使用） |
|---|---|---|
| value clipping | 有（Schulman recipe） | **无** |
| advantage 归一化 | minibatch 级（CleanRL 惯例） | **rollout 级 + per-channel** |
| uncertainty | entropy **bonus**（`- coef * U`） | uncertainty **floor hinge**（`+ coef * relu(floor-U)²`） |
| value loss | 合并进 total_loss | 独立 optimizer、独立 backward |
| KL 估计 | k3 | k3（一致） |

### 为什么危害不小

开源后读者会**先读 `algos/ppo.py`** —— 它 docstring 最漂亮、最像「这就是算法核心」，还写着 "matches CleanRL / SB3 conventions"。然后对着一份死代码理解错整个框架的算法行为。这会直接导致错误的 issue、错误的 PR、和「你们的 PPO 到底是哪个版本」的信任问题。

### 修复关键点

**选一条，不要都留：**

- **方案 A（推荐）：删掉 `algos/ppo.py`**，`compute_grpo_advantages` 一并删（GRPO 与本框架的多 channel 设计无关，留着是噪声）。`compute_returns_to_go` **保留** —— P1-1 的第 2 层修复需要它。
  - 删除时要更新 `algos/__init__.py` 的 `__all__`。
  - 在 CHANGELOG 里说明「这些是从未被使用的实验性代码」。

- **方案 B：让 `trainer.py` 真的调用 `ppo_loss`**。看起来更「干净」，但要先解决 5 处行为差异，且 trainer 的多 channel 结构（per-channel critic、独立 optimizer、masked loss）与 `ppo_loss` 的单 value 假设不兼容。**代价远高于收益，不建议。**

### 注意事项

- ⚠️ 删代码前 grep 确认没有测试在测它。`algos/test_ppo.py`（145 行）如果只测 `ppo_loss`，会一起删掉 —— 这会降低覆盖率数字，但**测试死代码的覆盖率本来就是虚的**。
- ⚠️ 如果团队认为 `ppo_loss` 有独立价值（比如给别人做参考实现），**移到 `examples/reference_ppo_loss.py` 并在文件顶部明确写「本文件不被框架使用，仅作教学参考」**。放在 `algos/` 下就会被当成生产代码。
- ⚠️ 同类清理：`policies/todo/` 下的 8 个策略族 + 9 份 DESIGN 文档。P0-6 已经证明「留在仓库里的半成品」会造成真实事故。

### 验收标准

- `grep -rn "ppo_loss" --include=*.py .` 无输出，或全部命中 `examples/` 且带免责声明
- `algos/__init__.py` 的 `__all__` 与实际导出一致

---

## P1-7. `hasattr(actor, "log_std")` —— 通用 trainer 里残留的策略族耦合

### 问题是什么

`trainer.py` L909-927 为了打印 log_std 的梯度诊断，在**通用** trainer 里嗅探 Gaussian 策略专有的属性：

```python
if hasattr(actor, "log_std") and mb_idx == 0:
    pol_grads = torch.autograd.grad(policy_loss, actor.log_std, retain_graph=True, ...)
    floor_grads = torch.autograd.grad(floor_loss, actor.log_std, retain_graph=True, ...)
```

讽刺的是，L750-755 的注释刚刚在夸自己把这类代码删掉了：

> *"The trainer used to compute them by reaching into `actor.log_std` directly, which (a) crashed for any actor without that exact attribute and (b) re-derived sigma with the trainer's own bounds instead of the actor's..."*

同一个文件，一边宣布已经解耦，一边留着新的耦合。

### 附带代价

- 每个 update 多两次 `torch.autograd.grad(retain_graph=True)`，对 mixture / flow 类策略是纯浪费（虽然 `hasattr` 会跳过）
- `retain_graph=True` 会让计算图多存一份，增加显存峰值
- 对没有 `log_std` 的策略族，这个诊断静默消失，用户不知道为什么日志里没有 `[GradDiag]`

### 修复关键点

1. **把诊断的所有权交还策略**。`ActorEval.stats` 已经是为此设计的通道（"policy-contributed, no contract"）。但注意：**梯度诊断需要 loss，而 loss 在框架侧** —— 所以不能简单搬走。

2. **推荐做法：给 `TrainablePolicy` 加一个可选的钩子**：
   ```python
   def exploration_grad_diagnostics(
       self, policy_loss: torch.Tensor, floor_loss: torch.Tensor,
   ) -> Optional[Dict[str, float]]:
       """可选：报告两个 loss 对策略探索参数的梯度贡献。

       默认返回 None（不报告）。策略自己决定「探索参数」是什么
       （Gaussian 的 log_std、mixture 的温度、flow 的 scale...）。
       """
       return None
   ```
   trainer 侧变成：
   ```python
   if mb_idx == 0 and uncertainty_coef > 0.0:
       d = actor.exploration_grad_diagnostics(policy_loss, floor_loss)
       if d: ...collect...
   ```
   框架不再知道 `log_std` 存在。

3. **或者更简单：把这段挪进一个 debug flag 后面**（`ppo_update(..., grad_diagnostics=False)`），默认关闭。这段代码明显是为了排查「uncertainty floor 为什么不起作用」而临时加的脚手架，不该是常开的生产路径。

### 注意事项

- ⚠️ 这个诊断**确实有用**（`floor_abs / pol_abs` 的比值能直接看出 floor 有没有被 advantage 压过去）。不要直接删掉，要迁移。
- ⚠️ `allow_unused=True` + `if pol_grads is not None` 的写法说明作者已经知道这里不可靠。迁移后新接口应该让「不支持」变成显式的 `return None` 而不是 `allow_unused`。
- ⚠️ 迁移后确认 `[GradDiag]` 日志行格式不变，否则会破坏已有的日志解析脚本。

### 验收标准

- `grep -n "log_std" trainer.py` 无输出
- 用 `TruncatedNormalPolicy` 跑 smoke，`[GradDiag]` 行依然出现且数值与修改前一致
- 用一个没有 `log_std` 的 fake 策略跑 smoke，不崩、不报错

---

## P1-8. `evaluate_actions` 里多算了一整遍 `net(obs)`

### 问题是什么

`policies/truncated_normal_mlp.py` L309-331：

```python
mean, eff_sigma = self.forward(obs, explore_factor=explore_factor)   # 内部：tanh(self.net(obs))
...
policy_mean = torch.tanh(self.net(obs))    # ← 与上面的 mean 数值完全相同
```

`mean` **不依赖** `explore_factor`（`explore_factor` 只影响 sigma），所以 `policy_mean is mathematically identical to mean`。这是每个 minibatch 一次**多余的完整 forward + 反向图**，actor 的训练开销直接翻倍。

### 修复关键点

一行改动：

```python
policy_mean = mean   # mean 与 explore_factor 无关，直接复用
```

### 注意事项

- ⚠️ **改之前先确认「mean 与 explore_factor 无关」这个性质在当前实现下成立**。看 `forward()`：`raw_mean = self.net(obs); mean = tanh(raw_mean); sigma = effective_sigma(ef)`。是的，成立。
  但这是**实现细节**，不是接口保证。如果未来有策略让 explore_factor 也影响 mean（比如注入定向噪声），这个复用就错了。建议加注释说明依赖：
  ```python
  # 复用 forward() 的 mean：本策略的 mean 不依赖 explore_factor
  # （explore_factor 只缩放 sigma）。若未来改变这个性质，此处必须同步。
  policy_mean = mean
  ```
- ⚠️ 改完要跑 `policies/test_truncated_normal.py`（300 行）确认 uncertainty 数值不变。这是纯性能优化，**任何数值变化都说明改错了**。
- ⚠️ 用 `loop.py` 已有的 timing 输出（`ppo=X.XXs`）量化收益。如果 `ppo` 时间占比本来就很小（被 `rollout` 主导），这条的优先级可以降低。

### 验收标准

- `test_truncated_normal.py` 全绿，且 uncertainty 的数值与修改前**逐位相同**
- smoke run 的 `ppo=` 时间下降（记录修改前后的数字）

---

## P1-9. `compute_gae` 是逐步 Python 循环

### 问题是什么

`algos/advantages.py` L129-136 是一个逐时间步的 Python for 循环。规模估算：

```
256 episodes × ~200 steps × N channels = 每 update 几十万次 Python 迭代
```

`compute_returns_to_go` 同样（P1-1 第 2 层修复会让它也进热路径）。

### 修复关键点

1. **先量化，再优化**。`loop.py` 已经打出了 `ppo=X.XXs` / `rollout=X.Xs`。如果 `ppo` 只占总时间的 5%，这条可以直接降级到 P2。**不要凭直觉优化。**

2. **向量化方案**：GAE 的递推 `A_t = δ_t + γλ·A_{t+1}` 是一个反向的一阶线性递推，可以用「乘以折扣幂 + 反向累积和 + 除以折扣幂」实现：
   ```python
   delta = rewards + gamma * np.append(values[1:], last_value) - values
   coef = (gamma * lam) ** np.arange(T)
   adv = np.cumsum((delta / coef)[::-1])[::-1] * coef
   ```
   ⚠️ **这个写法在 `T` 大、`γλ` 小的时候会数值爆炸**（`1/coef` 增长到 1e30+）。必须用 `float64` 中间计算，并对 `T` 设上限，或者改用 `scipy.signal.lfilter`（数值稳定的一阶 IIR 滤波，这是 OpenAI baselines 的经典做法）：
   ```python
   from scipy.signal import lfilter
   adv = lfilter([1.0], [1.0, -gamma * lam], delta[::-1])[::-1]
   ```
   `scipy` 已经是本项目的依赖（见 `pyproject.toml`），可以直接用。

3. **更大的收益可能在别处**：把 per-channel、per-segment 的双层循环批量化（把所有 segment padding 成 `(n_segs, T_max)` 矩阵一次算完），而不是只优化单条的内循环。

### 注意事项

- ⚠️ **GAE 是整个框架的算法核心，改它风险最高**。必须：
  - 保留原实现为 `_compute_gae_reference()`
  - 加一个 property-based 测试（用 `hypothesis` 或手写随机用例）断言新旧实现在 1e-5 内一致，覆盖 `λ∈{0, 0.5, 1}`、`γ∈{0, 0.9, 1}`、`T∈{1, 2, 1000}`、`last_value≠0`
  - 现有的 4 个 GAE 测试（`test_gae_lam_one_monte_carlo` / `test_gae_lam_zero_td0` 等）是很好的基础，保留
- ⚠️ `λ=0` 时 `coef = 0**arange(T)` 会产生 0 和除零。边界必须特殊处理（λ=0 时 `adv = delta`）。这是向量化 GAE 最经典的 bug。
- ⚠️ `advantages.py` 现在的注释质量很高（GAE 数学推导、terminated/truncated 语义），**优化时不要把注释删掉**。

### 验收标准

```python
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("gamma", [0.0, 0.9, 0.99, 1.0])
@pytest.mark.parametrize("T", [1, 2, 7, 1000])
def test_gae_vectorized_matches_reference(lam, gamma, T):
    ...
    np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-6)
```
+ 记录优化前后的 `ppo=` 时间

---

## P1-10. eval 的种子每轮都变，`is_new_best` 在比较不同的随机试验

### 问题是什么

`loop.py` L530：

```python
eval_seed = cp.seed + 100_000 + u * 97
```

每个 eval 轮用**不同**的种子集合。后果：

1. **跨 update 的 eval 分数不可比**。`on_eval` 里的 `is_new_best` 很大程度上在选「运气最好的那一轮」，尤其在 `eval_episodes=16` 这种小样本下。
2. **系统性高估 best 策略**：取 N 次含噪声测量的最大值，期望值高于真实最优。导出的「best-of-run」策略实际性能会低于日志显示的分数。
3. **课程阶段推进在噪声上触发**：`GUIDE.md` §5.1 的示例 `if survival_rate > 0.8 and self._phase == 0: self._phase = 1` —— 一次运气好的 eval 就会永久推进阶段（`_phase` 还会被 `state()` 持久化）。

### 修复关键点

1. **固定 held-out eval 种子集**：
   ```python
   # eval 使用固定种子集，使跨 update 的分数可比。
   # 与训练种子空间（cp.seed + u * episodes_per_update）不重叠。
   eval_seed = cp.seed + 1_000_000
   ```
   ⚠️ 注意 `build_jobs(bp, base_seed, n)` 的契约是「每个 job 用 `base_seed + i`」，所以固定 `base_seed` 就得到固定的 N 个种子。要检查各实验的 `build_jobs` 是否真的遵守这个契约（有没有额外引入随机性，比如 `random.choice` 选对手）。

2. **另设一个可变种子集用于泛化诊断**（可选）。固定集用于 best 选择和阶段推进；可变集用于「是否过拟合到这 16 个种子」的诊断。两组分数都进日志。

3. **在 `GUIDE.md` 里明确警告阶段推进的噪声风险**，建议用连续 K 次达标（而非单次）作为推进条件：
   ```python
   if survival_rate > 0.8:
       self._consecutive_good += 1
   else:
       self._consecutive_good = 0
   if self._consecutive_good >= 3 and self._phase == 0:
       self._phase = 1
   ```

### 注意事项

- ⚠️ **固定种子会引入过拟合风险**：策略可能学会利用这 16 个特定初始状态。`eval_episodes` 太小时这个风险是真实的。建议 `eval_episodes` 至少几十，或者用「固定的大种子集 + 每轮取固定子集轮转」。这是一个需要权衡的决策，**要在文档里说明选择和理由**。
- ⚠️ 改动会让新旧 run 的 eval 分数不可比。CHANGELOG 要写明。
- ⚠️ `rollout_seed = cp.seed + u * cp.episodes_per_update`（L469）**不要改** —— 训练数据每轮应该用新种子，这是对的。注意这里存在一个潜在的种子空间重叠：如果 `eval_seed = cp.seed + 100_000` 而训练跑到 `u = 100_000 / episodes_per_update` 轮，两者会撞上。用 `1_000_000` 或分离的种子命名空间（如高位偏移）更安全。

### 验收标准

```python
def test_eval_seeds_are_stable_across_updates():
    """不同 update 的 eval job 种子集合必须相同。"""
    seeds_u10 = [j.seed for j in build_eval_jobs_at_update(10)]
    seeds_u20 = [j.seed for j in build_eval_jobs_at_update(20)]
    assert seeds_u10 == seeds_u20

def test_eval_and_train_seed_spaces_disjoint():
    """训练与 eval 的种子空间在 max_updates 范围内不重叠。"""
```

---

## P1-11. `SIGINT`/`SIGTERM` 直接 `SIGKILL` 自己的进程组

### 问题是什么

`loop.py` L367-370：

```python
def _shutdown_handler(signum, frame):
    os.killpg(os.getpgrp(), signal.SIGKILL)
signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
```

两个问题：

1. **`SIGKILL` 自己 → Ctrl-C 一定丢掉最后一个 checkpoint**。checkpoint 只在 `u % eval_interval == 0` 时保存，所以最坏情况会丢掉 `eval_interval - 1` 轮的训练成果。对一个跑几十小时的训练，这很痛。
2. **在没有 job control 的环境下会杀掉调用者**。`sh -c "python train.py"`、CI runner、某些容器 entrypoint 下，`os.getpgrp()` 可能就是父进程组，`killpg` 会把调用者一起干掉。`--background` 模式用了 `setsid` 所以安全，**前台模式在非交互 shell 下有风险**。

### 修复关键点

改成两段式优雅停机：

```python
_stop_requested = False

def _shutdown_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        # 第二次信号：强制退出
        print("[shutdown] second signal, forcing exit", flush=True)
        os._exit(130)
    _stop_requested = True
    print(f"[shutdown] signal {signum} received; will save checkpoint and exit "
          f"after current update. Send again to force.", flush=True)
```

主循环末尾：

```python
if _stop_requested:
    save_checkpoint(ckpt_dir / f"checkpoint_u{u:05d}.pt", ..., update=u)
    print(f"[shutdown] checkpoint saved at update {u}, exiting", flush=True)
    break
```

worker 清理交给 `ParallelRollouter.__exit__`（`with` 语句已经保证会执行）。

### 注意事项

- ⚠️ **必须验证 `ParallelRollouter.__exit__` 真的能清干净子进程**。原来那句 `killpg(SIGKILL)` 存在的理由很可能就是「worker 会变成孤儿进程」。先读 `parallel_rollouter.py` 的清理逻辑，实测 `Ctrl-C` 后 `ps aux | grep` 确认没有残留。**如果 `__exit__` 不可靠，先修它，再改信号处理。**
- ⚠️ 一轮 update 可能要几分钟（rollout 主导）。用户 Ctrl-C 后要等一轮才退出，体验上要有明确提示（上面的 print 已包含），并且第二次信号要能立刻强杀。
- ⚠️ 信号处理器里**不要**做 I/O 之外的复杂操作（不要在里面存 checkpoint）。Python 的信号处理器在主线程的字节码间隙执行，在里面调 `torch.save` 可能遇到重入问题。只设 flag，在主循环里做事。
- ⚠️ `--background` 模式下 `SIGTERM` 是标准停止手段（`CLAUDE.md` 里写的 `kill <pid>`）。改完要确认这个路径依然能停下，且现在还能顺便存 checkpoint —— 这是净收益。

### 验收标准

- 前台跑 smoke，Ctrl-C 后：进程退出、checkpoint 已写、`ps` 无残留 worker
- 后台跑，`kill <pid>` 后同上
- 连按两次 Ctrl-C 能立刻退出

---

## P1-12. 磁盘无界增长

### 问题是什么

- `policy_exports/u{N:05d}/` **每个 update 都写一个目录**（L459），eval 轮再多一个 `u{N:05d}_eval`（L531）。`max_updates=5000` → 上万个目录，从不清理。
- `checkpoints/` 每 `eval_interval` 存一个，也不清理。
- `videos/` 每次都留 `.mp4` + `.log`。

`GUIDE.md` 完全没提这件事。`CLAUDE.md` 只说了 "runs/ 可以非常大"。

### 修复关键点

1. **`CommonParams` 加保留策略参数**：
   ```python
   keep_last_exports: int = 3          # 只保留最近 N 个 policy_exports
   keep_last_checkpoints: int = 5      # 最近 N 个
   keep_every_n_checkpoints: int = 0   # 额外保留每 N 轮的（0=不额外保留）
   ```

2. **实现一个 `_prune_dir(dir, keep_last, keep_every_n)` 工具函数**，在每次写完 export / checkpoint 后调用。

3. **`policy/`（best-of-run）永不删除**。这是最重要的产物。

### 注意事项

- ⚠️ **默认值要保守**。有人可能依赖历史 export 做分析或 resume。建议默认 `keep_last_exports=3`（export 只用于当轮 rollout，几乎没有保留价值），但 `keep_last_checkpoints` 默认给大一些（比如 20），并保留「每 100 轮留一个」的长期锚点。
- ⚠️ **删除操作要极度小心**。只删 `run_dir/policy_exports/` 和 `run_dir/checkpoints/` 下**符合命名模式**（`u\d{5}` / `checkpoint_u\d{5}\.pt`）的条目。绝不用 `shutil.rmtree` 删拼接出来的路径而不校验。写单元测试覆盖「目录里有意外文件时不删它」。
- ⚠️ resume 依赖 checkpoint 存在。如果用户 `--resume-from` 指向一个已被 prune 掉的 checkpoint，要给出清晰错误。
- ⚠️ 考虑加一个 `--no-prune` 开关给需要完整历史的场景。

### 验收标准

```python
def test_prune_keeps_last_n_and_ignores_unknown_files():
    # 造 10 个 checkpoint + 1 个 README.md，prune(keep_last=3)
    # 断言：留下 3 个 checkpoint + README.md 完好
```
+ smoke run 后 `policy_exports/` 目录数 ≤ `keep_last_exports`

---

## P1-13. 缺少 on-policy PPO 的标配组件（或缺少「不做」的声明）

### 问题是什么

框架**没有**：

| 组件 | 状态 | 影响 |
|---|---|---|
| observation normalization（running mean/std） | 无 | 用户换个 obs scale 就训不动，且不知道为什么 |
| reward / return scaling | 无 | 各 channel 的 reward 量级差异靠 per-channel z-score 兜住了（这是好设计），但 critic 的回归目标量级仍受影响 |
| LR / clip_eps 退火 | 无 | 后期训练不稳 |
| value loss clipping | `ppo_loss` 里写了但**没被用** | 见 P1-6 |
| `actor.train()` / `.eval()` 切换 | **从不调用** | 当前策略无 BN/Dropout 所以没爆；任何带 BN 的策略会静默错误 |
| gradient accumulation | 无 | minibatch 必须放得进显存 |

### 为什么这是个问题

**不是「必须全部实现」**，而是**必须明确声明**。外部用户会默认一个「PPO 框架」做了这些（SB3、CleanRL 都做），然后：
- 换个 observation scale 训不动，怀疑是自己的 reward 设计问题，浪费几天
- 提 issue 问「为什么没有 obs normalization」
- 或者更糟：以为框架做了，于是自己也不做

### 修复关键点

**第一优先：写清楚。** 在 README 加一节：

```markdown
## 框架不负责的事（Non-goals）

本框架刻意不实现以下常见组件，它们由实验（Experiment）自行负责：

- **Observation normalization**：框架不做任何 obs 变换。若你的 obs 量级不在
  O(1)，请在 `build_actor` 返回的策略内部做归一化（推荐把 running mean/std
  作为 buffer 存进 `state_dict`，这样导出的策略自带归一化，部署时行为一致）。
- **Reward scaling**：框架对每个 channel 的 advantage 做 z-score 归一化，
  所以**跨 channel 的 reward 量级差异会被自动吸收**。但单个 channel 内部的
  reward 量级仍会影响该 channel 的 critic 回归难度。
- **LR / clip 退火**：`CommonParams` 是 frozen dataclass，每个 update 重新读取，
  所以实验可以通过在 `common_params()` 里返回随 update 变化的值来实现退火。
  （⚠️ 需核实：loop 只在启动时调用一次 `common_params()`，见下方注意事项）
- ...
```

**第二优先：`actor.train()` / `.eval()`。** 这是一个 3 行的正确性修复：
- `ppo_update` 开始时 `actor.train()`，各 critic 也 `.train()`
- `PPOBuffer` 的批量 `evaluate_actions`（θ_old 测量点）应该在 `.eval()` 下
- 结束后恢复

**第三优先：value loss clipping**。如果 P1-6 选了删除 `ppo_loss`，考虑把 value clipping 加进 `trainer.py` 的 critic 更新（作为可选的 `PPOParams.value_clip`）。

### 注意事项

- ⚠️ **上面 README 草稿里关于「退火」的说法需要先核实**。看 `loop.py` L359-361：`cp = experiment.common_params()` 只在**循环外**调用一次。所以**当前实现下实验无法通过 `common_params()` 实现退火**。要么改成每轮重读（注意 `max_updates` 等字段每轮变会很怪），要么在文档里明确说「不支持，请自己在 optimizer 上改 LR」—— 但实验拿不到 optimizer。这是一个**真实的能力缺口**，需要单独决策（比如给 `ExperimentPPO` 加一个 `on_update` 之外的 `adjust_optimizers(actor_opt, critic_opts, update)` 钩子）。
- ⚠️ obs normalization 若要加，**必须让归一化参数随策略一起导出**。否则训练时归一化、部署时不归一化，是又一个静默失败（和 P0-5/P0-6 同类）。推荐用 `register_buffer` 让它自动进 `state_dict`。
- ⚠️ 不要为了「功能对齐 SB3」而堆功能。这个框架的价值在于多 channel + 课程控制的清晰抽象，**声明边界比扩大边界更有价值**。

### 验收标准

- README 有 "Non-goals" 一节，逐条说明并给出替代方案
- `actor.train()/.eval()` 切换已实现，并有一个用带 Dropout 的 fake 策略的测试验证行为差异
- 「实验如何做 LR 退火」有明确答案（实现钩子 or 文档说明不支持）

---

## P1-14. `TruncatedNormalPolicy` 的 action bound 硬编码，rollout 用全局 RNG

### 问题是什么

**(a) action bound 硬编码**（L130-133）：

```python
_ACTION_LOW = -1.0
_ACTION_HIGH = 1.0     # 注释诚实地写了 "hardcoded for humanoid21"
```

模块级常量，无法用于其他 action space。

**(b) 全局 RNG 污染**：
- `sample_action` 用 `torch.rand_like`（走全局 torch RNG）
- 导出策略的 `reset(seed)` 调 `torch.manual_seed(seed)` —— **在 worker 进程里污染全局状态**，影响该 worker 后续所有 job 的随机性

**(c) `uncertainty ∈ [0,1]` 的契约没有断言**。实测 `log_std=3.0 → U=0.9996`，行为正确，但靠约定而非检查。

### 修复关键点

**(a)**：从 action space 读边界。构造时接受 `action_low` / `action_high`，存为 buffer（这样会随 `state_dict` 导出，部署时一致）：
```python
self.register_buffer("action_low", torch.full((action_dim,), float(action_low)))
self.register_buffer("action_high", torch.full((action_dim,), float(action_high)))
```
⚠️ 这会改变 `state_dict` 的键集合 → 与 P0-5 的 `strict=True` 冲突 → **必须同时 bump `format_version` 并提供迁移**。建议和 P0-5/P0-6 同一批做。

**(b)**：每个策略实例持有自己的 `torch.Generator`：
```python
self._gen = torch.Generator(device=...)
def reset(self, seed=None):
    if seed is not None:
        self._gen.manual_seed(int(seed))
```
`torch.rand_like(mean)` → `torch.rand(mean.shape, generator=self._gen, device=..., dtype=...)`。

**(c)**：在 `ActorEval` 构造处（或 debug 模式下）断言：
```python
if __debug__:
    assert torch.all((uncertainty >= 0) & (uncertainty <= 1 + 1e-5)), ...
```
更好的做法是在 **框架侧** 校验（`trainer.py` 消费 `actor_eval.uncertainty` 处），因为这是框架对策略的契约要求，不该依赖每个策略自觉。

### 注意事项

- ⚠️ **`torch.Generator` 与 device 绑定**。CPU generator 不能用于 CUDA tensor。要么按 device 建 generator，要么统一在 CPU 上采样再搬（rollout 在 CPU，训练不采样，所以实际影响小）。
- ⚠️ 改随机数生成方式会**改变所有历史 run 的可复现性**。这是可接受的（换来的是真正的可复现），但要在 CHANGELOG 明确写。
- ⚠️ `_EXPLORE_K = log(3)` 这个映射（ei=±1 → σ×3 或 ÷3）是硬编码的策略族选择。它本身合理（框架只规定 `[-1,1]` 和中性点 0，具体映射归策略），但**应该做成构造参数**，让用户能调探索强度的动态范围。
- ⚠️ `policies/todo/` 下的其他策略族有同样问题，一并检查（但不要在这个 PR 里改 todo/ 的代码，见 P0-6 对该目录的处置决策）。

### 验收标准

```python
def test_sample_is_reproducible_without_touching_global_rng():
    torch.manual_seed(0); before = torch.rand(1).item()
    p = TruncatedNormalPolicy(8, 3, 16); p.reset(seed=123)
    p.sample(np.zeros(8, dtype=np.float32))
    torch.manual_seed(0); after = torch.rand(1).item()
    assert before == after          # 全局 RNG 未被污染

def test_uncertainty_in_unit_range():
    for ls in (-20.0, -1.0, 0.0, 3.0, 20.0):
        ...  assert 0.0 <= U <= 1.0
```

---

# P2 —— 开源可用性

---

## P2-1. 这个包无法被安装或独立使用（开源的头号阻塞项）

### 问题是什么

所有 import 都是仓库根的绝对导入：

```python
from baseline.framework.ppo.algos import compute_gae
from envs.framework.policy import Policy, PolicyBlueprint
```

而 `pyproject.toml` 的 `[tool.setuptools.packages].explicit` 列表里**根本没有 `combatbench.baseline.framework`**（只有过时的 `combatbench.baseline.common.*`、`combatbench.baseline.humanoid21.curriculum.framework`）。

同时 `package-dir = {"combatbench" = "."}` 意味着装完之后包名是 `combatbench.baseline.*`，而**代码里写的是 `baseline.*`** —— 两者不一致。

结果：
- 只能靠 `PYTHONPATH=<repo root>` 运行
- `pip install -e .` 装不到这个框架
- `CLAUDE.md` 里那段 *"CRITICAL — PYTHONPATH requirement…否则 `import baseline` 会静默失败（尤其在 `--background` 模式下 stderr 被重定向，错误不可见）"* 就是这个问题的症状 —— 一个需要在文档里用 CRITICAL 警告的安装方式，本身就是设计问题

**外部开发者的第一个动作是 `pip install`，然后 import 失败。**

### 修复关键点

1. **决定包的顶层名字并全仓库统一**。两个方向：
   - **(a) 统一到 `combatbench.*`**：把所有 `from baseline...` / `from envs...` 改成 `from combatbench.baseline...` / `from combatbench.envs...`。改动量大（几百处）但一次到位，且是标准做法。可以用 `ast`-based 工具或 `sed` + 人工审查。
   - **(b) 把 `baseline/` 和 `envs/` 变成真正的顶层包**：去掉 `package-dir` 映射，让 `baseline` 和 `envs` 直接成为顶层可安装包。改动小，但占用了两个非常通用的顶层名字（`envs`、`baseline`），在别人的环境里极易冲突。**不推荐用于开源。**

   **建议 (a)**。

2. **`packages` 改用自动发现**，避免手工列表漂移：
   ```toml
   [tool.setuptools.packages.find]
   include = ["combatbench*"]
   ```
   当前的 explicit 列表已经和目录结构脱节（列了不存在的 `baseline.common.*`，漏了 `baseline.framework.*`），这正是手工维护列表的必然结局。

3. **验证安装可用性**（这是唯一可靠的验收方式）：
   ```bash
   python3 -m venv /tmp/v && /tmp/v/bin/pip install -e .
   cd /tmp && /tmp/v/bin/python -c "from combatbench.baseline.framework.ppo import ExperimentPPO"
   ```
   注意 `cd /tmp` —— **必须在仓库外测试**，否则当前目录会掩盖问题。

4. **CI 里加一个 install 测试**，见 P2-3。

### 注意事项

- ⚠️ **`build-backend` 当前就是坏的，已确认**：
  ```
  $ python3 -c "import importlib; importlib.import_module('setuptools.backends._legacy')"
  ModuleNotFoundError: No module named 'setuptools.backends'
  ```
  `pyproject.toml` 里写的 `build-backend = "setuptools.backends._legacy:_Backend"` 指向一个不存在的模块（标准值是 `setuptools.build_meta`）。**这证明 `pip install` 从未被真正测试过。** 修这一行是本条目的第 0 步。
- ⚠️ 改 import 路径会影响**已导出的 81 个 artifact**（它们 import `baseline.framework.ppo.policies...`）。这与 P0-6 是同一个问题的两面：**先做 P0-6 让 artifact 自包含，再改 import 路径**，顺序不能反。
- ⚠️ 改完之后 `PYTHONPATH=.` 的运行方式**仍然应该能用**（开发时方便），两者不冲突。
- ⚠️ `CLAUDE.md` 里的所有训练命令示例都要同步更新。

### 验收标准

- 在**仓库外**的干净 venv 里 `pip install -e .` 后能 `from combatbench.baseline.framework.ppo import ExperimentPPO`
- CI 有一个 job 做这件事
- `pyproject.toml` 不再有手工维护的 packages 列表

---

## P2-2. 框架与具体环境强耦合，没有任何标准环境上的收敛证据

### 问题是什么

框架的核心模块直接依赖 CombatBench 环境：
- `experiment.py` / `loop.py` import `envs.framework.blueprint`
- `Job` 写死了 `policy_a_bp` / `policy_b_bp`（双智能体对战假设）
- `_spawn_video_render` 直接 `python -m envs.framework.round_runner`

所以**它无法在 Gym/MuJoCo 标准环境上做 sanity check**。

**这意味着：没有一个「CartPole / Pendulum 上 5 分钟收敛」的 reference run。** 外部开发者既无法验证安装成功，也无法信任这个自研 PPO 实现是对的。

### 为什么这是取得信任的最低门槛

对一个自研 RL 算法实现，社区的第一个问题必然是：**「怎么知道你的 PPO 是对的？」**

CleanRL 的全部说服力来自于「每个实现都附带在标准 benchmark 上与论文对齐的曲线」。没有这个，任何算法讨论都无法进行 —— 因为无法区分「设计有问题」和「实现有 bug」。而本文档已经找出 6 个正确性 bug，恰好证明了这个担忧是合理的。

### 修复关键点

1. **抽一层 rollout 后端协议**。定义框架真正需要的最小接口：
   ```python
   class RolloutBackend(Protocol):
       def collect(self, jobs: Sequence[JobLike]) -> List[Episode]: ...
   ```
   `loop.py` 依赖这个协议而不是 `ParallelRollouter` 具体类。

2. **提供一个 Gym 单智能体后端**（`examples/gym_backend.py`）：把 `gymnasium` 环境包成产出 `Episode` 的形式。
   ⚠️ 需要核实 `Episode` 的字段（`observations[agent_id]`、`agent_termination_reason`、`observer_outputs`）能否被单智能体场景自然填充。如果强耦合到多智能体 + observer plugin，这一步的工作量会显著上升 —— 那本身就是重要的架构发现。

3. **写 `examples/exp_pendulum.py`**：单 channel、单 agent、标准 `Pendulum-v1`。它同时充当：
   - reference run（README 里放收敛曲线）
   - CI 的端到端测试（可以只跑 2 update 做 smoke）
   - P1-5 的可执行示例
   - 外部开发者的 hello world

4. **README 里放曲线和对齐说明**：「Pendulum-v1，3 seeds，200 updates，达到 -200 左右，与 SB3/CleanRL 同量级」。

### 注意事项

- ⚠️ **这是本文档里工作量最大的一项**，但也是「可开源」和「不可开源」的分界线。如果时间有限，**优先做第 3 步的简化版**：哪怕先用一个纯 numpy 的 fake 环境（比如一个已知最优解的 LQR 问题）做端到端验证，也远好于什么都没有。LQR 的好处是 V(s) 有闭式解，可以直接验证 critic 的正确性。
- ⚠️ 不要为了解耦而过度抽象。只抽 `loop.py` 真正用到的那几个调用（`collect`），不要设计一个大而全的后端框架。
- ⚠️ 单 channel 的 Pendulum 例子会暴露一个有趣的问题：**单 channel 时 L1 归一化让 `aw` 恒为 1**（`aw/|aw| = 1`），所以 `actor_weight` 完全失效。这是正确行为（已有 `test_l1_normalization_single_channel_unchanged` 覆盖），但要在文档里说明，否则用户会以为参数没生效。

### 验收标准

- `examples/exp_pendulum.py` 能跑，README 有 3-seed 收敛曲线
- CI 跑它的 2-update smoke
- `loop.py` 不再直接 import `ParallelRollouter` 具体类

---

## P2-3. 没有 CI，没有 lint，有 5 处 undefined name

### 问题是什么

仓库没有 `.github/`（已确认）。`pyflakes` 一跑出 20 条，其中 **5 条是真实的 undefined name**：

```
experiment.py:539  undefined name 'RewardChannel'
experiment.py:707  undefined name 'Episode'
experiment.py:707  undefined name 'Trajectory'
experiment.py:748  undefined name 'Episode'
trainer.py:403     undefined name 'ExplorationSpec'
```

以及一批未使用的 import：`experiment.py` 的 `numpy` / `EnvBlueprint`、`loop.py` 的 `PPOParams` / `TrainablePolicy`、`trajectory.py` 的 `field` / `Any` / `List` / `Tuple`、`truncated_normal_mlp.py` 的 `StochasticPolicy`。还有 2 处 `f-string is missing placeholders`（`loop.py:262` / `595`）。

### 根因与影响

这些类型注解靠 `from __future__ import annotations` 侥幸不在运行时炸（注解变成字符串，不求值）。但会破坏：
- `typing.get_type_hints()` → 任何运行时类型校验、pydantic、序列化工具
- IDE 的跳转和补全（对外部开发者的体验影响很大）
- `mypy` / `pyright` 静态检查

`trainer.py:403` 尤其典型：`ppo_update(..., exploration: Optional[ExplorationSpec] = None)` 的注解引用了一个从未 import 的名字。而**同一个函数的返回注解写的是 `Dict[str, float]`，实际返回 `UpdateStats`** —— 这是另一处需要修的注解错误（本条顺手修）。

### 修复关键点

1. **修 undefined name**：用 `TYPE_CHECKING` 块导入
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from baseline.framework.rollout.episode import Episode
       from .trajectory import RewardChannel, Trajectory
   ```
   `trainer.py` 的 `ExplorationSpec` 可以直接实 import（`.experiment` 已经被 import 了，无循环依赖风险 —— 但要确认）。

2. **修 `ppo_update` 的返回注解**：`-> UpdateStats`。

3. **清理未使用 import**，删掉或加 `# noqa` 说明理由。

4. **建 CI**（`.github/workflows/ci.yml`），最小四件事：
   ```yaml
   - pip install -e ".[dev]"          # P2-1 的验收
   - ruff check baseline/framework/ppo
   - mypy baseline/framework/ppo      # 宽松配置起步
   - pytest baseline/framework/ppo -q
   ```

5. **加 `pyproject.toml` 的 lint 配置**（ruff 推荐，一个工具覆盖 pyflakes + isort + 部分 pylint）。

### 注意事项

- ⚠️ **mypy 从宽松配置起步**。直接开 `--strict` 会出几百个错误，然后被 `# type: ignore` 淹没，毫无价值。建议先只开 `--warn-unused-ignores --warn-redundant-casts`，把 `disallow_untyped_defs` 留到后面逐模块开启。
- ⚠️ **CI 不能依赖 GPU**。现有测试都能在 CPU 跑（102 passed 是在有 GPU 的机器上，但用的是 `torch.device("cpu")`）。P0-4 的 cuda 测试要加 `skipif`。
- ⚠️ **CI 是否需要 MuJoCo** 取决于 P2-2。如果 P2-2 做了 fake/gym 后端，CI 可以完全不装 MuJoCo，跑得又快又稳。这是 P2-2 的一个额外收益。
- ⚠️ 本仓库在中国大陆的 GPU 服务器上开发，CI 在 GitHub 上跑 —— 注意依赖安装的镜像源配置差异（`CLAUDE.md` 里有代理说明，但那是给本地开发用的）。
- ⚠️ 一次性把整个仓库纳入 lint 会产生海量错误。**先只对 `baseline/framework/ppo/` 开启**，用 ruff 的 `per-file-ignores` / 目录级配置逐步扩大范围。

### 验收标准

- `python3 -m pyflakes baseline/framework/ppo/{*.py,algos/*.py,policies/*.py}` 零输出
- CI 四个 job 全绿
- `python3 -c "from typing import get_type_hints; import ...; get_type_hints(ppo_update)"` 不报错

---

## P2-4. 文档结构不适合外部读者

### 问题是什么

| 问题 | 具体表现 |
|---|---|
| 无英文入口 | 中文文档 + 英文 docstring 混排，面向国际开源缺少英文 README |
| 示例不可执行 | 见 P1-4 / P1-5 |
| 缺 API reference | 有 `DESIGN_*.md`（设计意图）和 `GUIDE.md`（教程），但没有「每个方法的签名与契约」的参考文档 |
| **缺「已知局限」章节** | 见 P1-13。这是开源项目最重要、也最常被省略的一节 |
| **内部工单编号泄漏** | 代码注释里有 `B1:` `B2:` `B8:` `A4:`（`trainer.py` 5 处、`loop.py` 2 处），外部读者完全无法解读 |
| **changelog 式注释** | "Key differences from v1"、"Previously this used floor division…"、"No `_current_actor_weights` hack"、"No plateau detection" —— 对没见过 v1 的人是纯噪声 |
| `todo/` 目录公开可见 | 8 个策略族 + 9 份 DESIGN 躺在 `policies/todo/`，而 `__init__.py` 只导出 1 个。P0-6 已证明这会造成真实事故 |

### 修复关键点

1. **工单编号 → 自解释说明**。例如：
   ```python
   # B1: Critic updates are decoupled from actor KL early-stop.
   ```
   改成
   ```python
   # Critic updates are decoupled from the actor's KL early-stop: critics
   # regress against a fixed target (precomputed GAE returns) and are not
   # subject to the trust region, so truncating them early only degrades
   # value estimation.
   ```
   （原注释后面本来就有这段解释，只需删掉 `B1:` 前缀）

2. **changelog 式注释 → `CHANGELOG.md`**。`trainer.py` 和 `loop.py` 顶部的 "Key differences from v1" 整段搬走。代码注释应该说明「现在是什么、为什么」，不是「以前是什么」。
   ⚠️ 例外：**「为什么不用某个看起来更自然的做法」这类注释要保留**。比如 `_normalize_adv` 里解释「为什么只在 aw≠0 的帧上归一化」、L676-684 解释「为什么 L1 而非 L2」—— 这些是防止后人「优化」掉正确设计的护栏，价值极高。判断标准：**删掉它，后人会不会做错？**

3. **`todo/` 的处置**（与 P0-6 同一决策）：正式支持、移到 `experimental/` 并声明不稳定、或移出仓库到分支。**不要让公开路径指向一个叫 `todo` 的目录。**

4. **文档结构建议**：
   ```
   README.md                 (英文，含 quickstart / reference run 曲线 / Non-goals)
   README_zh.md              (中文)
   GUIDE.md                  (教程，示例引用 examples/)
   API.md                    (ExperimentPPO 每个方法的契约)
   CHANGELOG.md              (从代码注释里搬出来的历史)
   DESIGN_*.md               (设计决策记录，保留)
   examples/exp_minimal.py   (可执行 + 进 CI)
   examples/exp_pendulum.py  (reference run)
   ```

5. **「已知局限」必须写**。把本文档 P1-13 的表格、P1-1 的 EV 语义说明、P1-3 的 confidence-LR 耦合，全部诚实写进去。**主动声明局限比被 critique 出来强得多**，而且这会显著提高项目的可信度。

### 注意事项

- ⚠️ **`GUIDE.md` 的整体质量其实很高**（数据流图、职责分工表、`actor_weight` 四层语义表、调试建议），不要重写，只修错误的部分。
- ⚠️ 关于原创性的表述要谨慎：**「每个 reward 分量一个独立 value head」不是新东西**（Hybrid Reward Architecture, van Seijen et al. NeurIPS 2017；以及多家大规模 PPO 实践）。如果 README 宣称 multi-critic 是创新，会被立刻挑战。建议把叙事收缩到三条真正可主张的贡献上：
  1. `actor_weight` 的 per-frame L1 归一化（课程权重与有效学习率的正交化）
  2. `explore_factor` 作为 per-frame 数据契约（importance ratio 的正确性由数据而非时序保证）
  3. 用 critic 可信度加权 advantage 合并（**但必须先修 P1-1**，否则这条主张站不住）

### 验收标准

- `grep -rn "^\s*#.*\b[AB][0-9]:" baseline/framework/ppo/*.py` 零输出
- README 有 Non-goals / Known limitations 一节
- `policies/todo/` 已处置

---

## P2-5. 测试风格不惯用，缺覆盖率与 property-based 测试

### 问题是什么

2492 行测试里：
- 大量 `print(f"test_xxx: PASS")` —— pytest 已经报告结果，这些 print 是噪声（且 pytest 默认捕获输出，看不到）
- `def test_zero_variance_warning(capsys=None):` —— **把 pytest fixture 当默认参数**。这样写 pytest 不会注入 fixture，`capsys` 恒为 `None`，所以函数里针对 capsys 的分支永远走不到（又是一个「看起来在测、其实没测」的例子，与 P0-2 同类）
- 没有 `parametrize`，相似场景靠复制粘贴（这是 2492 行里很大一部分）
- 没有覆盖率数据
- GAE、L1 归一化这类纯函数没有 property-based 测试

### 修复关键点

**这是 P2，优先级最低，但有两处例外要提前做：**

1. **⚠️ 优先：`capsys=None` 这类 fixture 误用要立刻修**（`test_zero_variance_warning`、`test_confidence_cold_start_warning` 等 4 处），因为它们和 P0-2 是同一类问题 —— 静默失效的测试。改成正确的 fixture 注入：
   ```python
   def test_zero_variance_warning():
       stats = ppo_update(...)
       assert any("zero-variance advantages" in d for d in stats.diagnostics)
   ```
   注意：`ppo_update` 现在把诊断放进 `stats.diagnostics` 而不是 print（B8 改动），所以**根本不需要 capsys** —— 这些测试的 capsys 参数是重构遗留物。

2. **优先：给 GAE 和 L1 归一化加 property-based 测试**（P1-9 的验收标准已包含）。这两个是算法核心，值得最强的测试。

**其余（低优先）：**
3. 删掉 `print("...: PASS")`
4. 用 `parametrize` 合并重复用例，2492 行大概能压到 1500 行以内
5. CI 里加 `pytest --cov=baseline/framework/ppo --cov-report=term-missing`，设一个不下降的门槛

### 注意事项

- ⚠️ **不要在修 P0 的 PR 里顺手改测试风格**。会让 diff 无法审查，也会掩盖真正的行为变化。
- ⚠️ 用 `parametrize` 合并时要小心：有些看起来重复的测试其实在测不同的边界（比如 `test_l1_normalization_*` 那 5 个）。**先读懂再合并**，不确定就别合。
- ⚠️ 覆盖率数字有欺骗性。P0-2 那两个测试贡献了覆盖率但什么也没测。**追求覆盖率不如追求「删掉实现代码，测试会不会失败」**。可以考虑引入 mutation testing（`mutmut`）对 `trainer.py` 跑一次，会很有启发。

### 验收标准

- `grep -n "capsys=None\|capfd=None" tests/` 零输出
- GAE 有 parametrize 覆盖 `λ×γ×T×last_value` 的组合
- CI 报告覆盖率

---

# 附录 A：常用命令

```bash
cd /data1/mono/things/combatbench

# 全量测试（基线：102 passed）
PYTHONPATH=. python3 -m pytest baseline/framework/ppo -q

# 只跑 trainer
PYTHONPATH=. python3 -m pytest baseline/framework/ppo/tests/test_trainer.py -q

# lint
python3 -m pip install pyflakes ruff
python3 -m pyflakes baseline/framework/ppo/{*.py,algos/*.py,policies/*.py}

# smoke（2 updates）
PYTHONPATH=. python3 baseline/framework/train.py --experiment <name> --algo ppo --smoke

# 验证 pip 安装（必须在仓库外执行）
python3 -m venv /tmp/v && /tmp/v/bin/pip install -e .
cd /tmp && /tmp/v/bin/python -c "from combatbench.baseline.framework.ppo import ExperimentPPO"
```

---

# 附录 B：修复过程中最容易犯的错

按本文档发现问题的经验，列出几个反复出现的陷阱：

1. **用零输入测数值差异**（P0-5）。`obs=zeros` 时第一层的 weight 完全不影响输出，会得出「没问题」的错误结论。**测数值一致性一律用随机非零输入。**

2. **把「前置条件」写成 `if` 而不是 `assert`**（P0-2、P2-5）。`if condition: assert ...` 在条件不成立时静默通过。**前置条件用 `assert`，让它在环境变化时失败而不是消失。**

3. **改了行为却没改统计口径**（P0-1）。B1 改了控制流，但 `epoch_kl_stats[-1]` 的语义依赖旧控制流。**改控制流时，把所有依赖「循环如何结束」的聚合逻辑都过一遍。**

4. **文档和代码分两处维护**（P1-4、P1-5）。必然漂移。**让文档里的代码就是被测试的代码。**

5. **`x or default` 兜底 `None`**（`trainer.py` L448）。对 `0.0`、`""`、`[]` 也会取 default。**判 None 用 `if x is None`。**

6. **在通用层嗅探具体实现的属性**（P1-7）。`hasattr(actor, "log_std")` 这类代码写起来快，但会在抽象边界上开洞，而且下一个策略族接进来时会静默失去功能。**用显式的可选钩子代替。**

7. **重构时把半成品留在仓库里**（P0-6）。`policies/todo/` 直接导致 81 个 artifact 失效。**要么正式支持，要么移出仓库。**

8. **认为静默失败比崩溃温和**（P0-3、P0-5）。`strict=False`、`allow_unused=True`、宽容的 fallback，都是在用「不报错」换「错得看不见」。**在训练框架里，早崩比晚错好得多。**
