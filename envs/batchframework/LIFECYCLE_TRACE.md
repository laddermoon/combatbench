# LIFECYCLE_TRACE — CPU 生命周期权威时序核对（E2-W0 产出）

**状态**：已核对（2026-02-20），与 `envs/framework/RESET.md` 规范交叉验证一致
**核对对象**：`envs/framework/env_runtime.py`、`context.py`、`observer_plugin.py`、`episode_runner.py`、`common_plugins.py`、`baseline/framework/rollout/episode_recorder.py`、`episode.py`
**用途**：E2 设备侧生命周期实现的**语义来源**。此处记的是 CPU 实测行为，不是愿望。

## 1. 单个 action step 的权威时序（`_RuntimeCore.step` + `EnvRuntime.step`）

```
EpisodeRunner（collector 层）
 1. policy.act(obs_t) per agent                    # post_termination_action 见 §6
 2. runtime.step(action_a, action_b, extras)

EnvRuntime.step
 3. observation = simulator.get_observation()       # ★ pre-action obs_t，先于一切
 4. _core.step({"robot_a":a,"robot_b":b})
     a. mutator.set_action(action)                  # runtime 自己写，非插件
     b. on_pre_action_step(mutator按插件授予)        # 【屏障A】all_agents_terminated→提前return
     c. for s in range(S):                          # S = phy_steps_per_action
          on_pre_phy_step(mutator)                  # 【屏障B】→提前return
          simulator.physical_step()
          ctx.physics_step += 1                     # 只计实际执行的子步
          on_post_phy_step(mutator)                 # 【屏障C】→提前return
     d. ctx.episode_step += 1                       # ★ 子步循环完整跑完才+1
     e. on_post_action_step(无mutator)              # dispatcher(1e6)先刷新observers
        …                                           # 【屏障D】→_handle_termination
     f. _handle_termination():                      # 仅当 all_agents_terminated
          is_episode_active=False
          on_post_episode(无mutator)                # ★ dispatcher先跑→observers.on_post_episode
 5. _invoke_recorders("on_post_action_step", obs_t, extras)
     # ★ 无条件执行——即便 core.step 提前 return（子步内终止）也记录一帧
     # 帧内容: observation=obs_t(步骤3), action=simulator.get_action()(实际执行值),
     #        observer_outputs=此刻的值(若env已结束→是on_post_episode之后刷新过的)
     # 帧内顺带扫描 agent_termination_proposals → (reason, episode_step) 去重记录
 6. if ended: _invoke_recorders("on_post_episode")   # 捕获 final_obs=当前obs(未reset)
```

### 屏障语义（实测）

- 终止检查点 = **5 处屏障**：post-pre_action、每子步 post-pre_phy、每子步 post-post_phy、post-post_action（结尾处不 return，正常收尾）。
- 触发条件**只有** `all_agents_terminated`；单 agent 终止不结束 env（env 继续跑，见 §6）。
- 插件在 hook 中 `request_termination` **立即**写 `agent_terminated` bool + append proposals list——**同一 phase 后续插件能看到**（立即生效，非延迟到屏障）。
- 屏障只做一件事：`if all_agents_terminated: _handle_termination()`。proposals 的消费（去重/记录）在 **recorder 帧**里，不在屏障。

## 2. 终止数据模型（CPU 真实语义）

`SimContext` 持两份：

- `agent_termination_proposals[aid]`：**append-only list，提出端不去重**。同一 reason 可重复 append；env 级请求（`agent_id=None`）向两个 agent 各 append 一次。
- `agent_terminated[aid]`：bool，提出时即置 True。

**去重与记录发生在 recorder**（`episode_recorder.py:144-150`）：

```python
for aid in AGENT_IDS:
    for reason in ctx.agent_termination_proposals.get(aid, ()):
        if reason not in self._seen_reasons[aid]:        # 每 reason 只记首次
            self._agent_termination_proposal_records[aid].append(
                (reason, int(ctx.episode_step)))          # ★ 帧时刻的 episode_step
```

- **每帧扫描完整 proposals list**——某步新出现的 reason 以**该帧的 episode_step** 记录。pre_action/子步内终止时 `episode_step` 未 +1 → 记录值 = 已完成步数（即帧索引 k）。post_action 终止时已 +1 → 记录值 = k+1。
- `records[0]` = 首个被记录原因 → `episode.py:330-336`：`obs_a[:term_step]` 截断 + `is_true_terminated = (first_reason != "timeout")` 决定 bootstrap。
- **截断边界含义**：post_action 终止（如 timeout）→ term_step 含终止帧本身；子步内终止 → term_step 排除那个半截帧（数据不完整所以弃用，final_obs 补 bootstrap）。**这不是巧合约定，是 episode_step 自增时机造成的精确语义**。
- `on_post_episode` 校验：每个 agent 必须 ≥1 条 record，否则 RuntimeError。**CPU 无法产出 zero-frame episode**——reset 内即终止的 episode 在 `on_post_episode` 校验处直接报错（recorder 的 on_post_action_step 从未跑过，records 为空）。

## 3. 子步内终止的后果（INTRA_ACTION_END，实测）

`_core.step` 提前 return 时：

| 量 | 值 |
|---|---|
| `physics_step` | = 实际执行的子步数（未完成的 skipped） |
| `episode_step` | **不自增** |
| `on_post_action_step` 插件 | **不执行**（含 dispatcher→observers 不刷新） |
| `on_post_episode` 插件 | 执行（dispatcher 先跑→observers.on_post_episode） |
| recorder 帧 | **仍记录一帧**：obs=obs_t，observer_outputs=**post_episode 刷新后的值**（若 observer 在 on_post_episode 里改了输出） |
| 终止记录 step | 未自增的 episode_step（=帧索引） |

**对设备契约的确认**：D7.2 第 6 条预判正确——"兼容导出遵循 CPU recorder 在 `_core.step` 返回后读取的实际结果"，observer 输出确实可能是 post_episode 刷新值。

## 4. Reset 链（与 RESET.md §3 一致，代码复核通过）

```
EpisodeRunner.run_episode:
  base_seed=None→secrets.randbits(32)           # ★ 入口确定性解析
  spawn: [runtime, policy_a, policy_b, *plugins(按priority序)]
  plugin.set_episode_seed(...)  ← 先于 reset（on_pre_episode 里就要用 RNG）
  runtime.reset(seed=runtime_seed, options, base_seed):
    if episode_active: request_termination("abandoned")+handle   # 隐式abandon
    clear_episode_state()  # step计数器/metrics/events/proposals/options全清
    ctx.base_seed/episode_options 写入
    _is_episode_active=True
    simulator.reset(seed, options)          # backend 消费 sim 相关 options
    on_pre_episode(mutator)                 # dispatcher先→observers.on_pre_episode
                                            #   然后其他插件按priority
    if all_agents_terminated: _handle_termination()   # reset内即终止
  recorder on_pre_episode                    # 拍起始快照（observers已刷新）
  (若reset内已终止) recorder on_post_episode
  policy.reset(seed) per agent               # 策略在 runtime 外，不参与生命周期
```

**顺序不变式（实测确认）**：
- `simulator.reset` 先于插件 `on_pre_episode`（插件读到新初态）。
- **observer `on_pre_episode` 先于其他插件**（dispatcher priority=1e6）→ 插件在 on_pre_episode 写的 metrics，observer 本轮看不到（RESET.md §3.4 明载，为设计而非 bug）。
- 终止发生在 reset 内 → recorder 无帧 + `on_post_episode` 校验炸（zero-frame 不存在）。
- reset 时若有 active episode → **自动 abandon**（reason="abandoned"，on_post_episode 正常走，不是静默丢弃）。

## 5. Observer 刷新时机（实测）

`_ObserverDispatcherPlugin` 是普通插件（priority=1e6，require_mutator=False）：

- 刷新点：`on_pre_episode`、`on_post_action_step`、`on_post_episode` + 手动 `refresh(force)`。**不在 pre/post_phy_step、pre_action_step 刷新**（那些 hook 返回 None）。
- 去重 token = (trigger, episode_step, physics_step, proposals tuple, all_terminated)——**同 token 重复刷新被跳过**；metrics/events 不在 token 内（注释明确标注为契约）。
- observer 永远只读（`require_mutator=False`），拿 `ReadOnlySimContext`（`MappingProxyType` 包装的 metrics、tuple 化的 proposals）。
- `attach_observer_plugin` 在 episode 进行中 → 立即 `refresh(force=True)`。

## 6. post_termination_action（EpisodeRunner 层）

- 默认 `"policy"`：已终止 agent **继续被策略采样**、动作照常进 `set_action` 驱动物理（KO 机器人还在抽搐），帧照常记录——但 per-agent 轨迹在 term_step 截断，后续帧不进该 agent 的训练数据。
- `"hold"`：回放终止前最后一个 action（`extra=None`）。
- 这是 **runner/collector 策略，不是 runtime 语义**——runtime 只管 all_terminated 判终。

## 7. Seed 模型

`SeedSequence(base).spawn(1+2+n_plugins)`：slot 顺序 = `[runtime, robot_a_policy, robot_b_policy, 各seedable插件(按priority排序)]`。插件种子**绑 priority 序而非名字**；`set_episode_seed` 在 reset 前调用。

## 8. 设备侧现状对照与差距清单

| CPU 语义 | 设备现状 | 判定 |
|---|---|---|
| proposals append-only 不 dedup；recorder 帧扫描去重记首次 | `agent_term_reason` 单值覆盖写；recorder 只记 per-agent 首次 | **差距**：同帧多原因只活一个（后写覆盖）；已终止 agent 的新原因被 `_seen_term` 挡掉。CPU 语义=每 reason 首次都记 |
| 5 屏障 + 提出即生效（同 phase 后续可见） | step 末尾单屏障消费；`request_termination` 直写 `agent_terminated`（提出即生效 ✓ 一致） | **半等价**：屏障位置少（无子步内），但提出可见性一致 |
| 终止帧仍记录 + term_step 截断规则 | wave recorder 同规则（`env_term_step`/`t_use`） | **等价**（当前仅 post_action 粒度） |
| 子步内终止：episode_step 不自增、post_action 插件跳过、帧仍记 | 无子步屏障，无法发生 | **差距**→E2-W3（先表达不启用，J3） |
| on_post_episode 在 recorder 帧前 → final 帧 observer 输出为 post_episode 刷新值 | device：`_WaveRecorder` 帧扫描在自己的 `on_post_action_step`（post_episode 之前）→ final 帧 observer 输出是**刷新前值**。当前无实质差异（`DeviceStandup4StageRewarder.on_post_episode` 是 no-op），但次序是结构性差异 | **差距**：W1 须把 terminated 行的 final 帧扫描置于 post_episode schedule 之后，保持 CPU 序 |
| final_obs = 终止时刻当前 obs（无 reset） | `env_term` 首次即捕获 io.obs（obs_{t+1}）✓ | **等价** |
| reset 内终止 → 报错，无 zero-frame episode | wave 模型天然每行 ≥1 帧；reset 内终止语义未定义 | **差距**：reset-time term 需拒绝/报错路径 |
| reset 隐式 abandon | step 内 auto-reset（非等价物） | **重设计**：E2-W2 显式 abandon + sealed-ENDED |
| post_termination_action="policy" 继续采样已终 agent | device 两 agent 恒采样（policy 语义天然等价）✓；"hold" 无 | **等价**（默认路径）；hold 待 policy_eval_mask（W1） |
| 插件种子按 priority 序 spawn | seed_offsets 按行 | **映射差异**：设备以 job 为单位派生，unit salt 注册表序——W5 需保"同名 unit 同 salt"确定性 |
| observer 刷新点：pre_episode/post_action/post_episode | device dispatcher 同三点位 ✓ | **等价** |
| zero-frame 校验 RuntimeError | 无对应校验 | **差距**：W6 测试补 |

## 9. 对 D7 契约草案的裁决

**无需修订契约**。逐项核对结果：

- D7.2 时序图：与实测一致，包括第 6 条"observer 输出可能是 post_episode 刷新值"的预判。
- D7.3 "同原因重复请求不重复导出"：CPU 提出端不去重、记录端去重——契约只约束导出语义（records），不约束 ctx 内部表示 → 不冲突。设备 pending 队列 + barrier 时把"提出即生效的 `agent_terminated`"保留为立即写（CPU 行为），仅历史记录进屏障 → 完全对齐。
- D7.3 "新原因在 agent 已终止后仍记录"：实测确认（recorder 扫全 list）；设备当前 `_seen_term` 挡掉后续记录是**设备侧 bug 级差距**，W1/W6 修。
- D7.4 "INTRA_ACTION_END 表达但不启用"：实测确认 CPU 该路径存在且语义明确（episode_step 不自增、半截帧被截断排除）→ 维持 pending 标记。
- **新发现需补进实现细节（非契约层）**：proposals 在 CPU 中对"同 phase 后续单元立即可见"（`agent_terminated` bool 即时置位）——设备 pending 模型必须保留这一点：`request_termination` 立即写 `agent_done`，屏障只负责 history 归档与 env 判定，不能把提出效果也延迟。
- **确认项**：`_WaveRecorder` 的终止帧/records 扫描发生在 runtime `on_post_episode` **之前**，与 CPU（插件 post_episode → recorder 帧）相反。现无实质差异（standup rewarder `on_post_episode` 为 no-op），但 W1 实现时必须恢复 CPU 次序，否则未来"在 post_episode 里更新输出的 observer"会产生静默语义漂移。
