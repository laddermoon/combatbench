# LIFECYCLE_TRACE — CPU 生命周期权威时序核对（E2-W0 产出，终止帧契约修订后更新）

> 类型：契约

**状态**：已核对（2026-02-20），与 `envs/framework/RESET.md` 规范交叉验证一致；
终止帧契约修订（`082187be`）后时序描述已同步更新。
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
     b. on_pre_action_step(mutator按插件授予)        # 【屏障A】all_agents_terminated→跳过循环
     c. for s in range(S):                          # S = phy_steps_per_action
          on_pre_phy_step(mutator)                  # 【屏障B】→break
          simulator.physical_step()
          ctx.physics_step += 1                     # 只计实际执行的子步
          on_post_phy_step(mutator)                 # 【屏障C】→break
     d. ctx.episode_step += 1                       # ★ 无条件——step() 调用计数
     e. on_post_action_step(无mutator)              # ★ 恰好一次/进入的step，
        …                                           #   作用于步末态（含终止态）；
                                                  #   dispatcher(1e6)先刷新observers
     f. _check_and_handle_termination():            # ★ 唯一终止处理点（屏障D）
          is_episode_active=False
          on_post_episode(无mutator)                # dispatcher先跑→observers.on_post_episode
 5. _invoke_recorders("on_post_action_step", obs_t, extras)
     # ★ 无条件执行——即便 core.step 因终止 break（子步内终止）也记录一帧
     # 帧内容: observation=obs_t(步骤3), action=simulator.get_action()(实际执行值),
     #        observer_outputs=步末态 post_action 刷新值（终止帧=终止态值）
     # 帧内顺带扫描 agent_termination_proposals → (reason, episode_step) 去重记录
 6. if ended: _invoke_recorders("on_post_episode")   # 捕获 final_obs=当前obs(未reset)
```

### 屏障语义（实测）

- 终止检查点 = **循环内只读检查（`all_agents_terminated`→break）+ 尾部统一
  `_check_and_handle_termination`**：屏障位置仍为 pre_action 后、每子步
  pre_phy 后、每子步 post_phy 后；差别是**检查本身不再触发 post_episode**——
  post_episode 收敛到尾部 post_action 之后统一触发。
- 触发条件**只有** `all_agents_terminated`；单 agent 终止不结束 env（env 继续跑，见 §6）。
- 插件在 hook 中 `request_termination` **立即**写 `agent_terminated` bool + append proposals list——**同一 phase 后续插件能看到**（立即生效，非延迟到屏障）。
- proposals 的消费（去重/记录）在 **recorder 帧**里，不在屏障。

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

- **每帧扫描完整 proposals list**——某步新出现的 reason 以**该帧的 episode_step** 记录。`episode_step` 无条件递增后，扫描值 = 帧索引+1 = **含端点轨迹边界**，对所有提出点（pre_action/子步内/post_action）一致。
- `records[0]` = 首个被记录原因 → 消费侧以 `episode.agent_frame_boundary[aid]` 取轨迹边界 + `is_true_terminated = (first_reason != "timeout")` 决定 bootstrap。
- **边界含义**：中途终止帧（物理增量 ≥1）进入轨迹为终止 transition；**退化帧**（pre_action/首子步 pre_phy 全员终止，物理增量=0，action 未物理生效）记录在 Episode 但按 boundary 排除出轨迹。判定依据 = 帧的 `physics_step` 增量，不依赖 episode_step。
- `on_post_episode` 校验：每个 agent 必须 ≥1 条 record，否则 RuntimeError。**CPU 无法产出 zero-frame episode**——reset 内即终止的 episode 在 `on_post_episode` 校验处直接报错（recorder 的 on_post_action_step 从未跑过，records 为空）。

## 3. 子步内终止的后果（INTRA_ACTION_END，修订后语义）

`_core.step` 子步循环 break 时：

| 量 | 值 |
|---|---|
| `physics_step` | = 实际执行的子步数（未完成的 skipped） |
| `episode_step` | **照常 +1**（步调用计数，无条件） |
| `on_post_action_step` 插件 | **照常执行**——作用于终止态；dispatcher→observers 刷新终止态值 |
| `on_post_episode` 插件 | 执行（统一在 post_action 之后） |
| recorder 帧 | 仍记录一帧：obs=obs_t，observer_outputs=**终止态 post_action 刷新值** |
| 终止记录 step | 已递增的 episode_step（=帧索引+1 = 含端点边界） |

**对设备契约的确认**：终止帧的 observer 输出来自 post_action 刷新（终止态），不再是 post_episode 刷新值——设备侧 `_RecorderAdapter` 的 post_episode 覆写因此变冗余。

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
| 终止帧仍记录 + records=含端点边界（修订后：中途终止帧进轨迹，退化帧按 physics delta 排除） | wave recorder 现行 `env_term_step`/`t_use` 为旧排他语义 | **差距**→设备侧对齐（episode_steps 无条件计数后 env_term_step 自动含端点） |
| 子步内终止：episode_step 照常+1、post_action 插件照常执行（终止态）、帧仍记 | 无子步屏障，无法发生 | **差距**→需表达"子步内 ENDED 的行也收 post_action"（mask=本步运行行） |
| post_action 恒先于 post_episode（终帧 observer 输出=终止态 post_action 值） | device `_RecorderAdapter` 以 post_episode 值覆写末帧 | **冗余**→对齐后可删覆写 |
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
- D7.4 "INTRA_ACTION_END 表达但不启用"：契约修订后该路径语义变更为（episode_step 照常+1、post_action 作用于终止态、终止帧含端点进轨迹、退化帧按 physics delta 排除）→ 设备侧需对齐而非 pending。
- **新发现需补进实现细节（非契约层）**：proposals 在 CPU 中对"同 phase 后续单元立即可见"（`agent_terminated` bool 即时置位）——设备 pending 模型必须保留这一点：`request_termination` 立即写 `agent_done`，屏障只负责 history 归档与 env 判定，不能把提出效果也延迟。
- **确认项**：`_WaveRecorder` 的终止帧/records 扫描发生在 runtime `on_post_episode` **之前**，与 CPU（插件 post_episode → recorder 帧）相反。契约修订后终止帧 observer 值 = post_action 刷新值，该次序差异的主要场景消除；残余一致性项 = `on_post_episode` 里更新输出的 observer 仍需 CPU 次序。
