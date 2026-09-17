# dumpkit — PPO 训练调试工具集（AI 能力地图）

**这是什么**：一套用于观察、下钻、分析 PPO 训练过程的工具集。不是工作流——
每个工具独立回答一类问题，可在任意分析过程中自由组合。

**使用者注意**：所有读命令输出 JSON 到 stdout（`--pretty` 美化）。除 `viewer`
外全部离线可用——不需要启动 HTTP server。`<run>` 参数接受 run 目录路径，
或 `baseline/runs` 下的 run 名（`--runs-root` 可换）。

## 数据阶梯（粒度从粗到细）

```
run          train.log 里的 __RAW_STATS__ JSON 行 —— 每 update 一条扁平时序
  └─ dump    dumps/uNNNNN/ —— 单个 update 的完整截面（episodes.npz、buffer.npz、
             update.npz、manifest.json、policy 导出、episode_options.json）
      ├─ episode   环境对局的逐帧 obs/action/observer 输出
      ├─ trajectory 训练用轨迹段（训练目标：ret/adv/actor_weight/confidence）
      └─ timeline  update 内部动力学（minibatch/epoch 级的 KL、ratio、loss）
  └─ 派生产物  record/（render 的 PNG 帧）、delta/（跨代 policy 漂移）
```

## 指标命名空间（扁平化后的一级前缀 = 来源/归属）

| 前缀 | 来源 | 说明 |
|---|---|---|
| `stats.*` | `UpdateStats`（框架） | PPO 动力学：kl/ratio/clip/loss/grad/lr/epochs… |
| `ep.*` | `episode_stats` | 环境 episode 层统计（ep_len 等） |
| `pc.<m>.<ch>` | `buffer_stats.per_channel` | 分 channel 指标（reward/adv/vloss/ev…） |
| `time.*` | timing | 各阶段耗时 |
| `exp.*` | `experiment.on_update()` 返回值 | 实验自定义，每 update 一条 |
| `eval.*` | `experiment.on_eval()` info | **稀疏**——只在 eval_interval 的 update 上有值 |
| `policy.*` | `policy_stats` | policy 自定义诊断 |

扁平化代码只有一份：`viewer/server.py::RunData._flatten_update`。
CLI 与 HTTP API 共用——两者输出逐字节相同，不存在第二份解析。

## 指标语义：metric_catalog.py（单一信息源）

指标含义/读法/诊断倾向全部集中在 `metric_catalog.py`（chart 布局 +
hint 表 + 分区定义）。任何入口拿到的都是同一份：

- `debug.py catalog` —— 全量目录；`--key <flat_key>` 单键解析 {zone, hint}
- `debug.py metrics --docs` —— 数据旁附每个 key 的语义
- `debug.py summary` —— 每个 key 自带 zone+hint
- `GET /api/catalog` —— viewer 前端渲染 hint 的来源

**不要**把 hint 文本复制进别的文档——引用 catalog，永远同源。

## 命令速查（按问题组织）

```bash
# cd /data1/mono/things/combatbench && PYTHONPATH=.

# —— 有哪些 run？
debug.py runs [--status running] [--tail 5]
#   → {runs_root, runs:[{name,experiment,algo,status,update,max_updates,
#       eval_success,eval_pot,created,activity,n_dumps,n_videos}]}

# —— 这个 run 整体怎么样？（定向命令，先跑这个）
debug.py summary <run> [--keys kl]
#   → {run,status,n_updates,dumps:[...],metrics:{key:{zone,hint,n,
#       first_update,latest,latest_update,min,min_update,max,max_update}}}

# —— 指标逐 update 怎么变的？
debug.py metrics <run> [--keys eval.,actor_lr] [--tail 20] [--docs]
                        [--from-update A --to-update B]
#   → {run,n_updates,metrics:[{update, <flat_key>: val, ...}]}

# —— 这个指标是什么意思？
debug.py catalog --key stats.post_kl_mean
#   → {key, zone, hint}

# —— update N 内部发生了什么？（截面采集，需要 run 正在训练）
debug.py dump <run_dir> --hypothesis "why is KL high at u250"
#   → 写 sentinel，训练循环在下一 update 边界捕获到 dumps/uNNNNN/
#   ⚠ --hypothesis 必填：说不清在查什么就不该 dump

# —— 看 dump 的逐帧画面 / 验证回放一致性
debug.py render <dump_dir> --episode 0
#   → dumps/uNNNNN/record/episode_NNNNN/*.png + association.json + 校验日志

# —— 策略这几代漂了多少？（离线确定性回放，无需 run 在跑）
debug.py delta <dump_dir> --episode 0 --gens 3
#   → dumps/uNNNNN/delta/episode_NNNNN/（viewer episode 页自动显示）

# —— 交互界面（给人看）
debug.py viewer [run_dir|dump_dir|runs_root] --port 8766
```

## HTTP API（viewer 运行时等价物）

模式：`runs`（多 run 索引）/ `run`（单 run）/ `dump`（单 dump）。

```
GET /api/mode | /api/catalog | /api/render-status
GET /api/runs?q=&page=&size=&sort=&order=                     (runs 模式)
GET /run/<name>/api/run/{info,dumps,metrics,videos}           (runs 模式)
GET /api/run/{info,dumps,metrics,videos}                      (run 模式)
GET /api/dump/<d>/<ep> 或 run 模式 /run/<n>/api/dump/<d>/<ep>：
    manifest | episode_list | traj_map
    episode/<pos>/frame/<f> | episode/<pos>/delta | image/<a>/<b>
    trajectory/<i> | trajectory/<i>/frame/<f>
    trajectory/<i>/epoch/<e>/overview|frame/<f> | trajectory/<i>/epoch_compare
    timeline/overview | timeline/step/<s>
POST /run/<name>/api/run/dump-request   {hypothesis}          (running run)
POST /api/dump/<d>/render|delta         {episode[,gens]}      (单 job 槽)
```

## Gotchas

- `metrics`/`summary`/`catalog`/`runs` **完全离线**；只有 `dump`（需训练中的 run）、
  `viewer`（起服务）、`render`/`delta` 的**前端触发**需要 server——CLI 直接跑不需要。
- `summary` 里 `eval.*` 的 `n` 远小于 `n_updates` 是稀疏的正常表现，不是丢数据。
- `render`/`delta` 在 server 内共享**单个后台 job 槽**（409 = 有任务在跑）；
  CLI 直接调用不受此限。
- `render` 校验 FAIL 的帧仍产出 PNG，但 association 标记非 ground truth。
- run 名解析只允许 `runs_root` 的直接子目录且须含 config.json 或 train.log。
- 新起的 run 才有新字段（如 exp.*/post_kl_*/uncertainty_floor）——老 run
  日志缺字段属正常向后兼容，不是解析失败。

## 文档索引

- `DATA_FLOW.md` —— dump 捕获管线数据流
- `DESIGN_viewer_overview.md` + `DESIGN_viewer_scene1-4.md` —— viewer 设计
- 每个 dump 内的 `RECORD_GUIDE.md` —— 该截面的录制/复现命令
- 项目根 `CLAUDE.md` —— 训练框架总览；`experiments_ppo/README.md` —— 实验注册
