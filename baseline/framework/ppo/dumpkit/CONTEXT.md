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
             update.npz、gradsig.npz、manifest.json、policy 导出、episode_options.json）
      ├─ episode   环境对局的逐帧 obs/action/observer 输出
      ├─ trajectory 训练用轨迹段（训练目标：ret/adv/actor_weight/confidence）
      ├─ timeline  update 内部动力学（minibatch/epoch 级的 KL、ratio、loss）
      └─ gradsig   gradsig.npz —— 该 update 的 ADV 梯度信号分布
                   （cos(g_i,G) × ‖g_i‖ 分箱 + 逐帧 proj/cos 数组），
                   由 dump 请求触发；run 级 gradsig/meta.json 冻结
                   norm 分箱边界（norm_axis: per_frame_norm）
  └─ 派生产物  record/（render 的 PNG 帧）、delta/（跨代 policy 漂移）
```

注：旧 run 可能仍有 run 级 `gradsig/uNNNNN.npz`（dump-only 改造前
的周期性工件），读取 API 保留但新 run 不再产生。

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

指标含义/读法/诊断倾向全部集中在 `metric_catalog.py`（**六分区
sectioned 布局**：task → policy_update → signal → explore → sampling →
cost；每区 curated 精选图 + expanded 折叠辅助组；图 spec 携带
subtitle/guide/zone_pick/spike_keys 元数据）。任何入口拿到的都是同一份：

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
debug.py dump <run_dir> [--hypothesis "why is KL high at u250"]
#   → 写 sentinel，训练循环在下一 update 边界捕获到 dumps/uNNNNN/
#   --hypothesis 可选（dump 是通用工具，写上让产物自描述）

# —— 预约 dump：启动时指定 update（可配合 --resume-from 精确复现）
train.py ... --resume-from ckpt_u280 --dump-at 282 [--dump-hypothesis "..."]
#   → 该 run 到 u282 时自动完整捕获（含 gradsig），request.json 记 source=cli

# —— 看 dump 的逐帧画面 / 验证回放一致性
debug.py render <dump_dir> --episode 0
#   → dumps/uNNNNN/record/episode_NNNNN/*.png + association.json + 校验日志

# —— 策略这几代漂了多少？（离线确定性回放，无需 run 在跑）
debug.py delta <dump_dir> --episode 0 --gens 3
#   → dumps/uNNNNN/delta/episode_NNNNN/（仅 CLI；viewer 不展示）

# —— 交互界面（给人看）
debug.py viewer [run_dir|dump_dir|runs_root] --port 8766

# —— 分析一个已捕获的 dump（dump_analysis.py，与 HTTP 端点同一实现）
debug.py inspect  <dump>                    # 总览：能力矩阵+各阶段摘要
debug.py samples  <dump> --sort neg_proj --limit 20 [--sign neg]
                                            [--group-by episode]   # 梯度样本表
debug.py trace    <dump> --buffer-index N   # 一帧贯穿 buffer→GAE→combine
                                            #   →gradsig→epoch→timeline
                  [--frame ep0007:robot_a:123]
debug.py timeline <dump> [--step N | --key-steps]
#   <dump> = dump 目录路径，或 <run>:u<N> 简写（--runs-root 下解析）
```

## HTTP API（viewer 运行时等价物）

模式：`runs`（多 run 索引）/ `run`（单 run）/ `dump`（单 dump）。

```
GET /api/mode | /api/catalog | /api/render-status
GET /api/runs?q=&page=&size=&sort=&order=                     (runs 模式)
GET /run/<name>/api/run/{info,dumps,metrics,videos}           (runs 模式)
GET /run/<name>/api/run/gradsig/<update>                      (runs 模式, 旧 run)
GET /api/run/{info,dumps,metrics,videos}                      (run 模式)
GET /api/run/gradsig/<update>                                 (run 模式, 旧 run)
GET /api/dump/<d>/<ep> 或 run 模式 /run/<n>/api/dump/<d>/<ep>：
    manifest | episode_list | traj_map | gradsig
    episode/<pos>/overview | episode/<pos>/series?keys=k1,k2
    episode/<pos>/frame/<f> | episode/<pos>/delta | image/<a>/<b>
    trajectory/<i> | trajectory/<i>/frame/<f>
    trajectory/<i>/epoch/<e>/overview|frame/<f> | trajectory/<i>/epoch_compare
    timeline/overview | timeline/step/<s>
    inspect                                   # dump_analysis.inspect_dump
    adv/hist                                  # ADV 变换链各阶段直方图
    gradsig/samples?sort&sign&limit&offset&group_by=episode
    trace/<buffer_idx>                        # 跨阶段单帧溯源
    pipeline                                  # dump 主页管线卡的汇总
    trajectory/<i>/gae?channel&gamma&lam      # 服务端复用 compute_gae 重算
    advnorm?channel&method&traj               # 复用 normalize_advantages 试算
    merge                                     # 通道合并：normed×conf×aw→combined
POST /run/<name>/api/run/dump-request   {hypothesis}          (running run)
POST /api/dump/<d>/render|delta         {episode[,gens]}      (单 job 槽)
```

`/compare/<runA>,<runB>,...`（runs 模式 SPA 页，无专属 API）——多 run
对比页：与 run 首页相同的六分区仪表板，每条线=一个 run（固定调色板、
跨图同色），x 轴=绝对 update（有数据就画、缺段自然空缺，不对齐长度）。
顶部 chips 增删 run（URL 同步可分享）+ 摘要表（succ≥0.5/0.9 的 update、
last10 eval、早停率、耗时）。单图 ≤12 线叠加（run 色+key 虚线），>12 线
按 run 拆并排面板（恢复 channel 配色）。入口：runs 索引 checkbox /
run 首页 "compare →" / 页内 chips。

## Viewer 页面结构（dump 层，2026-02 重构）

dump 以下按**变换管线**组织，不再是 episode/traj/timeline 平铺：

- **dump 主页 = 管线看板**：①Episode→Trajectory ②Reward→ADV ③ADV
  Normalization ④Channel Merge 四张汇总卡（`/api/pipeline`）+
  Update Process（timeline 摘要，"full timeline →" 下钻）+ Gradient
  Signal。每卡只放汇总 + `open tool →` 入口，**没有 episode 列表**。
- **工具页**（各自独立 URL，结构同构：traj picker + 大图 + 缩略图
  时间线 + scrubber + 页专属曲线）。分工原则：主页看汇总/分布，
  工具页只做 trajectory 级下钻——分布图不进工具页。
  - `/episode/<pos>` —— ①的钻取：与其它工具页同一个视频编辑器骨架
    （大图+缩略图+scrubber，帧游标共享）。下方 Tracks 卡 = episode
    轴上的多轨道区：先 scalar key 选择器（只枚举 episode 侧列：
    `observer.<o>.<f>` 标量 + `obs/actions/explore_factors.<aid>[i]`
    向量，`/episode/<pos>/overview` 给清单、`/series?keys=` 取序列），
    每条 key 一条 sparkline 轨；再每条 traj 一条 lane——lane 内画
    该 traj 的逐帧信号（每通道 reward 实线 + actor_weight 虚线，各自
    min-max 归一化到 lane 高，clip=[t_start,t_start+len) 之外留空），
    标签列显示 traj·agent·当前帧值，点击 lane 跳 `/gae/<i>`。
    所有轨道共享 playhead，scalar 轨点击设帧。帧级原始数据只在
    最底部 frame readout（obs/actions/observer dict 按需展开）。
  - `/gae/<i>` —— ②的钻取：视频编辑器布局（大图 + 缩略图时间线 +
    reward/value/δ/adv 曲线），γ/λ 滑杆触发服务端
    `/trajectory/<i>/gae` 用**训练同款 `compute_gae`** 重算
    （验证过与 dump 存储值逐位一致）。旧 `/trajectory/<i>` URL 仍
    兼容落到此页。
  - `/advnorm` —— ③的钻取：通道选择 + method tabs（trained 方法高亮），
    视频编辑器 stage + 单 traj raw vs normed 曲线，服务端
    `normalize_advantages` 现算。分布直方图只在主页汇总卡。
  - `/merge` —— ④的钻取：视频编辑器 stage + 单 traj per-channel
    normed/aw/combined 曲线 + 帧读数（每通道 conf/ev/terminated 在
    traj 表内）。合并的汇总与分布只在主页汇总卡。
- `/timeline` 保留为 update 内部 step 级钻取页。

复用红线：GAE 与 adv 归一化预览**只能**调
`baseline/framework/ppo/algos/advantages.py` 的 `compute_gae` /
`normalize_advantages`（`_normalize_adv` 已从 trainer 移入 algos，
trainer 与 viewer 共用同一实现），前端不做算法重写。

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
- `stats.grad_sig_*`（ADV 梯度信号诊断）：**dump-only**——只在带 dump
  请求的 update 上运行（内部固定 2000 帧采样，`sample_size`/`interval`
  公共参数已删除），数据并入 `dumps/uNNNNN/gradsig.npz`（hist +
  逐帧数组自包含），经 `_dump_gradsig(ds)` / `/api/dump/<d>/gradsig`
  读取。非诊断 update 不输出 `grad_sig_*` 字段（`grad_sig_ran` 标记）。
  Dashboard 不出现（suppress_prefixes 挡掉旧 run 的残留键）。
  `/api/run/gradsig/<u>` 只为旧 run 的 `gradsig/u*.npz` 服务，404
  `{available:false}` 不是错误。零范数帧不计入投影/余弦统计；
  旧 pairwise 格式的工件被当作不可用而非误读。诊断约 +5s/update，
  这也是它不属于常规遥测的原因。
- `param_overrides` 是行级元数据（dict），经 `_flatten_update` 透传、
  只在 Update Detail 显示，不进图、不在 `stats.*` 下。Update Detail
  现在只回答"本 update 生效参数"：调度值（stats 里的
  uncertainty_floor/coef、actor_lr、critic_lr）+ param_overrides；
  stats/pc/ep/time 数字表已删（信息与曲线重复）。Run Info 渲染
  common_params/ppo_params/reward_channels 全量字段，被 override
  过的字段打 ▲ 标记（依赖 metrics 加载完重渲染）。
- `dump_analysis.py` 是 dump 分析的**唯一计算源**：HTTP 端点、前端、
  CLI 都调它。语义红线（写进响应 meta）：gradsig 是 ≤2000 帧 θ_old
  抽样、proj<0=反向而非"坏样本"、dtheta_* 是 Adam 后的实际位移而非
  预更新梯度方向、缺失字段显式缺席不填 0、frame_id 为 flat:* 的帧
  无 episode 映射（`mapped:false`）。timeline overview 现已透传全部
  已采字段（dtheta_*/adv_*/argmax|min_ratio_bufidx/dual_clip_frac/
  floor_loss/mb_size/n_ratio_*），另派生 `key_steps` 快查索引。
- compare 页的 `buildMetricsCharts(null, cmp)` 与单 run 共用一套
  catalog 解析；cmp 序列带 `run/sub/subColor/subGroup` 字段供标签与
  拆分逻辑使用——改 emit/emitSpec 时两种模式都要过一遍。

## 文档索引

- `DATA_FLOW.md` —— dump 捕获管线数据流
- `../DESIGN_run_dashboard_zones.md` —— Run 页六分区指标系统设计文档
- `DESIGN_viewer_overview.md` + `DESIGN_viewer_scene1-4.md` —— viewer 设计
- 每个 dump 内的 `RECORD_GUIDE.md` —— 该截面的录制/复现命令
- 项目根 `CLAUDE.md` —— 训练框架总览；`experiments_ppo/README.md` —— 实验注册
