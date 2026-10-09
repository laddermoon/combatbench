# Floor 实验报告 — uncertainty_floor=0.4/0.6 sweep（vs uf0 / ef05 基线）

> 类型：产物

日期：2026-09-30
状态：24 run（8 格 × 3 seed）全部进入 eval≥95% 平台；因计算调度在 u355–568 提前停止
（中位 u475），统计按 resume 代际串联、重叠 update 由后代覆盖；u@100% 存在右删失。

## 1. 实验设计

- **任务**：`standup_floor04`（humanoid21 站立），PPO，目标 600 updates，eval 每 5 updates
- **处理**：`uncertainty_floor=0.4`（单分量格）/ `0.6`（MoG 格），`explore_factor=0`
- **对照臂**（同任务同格同 seed，来自 `RESULTS_truncnorm_sweeps.md`）：
  `sweep_uf0_*`（floor=0, ef=0）与 `sweep_ef05_*`（floor=0, ef=0.5）
- **指标**：
  - `u@50/u@95/u@100`：首个 eval success≥0.5/0.95/=1.0 的 update（eval=确定性策略，
    判据 `max_pot≥0.9` = episode 中**曾站起来过**，n=128 traj）
  - `online_success`：rollout 随机轨迹 `final_pot≥0.9`（**结尾还站着**，n=1024 traj/update）
  - AUSC：eval success 曲线均值（配对时截断到 floor run 的 last_u）
- **统计前提**：每格 n=3，格均值 SE≈50–75 update，单格排名不可信；
  结论建立在**配对差**（floor − 基线，同 cell 同 seed）与定性事实上。

## 2. 结论分级

### A 级 — 可信（CI 排 0 或为定性事实）

| # | 结论 | 证据 |
|---|------|------|
| A1 | **floor 不慢"站起来"的学习**：eval 里程碑与 uf0 无差异 | floor−uf0 配对 24 对：Δu50=+12 中位（CI[−17,+21]）、Δu95=+0（CI[−34,+15]）、Δu100=−25（CI[−58,+8]，21/24 对可比）——全部 CI 跨 0。地板提供的持续探索噪声没有加速 eval 到标 |
| A2 | **floor 显著慢于 ef05** | floor−ef05 配对：Δu50=+44（CI[+16,+73]✅）、Δu95=+43（CI[+9,+78]✅）、Δu100=+36（CI[−9,+80]）。ef 乘法增强（σ×1.5）在前期给的探索恰好够，恒定地板不是替代品 |
| A3 | **floor 破坏"站稳"质量**：online_success 大面积塌陷 | 7/24 run 全程未到 online≥0.95（uf0: 0/24、ef05: 1/24）；plateau online 均值 0.859 vs uf0 0.958 / ef05 0.950；配对窗内 Δonl 出现 −0.39 到 −0.67 的深亏（boundedstd_s42/s44、msb_s44、plain_s42、mixture_s43） |
| A4 | **eval=1.0 可以是"瞬间站立"的假象** | eval 判据是 `max_pot≥0.9`（曾站起，允许之后摔倒）。floor 臂出现极端分裂：`mixture_shared_boundedstd_s42` eval=1.0@u480 时 online≈0.10；`boundedstd_s44` eval=1.0@u465 时 online≈0.09——能瞬间站起但几乎从不在 episode 结尾仍站着。**达标判定必须看 final-pot/online 口径** |
| A5 | **地板把 σ/U 终态钉死在高位** | 末段 σ：floor 0.27–0.43 vs uf0 0.14–0.19；U：floor 0.30–0.53 vs uf0 0.16–0.30。机理上解释 A3：σ≥0.4/0.6 的恒定探索噪声使随机轨迹的末态站立率被机械性压低 |

### B 级 — 线索级

| # | 结论 | 证据与保留 |
|---|------|-----------|
| B1 | eval→online 的转化滞后是通用现象，但 floor 的缺口持久不收敛 | uf0 也有转移期分裂（`mix_shared_s42` 窗口 online=0.003、`mbs_s42` =0.041），但随后闭合到 0.9+；floor 的 boundedstd_s42/s44、msb_s42 的缺口直到被杀都没合上。方向一致、机理吻合，但"持久不收敛"单点观察，需更久窗口复证 |
| B2 | floor 伤害集中在 bounded-σ / shared-σ 格 | 受损最深的 run（Δonl −0.4~−0.67）全在 boundedstd、mixture_shared_boundedstd、plain 格——这些格自然 σ 收敛最小，恒定地板相对抬升最大。机制一致但非配对显著 |
| B3 | floor 组提前终止不影响方向结论 | 末窗（各 run 最后 100u）口径 floor eval 0.773/online 0.627 vs uf0 0.975/0.975：多个 run 末段仍在下滑（boundedstd 0.402、msb 0.369），不是"还没来得及学会"。但 P(eval=1.0) 对比（0.20 vs 0.47）受短 plateau 窗口影响，不单独作据 |

### C 级 — 观测记录，不作结论

- `mixture` 格是唯一 floor ≥ ef05 的 cell（u@100 中位 325 < 345）——单格 n=3，可能噪声
- 95→100 徘徊间隔三臂一致 ~30–40u：到不了 1.0 是 ~2% 残余失败率等 eval 采样全过的现象，与探索机制无关
- floor plateau P(eval=1.0)=0.20 vs uf0 0.47 / ef05 0.59——受窗口长度污染，仅记录
- `boundedstd_statesig_s44` 是唯一 eval≥95% 后 70+u 未打 1.0 的 run

## 3. 合成判断（给决策用的摘要）

**恒定 uncertainty_floor 在本任务上是净负收益，不建议按当前形态使用。**

eval 侧它与什么都不加（uf0）打平、显著慢于 ef05；online 侧它把 ~1/3 的 run 的
"结尾站稳率"永久压低。机理清楚：恒定 σ 地板=随机策略永远带噪采样，"瞬间站起"
（eval 能抓到）与"站稳到结尾"（任务的真语义）被解耦——地板越高于该格自然收敛
σ，损伤越大。

**若要保留 floor 思想**：恒定地板不行，可行方向是退火（floor 随 update 线性→0，
前期保探索、后期释放终态质量）或仅 warmup 期启用。

**方法论副产物**：eval `max_pot≥0.9` 判据过度授信——"曾在 episode 中站起"
不等于"站稳"。达标判定应改/加 `final_pot≥0.9` 或直接用 `online_success`。

## 4. 附录：里程碑原始数据

### floor 批次（sweep_floor_*，各 run 串联 2–5 代 resume）

| 格 | s42 u50/u95/u100 | s43 u50/u95/u100 | s44 u50/u95/u100 | best(3sd) | AUSC 中位 | onl_plat 中位 | onl@95 |
|---|---|---|---|---|---|---|---|
| plain | 505/525/n/r | 295/330/365 | 380/420/480 | 0.977/1.0/1.0 | 0.239 | 0.964 | 1 n/r |
| statesig | 350/460/525 | 310/405/445 | 470/490/525 | 1.0×3 | 0.334 | 0.948 | 3/3 |
| boundedstd | 450/455/490 | 285/320/365 | 450/465/465 | 1.0×3 | 0.250 | 0.427 | 1/3 |
| bnd_statesig | 380/450/535 | 325/430/545 | 445/490/n/r | 1.0/1.0/0.984 | 0.296 | 0.854 | 2/3 |
| mixture | 270/290/290 | 370/405/475 | 285/315/325 | 1.0×3 | 0.267 | 0.939 | 3/3 |
| mix_bnd_statesig | 275/310/330 | 320/380/475 | 325/360/380 | 1.0×3 | 0.251 | 0.968 | 2/3 |
| mix_shared | 360/365/420 | 385/415/445 | 310/335/360 | 1.0×3 | 0.160 | 0.875 | 3/3 |
| mix_shared_bnd | 455/465/480 | 320/350/395 | 320/325/335 | 1.0×3 | 0.283 | 0.450 | 1/3 |

### 配对差汇总（floor − 基线；Δ>0 = floor 更慢/更差）

vs **uf0**（24 对）：Δu50=+12 中位（CI[−17,+21]）｜Δu95=+0（CI[−34,+15]）｜
Δu100=−25（CI[−58,+8]，21/24）｜ΔAUSC=−0.024 中位

vs **ef05**（23 对）：Δu50=+35 中位 / 均值+44（CI[+16,+73]✅）｜Δu95=+45 中位 /
均值+43（CI[+9,+78]✅）｜Δu100=+22 中位 / 均值+36（CI[−9,+80]，20/24）

online plateau 配对窗 Δonl（floor−uf0，窗=[floor 首达 ev95, floor last_u]）：
13 负 / 11 正但深亏集中：msb_s44 −0.53、bnd_s44 −0.67、bnd_s42 −0.44、
plain_s42 −0.48、mixture_s43 −0.39、bs_s42 −0.14、bs_s44 −0.17、mbs_s44 −0.20

### σ/U 终态（末 20 update 均值，3 seed 平均）

| 格 | uf0 σ / U | floor σ / U |
|---|---|---|
| plain | 0.167 / 0.195 | 0.281 / 0.308 |
| statesig | 0.139 / 0.163 | 0.388 / 0.340 |
| boundedstd | 0.170 / 0.199 | 0.272 / 0.305 |
| bnd_statesig | 0.157 / 0.183 | 0.357 / 0.346 |
| mixture | 0.167 / 0.266 | 0.425 / 0.535 |
| mix_bnd_statesig | 0.154 / 0.248 | 0.393 / 0.532 |
| mix_shared | 0.184 / 0.291 | 0.326 / 0.527 |
| mix_shared_bnd | 0.194 / 0.303 | 0.332 / 0.517 |

## 5. 方法学注记

- run 串联：base→resumed→resumed2/3/4 各代在重叠 update 处**后写代际覆盖**
  （resume 从 ckpt 重算区间以续跑值为准，等价性已由 r1/r2 u56 位级一致实测背书）
- `u@100%` 右删失：floor 组 2/24 run（plain_s42、bs_s44）被杀时未达 1.0，
  配对差中排除；真实 Δu100 只能更差
- eval=128 traj（64 ep × 2 agents），粒度 1/128≈0.0078；u 里程碑 ±5u 量化误差
- **eval 与 online 语义不同**：eval=max-pot（曾站起）、online=final-pot（结尾站着）
  +随机探索策略。两者在 plateau 上分裂到 0.4–0.9 的 run 存在——跨口径对比前先确认语义
- uf0 批为 2 并发×96 worker、ef05/floor 为 24 并发×8 worker——调度差异不影响
  实验条件（每 update 512 ep 数据量相同、seed 派生链与调度无关）
