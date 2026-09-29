# M1：独立验证工具与 AI 转换输入规范

## 1. 入口与证据边界

参考任务与四项训练验收门槛见 [M0_BASELINE.md](M0_BASELINE.md)。本工具只提供**单样例**参考采集、候选执行、分字段比较与重放；`pass` 不代表整个 V1/V2 层级通过，更不代表训练或加速通过。当前没有接入 MJX 候选，也未执行 GPU 基准或训练。

实现入口：

- `validation.py`：typed JSON 样例、来源指纹、严格比较、JSON 报告、CLI。
- `validation_cpu.py`：调用现有 CPU rewarder / Standup trajectory 构造器 / simulator 的独立 oracle，不能复制它们的业务公式代替参考。
- `../../tests/test_batch_validation.py`：正确样例与注入错误；不加载 MJX/JAX，不启动 rollout worker。

## 2. 已实现的支持边界

| 输入 kind / operation | 执行器 | 范围 | 明确未覆盖 |
|---|---|---|---|
| `logic / reward`（V2） | `StandingBalance4StageRewarder` | 相同 body 位置、四元数、接触分类和力，比较全部 9 个输出；a/b 均支持 | 不证明接触力本身的物理正确性 |
| `logic / trajectory`（V2） | `StandupFloor04.build_trajectories` | 合成双 agent 两帧 Episode，timeout、final obs、reward、sampling ctx | 完整 collector、GAE/PPO 更新、真实 200 步 episode |
| `physics` 或 `action_sequence`（V1） | `Humanoid21Simulator` | 完整 `mjSTATE_INTEGRATION` + 控制目标 + 固定双机动作，1/25 子步；qpos/qvel、core、96 维观测和空 sensor | 原始 contacts 语义匹配、全部 derived 字段、外力接口、batch 隔离 |
| `policy_eval`（预留 V3 输入类别） | 无 | 明确 `unsupported` | 策略推理、reset 分布、交叉评估需 M4 扩展 |
| 未登记逻辑 operation / 非空 physics plugins | 无 | 明确 `unsupported`，不静默跳过 | RandomFallen/Timeout 等不在物理探针执行循环内 |

物理探针的状态包含 MuJoCo integration state（包括控制、外力、warm-start 等），另外保存 simulator 的归一化动作目标。恢复时 `mj_setState → mj_forward → mj_setState`，重建 derived 数据并恢复 forward 可能改变的积分输入。它定义的是**规范化恢复后的输入**，不是任意运行现场 Python 缓存的逐位恢复。不可把只保存 core state 当作完整物理状态。

奖励样例覆盖四阶段、1N/10N 力阈值、0.52m 距离门槛、0.15/1.28m 高度、f_score=0.8、重复 body 去重、墙不等于地面和 robot_b。它们是人工构造的**逻辑特征**，不是宣称物理可达的姿态。

CPU 自比较只证明重放与工具链自洽；另外用手工已知答案验证基础阶段和阈值，并注入错误验证比较器。不能把 CPU 自比较冒充 CPU/MJX 对照。

## 3. 可直接运行的命令

从 `things/combatbench` 运行；禁用 CUDA、限制 BLAS 线程，避免挤占当前训练资源：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B -m pytest tests/test_batch_validation.py -q
```

采集一个奖励参考（父目录必须存在，目标文件必须不存在；不会覆盖原有证据）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B -m envs.batchframework.validation capture /tmp/m1_reward.json \
  --case reward --index 2
```

stdout 返回 `fixture` 路径与 SHA-256 `digest`。由验收方将 digest 另存到受控记录；下面的 `<approved-sha256>` 使用该值：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B -m envs.batchframework.validation replay /tmp/m1_reward.json \
  --digest <approved-sha256> --candidate envs.batchframework.validation_cpu \
  --output /tmp/m1_reward_report.json
```

也可以用 `--index <index.json>` 代替 `--digest`：按 fixture 文件名在索引中查找审核值。

其他参考入口：`capture ... --case physics --steps 1`、`--steps 25`、`capture ... --case trajectory`。reward `--index` 范围为 0–24（具体 ID 以 `reward_cases()` 为准）。

已提交的最小参考集在 `validation_fixtures/`（28 个 fixture + `index.json`）。源文件或环境变化后由验收方重新生成并审核 index diff：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B -m envs.batchframework.validation make-fixtures envs/batchframework/validation_fixtures
```

逐条重放该集合（以 index 为审核 digest 来源，CPU 自重放只验证工具链自洽）：

```bash
for f in envs/batchframework/validation_fixtures/*.json; do
  [ "$f" = "envs/batchframework/validation_fixtures/index.json" ] && continue
  python3 -B -m envs.batchframework.validation replay "$f" \
    --index envs/batchframework/validation_fixtures/index.json \
    --candidate envs.batchframework.validation_cpu || break
done
```

`replay` 不提供 `--candidate` 时返回 `not-run`。仅 `pass` 的退出码为 0；fail/unsupported/not-run/stale 均为 1。输入 JSON/schema 无效或文件错误直接报错，不转换成成功。报告包含完整重放命令；不自动重跑昂贵训练。

## 4. 样例、适配器与报告契约

每个 fixture 包含：

- `schema_version=1`；case 固定字段 `id/kind/level/seed/episode/frame/input`。
- `expected`：参考执行器产出的树状数据。ndarray 保存 `dtype/shape/__array__`，不丢 dtype、不用 pickle。
- `source`：backend、版本标签、Git commit、明确列出的源码/模型/配置文件 SHA-256、Python 与相关依赖版本。
- `validator`：比较器自身 SHA-256。
- `tolerance` 和 `field_tolerances`：由验收方冻结。M1 CPU 自重放默认 atol=rtol=0；这是数值精确比较，不是原始内存字节比较（例如 ±0）。

Git commit 是采集时 HEAD 的出处信息；实际工作区代码由文件指纹标识，不能用 commit 替代指纹。CPU manifest 是**按这三类 fixture 的语义链策展**的清单：`envs/framework` 与 `envs/humanoid21` 全部源码、trajectory 契约实际加载的 baseline 模块（`ppo` 的 experiment/sampling_context/stochastic_policy/trajectory、`rollout` 的 episode/job/observer_utils、`critic_mlp`、`experiments_ppo` 的 base/exp_standup/exp_standup_floor04）、rewarder、blueprint、XML 及其引用纹理、`M0_BASELINE.md` 和两个验证文件。刻意不纳入 PPO trainer/loop/debug 与无关 experiment——它们不改变这三类 fixture 的参考输出，纳入只会产生误报 stale。适配器新增 case 类型时必须重新审查并扩展自己的依赖声明（M5 的 PPO 更新级 fixture 需补 trainer/algos）；M1 不做任意 Python 的完整依赖分析。

候选模块必须导出 `Adapter()`；实例实现以下显式协议：

```python
backend = "your-backend"
version = "your-version"
dependencies = [source_path, model_path, config_path]

def execute(self, case):
    ...
```

`execute` 只收到 case 的独立副本，**不会收到 expected 或 tolerance**。只返回数据，不返回自评 pass/fail；不支持时必须抛出 `envs.batchframework.validation.Unsupported`——从规范模块路径导入，不得自行定义同名异常（`python -m` 入口已对 `__main__` 建立模块别名保证类同一性，但自定义入口不经该别名时必须确保同一模块实例）。其他异常转换为 `fail/execution`，保留异常类型与消息。候选 CLI 模块是用户选择的可信 Python 代码；这不是防恶意程序的进程沙箱。

比较规则：

- key 集合、type、shape、dtype、序列长度必须相同；不广播、不截断、不填默认值。
- bool/int 离散值精确比较；浮点要求 finite 且 `abs(actual-expected) <= atol + rtol*abs(expected)`。
- `field_tolerances` 使用 `output.*` 路径，最长前缀匹配；例如 `output.robot_a.potential`。拼错且未使用的路径报错。
- 每条失败记录包含 field（含 agent/数组索引）、expected、actual、reason、tolerance；报告顶层包含 case ID、seed、episode、frame、level、source、target、fixture digest 和 replay 命令。
- 输出 `scope=single-case`；列表帧下标写在字段路径中。不存在的字段显式标为 absent，不与零值混淆。

## 5. 来源过期与审核责任

以下情况在候选执行前返回 `stale`：fixture digest 不等于审核值、参考源/模型/配置/validator 文件变化或丢失、参考依赖版本变化。不可通过重采集覆盖旧样例来绕过失败。

M1 指纹使用当前 checkout 的绝对路径；移动 checkout 后先由验收方核验内容并迁移记录，不能自动认可新位置。目标依赖版本与文件指纹写入每次报告；目标修改后必须重新执行，旧报告不是自动刷新状态的证书。完整跨机器 manifest、旧报告失效扫描与能力注册表产品化属于 M8。

容差变更、oracle 修正或增补用例由验收方审核，新 digest 独立登记。转换 Agent 不得改比较器、参考实现、fixture、index、容差或测试预期来取得通过。digest 与文件都由同一 Agent 随意修改就没有独立验收意义；此工具提供检测机制，不代替仓库审核权限。

## 6. 给转换 Agent 的输入包和工作流

开始前必须获得：

1. 源实验标识、代码版本与主参考 Run；M0 契约及当前审核 fixture index。
2. 完整 blueprint（解析后的参数与 episode options）、模型、meta、插件和 observer 列表。
3. 目标 backend/版本/精度/batch 形状及明确支持矩阵，允许改动的候选文件范围。
4. 所需等级（V0–V5）、审核容差和训练级门槛；可用计算资源。
5. 策略/状态/动作样例、随机流约定、控制与 observer 时序、timeout/bootstrap 约定。

执行顺序：

1. 逐项列出必需能力：原生 / 宿主兼容 / 待转换 / 不支持。未知插件或配置不得省略；不能因为同名类存在就宣称支持。
2. 先运行 CPU oracle 重放验证工具和输入来源；stale 时停止并报告，不自行刷新基准。
3. 只实现候选适配器及获准的执行代码；复用 PPO 与源任务参数，不复制一套独立演化的奖励规则。
4. 先运行逻辑特征对照，再运行受控状态物理对照；把公式错误与动力学差异分开定位。
5. 按报告的 case/field/index 修正候选并重放同一输入。对于未知能力抛 Unsupported；对于 NaN/缺字段/异常明确 fail。
6. 交付候选指纹、命令、报告和未执行清单；只声称实际通过的样例和等级范围。

禁止：补零、漏插件、减少子步/接触、偷换初始化分布、把双机 min-height 早停改为各自早停、丢 final observation、timeout 清零 bootstrap、按测试结果调宽门槛。不能用“PPO 后来学会了”解释环境对照失败。

必须升级人工处理：源实现疑似有 bug；后端不支持真实模型；语义只有改任务才能实现；存在未解释的阈值跨越；参考策略系统性退化；依赖/模型/审核来源过期；需要扩大计算资源。

## 7. 后续扩展入口

M2：补 contacts 按 geom/body/位置匹配和容量检测、完整 derived/外力/写后刷新/部分 reset/batch 隔离、FP32/FP64 容差的独立批准。当前泛型数组比较不能用于假设接触数组同序的跨后端验收。

M4：CPU fallen-state 与真实策略中间帧采集、设备原生 reset 分布、policy_eval 执行器与固定策略交叉评估。正式留出种子集未生成，不能把当前 seed=42 的诊断样例称作留出集。

M5：完整 Job→Episode 采样时序、policy log-prob、GAE/PPO 和恢复测试。当前两帧 timeout 样例只防止局部数据契约错误。

M7：四项冻结门槛的多 seed 验收与端到端计时；M1 报告不能提供训练/加速成功结论。
