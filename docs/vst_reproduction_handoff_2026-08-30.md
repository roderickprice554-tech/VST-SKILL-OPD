# VST 3B 非 Ego4D 两阶段复现实验交接文档

更新时间：2026-08-30 20:41（Asia/Shanghai）
目标读者：接手该任务的新 Codex 对话
执行位置：远程服务器 `bujunru@10.130.140.10:52234`，项目文件不在本机

## 1. 新对话应先理解的结论

本实验目标是复现 VST 3B 的两阶段训练与完整 OVO-Bench 评测，但用户已经明确决定跳过所有 Ego4D 来源，因此实验的准确名称必须是：

> **VST 3B 非 Ego4D 复现实验**

不得把最终结果描述成“完整官方数据复现”。SFT 和 RL 都要排除任意路径层级中名为 `ego4d`（大小写不敏感）的媒体引用，包括嵌套在 `LLaVA-Video-178K/.../ego4d/...` 中的样本。

VST 固定下载清单已经完成，当前正在拼接、校验和解压训练媒体；OVO 下载仍暂停。尚未运行正式 smoke、SFT、RL 或完整 OVO 评测。不要根据 OVO 测试结果选择 checkpoint、prompt 或超参数。

### 1.1 当前进度摘要

已完成：

- 官方代码固定到 `f2500bb...` 并建立隔离 worktree；
- VST 官方数据固定清单 197/197 文件按路径和大小验证完成；
- 完整 SFT/RL 标注已到齐并完成规模/Ego4D 统计；
- 非 Ego4D 路径段过滤逻辑及测试已提交；
- 完整 OVO 3,035 条官方标注及哈希已确认；
- 每小时下载/阶段编排器持续运行。

当前运行：

- `setup_dataset.py` 正在拼接、SHA256 校验和解压 VST 媒体；
- OVO 下载进程仍为 SIGSTOP 暂停状态；
- 正式训练尚未启动。

未完成：

- VST 媒体准备完成 marker；
- 非 Ego4D 正式 manifest、seek 索引、媒体存在性和 split 泄漏终审；
- 因果流式 smoke runner；
- 3B 两卡 ZeRO-3 配置和正式 SFT launcher；
- 完整 OVO 剩余 13 个文件、解压和 evaluator smoke；
- SFT、SFT 后完整 OVO、RL、RL 后完整 OVO 和最终报告。

### 1.2 关键数据目录

```text
VST 原始固定快照:
/home/bujunru/vlm-repro/VST-full-reproduction/data/VST-Training-Data-official-5647583491c2

正在准备的非 Ego4D 媒体根:
/home/bujunru/vlm-repro/VST-full-reproduction/data/vst-training-media-no-ego4d

非 Ego4D manifest 目标:
/home/bujunru/vlm-repro/VST-full-reproduction/data_manifests/vst_no_ego4d_aaef152e

OVO 原始固定快照:
/home/bujunru/vlm-repro/VST-full-reproduction/data/OVO-Bench-official-fec29e3

OVO 解压目标:
/home/bujunru/vlm-repro/VST-full-reproduction/data/OVO-Bench-prepared-fec29e3

完整 OVO 官方标注:
/home/bujunru/vlm-repro/VST-full-reproduction/eval/eval_data/anno/eval/OVOBench/json
```

### 1.3 关键代码入口

```text
SFT 官方入口: VST-SFT/train.py、VST-SFT/run.sh
RL 官方入口: VST-RL/run.sh、VST-RL/recurrent/、VST-RL/verl/
官方评测入口: eval/eval_entry.py、eval/scripts/ovobench.sh
流式 vLLM 评测: eval/vllm_eval/vllm_eval_engine_stream.py
下载/阶段编排: audit/vst_download_orchestrator.py
每小时循环: audit/run_vst_download_monitor.sh
VST 准备: audit/prepare_vst_official.sh
非 Ego4D 审计: audit/audit_vst_no_ego4d.py
OVO 下载: audit/download_ovobench_official.sh
OVO 准备: audit/prepare_ovobench_official.sh
smoke 包装器: audit/run_vst_smoke.sh（其 Python runner 尚缺失）
```

### 1.4 远程连接

本机 PowerShell：

```powershell
ssh -p 52234 bujunru@10.130.140.10
```

进入项目：

```bash
cd /home/bujunru/vlm-repro/VST-full-reproduction
```

## 2. 实验目的与最终交付

目标是产生可追溯、可复跑的以下结果：

1. 非 Ego4D 全量 SFT 完成后的 3B checkpoint，在完整 OVO-Bench 上的结果；
2. 从冻结的 SFT checkpoint 启动非 Ego4D RL 后的 3B checkpoint，在同一完整 OVO-Bench 上的结果；
3. 数据来源、过滤规则、manifest、训练配置、代码版本、随机种子、checkpoint、评测协议和逐题结果全部可追溯。

最终约定路径：

- SFT 选定 checkpoint：`/home/bujunru/vlm-repro/VST-full-reproduction/checkpoints/vst_full/sft_selected/`
- SFT 后完整 OVO 结果：`/home/bujunru/vlm-repro/VST-full-reproduction/results/ovo_full_after_sft/`
- RL 选定 checkpoint：`/home/bujunru/vlm-repro/VST-full-reproduction/checkpoints/vst_full/rl_selected/`
- RL 后完整 OVO 结果：`/home/bujunru/vlm-repro/VST-full-reproduction/results/ovo_full_after_rl/`
- 最终报告：`/home/bujunru/vlm-repro/VST-full-reproduction/docs/vst_full_reproduction_report.md`

## 3. 不可违反的实验边界

- 基座固定为 3B：`/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct`。
- SFT 固定训练 1 epoch，不根据 OVO 分数选 checkpoint。
- SFT、RL、验证和测试严格隔离；OVO 的样本、答案、选项和人工分析不得进入训练、验证、prompt 调优或 checkpoint 选择。
- 不允许在 VST 训练媒体未完整到齐时按“已下载分片”启动正式 SFT，否则会改变全量 shuffle 和 1 epoch 语义。
- VST 数据完成准备、过滤、审计和 smoke 后，允许启动 SFT，并同时低优先级下载完整 OVO-Bench。
- RL 必须在 SFT 完成并冻结 `sft_selected` 后顺序启动，不能与 SFT 并行。
- 流式推理必须保持因果性：每步只能看到当前及之前帧，不能使用未来帧补记忆。
- Direct、SFT 和 RL 对比必须使用同一完整 OVO manifest、视频前缀、FPS、chunk、截止时间、解析和评分规则。
- 所有新产物写入新目录；不要覆盖旧 checkpoint、日志、manifest 或此前结果。

## 4. 远程仓库、版本与硬件

SSH：

```bash
ssh -p 52234 bujunru@10.130.140.10
```

干净官方复现 worktree：

```text
/home/bujunru/vlm-repro/VST-full-reproduction
branch: codex/vst-full-reproduction
HEAD at status capture: e55e71abad0c9f0b8041322f7118a1bcd431bea0
official base: f2500bb8699d59a13b96ec4229fc4cd643a96207
origin: https://github.com/1ranGuan/VST.git
```

关键提交：

- `0153358`：下载监控设计
- `a12a496`：下载监控实现计划
- `ceddaca`：VST/OVO 下载编排器
- `e0189ce`：非 Ego4D 数据准备与审计框架
- `20ae757`：修复嵌套 Ego4D 路径漏过滤
- `e55e71a`：新增完整复现实验交接文档

硬件：2 张 NVIDIA A100 PCIe 40GB。文件系统 `/dev/sdb1` 总计约 19TB；最近检查可用约 6.3TB，inode 充足。

Conda 环境：

```text
SFT:  /home/bujunru/.conda/envs/vst-sft311
RL:   /home/bujunru/.conda/envs/vst-rl
Eval: /home/bujunru/.conda/envs/vst-eval
现有审计工具: /home/bujunru/.conda/envs/vision-se
```

本地模型：

```text
SFT 基座: /home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct
官方发布 VST-3B: /home/bujunru/vlm-repro/models/VST-3B
```

## 5. 已完成事项

### 5.1 仓库与数据版本固定

- VST 官方数据源：ModelScope `catalan/VST-Training-Data`
- 固定 ModelScope revision：`aaef152ea68ffa0e9d9f7367ccf871ea2f699693`
- 对应 Hugging Face revision：`5647583491c298aa8b2926fe9910f651fc0e692d`
- VST 固定清单：197 个文件，1,697,359,144,663 bytes
- VST 清单文件：`audit/inventories/vst_modelscope_aaef152e.json`
- VST 清单 SHA256：`6279d48fae62ce61eb9fe7e92f33852744b6ca4526eb4f3f0b1075df73275d24`

- OVO 数据源：Hugging Face `JoeLeelyf/OVO-Bench`
- 固定 revision：`fec29e3`
- OVO 固定清单：22 个文件，199,618,901,601 bytes
- OVO 清单文件：`audit/inventories/ovo_hf_fec29e3.json`
- OVO 清单 SHA256：`a1bad9dc33c7c313b4c950b2fccc4ceba034bcdbda8ab3d9a64fe386ddb7dd70`

### 5.2 下载与每小时监控

后台编排器已部署：

```text
script: /home/bujunru/vlm-repro/VST-full-reproduction/audit/run_vst_download_monitor.sh
controller: /home/bujunru/vlm-repro/VST-full-reproduction/audit/vst_download_orchestrator.py
monitor PID at handoff: 1237826
```

状态与日志：

```text
logs/vst_download_orchestrator/status.json
logs/vst_download_orchestrator/events.jsonl
logs/vst_download_orchestrator/monitor.log
logs/vst_training_data_modelscope.log
```

下载器第一次遍历 198 个远端条目后有 28 个失败，编排器自动续传后已经完成 VST 固定清单 197/197。VST 下载进程已退出；不要再使用旧 PID。当前进入媒体准备阶段。

### 5.3 非 Ego4D 过滤规则

`audit/audit_vst_no_ego4d.py` 已按路径段、大小写不敏感地排除 Ego4D，相关测试通过。RL 中直接 `Ego4D/` 前缀为 1,080 行，但完整路径段统计为 1,094 行；必须采用 1,094，不能漏掉 14 条嵌套引用。

### 5.4 已确认的完整 OVO 标注

固定官方提交自带的完整 OVO 标注位于：

```text
eval/eval_data/anno/eval/OVOBench/json/
```

完整规模为 3,035 条：

| 类别 | 条数 | SHA256 |
| --- | ---: | --- |
| backward tracking | 631 | `6f039894736989676c95c4e1077c79b9e64d38ca411a09bfc976732ed34233d2` |
| real-time visual perception | 837 | `a83c1585a36cad22f0d79097d0124a15a16e079c2e09e1273f2310e509e58a23` |
| forward active responding | 1,567 | `7527112e31508207c10c43cec3a182410a5b62bab7fc475c229e79d8945c5944` |

此前 `/home/bujunru/.conda/envs/vision-se/ovo800/test387` 的 780 条（250/230/300）只是子集，禁止用于正式完整评测。

## 6. 当前正在运行的工作

状态文件最近一次记录（2026-08-30 20:02 Asia/Shanghai）：

```text
state: vst_preparing
VST: 197/197 文件按路径和大小验证，complete=true
VST directory bytes: 2,061,352,714,578（包含 parts、已拼接 zip 和缓存，不等于清单总量）
OVO: 9/22 文件按大小验证，103,536,398,997 bytes
OVO stopped: true
```

接手时进程：

```text
VST prepare wrapper: PID 2374820
setup_dataset.py: PID 2374822（运行中）
OVO downloader: PID 1162909（SIGSTOP 暂停，STAT 应为 T/Tl）
hourly monitor: PID 1237826
```

当前没有 `.complete` marker。VST 下载清单虽已完成，但媒体准备仍在运行：`hdvila.zip` 已由分片拼接成约 540GB 文件；`setup_dataset.py` 已累计读取约 598.6GB、写入约 540.0GB，正在阻塞式磁盘 I/O/校验阶段。`prepare_vst.log` 为空是 Python 重定向输出缓冲造成的，不能据此判断失败。媒体目标目录暂时仍接近空目录，说明尚未进入或完成正式解压。

检查最新状态：

```bash
cd /home/bujunru/vlm-repro/VST-full-reproduction
/home/bujunru/.conda/envs/vision-se/bin/python audit/vst_download_orchestrator.py --status
pgrep -af 'modelscope download.*catalan/VST-Training-Data'
pgrep -af 'hf download.*JoeLeelyf/OVO-Bench'
pgrep -af 'prepare_vst_official|setup_dataset.py'
tail -f logs/vst_download_orchestrator/prepare_vst.log
```

## 7. 实验数据接口

### 7.1 SFT

源目录：

```text
data/VST-Training-Data-official-5647583491c2/vst_sft_data/
```

每行 JSONL 是一个消息列表。典型结构：

```text
[
  {role: user, content: [{type: video, video, video_start, video_end}, {type: text, ...}]},
  {role: assistant, content: [{type: text_stream, ...}]}
]
```

完整官方发布标注现已包含 10 个 train 和 10 个 valid JSONL。按“行中包含 Ego4D 路径段”统计：

| split | 原始行数 | 排除 Ego4D | 非 Ego4D 保留 |
| --- | ---: | ---: | ---: |
| SFT train | 154,785 | 19,085 | 135,700 |
| SFT valid | 5,107 | 487 | 4,620 |

此前基于尚未完整下载的 7 个文件得到的 143,665 行统计已经过时，不得继续引用。

过滤后目标目录：

```text
data_manifests/vst_no_ego4d_aaef152e/
```

注意：当前过滤器会写 JSONL，但还没有实现官方所需的对应 `*_seeks.jsonl` 字节偏移索引生成。下载完成后的官方 `vst_sft_data/prepare_data.py` 只对其自身目录工作；接手者应在过滤目录中以相同格式生成 seek 索引，并加测试。

### 7.2 RL

源文件：

```text
data/VST-Training-Data-official-5647583491c2/vst_rl_data/train.parquet
```

字段：

```text
data_source: string
prompt: string
question: string
video_context: string
reward_model: {ground_truth: list[string], style: string}
extra_info: {duration: double, origin_id: string}
```

规模：

| split | 原始行数 | 排除 Ego4D | 非 Ego4D 保留 |
| --- | ---: | ---: | ---: |
| RL train | 11,000 | 1,094 | 9,906 |

SFT train + RL train 合计保留 145,606 / 165,785 行（87.8282%）。SFT 与 RL 之间媒体重合允许存在；需要严格检查的是 train/validation/OVO test 污染。

### 7.3 OVO-Bench

下载目录：

```text
data/OVO-Bench-official-fec29e3/
```

主要下载内容是 `chunked_videos.tar.part??` 和 `src_videos.tar.part??`。当前准备脚本只拼接并解压 `chunked_videos.tar.part??`：

```text
audit/prepare_ovobench_official.sh
目标: data/OVO-Bench-prepared-fec29e3/
```

标注字段：

- backward/real-time：`video, start, end, task, subtask, question, candidates, answer`
- forward：`video, start, end, task, subtask, question, answer`，没有 candidates

官方 evaluator 明确规定：backward 和 real-time 是 MCQ 字母评分；forward 不是统一 MCQ，`SSR/CRR` 使用字符串包含匹配，`REC` 提取唯一整数后精确匹配。用户最初要求“三类总体 MCQ letter accuracy”与官方 forward 实现存在冲突；必须以官方 evaluator 为准并在报告中披露，不能把 forward 静默改造成字母选择题。

## 8. 官方训练协议与当前适配决定

### 8.1 SFT 官方参数

官方 `VST-SFT/run.sh`：

- epoch：1
- per-device batch：1
- gradient accumulation：8
- intended topology：2 nodes × 8 GPUs
- intended effective global batch：128
- learning rate：`5e-6`
- warmup ratio：`0.03`
- optimizer：AdamW Torch
- scheduler：cosine
- bf16/tf32/gradient checkpointing：开启
- save steps：25；eval steps：50
- `text_sink=512`
- `TEXT_SLIDING_WINDOW=32768`
- `FPS_MAX_FRAMES=384`
- 训练代码会冻结 `visual`/`vision_tower`，其余为全参训练，不是 LoRA

用户固定使用 3B，因此基座必须改为：

```text
/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct
```

两张 GPU 若保持官方有效 global batch 128，应使用：

```text
2 GPUs × per_device_batch 1 × gradient_accumulation 64 = 128
```

这只是等价硬件适配，必须写入配置差异。

### 8.2 SFT 官方代码缺口

正式 SFT 启动器尚未实现，原因如下：

1. 官方 `run.sh` 默认模型是 `Qwen2.5-VL-7B-Instruct`，不是目标 3B；
2. 数据路径仍是 `/workspace/...` 占位符；
3. `TRAIN_DATASET_NAMES` 仍是假定的单文件 `train_with_seeks.jsonl`，而发布数据是多个 JSONL；
4. 引用了 `./scripts/zero3.json`，但固定官方提交没有发布 `VST-SFT/scripts/zero3.json`，服务器也未找到副本；
5. 当前 `audit/run_vst_smoke.sh` 只检查 `audit/run_vst_smoke.py`，而该 Python runner 尚不存在，会以 exit 78 退出。

不得静默编造官方 ZeRO-3 配置。需要选择一个有来源的标准 ZeRO-3 配置，记录其来源和仅影响分布式基础设施、不改变优化算法的差异，然后先 smoke。

### 8.3 RL 官方参数与未决适配

官方 `VST-RL/run.sh` 关键参数：

- 4 nodes × 8 GPUs（服务器只有 2 GPUs）
- GRPO；rollout group `n=8`
- train batch 256；PPO mini batch 64
- actor LR `5e-7`
- rollout temperature 1，top-p 0.98
- KL loss coefficient 0.01；KL controller coefficient 0.001
- vision tower frozen
- save/test frequency 10；total epochs 1
- FSDP size 8；CPU parameter/optimizer offload

两卡 RL 等价适配尚未完成。特别是 `fsdp_size=8`、batch/group/update 语义和 vLLM rollout 并行必须先审计，不能直接把 GPU 数改成 2。

## 9. 当前编排器的真实行为与最新用户意图的差异

当前 `decide()` 顺序是：

```text
VST snapshot complete
→ prepare VST
→ audit VST
→ 完成/恢复 OVO 下载
→ prepare OVO
→ smoke
→ terminal smoke_complete
```

它**不会启动 SFT/RL**，并且当前会等待 OVO 全部完成后才 smoke。

用户最新意图是：

```text
优先完成 VST
→ 非 Ego4D 准备/审计
→ smoke
→ 启动 3B SFT
→ SFT 运行期间同时恢复完整 OVO 下载
```

因此接手者在实现并验证 smoke/SFT launcher 后，还需测试驱动地修改编排器顺序。不要在 launcher 未审核前仅修改状态机，否则下载完成时可能自动启动错误训练命令。

## 10. 未完成事项与推荐执行顺序

1. 不要中断当前 `prepare_vst_official.sh`/`setup_dataset.py`。先观察进程、I/O 和目标目录增长，等待 `vst_prepare.complete`。
2. `setup_dataset.py` 会逐个拼接、按 sidecar SHA256 校验并解压归档；VST 固定清单的路径/大小检查已完成，但归档级完整性仍以该脚本结果为准。
3. 准备完成后运行正式非 Ego4D 审计，生成 SFT/RL manifest、seek 索引、唯一性、缺字段、空媒体、视频时长和 split 泄漏报告。**不要在媒体准备完成前运行现有 `audit_vst_no_ego4d.py`**：它会把暂时性 missing 缓存到正式 `MANIFEST_ROOT`。
4. 实现只读 smoke：数据读取、官方 FPS/chunk、截止时间、逐步因果 memory、三类 evaluator；smoke 只能用隔离的小样本验证流程，不能改变最终参数。
5. 补齐并记录 3B、2×A100 的 SFT ZeRO-3 基础设施配置；实现 `run_vst_sft.sh`，固定 1 epoch、有效 global batch 128、独立输出目录和完整 provenance。
6. 修改编排器为审计+smoke 通过后启动 SFT，并同时 `SIGCONT`/续传 OVO；OVO 下载应降低 CPU/磁盘优先级，若影响训练吞吐可暂停。
7. 完成 OVO 剩余 13/22 文件、拼接、解压和完整 3,035 条 manifest 校验。
8. SFT 结束后只按训练/独立验证选择并冻结 `sft_selected`；完整 OVO 只读测试一次。
9. 完成两卡 RL 等价语义审计与 smoke，再从 `sft_selected` 启动 1 epoch 非 Ego4D RL。
10. 用完全相同的完整 OVO manifest 和 evaluator 测试 `rl_selected`，写最终报告。

## 11. 已知风险与必须披露的偏差

- Ego4D 被主动移除：这是数据组成偏差，正式命名必须保留“非 Ego4D”。
- 完整官方 SFT 标注比此前部分下载统计多 6 个 JSONL；旧总数不可用。
- 官方 SFT 缺失 `zero3.json`，且 run.sh 含模型/路径/文件名占位符。
- 官方 32-GPU RL 到 2-GPU 的等价更新语义尚未解决。
- 固定官方 OVO 标注为 3,035 条；780 条缓存是子集。
- OVO forward 的官方评分不是 MCQ 字母评分，和用户最初描述有口径冲突。
- 当前固定清单验证只检查路径和文件大小，最终报告需要 SHA256/归档验证。
- 文件系统为共享盘；最近约 6.3TB 可用。当前同时保留分片和已拼接 zip，因此准备期间占用会明显增长。需要持续监控解压和 ZeRO-3 checkpoint 占用，建议低于 2TB 时阻止下一阶段启动。

## 12. Git 与文件安全

当前 worktree 有用户/实验产生的未跟踪目录和文件，必须保留：

```text
.venv-modelscope/
data/
audit/download_ovobench_official.sh
audit/run_full_inventory.py
audit/vst_full_20260828_audit_v1/
docs/vst_full_reproduction_plan.md
tests/test_run_full_inventory.py
```

不要执行 `git reset --hard`、`git clean` 或删除下载缓存。提交时只精确 `git add` 本次修改文件。

## 13. 新 Codex 的首轮检查清单

接手后先运行只读命令：

```bash
cd /home/bujunru/vlm-repro/VST-full-reproduction
git status --short
git rev-parse HEAD
/home/bujunru/.conda/envs/vision-se/bin/python audit/vst_download_orchestrator.py --status
pgrep -af 'modelscope download.*catalan/VST-Training-Data'
pgrep -af 'hf download.*JoeLeelyf/OVO-Bench'
df -h .
```

然后回答以下问题，确认文档没有被误读：

1. 当前实验为什么不能叫完整官方数据复现？
2. 正式 OVO 是多少条，780 条目录能否使用？
3. 当前有没有启动训练？
4. 为什么不能在 VST 媒体未完整时训练已下载分片？
5. 哪些审计可以与下载并行，哪些必须等待下载/解压完成？
6. SFT 为什么使用 gradient accumulation 64？
7. 官方 SFT 启动前还有哪些阻塞？
8. 当前编排器是否已经实现 SFT 与 OVO 下载并行？
9. forward active responding 使用什么评分，不应被改成什么？
10. 下一步的只读检查和安全动作是什么？

若新对话无法从本文准确回答这些问题，应先补充交接信息，不要启动训练。

## 14. 2026-08-31 训练优先编排更新

用户已确认训练优先，并授权在审计与 smoke 通过后自动启动固定的 Qwen2.5-VL-3B-Instruct 两卡 SFT。OVO 只能使用未分配给训练的整张物理 GPU，不能使用训练卡上的剩余显存。

新增提交：

- `56ff9a4`：训练优先编排设计；
- `d0a3140`：测试驱动实施计划；
- `eb23192`：训练优先状态机与 30 分钟轮询；
- `9c82326`：非 Ego4D seek 与缺失媒体门禁；
- `9e5f683`：修复 OVO 包装器重试间隙被误判、重复启动；
- `84b6029`：真实数据解码与 3B 权重加载 smoke gate；
- `81ce0b9`：固定 3B、2×A100、1 epoch、有效 global batch 128 的 SFT launcher；
- `4b9ea30`：完整 3,035 条 OVO 验证和 Qwen 3B 基线评测包装器。

最新验证（2026-08-31 00:23 Asia/Shanghai）：

- `pytest tests -q`：43 passed；
- 所有修改的 Bash 包装器通过 `bash -n`；
- 所有修改的 Python 文件通过 `py_compile`；
- DeepSpeed 0.17.1 自带 `autotuning/config_templates/template_zero3.json` 是新增 ZeRO-3 配置的记录来源；
- 30 分钟监控主 PID：2507143；`sleep 1800` 子 PID：2507176；
- VST 准备仍在运行：包装器 PID 2374820，`setup_dataset.py` PID 2374822；尚无 `vst_prepare.complete`；
- OVO 单下载链仍在运行，控制器 dry-run 识别 PID 2507154；9/22 固定文件完整；
- 当前无 smoke、SFT 或 OVO 推理进程；两张 GPU 空闲；
- 文件系统剩余约 5.5TB。

自动顺序现为：VST 准备完成 → 非 Ego4D 审计/seek → 因果 smoke → 两卡 SFT；OVO 下载独立维护，SFT 完成且完整 OVO 校验通过后，才在空闲整卡运行 Qwen 3B 基线评测。任一门禁失败都会停止推进并保留日志，不会绕过。
