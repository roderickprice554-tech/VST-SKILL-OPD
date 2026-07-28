## 🚀 RLSD 评估指南

本目录是 VST 模型评估体系：基于 **vLLM** 的多GPU数据并行评估引擎，覆盖 StreamingBench / OVOBench / VideoHolmes / LongVideoBench / Video-MME 五个视频理解benchmark，支持标准单轮问答评估和 **StreamThink 流式推理评估** 两条路径。

> 注：这不是 `VST/eval` 那套基于 `lmms-eval` + `accelerate` + HuggingFace `transformers.generate()` 的旧流程（那套走的是 `qwen*_vl_stream_think` 模型类）。本目录是完全独立的、面向 verl/RLSD checkpoint 的 vLLM 原生实现，二者的分段/并行/加速机制均不同，不要混用命令。

---

### 1. 目录结构

```
RLSD/eval/
├── eval_config.py              # 统一配置：GPU列表、采样参数、流式分段参数、数据集路径、prompt模板
├── checkpoint_queue.csv         # 待评估ckpt队列（exp_name/key_config/ckpt_path/created_at/evaluated）
├── vllm_eval/
│   ├── vllm_eval_engine.py      # 【标准】非流式batch评估引擎 CLI 入口
│   ├── vllm_eval_engine_stream.py  # 【流式】StreamThink评估引擎 CLI 入口（本目录核心）
│   ├── worker.py                 # 标准评估的单GPU worker子进程（llm.chat批量提交）
│   ├── worker_stream.py          # 流式评估的单GPU worker子进程（分段think+并发样本）
│   ├── async_chat_client.py      # 基于 vLLM AsyncLLM 的封装，用于测量真实TTFT
│   ├── tasks.py                  # 任务注册表：doc_to_text / resolve_video / score_one / aggregate
│   ├── ckpt_utils.py              # verl FSDP分片ckpt自动合并成HF safetensors
│   ├── checkpoint_queue.py        # 读取/回写 checkpoint_queue.csv 的 evaluated 标记
│   ├── progress.py                # 多进程JSON进度文件读写+轮询汇总
│   ├── result_store.py            # 汇总结果落盘（CSV + 免依赖XLSX）
│   └── demo/render_sample_walkthrough.py  # 把某条样本的完整流式过程渲染成可读Markdown
├── results/eval_summary.{csv,xlsx}   # 每次评估的累计汇总（跨ckpt/跨task）
├── eval_outputs/                 # 各次运行的详细产出（shard结果、samples.jsonl、summary.json）
└── eval_data/anno/eval/           # 标注JSON（含全量数据 + 按比例抽样的 subset7500 子集）
```

两套引擎共享同一个 `tasks.py` 任务定义/评分逻辑和 `eval_config.py` 参数源，区别仅在于是否做"分段流式思考"：

| 引擎 | Worker | 是否流式思考 | 适用场景 |
|---|---|---|---|
| `vllm_eval_engine.py` | `vllm_eval.worker`（同步 `llm.chat()` 批处理） | ❌ 整段视频一次性问答 | 快速跑分、对照实验 |
| `vllm_eval_engine_stream.py` | `vllm_eval.worker_stream`（`AsyncLLM` + asyncio） | ✅ 分段"看一段想一段"，复刻VST StreamThink | 评估流式/长视频理解能力，测TTFT |

---

### 2. 环境准备

- vLLM 环境即可运行（不再依赖 `lmms-eval`/`accelerate`/`transformers.generate()`）。
- 数据集路径统一在 `eval_config.py` 的 `TASK_PATHS` 中配置（`anno_path` 标注JSON + `video_root` 视频目录）；若数据集迁移到别的挂载点，只需改这里，不用动 `tasks.py`。
- 待评估 checkpoint 可以是：
  1. 已经是标准 HF 格式（含 `config.json` + `*.safetensors`）的目录；
  2. verl actor 保存目录（`{save_checkpoint_path}/global_step_{N}/actor`，含 `model_world_size_*_rank_*.pt` 分片 + `huggingface/` 子目录）—— `ckpt_utils.resolve_hf_model_dir()` 会自动检测并调用 `scripts/model_merger.py` 合并成 safetensors（幂等，合并过一次后续直接复用）。

---

### 3. 任务名与任务组

单个任务名（`TASK_REGISTRY`，见 `vllm_eval/tasks.py`）：

- `streamingbench`
- `ovobench_real_time_visual_perception` / `ovobench_backward_tracking` / `ovobench_forward_active_responding`
- `videoholmes`
- `longvideobench`
- `videomme`
- 以上每个任务都有一个 `_subset7500` 版本，指向按比例抽样的 7500 条子集标注（`eval_data/anno/eval/subset7500/`），用于快速跑分对齐口径。

`--tasks` 支持逗号分隔多个任务名，也支持预定义任务组（自动展开）：

| 任务组 | 展开内容 |
|---|---|
| `ovobench` | 3个 OVOBench 子任务 |
| `all` | streamingbench + 3个ovobench + videoholmes + longvideobench + videomme（全量标注） |
| `all_subset7500` | 上面7个任务的 `_subset7500` 版本 |

`vllm_eval_engine.py` 的 `all` 组不含 longvideobench/videomme（早期版本），`vllm_eval_engine_stream.py` 的 `all`/`all_subset7500` 组是全量7个任务，以 stream 版本为准。

---

### 4. 标准（非流式）评估：`vllm_eval_engine.py`

整段视频一次性喂给模型，单轮 `llm.chat()` 问答，用于快速跑分/对照实验。

```bash
python vllm_eval/vllm_eval_engine.py \
    --model /path/to/global_step_93/actor \
    --tasks streamingbench,ovobench \
    --gpus 0,1,2,3,4 \
    --output-dir ./eval_outputs/my_run \
    --batch-size 64 \
    --max-num-frames 32 \
    --max-model-len 20480 \
    --gpu-memory-utilization 0.75
```

关键参数：

- `--model`：省略时会扫描 `--checkpoint-csv`（默认 `eval_config.CHECKPOINT_QUEUE_CSV`）中 `evaluated=0` 的行，逐个评估并在成功后回写 `evaluated=1`；显式传入则只评估这一个ckpt，不碰队列。
- `--gpus`：参与评估的物理GPU编号列表，**数据并行**——每张卡起一个独立 vLLM 引擎进程（`subprocess.Popen`，非 `multiprocessing`，保证每个worker拿到干净的CUDA上下文），按 `index % num_shards` 切分该任务的样本，互不通信。Qwen3-VL-8B 单卡（H800）就能放下，因此这里**不用张量并行（TP）**——之前的吞吐测试显示TP在这个模型量级上通信开销大于收益。
- `--batch-size`：单个worker一次 `llm.chat()` 提交的请求数，vLLM 的 continuous batching + PagedAttention 调度器会在这批请求内做并发解码，而不是一条条排队；视频解码（CPU-bound，opencv backend）也是在这次调用里触发的。
- `--max-num-frames` / `--max-model-len` / `--gpu-memory-utilization`：默认值取自 `eval_config.DEFAULT_SAMPLING` / `DEFAULT_ENGINE`，命令行传参可覆盖。
- `--limit`：调试用，切分shard前先截取每个任务的前N条样本。

执行链路：`vllm_eval_engine.py` 为每个 `(task, gpu)` 组合拼出一条 `python -m vllm_eval.worker ...` 命令 → 各 worker 独立加载模型、消费自己的shard、把结果写入 `shard_{id}.json` → 引擎轮询各shard的 `progress.json` 等待全部完成 → 合并所有shard的 `rows`、调用该任务的 `aggregate()` 算总分 → 写 `summary.json` + 追加一行到 `results/eval_summary.{csv,xlsx}`。

---

### 5. 流式（StreamThink）评估：`vllm_eval_engine_stream.py`

这是当前重点维护的评估路径，复刻 VST 原始 StreamThink 评估逻辑（视频切段 → 分段"预思考"生成文字摘要 → 摘要作为记忆参与最终问答），但用 vLLM `AsyncLLM` 重新实现，并针对多视频批量评估做了专门的效率优化。

```bash
python vllm_eval/vllm_eval_engine_stream.py \
    --model /path/to/VST-7B \
    --tasks all_subset7500 \
    --gpus 0,1,2,3,4,5,6,7 \
    --output-dir ./eval_outputs/subset7500_batch32_8gpu \
    --chunking-mode vst_original \
    --sample-concurrency 8 \
    --decode-timeout-s 180 \
    --max-model-len 98304
```

#### 5.1 推理主流程（单个样本）

1. **一次性决定全局帧预算**（`_sample_global_frame_indices`）：先用 `decord` 探测视频总帧数/fps，按 `--sample-fps` 在整段时间轴上均匀撒候选帧，候选数超过 `--max-num-frames` 再均匀抽稀到该上限。这一步只做一次，得到的是**整个样本**（所有中间思考请求+最终回答请求加起来）的帧数硬上限，不会因为切了几个chunk而成倍增加——这与非流式路径看到的总帧量是可比的。
2. **切分chunk**（两种模式，见5.2），得到若干 `{start_s, end_s, frame_indices, estimated_visual_tokens}` 描述的片段。
3. **全局单次解码**（`_decode_global_video_once`，当 `--use-predecoded-video 1` 时）：把第1步选中的帧号一次性用 `decord` 解码成 ndarray，按需做 `max_pixels` 限制的等比缩放；后续每个chunk只是对这个ndarray做切片，不再重新打开/解码mp4文件。
4. **逐个chunk做中间思考**：除最后一个chunk外，每个chunk单独发一次请求（`system prompt + 累积文本memory + 该chunk视频片段 + "描述这段视觉证据"的指令`），用 `chat_with_predecoded_video_ttft()` 提交给vLLM，取回的文本追加进 `textual_memory`；memory超过截断阈值（token_budget模式按字符数，vst_original模式按轮数）时做截断，只保留首段+最近若干段。
5. **最终回答**：最后一个chunk + 累积memory + 官方问题一起提交，记录这次请求的 TTFT（从"该样本视频输入完毕"到"模型吐出第一个token"的延迟）——这是本评估**唯一上报的headline延迟指标**，中间思考请求的TTFT只记录不计入统计。
6. 用该任务的 `score_one()` 打分，把完整过程（每个chunk的prompt/输出/memory变化、最终prompt/输出/GT/分数、耗时）写成一行JSON，追加到 `samples.jsonl`。

#### 5.2 两种切段模式（`--chunking-mode`）

| 模式 | 切段依据 | 说明 |
|---|---|---|
| `token_budget`（默认） | 按估算视觉token累积量切段：`frames_per_chunk = stream_token_budget // tokens_per_frame`（`--tokens-per-frame` 为单帧估算视觉token数），累积满一个 `--stream-token-budget` 就切一刀 | RLSD自有实现；`--max-chunks` 是安全上限，理论chunk数超过时把多余边界并入最后一段；memory按 `--max-memory-chars` 字符数截断 |
| `vst_original` | 按视频时长分档查表决定分几段（`--stream-think-times`，如 `2-3-5-5` 对应 ≤0.5min/0.5-4min/4-30min/≥30min 四档），再把固定帧集合均分成N段 | 复刻VST官方 `qwen2_5_vl_stream_think.py` 的分档逻辑，用于口径对齐/复现VST论文结果；`--max-stream-vid-tokens`/`--max-keep-memory` 仅在此模式下生效（分别对应VST的单轮像素预算基数、按轮数截断memory）；本例命令用的就是这个模式 |

两种模式共享同一份"全局帧预算"（步骤1），只是分段规则不同，因此在相同 `--max-num-frames` 下结果可直接对比。

#### 5.3 面向多视频批量评估的效率优化（重点）

这是相对于naive多轮`generate()`实现的关键加速点：

1. **全局单次解码 + chunk切片复用**（`--use-predecoded-video 1`，默认开）：避免vLLM每次 `llm.chat()` 收到 `file://` video_url 都要重新用 opencv/decord 打开并解码整段mp4——一个样本的所有请求（N个think + 1个final）共用同一次 `decord.VideoReader.get_batch()` 结果，chunk间只是数组切片。对于多chunk的长视频，这是最大的CPU侧节省。
2. **样本级并发（`--sample-concurrency`）**：每个GPU worker内部用 `asyncio` 维护一个"最多同时在飞 N 个样本"的调度器（`_run_concurrent_samples`）——样本A在等自己的下一个chunk的GPU生成结果时，样本B、C的视频解码/prompt构造（CPU侧）可以同时进行，vLLM的 `AsyncLLM` 引擎则把多个样本、多个chunk的生成请求放进同一个continuous batching队列里并发处理，而不是一个worker只能串行跑完一个样本的全部chunk再跑下一个样本。`--sample-concurrency 1` 退化为纯串行（等价于VST原始 `batch_size=1` 的语义），`8` 表示每卡最多8个样本同时处于"某个chunk请求在飞"的状态。这与"多GPU数据并行分shard"是两层独立的并行：GPU间分片 × GPU内样本并发。
3. **真实TTFT测量走 AsyncLLM 流式接口**（`async_chat_client.StreamChatClient`）：vLLM 0.11.0 同步 `LLM.chat()`/`LLM.generate()` 路径不会填充 `RequestOutput.metrics.first_token_time`，因此这里绕过同步API，直接用 `AsyncLLM.generate()` 的流式生成器，在收到第一个带 token 的 `RequestOutput` 时打时间戳，得到与引擎内部一致但可自行控制的TTFT。
4. **`--decode-timeout-s`**：给单样本的"探测视频信息"和"全局解码"两步加超时保护（`asyncio.wait_for`），避免个别损坏/超大视频文件卡死整个worker——超时会被 `_launch()` 的异常处理捕获，该样本记0分并继续处理队列里的其它样本，不影响整体进度。
5. **worker进程组隔离 + 级联清理**：每个GPU worker用 `start_new_session=True` 起独立进程组，引擎进程捕获 SIGINT/SIGTERM 时会对所有still-alive的worker进程组先 SIGTERM 再（超时后）SIGKILL，确保Ctrl-C不会留下孤儿vLLM EngineCore进程占着显存。

关键参数速查：

| 参数 | 作用 |
|---|---|
| `--sample-concurrency` | 单卡内并发样本数；1=串行基线，越大CPU/GPU重叠越充分，但受限于`--max-model-len`和显存 |
| `--decode-timeout-s` | 单样本视频探测/解码超时（秒），<=0禁用 |
| `--use-predecoded-video` | 1=全局单次decord解码+切片复用（推荐）；0=退化为每次请求走`file://`让vLLM自己解码 |
| `--decode-num-threads` | decord解码线程数，CPU核数充足可设4~8（`eval_config.py`里默认更激进，24） |
| `--max-model-len` | vLLM引擎的最大上下文长度；流式评估因为要塞多段视频token+累积memory，通常比非流式设置更大（本例98304） |
| `--think-max-tokens` | 中间思考请求的最大生成token数 |
| `--max-pixels` | 单帧像素上限，传给Qwen-VL的`mm_processor_kwargs`，避免原始高分辨率帧把视觉token撑爆`max_model_len` |

#### 5.4 输出产物

```
<output-dir>/<exp_name>/<chunking_mode>/concurrency_<N>/<task_name>/
├── shard_{i}.json              # 第i张卡的原始结果（rows）
├── shard_{i}.samples.jsonl     # 第i张卡每个样本的完整流式过程记录
├── shard_{i}.log                # 第i张卡worker进程的stdout/stderr
├── shard_{i}.progress.json      # 引擎轮询用的实时进度
├── samples.jsonl                # 所有shard合并后的完整样本记录（供分析/render_sample_walkthrough.py使用）
└── summary.json                 # 该task的聚合指标（accuracy、TTFT分位数、耗时、rows）
<output-dir>/<exp_name>/<chunking_mode>/concurrency_<N>/all_tasks_summary.json  # 该次运行所有task汇总
```

注意输出目录按 `chunking_mode` + `sample_concurrency` 分层，同一个 `exp_name` 跑 `token_budget` vs `vst_original`，或串行vs并发，不会互相覆盖。

想直观看某一条样本从头到尾发生了什么，可以用：

```bash
python -m vllm_eval.demo.render_sample_walkthrough \
    --samples-jsonl eval_outputs/subset7500_batch32_8gpu/<exp_name>/vst_original/concurrency_8/streamingbench_subset7500/samples.jsonl \
    --min-chunks 4 \
    --out-md /tmp/sample_walkthrough.md
```

---

### 6. 评分逻辑与任务定义（`vllm_eval/tasks.py`）

所有任务共享的核心概念：`TaskSpec`（一个benchmark JSON标注 + 视频根目录 + `build_prompt`/`resolve_video`/`score_one`/`aggregate` 四个回调）。评分口径与原始 `lmms-eval` tasks/*/utils.py 保持一致：

- **StreamingBench / OVOBench(MCQ子任务) / LongVideoBench / Video-MME**：标准MCQ，正则匹配裸字母答案，与GT字母比较。
- **OVOBench forward_active_responding**：非选择题，`SSR`/`CRR` 子任务走字符串包含匹配，其余子任务走"提取唯一整数"精确匹配（`far_acc`）。
- **VideoHolmes**：强制 `<think>...</think><answer>...</answer>` 格式，从 `<answer>` 标签中提取选项字母（找不到标签则退化用裸文本匹配）。

聚合统一走 `aggregate_task_subtask()`：整体准确率 = 正确数/已回答数（空答案不计入分母），并按 `task`/`subtask`/`subsubtask` 三级细分统计，同时给出各子任务准确率的算术平均（`subtask_acc_avg`，弥补子任务样本量不均衡）。

---

### 7. 汇总结果与ckpt队列

- 每次任务跑完都会往 `results/eval_summary.csv` 追加一行，同时写入 `results/eval_summary.xlsx`（每个task一个sheet，按accuracy从高到低重新排序；XLSX用标准库`zipfile`手搓，不依赖openpyxl）。
- 省略 `--model` 时走 `checkpoint_queue.csv` 队列模式：扫描 `evaluated=0` 的行，逐个（内部多GPU数据并行）评估，成功后回写 `evaluated=1`（带文件锁，避免并发写队列文件冲突）。同一时间建议只跑一个引擎进程消费队列。
- verl actor目录（含 `model_world_size_*_rank_*.pt` 分片）会被 `ckpt_utils.resolve_hf_model_dir()` 自动检测并调用 `scripts/model_merger.py` 合并成HF safetensors，合并结果幂等缓存在 `huggingface/` 子目录下，后续复用无需重新合并。

---

### 8. 并发推理与精度波动：原理分析

在贪婪解码（`temperature=0`, `top_k=1`）且数据预处理完全一致的前提下，**相同模型 + 相同数据的两次 vLLM 评估结果仍可能存在微小差异**。以下是实测案例和根因分析。

#### 8.1 实测现象

两次完全相同配置（同模型、同数据集、同 `sample_concurrency=8`、同GPU）的评估结果：

| Run | OVO-Bench | VideoMME | LongVideoBench | VideoHolmes | Overall |
|-----|-----------|----------|----------------|-------------|---------|
| Run A | 56.145 | 63.246 | 59.259 | 42.515 | 55.948 |
| Run B | 56.083 | 62.764 | 59.375 | 42.896 | 55.872 |
| Δ | 0.062 | 0.482 | -0.116 | -0.381 | 0.076 |

Overall 差异仅 0.076pp，对应约 **6-7 个样本**（/8909）的答案翻转。

#### 8.2 根本原因：浮点运算的非结合性

IEEE 754 浮点数不满足结合律：`(a + b) + c ≠ a + (b + c)`。这意味着相同数值按不同顺序累加会得到不同结果（差异在 ~1e-7 量级），但在 28 层 Transformer 中逐层放大后足以翻转 argmax。

#### 8.3 vLLM 并发引入非确定性的具体路径

**① PagedAttention 内部的并行归约**

vLLM 的 PagedAttention kernel 使用 CUDA parallel reduction 计算 softmax 分母和加权求和。GPU warp/block 的调度顺序是非确定性的——同一个 attention head 内，partial sum 的累加顺序在每次 kernel launch 时可能不同，导致输出在最低有效位（ULP）上产生差异。

**② Continuous Batching 的动态组 batch**

vLLM 的核心调度器根据当前队列状态动态组成每一步的 batch：
- `sample_concurrency=8` 时，同一时刻有多个样本的多个 chunk 请求同时排队
- 请求到达顺序受 CPU 侧 asyncio 调度影响（视频解码耗时的微小波动 → 请求入队顺序变化）
- 不同的 batch 组成 → cuBLAS GEMM 选择不同的 tiling/分块策略 → 矩阵乘法内部累加顺序不同

**③ cuBLAS GEMM 算法选择**

cuBLAS 在不同 batch size / 矩阵维度下可能选用不同的内部算法（`CUBLAS_MATH_MODE`）。即使 batch size 相同，当 GPU 上同时有其他 kernel 在执行时，runtime autotuning 可能选择不同路径。

**④ `atomicAdd` 的执行顺序**

Reduction kernel 中的 `atomicAdd` 操作（如 Flash Attention 的 split-K 合并）依赖于 warp 调度顺序。多个 warp 同时对同一地址做原子加法时，执行顺序不确定 → 最终累加结果在 ULP 级别不同。

#### 8.4 差异放大机制

```
Layer 1:   attention 输出差异 ~1e-7 (1 ULP)
Layer 2:   经过 LayerNorm 归一化 + SwiGLU → ~1e-6
  ...       逐层残差连接累积
Layer 28:  hidden state 差异 → ~1e-3 ~ 1e-2
LM Head:   某些样本的 top-1 和 top-2 logit 差距 < 1e-2
           → 微小扰动翻转 argmax → 选出不同 token
           → 后续 autoregressive 生成完全分叉
```

绝大多数样本的 logit margin 足够大（>0.1），贪婪解码结果完全稳定。只有极少数"边界样本"（top-1 与 top-2 logit 差距在 1e-3 量级）会被浮点噪声翻转——这解释了为什么 8909 个样本中只有 6-7 个结果不同。

#### 8.5 与 `sample_concurrency` 的关系

| 并发度 | 非确定性来源 | 预期波动 |
|--------|-------------|---------|
| `1`（纯串行） | 仅 kernel 级（PagedAttention reduction、cuBLAS），batch 组成固定 | 最小，但仍非零 |
| `8`（默认） | kernel 级 + batch 组成变化 + 请求到达顺序变化 | ~0.1-0.5pp |
| `16+` | 同上但 batch 更大更不稳定 | 可能略增 |

注意：即使 `sample_concurrency=1`，由于 CUDA kernel 本身的非确定性，结果也**不能保证**完全一致。只是波动幅度更小。

#### 8.6 如何获得完全确定性结果（如需要）

| 方法 | 效果 | 性能代价 |
|------|------|---------|
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 强制 cuBLAS 使用确定性算法 | -10~20% 吞吐 |
| `torch.use_deterministic_algorithms(True)` | 禁用所有非确定性 CUDA op | 部分 op 会 fallback CPU |
| 固定 batch（禁用 continuous batching） | 消除 batch 组成差异 | 丧失并发优势 |
| `--sample-concurrency 1` | 串行处理，减少调度抖动 | 回到串行速度 |

**实际建议**：0.1-0.5pp 的波动对评估结论无影响，不需要特殊处理。报告结果时使用 Overall（样本加权平均），其波动比单数据集小一个数量级（本例仅 0.076pp）。如需严格可复现性（如发表论文中的消融实验），建议跑两次取平均。

---

---

### 9. 常用命令示例

```bash
# 流式评估：8卡跑全量subset7500任务组，vst_original切段口径，每卡8样本并发
python vllm_eval/vllm_eval_engine_stream.py \
    --model /path/to/VST-7B \
    --tasks all_subset7500 \
    --gpus 0,1,2,3,4,5,6,7 \
    --output-dir ./eval_outputs/subset7500_batch32_8gpu \
    --chunking-mode vst_original \
    --sample-concurrency 8 \
    --decode-timeout-s 180 \
    --max-model-len 98304

# 标准评估：单个verl checkpoint跑ovobench三个子任务
python vllm_eval/vllm_eval_engine.py \
    --model /path/to/checkpoint/actor \
    --tasks ovobench \
    --gpus 0,1,2,3

# 消费checkpoint_queue.csv中所有待评估的ckpt（不传--model）
python vllm_eval/vllm_eval_engine_stream.py --tasks all_subset7500 --gpus 0,1,2,3,4,5,6,7
```
