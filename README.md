<div align="center">

# 🎬 Video Streaming Thinking

### VideoLLMs Can Watch and Think Simultaneously

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-9b59b6?style=flat-square)]()
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2603.12262)
[![Homepage](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://1ranguan.github.io/VST/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)]()

[![Model 3/7/32B](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow?style=flat-square)](https://huggingface.co/Catalan258/VST-7B)
[![Training Data](https://img.shields.io/badge/🤗%20Training%20Data-HuggingFace-yellow?style=flat-square)](https://huggingface.co/datasets/Catalan258/VST-Training-Data)
[![Training Data MS](https://img.shields.io/badge/🤖%20Training%20Data-ModelScope-624aff?style=flat-square)](https://www.modelscope.cn/datasets/catalan/VST-Training-Data)

**🎉 VST has been accepted to ECCV 2026!**

</div>

> **Video Streaming Thinking** introduces a new paradigm for streaming video understanding that interleaves active reasoning with continuous video consumption, enabling amortized test-time scaling with real-time responsiveness.

---

## 🔍 Overview

Existing online VideoLLMs focus on efficient streaming perception but lack explicit analytical reasoning. Offline VideoLLMs with Chain-of-Thought (CoT) can reason deeply, but incur high query-answer (QA) latency that violates real-time constraints. **VST bridges this gap** by shifting the LLM backend from passive waiting to active, intermittent reasoning *during* video consumption, implementing a **thinking-while-watching** mechanism inspired by human neural coupling.

https://github.com/user-attachments/assets/49846db5-bf76-4cf8-b923-4b9b88117482

### ✨ Key Idea

Instead of deferring all reasoning until a user query arrives, VST continuously processes incoming video clips and produces **intermediate streaming thoughts** in real time. This front-loads and amortizes the reasoning cost, so the final response is both **deeply grounded** and **instantly available**.

## 🏗️ Model Zoo

| **Model** | **HuggingFace** | **OVO-Bench** | **StreamingBench** | **VideoMME** | **LongVideoBench** | **VideoHolmes** |
|---|---|---|---|---|---|---|
| VST-3B | [🤗 Link](https://huggingface.co/Catalan258/VST-3B) | 56.2 | 75.5 | 59.5 | 54.1 | 36.1 |
| VST-7B | [🤗 Link](https://huggingface.co/Catalan258/VST-7B) | 59.3 | 79.5 | 64.9 | 58.0 | 41.9 |
| VST-32B | [🤗 Link](https://huggingface.co/Catalan258/VST-32B) | 63.5 | 80.7 | 67.2 | 60.7 | 45.1 |

## 📦 Training Data

We release the full training data used for both SFT and RL stages on HuggingFace and ModelScope:

| **Dataset** | **HuggingFace** | **ModelScope** | **Description** |
|---|---|---|---|
| vst_sft_data | [🤗 Link](https://huggingface.co/datasets/Catalan258/VST-Training-Data/tree/main/vst_sft_data) | [🤖 Link](https://www.modelscope.cn/datasets/catalan/VST-Training-Data/files/vst_sft_data) | SFT data including video-text pairs from multiple sources |
| vst_rl_data | [🤗 Link](https://huggingface.co/datasets/Catalan258/VST-Training-Data/tree/main/vst_rl_data) | [🤖 Link](https://www.modelscope.cn/datasets/catalan/VST-Training-Data/files/vst_rl_data) | RL data for reinforcement learning stage |

## 🧪 3B 非 Ego4D 复现运行手册（服务器交接）

> 本节对应 **Qwen2.5-VL-3B-Instruct、非 Ego4D** 复现实验，不要求使用特定 GPU。当前服务器正在运行的实例恰好使用 2×A100 40GB；接手者必须根据自己的 GPU 数量和显存重新做分布式资源适配。Ego4D 已按用户决定从 SFT 和 RL 数据中排除，因此结果必须称为“VST 3B 非 Ego4D 复现”，不能称为完整官方数据组成复现。官方算法、有效 global batch、2 FPS、chunk 语义、1 epoch 和流式因果约束保持不变。

### 1. 服务器与仓库

从本机连接服务器：

```bash
ssh -p 52234 bujunru@10.130.140.10
```

服务器上的工作目录和分支：

```text
/home/bujunru/vlm-repro/VST-full-reproduction
branch: codex/vst-full-reproduction
official base commit: f2500bb8699d59a13b96ec4229fc4cd643a96207
```

GitHub 复现仓库：

```bash
git clone git@github.com:roderickprice554-tech/VST.git
cd VST
git checkout main
```

数据、模型、checkpoint 和日志体积很大，不在 Git 中。下文 `/home/bujunru/...` 均为当前服务器实例路径，不是代码要求；师兄在自己的机器上必须替换为自己的绝对路径，并把最终路径写入 provenance。服务器上已有目录只是当前实验的权威副本。

迁移到新机器前可用下面的命令定位所有当前服务器硬编码路径，逐项做纯路径适配：

```bash
rg -n '/home/bujunru|VST-full-reproduction' audit VST-SFT VST-RL eval
```

### 2. Python 环境与模型

```text
SFT:   /home/bujunru/.conda/envs/vst-sft311
RL:    /home/bujunru/.conda/envs/vst-rl
Eval:  /home/bujunru/.conda/envs/vst-eval
Audit: /home/bujunru/.conda/envs/vision-se

3B SFT base: /home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct
released VST: /home/bujunru/vlm-repro/models/VST-3B
```

在新服务器获取模型：

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct

hf download Catalan258/VST-3B \
  --local-dir /home/bujunru/vlm-repro/models/VST-3B
```

网络不稳定时可在命令前设置 `HF_ENDPOINT=https://hf-mirror.com`，下载器支持断点续传。

### 3. 数据来源、固定版本和本地目录

| 数据 | 官方来源 | 固定 revision | 服务器路径 |
|---|---|---|---|
| VST SFT/RL 原始数据 | ModelScope `catalan/VST-Training-Data` | `aaef152ea68ffa0e9d9f7367ccf871ea2f699693` | `/home/bujunru/vlm-repro/VST-full-reproduction/data/VST-Training-Data-official-5647583491c2` |
| VST 对应 Hugging Face 数据 | `Catalan258/VST-Training-Data` | `5647583491c298aa8b2926fe9910f651fc0e692d` | 同上，二选一下载，不要重复保存 |
| 非 Ego4D 解压媒体 | 由官方 `vst_video/setup_dataset.py` 生成 | 跟随上述 revision | `/home/bujunru/vlm-repro/VST-full-reproduction/data/vst-training-media-no-ego4d` |
| 非 Ego4D 审计 manifest | 本仓库审计脚本生成 | 配置名 `vst_no_ego4d_aaef152e` | `/home/bujunru/vlm-repro/VST-full-reproduction/data_manifests/vst_no_ego4d_aaef152e` |
| 完整 OVO-Bench 媒体 | Hugging Face `JoeLeelyf/OVO-Bench` | `fec29e3` | `/home/bujunru/vlm-repro/VST-full-reproduction/data/OVO-Bench-official-fec29e3` |
| OVO 解压媒体 | 由准备脚本生成 | `fec29e3` | `/home/bujunru/vlm-repro/VST-full-reproduction/data/OVO-Bench-prepared-fec29e3` |
| 完整 OVO 标注 | 固定代码提交自带 | 仓库 commit | `/home/bujunru/vlm-repro/VST-full-reproduction/eval/eval_data/anno/eval/OVOBench/json` |

当前非 Ego4D 数据规模：SFT train 135,700 行、SFT valid 4,620 行、RL train 9,906 行。完整 OVO 标注为 3,035 条：Backward 631、Real-time 837、Forward 1,567。旧目录 `.../ovo800/test387` 的 780 条只是子集，禁止用于正式完整评测。

固定文件清单：

```text
audit/inventories/vst_modelscope_aaef152e.json
audit/inventories/ovo_hf_fec29e3.json
```

### 4. 下载与准备数据

VST 数据优先使用 ModelScope：

```bash
cd /home/bujunru/vlm-repro/VST-full-reproduction

./.venv-modelscope/bin/modelscope download \
  --repo-type dataset \
  --revision aaef152ea68ffa0e9d9f7367ccf871ea2f699693 \
  --local-dir data/VST-Training-Data-official-5647583491c2 \
  --max-workers 8 \
  catalan/VST-Training-Data
```

也可以从 Hugging Face 下载相同发布数据，但不要和 ModelScope 副本重复保存：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  Catalan258/VST-Training-Data \
  --repo-type dataset \
  --revision 5647583491c298aa8b2926fe9910f651fc0e692d \
  --local-dir data/VST-Training-Data-official-5647583491c2
```

准备媒体、过滤 Ego4D、生成 seek/manifest 并执行 smoke：

```bash
bash audit/prepare_vst_official.sh
bash audit/audit_vst_no_ego4d.sh
bash audit/run_vst_smoke.sh
```

下载并准备完整 OVO-Bench：

```bash
bash audit/download_ovobench_official.sh
bash audit/prepare_ovobench_official.sh
/home/bujunru/.conda/envs/vst-eval/bin/python audit/validate_ovobench.py
```

不要通过降低 FPS、改变 chunk 或裁剪 OVO 样本来加速。Forward 样本只能看到标注规定的 `end` 之前内容。

### 5. 代码入口

| 阶段 | 官方入口 | 本次复现入口/配置 |
|---|---|---|
| SFT | `VST-SFT/train.py` | `audit/run_vst_sft.sh`、`audit/validate_vst_sft_launch.py`、`VST-SFT/scripts/zero3-vst-2xa100.json` |
| RL | `VST-RL/run.sh`、`VST-RL/recurrent/impls/video_memory.py` | 两卡 launcher 尚未完成；官方脚本仍是 4 nodes × 8 GPUs，禁止原样运行 |
| OVO Eval | `eval/eval_entry.py` | `audit/run_ovo_qwen3b_eval.sh`、`audit/validate_ovobench.py` |
| 数据编排 | 不属于官方算法 | `audit/vst_download_orchestrator.py`、`audit/run_vst_download_monitor.sh` |
| 审计/Smoke | 不属于官方算法 | `audit/audit_vst_no_ego4d.py`、`audit/run_vst_smoke.py` |

`audit/run_ovo_qwen3b_eval.sh` 当前指向未训练的 Qwen 3B，仅用于 Direct/base baseline。评测 SFT 或 RL checkpoint 时必须固定同一 OVO manifest、媒体前缀、帧率、chunk、推理参数和 scorer，只替换 checkpoint 与独立输出目录。

### 6. 按目标 GPU 适配 SFT

算法侧固定配置：Qwen2.5-VL-3B、有效 global batch 128、学习率 `5e-6`、1 epoch、2 FPS、最多 384 帧、vision tower 冻结、语言侧全参数训练（不是 LoRA）。GPU 拓扑、ZeRO/FSDP 分片、CPU/NVMe offload 和 gradient accumulation 属于硬件适配，必须记录差异并先做资源 smoke。

当 `per_device_batch=1` 时，应按下式保持有效 global batch 128：

```text
gradient_accumulation_steps = 128 / GPU数量
```

只有结果为整数时才能直接使用。常见配置：

| GPU 数量 | per-device batch | gradient accumulation | effective global batch |
|---:|---:|---:|---:|
| 1 | 1 | 128 | 128 |
| 2 | 1 | 64 | 128 |
| 4 | 1 | 32 | 128 |
| 8 | 1 | 16 | 128 |
| 16 | 1 | 8 | 128 |
| 32 | 1 | 4 | 128 |

本仓库的 `audit/run_vst_sft.sh` 和 `VST-SFT/scripts/zero3-vst-2xa100.json` 是当前 **2×A100 40GB** 实例配置，不能在未知硬件上原样运行。接手者应复制为新配置文件，至少调整以下项目并保留原文件：

```text
CUDA_VISIBLE_DEVICES
torchrun --nproc_per_node
gradient_accumulation_steps
ZeRO-3 bucket、parameter/optimizer offload（仅在显存需要时）
root、model、data、output、log 的绝对路径
```

禁止为了适配显存而降低 FPS、384 帧上限、缩小正式数据、改变 chunk 语义或改成 LoRA。若目标 GPU 无法在这些约束下完成一个真实 optimizer-step smoke，应先报告硬件阻塞，不要静默改变算法。

当前服务器的两卡启动方式如下，仅用于继续当前机器上的实验。

启动前必须确认没有现有训练，且 audit/smoke gate 已通过：

```bash
cd /home/bujunru/vlm-repro/VST-full-reproduction
pgrep -af 'run_vst_sft.sh|torchrun.*train.py|VST-SFT/train.py'
/home/bujunru/.conda/envs/vst-sft311/bin/python \
  audit/validate_vst_sft_launch.py \
  --validate-zero VST-SFT/scripts/zero3-vst-2xa100.json
```

仅在当前服务器没有训练进程时启动一次：

```bash
nohup bash audit/run_vst_sft.sh \
  > logs/vst_download_orchestrator/run_sft_2gpu_manual.log 2>&1 < /dev/null &
```

当前 launcher 会生成新的时间戳 run，不能把“重新运行 launcher”当作断点恢复。进程异常退出时，先检查最后一个 `checkpoint-*` 和日志，再明确指定恢复策略。师兄机器的精确命令必须在确认 GPU 型号、GPU 数量、单卡显存、主机内存和本地磁盘后生成。

### 7. 当前 2×A100 服务器实例状态（2026-09-01）

当前两卡 SFT 已启动：

```text
run: /home/bujunru/vlm-repro/VST-full-reproduction/checkpoints/vst_full/sft_runs/vst_3b_no_ego4d_2xa100_20260901_064628
log: /home/bujunru/vlm-repro/VST-full-reproduction/logs/vst_download_orchestrator/run_sft_2gpu_20260901_0645.log
source identity: <run>/source_identity.json
total optimizer steps: 1060
checkpoint interval: 25 steps
validation interval: 50 steps
```

实时查看：

```bash
tail -f /home/bujunru/vlm-repro/VST-full-reproduction/logs/vst_download_orchestrator/run_sft_2gpu_20260901_0645.log
nvidia-smi
df -h /home/bujunru
```

当前原始 VST 固定清单 197/197、OVO 固定清单 22/22 已下载。OVO 解压目录尚未生成；SFT 训练期间不要启动占用 GPU 的 OVO 推理。

### 8. 实验顺序与隔离原则

1. 固定源码、数据 revision、模型和环境。
2. 下载并准备非 Ego4D 媒体，运行完整审计和 causal smoke。
3. 完成 1 epoch SFT；只能依据训练/独立验证选择并冻结 `checkpoints/vst_full/sft_selected/`。
4. 使用完整 3,035 条 OVO，只读评测一次，写入 `results/ovo_full_after_sft/`。
5. 完成两卡 RL 等价语义和 one-update smoke 后，才从冻结的 `sft_selected` 启动 RL。
6. 冻结 `checkpoints/vst_full/rl_selected/`，使用完全相同的 OVO 协议评测到 `results/ovo_full_after_rl/`。

OVO 测试样本、答案、选项和失败分析不得用于训练、prompt 调优、checkpoint 选择或超参数选择。流式 memory 的每一步只能读取当前及过去帧。Backward/Real-time 使用官方 MCQ 字母评分；Forward 必须使用官方 SSR/CRR 字符串匹配和 REC 整数匹配，不能静默改成统一字母评分。

详细审计、偏差和接手检查清单见：

```text
docs/vst_reproduction_handoff_2026-08-30.md
docs/vst_full_reproduction_plan.md
```

## 📅 TODO
- [x] Release the paper.
- [x] Release checkpoint and eval code.
- [x] Release training code.
- [x] Release training data.

## 👍 Acknowledgement
We thank the following great works and open-source repositories:
- [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)
- [Streamingthinker](https://github.com/EIT-NLP/StreamingLLM)
## 📖 Citation
```
@inproceedings{guan2026videostreamingthinking,
      title={Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously}, 
      author={Yiran Guan and Liang Yin and Dingkang Liang and Jianzhong Ju and Zhenbo Luo and Jian Luan and Yuliang Liu and Xiang Bai},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2026},
}
```
