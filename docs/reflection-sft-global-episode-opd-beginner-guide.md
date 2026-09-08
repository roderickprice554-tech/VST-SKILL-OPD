# VST-SKILL-OPD 全流程新手指南

本文说明当前代码从“视频分段记忆”到“反思 SFT”，再到“在线强化学习 + OPD”的完整流程。阅读本文不要求熟悉 VST、PPO 或知识蒸馏。

> 当前实现目标是验证代码链路，而不是证明训练效果。已经完成 CPU code smoke；尚未运行真实外部大模型、完整 SFT、GPU/Ray/DeepSpeed 全量训练或效果评测。

## 1. 一句话理解整个系统

系统先让 Actor 学会逐段观看视频并写记忆，最后回答问题；然后用不含正确答案的事后反思数据做一次 SFT，教会 Actor 输出结构化“技能”；进入在线训练后，冻结的当前 Actor 充当 Analyzer，为完整轨迹生成技能，再把技能作为教师侧额外提示，通过 OPD 与原 VST-RL loss 一起更新 Actor。

```text
离线轨迹
  -> 外部 Teacher 事后反思（只见预测是否正确，不见正确答案）
  -> 反思 SFT 数据
  -> SFT 后的 Actor
  -> 在线视频轨迹（T-1 个 memory turn + 1 个 final turn）
  -> 环境只给 final turn reward
  -> reward 按 trajectory_uid 回填并转成 is_correct
  -> 冻结当前 Actor 作为 Analyzer，生成 episode skill / step skill
  -> 教师侧带技能再前向一次，截取 top-100 分布
  -> 所有有效的非最终 memory token 计算 OPD
  -> 原 VST-RL loss + lambda_opd * LOPD 更新 Actor
```

这里有两个很容易混淆的结论：

1. **外部大模型只用于离线反思 SFT 数据准备。** 在线 OPD 不调用外部模型，Analyzer 是冻结的当前 Actor。
2. **OPD 不只训练 Analyzer 选中的关键步。** 所有有效的非最终 memory 行都有 episode skill，因此都进入 OPD；关键步只是再附加一个更具体的 step skill。final turn 永远不进入 OPD。

## 2. 核心名词

- **trajectory / episode**：一个样本从第一段视频到最终回答的完整过程。
- **memory turn**：非最终轮。输入是旧记忆、当前视频块和与问题无关的指令，输出是本轮记忆文本 \(Y_t\)。
- **final turn**：最终轮。输入是 \(M_{T-1}, C_T, Q\)，输出最终答案，只参与环境 reward 和原 VST-RL。
- **Actor**：被训练的模型，既生成记忆和答案，也在冻结状态下充当在线 Analyzer。
- **Analyzer**：读取完整轨迹并输出结构化反思技能的角色，不是单独的外部服务。
- **episode skill**：适用于整条轨迹所有 memory turn 的全局技能。
- **step skill**：只附加到 Analyzer 判定为关键的某个 memory turn。
- **OPD**：让原始输入上的学生分布模仿“加入技能提示后”的冻结教师分布。

## 3. 阶段总览

| 阶段 | 目的 | 主要输入 | 主要输出 |
|---|---|---|---|
| 0. 隔离与复现记录 | 不污染原 VST 实验 | 基线提交、已有模型/数据绝对路径 | 独立 branch/worktree、manifest |
| 1. 离线轨迹转反思请求 | 给 Teacher 准备无答案信息 | 完整轨迹、预测、final reward | Analyzer prompt；reward 仅转成布尔值 |
| 2. 外部 Teacher 生成反思 | 产生 SFT 标签 | 视频、轨迹、问题/选项、预测、is_correct | 严格 JSON 反思 |
| 3. 反思 SFT | 教会 Actor 生成结构化反思 | VST-SFT conversation JSONL | SFT checkpoint |
| 4. 在线视频 rollout | 生成记忆轨迹和最终答案 | 视频、问题、batch 起始 policy version | \(T-1\) 个 transition + 1 个 final row |
| 5. reward 回填与轨迹聚合 | 把 final reward 对齐到整条轨迹 | rollout rows、final reward | 完整 `ReflectionTrajectory` |
| 6. 冻结 Actor Analyzer | 为本次轨迹生成技能 | 完整轨迹 + is_correct | episode skill、关键步及 step skill |
| 7. 教师 top-100 与 OPD | 用技能信号监督全部 memory 步 | 原 batch、技能增强教师 batch | \(L_{OPD}\)、统计指标 |
| 8. 联合更新 | 保留原 RL 目标并加入 OPD | VST-RL loss、OPD loss | 更新后的 Actor |

## 4. 阶段 0：隔离代码和记录复现来源

### 目的

从指定基线创建独立分支和 worktree，不继承其他 VST 目录的未提交文件，也不复制大型数据、模型和 checkpoint。

### 输入

- 源提交：`26f31d3`
- 独立工作目录：`/home/bujunru/vlm-repro/VST-skill-opd`
- 外部模型、数据和 checkpoint：使用绝对只读路径引用

### 输出

- 独立分支：`codex/vst-skill-opd`
- 来源与资产哈希 manifest

### 代码/记录位置

- [`manifests/opd-foundation.json`](../manifests/opd-foundation.json)
- [`VST-RL/recurrent/test/test_opd_trajectory.py`](../VST-RL/recurrent/test/test_opd_trajectory.py)

### 验证重点

- worktree 创建时 HEAD 精确指向基线。
- 原目录未跟踪数据不会进入新分支。
- 大型资产不复制到 Git。

## 5. 阶段 1：把离线轨迹转换为反思请求

### 目的

把已有 Actor 轨迹整理成外部 Teacher 可以读取的 Analyzer 请求，同时保证 Teacher 看不到正确答案和原始 reward。

### 输入

每条离线 JSONL 记录必须包含：

```json
{
  "trajectory_uid": "traj-001",
  "policy_version": 10,
  "observed_video": "https://.../video.mp4",
  "transitions": [
    {
      "transition_index": 0,
      "previous_memory_tokens": [],
      "current_chunk_boundary": {"frames": [0, 16], "seconds": [0.0, 2.0]},
      "generated_y_t_tokens": [1, 2],
      "updated_memory_tokens": [1, 2]
    }
  ],
  "query": "问题和选项",
  "prediction": "Actor 的答案",
  "final_reward": 1.0
}
```

### 处理规则

- 本地转换器读取 `final_reward`，只用 `reward > 0` 得到 `is_correct: true/false`。
- 发送给 Teacher 的 payload 只含：轨迹 ID、policy version、视频、transitions、query、prediction、is_correct。
- 原始 reward、正确选项、标准答案、解析和 solution 都不会进入 Teacher 请求。
- 输入字段是严格白名单；多余字段会被拒绝，以防答案意外泄漏。

### 输出

- `export` 模式：输出待调用请求 JSONL，不调用外部 API。
- `import` 模式：读取已生成的响应，严格校验后输出训练 JSONL 和 seeks 文件。
- `call` 模式：才会实例化 OpenAI-compatible 外部客户端并实际调用。

### 代码位置

- [`VST-RL/recurrent/reflection_sft.py`](../VST-RL/recurrent/reflection_sft.py)：输入校验、请求构造、外部客户端、SFT 记录转换。
- [`VST-RL/scripts/build_reflection_sft_data.py`](../VST-RL/scripts/build_reflection_sft_data.py)：`export`、`import`、`call` 三种命令入口。

### 可调整参数

- `call --model`：外部 Teacher 模型名。
- `call --api-key`：API key。
- `call --base-url`：OpenAI-compatible 服务地址。
- 输入/输出 JSONL 路径。

当前外部请求固定使用确定性生成（temperature 为 0）。smoke 使用 fixture，**不会读取 API key，也不会访问网络**。

## 6. 阶段 2：外部 Teacher 生成事后反思

### 目的

在正式在线 OPD 前，先教会 Actor 按严格格式分析“这条轨迹的记忆过程哪里值得保持或纠正”。

### Teacher 能看到什么

- 已观察的视频。
- 全部 memory transitions。
- 问题和选项。
- Actor 的预测。
- 预测是否正确 `is_correct`。

### Teacher 看不到什么

- 正确答案或正确选项。
- 标准解析、solution。
- 原始 reward 数值。
- 最终视频块之外的任何未观察视频。

这与在线 Analyzer 的可见信息保持一致，减少离线 SFT 与在线 OPD 的信息分布差异。

### 输出格式

需要应用 OPD 时：

```json
{
  "apply_opd": true,
  "episode_skill": "整条轨迹都应遵循的技能",
  "key_transitions": [
    {
      "transition_index": 0,
      "kind": "preserve",
      "memory_attribute": "temporal_order",
      "step_skill": "该步的具体技能"
    }
  ]
}
```

不应应用 OPD 时：

```json
{
  "apply_opd": false,
  "episode_skill": null,
  "key_transitions": [],
  "skip_reason": "answer_only_error"
}
```

允许的 `kind` 是 `preserve` 或 `correct`。允许的 `memory_attribute` 包括实体身份、状态变化、时间顺序、事件存在性、计数、空间关系、可见文字和压缩。响应必须是纯 JSON；Markdown 代码围栏、尾随文字、未知字段、重复/越界 transition、policy mismatch 或 query leakage 都会被拒绝。

### 代码位置

- [`VST-RL/recurrent/reflection.py`](../VST-RL/recurrent/reflection.py)：Analyzer prompt、JSON schema、严格解析和泄漏检查。
- [`VST-RL/recurrent/reflection_sft.py`](../VST-RL/recurrent/reflection_sft.py)：Teacher 调用与响应接收。

## 7. 阶段 3：反思 SFT

### 目的

让 Actor 学会根据视频和轨迹生成上述结构化反思。这样进入在线训练后，可以直接冻结当前 Actor 作为 Analyzer，不再依赖外部模型。

### 输入

转换后的 VST-SFT conversation JSONL：

- user：视频 + Analyzer prompt。
- assistant：外部 Teacher 返回的严格 JSON 反思。

训练数据仍沿用现有 VST-SFT 格式和 seeks 索引。训练只对 assistant token 计算监督损失。

### 输出

- 反思 SFT checkpoint。
- 训练完成后，把 RL 配置中的 `actor_rollout_ref.model.path` 指向该 checkpoint，再启动在线 RL。

### 代码位置

- [`VST-SFT/run_reflection_sft.sh`](../VST-SFT/run_reflection_sft.sh)：最小训练启动脚本。
- [`VST-SFT/train.py`](../VST-SFT/train.py)：复用现有 Hugging Face Trainer。
- [`VST-SFT/streaming_vlm/data/lmm_dataset.py`](../VST-SFT/streaming_vlm/data/lmm_dataset.py)：conversation 数据读取与 assistant-only label。

### 可调整参数

启动脚本必须提供三个绝对路径：

- `TRAIN_JSONL`：反思训练集。
- `BASE_MODEL`：基础模型或已有 checkpoint。
- `OUTPUT_DIR`：SFT 输出目录。

可选：

- `NPROC_PER_NODE`：每节点进程数，默认 1。

当前脚本内训练参数为：batch size 1、gradient accumulation 8、learning rate `5e-6`、1 epoch、BF16、gradient checkpointing、每 25 step 保存。需要改实验规模时，可直接修改 [`run_reflection_sft.sh`](../VST-SFT/run_reflection_sft.sh) 中这些参数。

## 8. 阶段 4：在线 Actor 生成视频轨迹

### 目的

严格区分“逐段写记忆”和“最终回答”，并为之后 reward 回填、Analyzer 和 OPD 保留逐行元数据。

### memory turn 接口

```text
input  = previous memory + current chunk + query-independent instruction
output = generated Y_t tokens + updated memory tokens
```

memory turn 不读取也不传递 `prompt_ids`/`question_ids`，因此不含问题、选项或答案。`TEMPLATE_TYPE_2` 已删除 `<problem>{prompt}</problem>`。

### final turn 接口

```text
input  = M_(T-1) + final chunk C_T + question/options
output = final answer
```

只有 final 分支才读取 `prompt_ids`。它带 `final_mask=True`，只用于答案 reward 和原 VST-RL，不进入 OPD。

### 每个非最终 transition 的输出元数据

```text
trajectory_uid
group_uid
sample_index
transition_index
previous_memory_tokens
current_chunk_boundary  # frames + seconds
generated_y_t_tokens
updated_memory_tokens
final_mask=False
policy_version
```

`policy_version` 使用本 batch rollout 开始前的 `global_step`，同一 batch 中冻结不变。

### 代码位置

- [`VST-RL/recurrent/impls/video_memory.py`](../VST-RL/recurrent/impls/video_memory.py)：视频切块、memory/final prompt、transition 元数据。
- [`VST-RL/recurrent/generation_manager.py`](../VST-RL/recurrent/generation_manager.py)：`run_llm_loop()` 收集各 turn、写入 policy version、拼接输出。
- [`VST-RL/recurrent/interface.py`](../VST-RL/recurrent/interface.py)：turn 校验、reward 回填、轨迹聚合。

### 可调整参数

`VideoMemoryConfig` 相关参数：

- `video_clip_token_size`：每个视频块的 token 大小，直接影响 transition 数量。
- `max_memorization_length`：记忆最大长度。
- `max_video_clips`：最大视频块数。
- `max_final_response_length`：最终答案最大长度。
- `max_prompt_length`：问题 prompt 最大长度。
- `max_video_frame`：最大视频帧数。
- `prompt_type`：使用哪套 memory/final template。
- `video_key`、`video_root`：视频字段和根路径。

改变切块参数会改变 \(T\)，因此必须重新检查每条轨迹是否仍是恰好 \(T-1\) 个 memory transitions 加 1 个 final turn。

## 9. 阶段 5：final reward 回填并聚合完整轨迹

### 目的

环境只对最终答案打分，但 Analyzer 需要查看完整 episode，所以必须把 final reward 与同一 `trajectory_uid` 的全部行对齐。

### 输入

- `DataProto.concat()` 后的所有 memory/final rows。
- final rows 上计算出的环境 reward。
- 每行 `trajectory_uid`、`transition_index`、`final_mask`、`policy_version`。

### 处理

1. 校验每条 trajectory 恰好一个 final row。
2. 校验 memory transition index 从 0 连续增长。
3. 校验同一轨迹 policy version 一致。
4. 按 `trajectory_uid` 把 final reward 回填到整条轨迹。
5. 把 reward 转为 `is_correct`；Analyzer payload 不保留原 reward。
6. 按 transition index 排序，聚合为完整 `ReflectionTrajectory`。

### 输出

- 每条样本一份完整轨迹，含 transitions、query、prediction 和 `is_correct`。

### 代码位置

- [`VST-RL/recurrent/interface.py`](../VST-RL/recurrent/interface.py)：结构校验、聚合、reward propagation。
- [`VST-RL/recurrent/skill_opd.py`](../VST-RL/recurrent/skill_opd.py)：`reward_to_is_correct()` 和 Analyzer 输入白名单。
- [`VST-RL/verl/trainer/ppo/ray_trainer.py`](../VST-RL/verl/trainer/ppo/ray_trainer.py)：在线训练中的实际串联。

## 10. 阶段 6：冻结当前 Actor 作为 Analyzer

### 目的

不调用外部模型，用当前 batch 更新前的 Actor 生成结构化技能。因为它在本轮更新前被冻结，所以反思和教师分布来自同一个 `policy_version`。

### 输入

- observed video。
- 全部 memory transitions。
- question/options。
- Actor prediction。
- `is_correct`。

### 输出

- `apply_opd`。
- episode skill。
- 最多 `max_key_transitions` 个关键 transition 及其 step skill。
- 或一个明确的 `skip_reason`。

### 代码位置

- [`VST-RL/recurrent/reflection.py`](../VST-RL/recurrent/reflection.py)：Analyzer 输入和输出 schema。
- [`VST-RL/recurrent/skill_opd.py`](../VST-RL/recurrent/skill_opd.py)：`SkillOPDManager`、多模态 Analyzer batch、生成和解析。
- [`VST-RL/verl/trainer/ppo/ray_trainer.py`](../VST-RL/verl/trainer/ppo/ray_trainer.py)：在 Actor update 前调用 Analyzer。

### 可调整参数

主配置位于 [`VST-RL/verl/trainer/config/ppo_trainer.yaml`](../VST-RL/verl/trainer/config/ppo_trainer.yaml)：

```yaml
skill_opd:
  enable: false
  smoke_metrics_path: null
  mode: global_episode
  top_k: 100
  teacher_temperature: 1.0
  lambda_opd: 0.01
  max_key_transitions: 3
  reflection:
    temperature: 0
    top_p: 1.0
    max_tokens: 256
    max_context_tokens: 32768
```

- `enable`：总开关，默认关闭；关闭时保持原 VST-RL 路径。
- `max_key_transitions`：Analyzer 最多返回多少个关键步。
- `reflection.max_tokens`：反思最大生成 token 数。
- `reflection.max_context_tokens`：Analyzer 最大上下文长度。
- `smoke_metrics_path`：可选的 smoke 指标输出位置。

当前实现强制 Analyzer 确定性生成：`do_sample=False`、temperature 0、top-p 1。YAML 中 temperature/top-p 记录了这一意图，但当前管理器并未把它们作为可变生成参数传入；只改 YAML 不会改变行为。

## 11. 阶段 7：技能增强教师与全局 episode OPD

### 目的

比较同一个 Actor 在“原始 memory 输入”和“加入反思技能的 memory 输入”上的分布，让学生在不需要推理时持续看到技能提示的情况下，也学到技能带来的行为变化。

### 技能如何附加

- Analyzer 接受的轨迹中，**每个非最终 memory row** 都附加 `Episode skill: ...`。
- Analyzer 指定的关键 row 再附加 `Step skill: ...`。
- final row 不附加技能，也不进入 OPD。

因此：

```text
非关键 memory row = episode skill -> 计算 OPD
关键 memory row   = episode skill + step skill -> 计算 OPD
final row         = 不计算 OPD
```

Analyzer 的关键步选择控制的是 step skill 放在哪里，不是 OPD 总开关。

### 教师和学生输入

- **学生**：原始 memory prompt。
- **教师**：同一原始 prompt 的深拷贝，加上 episode skill，关键步再加 step skill。
- 教师是更新前的同一个 Actor；教师 logits 立即 `detach`，不会从 OPD 反向传播到教师分支。

### 有效 token 掩码

只有同时满足以下条件的 token 才计算 OPD：

```text
response_mask
AND memory_mask
AND episode_mask
AND reflection_mask
AND metadata_mask
```

`opd_key_mask` 仅用于表示关键步和统计，不在全局 OPD 的 loss gate 中。

### top-100 教师分布

1. 教师 logits 除以 `teacher_temperature`。
2. 每个有效 token 只保留概率最大的 `top_k=100` 个词。
3. 在这 100 个词上重新归一化教师分布。
4. 从学生 logits 中抽取相同词表位置。
5. 计算教师到学生的 forward KL。

输出还记录 top-k retained mass，用于观察 top-100 覆盖了教师原分布多少概率质量。

### 代码位置

- [`VST-RL/recurrent/skill_opd.py`](../VST-RL/recurrent/skill_opd.py)：为 episode/key rows 构建 annotation 和技能 prompt。
- [`VST-RL/verl/trainer/ppo/skill_opd_loss.py`](../VST-RL/verl/trainer/ppo/skill_opd_loss.py)：top-k 教师分布、有效掩码和 OPD KL。
- [`VST-RL/verl/workers/actor/dp_actor.py`](../VST-RL/verl/workers/actor/dp_actor.py)：学生前向、联合 loss 和反向传播。
- [`VST-RL/verl/trainer/ppo/ray_trainer.py`](../VST-RL/verl/trainer/ppo/ray_trainer.py)：技能增强 batch 和教师 cache。

### 可调整参数

- `top_k`：教师保留多少个词，默认 100。
- `teacher_temperature`：教师分布温度，默认 1.0。
- `lambda_opd`：OPD loss 权重，默认 0.01。
- `max_key_transitions`：最多多少个 memory row 获得 step skill，默认 3。

## 12. 阶段 8：联合 loss 更新 Actor

原 VST-RL 的 Actor loss 保持原结构：

```text
L_VST-RL = policy_gradient_loss
           - entropy_coefficient * entropy
           + optional_reference_KL
```

新增 OPD 后：

```text
L_total = L_VST-RL + lambda_opd * L_OPD
```

其中：

- `L_OPD` 是教师 top-k 分布到学生分布的 forward KL。
- final answer token 的 OPD mask 为 0。
- 没有有效 OPD token 时，`L_OPD` 返回可微的 0，不影响原训练。
- `skill_opd.enable=false` 时走原 VST-RL 路径，避免改变既有实验。

代码位置：

- [`VST-RL/verl/workers/actor/dp_actor.py`](../VST-RL/verl/workers/actor/dp_actor.py)
- [`VST-RL/verl/trainer/ppo/skill_opd_loss.py`](../VST-RL/verl/trainer/ppo/skill_opd_loss.py)

## 13. 当前 CPU code smoke 验证了什么

### 运行入口

- [`VST-RL/scripts/run_skill_opd_code_smoke.py`](../VST-RL/scripts/run_skill_opd_code_smoke.py)
- [`VST-RL/scripts/audit_skill_opd_smoke.py`](../VST-RL/scripts/audit_skill_opd_smoke.py)
- [`docs/smoke/skill-opd-cpu-code-smoke.md`](smoke/skill-opd-cpu-code-smoke.md)

### 已验证

当前 smoke 在 CPU 上用极小 fixture/toy model 跑通了：

1. 一条轨迹恰好包含 2 个 memory transitions 和 1 个 final turn。
2. final reward 能按 trajectory 映射，并转成 `is_correct`。
3. 外部 Teacher 请求中没有原 reward 或正确答案。
4. fixture 反思能通过严格 JSON/schema/leakage 校验。
5. 反思能转换为现有 VST-SFT conversation 格式。
6. toy SFT 只训练 assistant 反思 token，并发生了参数更新。
7. 在线教师使用冻结 Actor，teacher logits 已 detach。
8. 2 个 memory row 都属于 episode OPD；其中 1 个是关键步，另 1 个是非关键步。
9. final turn 的 OPD token 数为 0。
10. top-100 分布、retained mass、OPD KL 都是有限值。
11. 原 RL loss、加权 OPD loss 和总 loss 能完成 backward/optimizer step，Actor 参数发生变化。
12. `skill_opd.enable=false` 与原路径等价。
13. smoke 全程没有调用外部 API，也没有使用 GPU。

一次已记录的 smoke 关键结果：

| 指标 | 值 |
|---|---:|
| trajectory_count | 1 |
| memory_transition_count | 2 |
| final_turn_count | 1 |
| episode_memory_row_count | 2 |
| key_memory_row_count | 1 |
| non_key_episode_row_count | 1 |
| valid_token_count | 3 |
| final_opd_token_count | 0 |
| top_k | 100 |
| retained_mass | 0.9191031 |
| sft_loss | 5.2946568 |
| rl_loss | 0.3437940 |
| lopd_loss | 0.0342918 |
| weighted_lopd_loss | 0.0003429 |
| total_loss | 0.3441369 |
| external_api_called | false |
| gpu_used | false |

完整测试结果为 `138 passed, 2 skipped`。两个 skip 是当前 CPU 环境不具备的 CUDA 路径；CPU/BF16 相关逻辑已有覆盖。

Smoke 结果文件保存在仓库外：

```text
/home/bujunru/vlm-repro/skill-opd-smoke/reflection-sft-global-episode-opd-cpu.json
SHA256: d6b9ba9ef2a868922594cb38ab83f55953d4b5d7bf7f1a15ebece369990a5b33
```

## 14. 当前 smoke 没有验证什么

以下内容不能从目前 smoke 结果推断：

- 没有真实调用外部 Teacher；只验证了客户端代码存在、请求构造正确，smoke 使用 fixture。
- 没有用真实视频做完整解码和长上下文 rollout。
- 没有运行真实 Qwen/VST 模型的反思 SFT，也没有产出可用 checkpoint。
- 没有评估 Actor 是否真的学会高质量反思。
- 没有运行完整 GPU、CUDA、BF16、DeepSpeed、Ray 分布式训练。
- 没有验证训练收敛、最终准确率提升、超参数最优性或大规模数据吞吐。
- 没有验证外部服务的鉴权、限流、费用和网络稳定性。

所以当前结论应表述为：**代码接口、数据对齐、掩码、loss 和 CPU 最小优化步骤已跑通；真实训练效果仍需后续实验验证。**

## 15. 最小使用顺序

### 15.1 导出 Teacher 请求，不调用外部模型

```bash
cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL
python scripts/build_reflection_sft_data.py export \
  --trajectory-jsonl /absolute/path/trajectories.jsonl \
  --requests-jsonl /absolute/path/teacher_requests.jsonl
```

### 15.2 导入外部生成的反思

```bash
python scripts/build_reflection_sft_data.py import \
  --trajectory-jsonl /absolute/path/trajectories.jsonl \
  --responses-jsonl /absolute/path/teacher_responses.jsonl \
  --output-jsonl /absolute/path/reflection_sft.jsonl
```

如需让代码直接调用 OpenAI-compatible 外部服务，使用 `call` 子命令并提供 model/API key/base URL。该路径未被当前 smoke 实际调用。

### 15.3 运行反思 SFT

```bash
cd /home/bujunru/vlm-repro/VST-skill-opd/VST-SFT
TRAIN_JSONL=/absolute/path/reflection_sft.jsonl \
BASE_MODEL=/absolute/path/base_model \
OUTPUT_DIR=/absolute/path/reflection_sft_checkpoint \
NPROC_PER_NODE=1 \
bash run_reflection_sft.sh
```

### 15.4 启动在线 OPD 前的配置

1. 把 `actor_rollout_ref.model.path` 指向反思 SFT checkpoint。
2. 设置 `skill_opd.enable=true`。
3. 保持 `skill_opd.mode=global_episode`。
4. 首次真实运行建议保持 `top_k=100`、`lambda_opd=0.01`，先观察 retained mass、有效 token 数和 loss 比例。

### 15.5 重新运行 CPU smoke

```bash
cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL
python scripts/run_skill_opd_code_smoke.py \
  --output /absolute/path/smoke.json
python scripts/audit_skill_opd_smoke.py \
  --metrics /absolute/path/smoke.json
pytest -q
```

## 16. 新手排查清单

如果在线实验不符合预期，按以下顺序检查：

1. 每条 trajectory 是否恰好有一个 `final_mask=True`。
2. transition index 是否从 0 连续到 \(T-2\)。
3. memory prompt 是否意外含有问题、选项或答案。
4. final chunk \(C_T\) 是否只在 final turn 出现。
5. 同一 batch 的 policy version 是否一致。
6. Analyzer 输入是否只有 `is_correct`，没有正确答案或 raw reward。
7. reflection 是否被 schema 或 leakage validator 拒绝。
8. 非关键 memory row 是否仍有 episode skill 和有效 OPD mask。
9. final row 的 OPD token 数是否为 0。
10. retained mass 是否过低；若过低再考虑提高 `top_k`。
11. `lambda_opd * L_OPD` 是否与原 RL loss 同量级或过大。
12. 真实训练前先确认 `skill_opd.enable=false` 能复现原 VST-RL 行为。

## 17. 设计与测试的进一步资料

- [`docs/superpowers/specs/2026-09-08-reflection-sft-global-episode-opd-design.md`](superpowers/specs/2026-09-08-reflection-sft-global-episode-opd-design.md)：设计决策和边界。
- [`docs/superpowers/plans/2026-09-08-reflection-sft-global-episode-opd.md`](superpowers/plans/2026-09-08-reflection-sft-global-episode-opd.md)：实现计划。
- [`VST-RL/tests/test_reflection_sft.py`](../VST-RL/tests/test_reflection_sft.py)：离线 Teacher/SFT 数据测试。
- [`VST-RL/tests/test_skill_opd_loss.py`](../VST-RL/tests/test_skill_opd_loss.py)：top-k OPD loss 测试。
- [`VST-RL/tests/test_skill_opd_smoke_audit.py`](../VST-RL/tests/test_skill_opd_smoke_audit.py)：smoke 审计测试。
- [`VST-RL/recurrent/test/test_opd_trajectory.py`](../VST-RL/recurrent/test/test_opd_trajectory.py)：轨迹、边界、manifest 测试。
