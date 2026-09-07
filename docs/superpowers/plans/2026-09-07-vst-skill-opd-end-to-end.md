# VST Backward Reflection + Localized Top-100 OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Every production change follows RED → GREEN → refactor and is committed only after focused regression tests pass.

**Goal:** Extend the isolated VST recurrent branch into a real Backward Reflection + Localized teacher-selected Top-100 OPD training loop and prove it with one real GPU rollout-batch optimizer smoke.

**Architecture:** Keep native append-only recurrent memory and the existing GRPO/PPO/reference-KL path. Add an explicit trajectory identity/assembly layer, a pure reflection schema/leakage layer, a pure localized top-k KL layer, then connect reflection generation and pre-update teacher cache to the current Actor snapshot. Pass only compact teacher-selected indices/log-probabilities and masks into the existing actor update.

**Tech Stack:** Python, PyTorch/FSDP, VERL DataProto/Ray workers, Qwen2.5-VL, OmegaConf/Hydra, pytest, two NVIDIA A100-PCIE-40GB GPUs.

## Global constraints

- Worktree: `/home/bujunru/vlm-repro/VST-skill-opd`; branch: `codex/vst-skill-opd`.
- Baseline ancestry: `26f31d36eb8bcc0b480b43eaaed1277b33a10488`.
- Do not alter VST, VST-official, VST-SE, or VST-full-reproduction worktrees.
- Do not copy model weights, videos, datasets, checkpoints, or caches into Git.
- No optimizer step may occur before reflection and all OPD teacher caches for the rollout batch are complete.
- The OPD teacher and Reflection Analyzer are the rollout-producing current Actor snapshot; the original reference policy remains dedicated to the original KL.
- Memory turns never read query/options; final answer tokens always have zero OPD mask.
- Total context budget is 32K shared by visual tokens, memory, templates, query, and output.
- Smoke defaults: localized mode, top-k 100, teacher temperature 1.0, lambda 0.01, at most three key transitions, greedy reflection with 256 output tokens.
- `skill_opd.enable=false` must preserve the original VST-RL behavior.

---

### Task 1: Correct rollout identity and finish stage-0 provenance

**Files:**
- Modify: `VST-RL/recurrent/interface.py`
- Modify: `VST-RL/recurrent/impls/video_memory.py`
- Modify: `VST-RL/verl/trainer/ppo/ray_trainer.py`
- Modify: `VST-RL/recurrent/test/test_opd_trajectory.py`
- Modify: `manifests/opd-foundation.json`
- Create: `docs/smoke/opd-baseline.md`

**Interfaces:**
- `group_uid`: UUID shared by the `n` rollout samples generated for one prompt; remains the existing GRPO grouping identity.
- `trajectory_uid`: `"{group_uid}:rollout-{ordinal}"`, unique for one rollout.
- `make_repeated_rollout_ids(group_uids, repeat_times) -> tuple[np.ndarray, np.ndarray]` returns interleaved group and trajectory arrays.

- [ ] Add failing tests with two prompts and two rollouts each. Assert repeated rows have two distinct group IDs, four distinct trajectory IDs, two trajectory IDs per group, and one final row per trajectory. Assert `group_uid != trajectory_uid` for every row.
- [ ] Run the focused tests and confirm current code fails because `trajectory_uid` is copied from `uid`.
- [ ] Implement the pure ID helper; assign one group ID before repeat, then assign the rollout ordinal immediately after `repeat(interleave=True)` in both training and validation paths. Keep `uid == group_uid` for existing GRPO code.
- [ ] Add `group_uid` and non-tensor `final_mask` to every VideoMemory turn; require both in metadata validation and verify non-tensor/tensor final masks agree.
- [ ] Record CUDA, PyTorch, GPU models, source/worktree relationship, absolute model/data paths, and manifest hashes in the baseline report/manifest.
- [ ] Run focused tests plus the existing recurrent baseline and commit.

### Task 2: Assemble reward-bearing reflection trajectories without labels

**Files:**
- Create: `VST-RL/recurrent/skill_opd.py`
- Create: `VST-RL/recurrent/test/test_skill_opd_trajectory.py`
- Modify: `VST-RL/verl/trainer/ppo/ray_trainer.py`

**Interfaces:**
- `ReflectionTrajectory` contains group/trajectory/policy IDs, ordered memory transition row indices, final row index, query tokens/text, predicted answer tokens/text, scalar environment reward, and references to already-observed full-video data.
- `assemble_reflection_trajectories(...) -> list[ReflectionTrajectory]` rejects missing reward, incomplete rollouts, duplicate IDs, final/query mismatch, and policy-version mismatch.

- [ ] Add failing tests proving same-group rollouts may receive different rewards, answers/rewards never cross trajectory IDs, a missing reward raises instead of becoming zero, and `ground_truth`/`correct_answer`/`answer_key` fields are absent from the resulting analyzer input object.
- [ ] Implement assembly as a pure function after final-only reward computation. Use sample index only as a checked lookup; trajectory UID is the identity key.
- [ ] Preserve current final reward, advantage, PPO, entropy, and reference-KL operations.
- [ ] Run focused tests and commit.

### Task 3: Strict reflection schema, prompt, and leakage validator

**Files:**
- Create: `VST-RL/recurrent/reflection.py`
- Create: `VST-RL/recurrent/test/test_reflection.py`

**Interfaces:**
- `build_reflection_prompt(trajectory, max_key_transitions=3) -> str` includes observed video marker, incremental `Y_k`, Q/options, prediction, and reward; excludes ground truth.
- `parse_and_validate_reflection(text, trajectory, policy_version) -> ReflectionEnvelope` accepts exactly one JSON object and returns `reflection_valid=False` with a reason on validation failure.
- Schema version is `skill_opd.reflection.v1`; payload and enum sets exactly match the supplied specification.

- [ ] Add failing schema tests for valid apply/skip payloads and every hard-reject class: fences/trailing text, extra fields, invalid enums, conditional-field violations, duplicate/out-of-range/final transition indices, ID/version mismatch.
- [ ] Add failing leakage tests after NFKC/case-fold/punctuation/whitespace normalization: full query/option, query four-word n-gram, option fragment of at least eight characters, answer-indicator patterns, and query-specific number/time/count copying.
- [ ] Implement strict JSON decoding with `JSONDecoder.raw_decode`, exact-key schema checks, normalization, n-gram/fragment/number checks, and deterministic rejection reasons. No semantic-leakage claims are made by this lexical validator.
- [ ] Add prompt-budget accounting with a hard 32768 total-token ceiling and no fixed memory-only allocation.
- [ ] Run tests and commit.

### Task 4: Pure teacher-selected localized Top-100 KL

**Files:**
- Create: `VST-RL/verl/trainer/ppo/skill_opd_loss.py`
- Create: `VST-RL/tests/test_skill_opd_loss.py`

**Interfaces:**
- `build_teacher_topk(teacher_logits, top_k, temperature) -> (indices, log_probs, retained_mass)` selects support only from detached teacher logits and normalizes on that support.
- `localized_topk_opd_loss(student_logits, teacher_topk_indices, teacher_topk_log_probs, response_mask, memory_mask, key_mask, reflection_mask, metadata_mask) -> (loss, metrics)` gathers student logits on teacher support, normalizes there, and computes forward KL.

- [ ] Add RED tests: exact teacher-selected support, detached teacher/cache, gradient only on student, mask product semantics, differentiable zero for no valid tokens, final-answer zero mask, finite loss, K exactly 100 when vocab permits, and equivalence with full KL when vocab <= 100.
- [ ] Implement the two pure functions with `log_softmax`, `gather`, and `sum(p_teacher * (logp_teacher - logp_student))`; denominator is `max(1, valid_token_count)`.
- [ ] Record valid/final token counts and retained-mass metrics.
- [ ] Run CPU and CUDA unit tests and commit.

### Task 5: Generate Actor reflection before any optimizer step

**Files:**
- Modify: `VST-RL/recurrent/skill_opd.py`
- Modify: `VST-RL/recurrent/generation_manager.py`
- Modify: `VST-RL/verl/trainer/ppo/ray_trainer.py`
- Modify: `VST-RL/recurrent/test/test_skill_opd_trajectory.py`

**Interfaces:**
- `SkillOPDManager.generate_reflections(trajectories, actor_rollout_wg) -> list[ReflectionEnvelope]` uses greedy generation (`temperature=0`, `top_p=1`, `max_tokens=256`) with the still-current rollout Actor.
- Invalid/rejected reflection produces an all-zero OPD mask while leaving the RL batch untouched.

- [ ] Add failing ordering tests with a recording worker: rollout → reward → reflection must occur before `update_actor`; no ground-truth field may cross the manager boundary.
- [ ] Retain the original per-turn `multi_modal_inputs` in the recurrent output so analyzer and teacher can consume observed video/chunk metadata.
- [ ] Build real Qwen-VL chat/video inputs under the shared 32K budget; decode model output without rewriting it, then apply the strict validator.
- [ ] Produce critical transition masks from at most three validated, non-final indices and metrics for accepted/rejected, preserve/correct, attribute counts, and context usage.
- [ ] Run mocked ordering tests plus a real-model reflection-only probe; commit after a valid strict JSON sample is observed without manual payload injection.

### Task 6: Pre-update current-Actor teacher Top-100 cache

**Files:**
- Modify: `VST-RL/verl/workers/actor/dp_actor.py`
- Modify: `VST-RL/verl/workers/fsdp_workers.py`
- Modify: `VST-RL/verl/trainer/ppo/ray_trainer.py`
- Modify: `VST-RL/recurrent/skill_opd.py`
- Create: `VST-RL/tests/test_skill_opd_cache.py`

**Interfaces:**
- Worker RPC `compute_opd_teacher_cache(data)` runs the current actor under `eval()` and `torch.no_grad()` before update.
- Cache tensors: `opd_topk_indices [B,R,100]`, `opd_teacher_topk_log_probs [B,R,100]` (BF16 allowed), `opd_valid_token_mask [B,R]`, plus row-aligned trajectory/policy/transition metadata and retained mass.

- [ ] Add failing tests for teacher-only top-k selection, exact shape, detached cache, UID/version/transition/shape rejection, BF16 finite normalization, and cache-before-update ordering.
- [ ] Extend the actor forward path minimally to return response-position logits or teacher-selected top-k; do not reuse the reference policy and do not select support from student logits.
- [ ] Build skill-augmented teacher prompts for critical memory rows only; non-critical/final rows carry zero masks.
- [ ] Cache only indices/log-probabilities/masks, measure CPU cache bytes, and reject an entire trajectory on any metadata mismatch.
- [ ] Run tests and a one-batch no-optimizer GPU cache probe; commit.

### Task 7: Add localized OPD to the existing actor update

**Files:**
- Modify: `VST-RL/verl/workers/actor/dp_actor.py`
- Modify: `VST-RL/verl/workers/fsdp_workers.py` only if dispatch keys require it.
- Modify: `VST-RL/verl/trainer/config/ppo_trainer.yaml`
- Create: `VST-RL/recurrent/test/test_skill_opd_actor.py`

**Interfaces:**
- Config namespace `skill_opd`: enable, mode, top_k, teacher_temperature, lambda_opd, max_key_transitions, reflection settings.
- `L_total = L_VST-RL + lambda_opd * L_LOPD`; existing PPO clipping, GRPO advantage, entropy, and reference KL code remains intact.

- [ ] Add RED tests for disabled-path equivalence, localized masks, zero final OPD tokens, finite RL/OPD/total losses, and parameter update driven by combined loss.
- [ ] Select cache tensors into actor mini/micro batches. Gather current student logits on cached teacher indices and call the pure OPD loss.
- [ ] Add `lambda_opd * lopd_loss` to the existing `policy_loss` after existing entropy/KL composition and before backward. With enable false, do not require cache keys and execute the original branch exactly.
- [ ] Emit all required metrics, including original RL loss, weighted OPD, total, valid/final tokens, cache bytes, retained mass, and reflection distributions.
- [ ] Run actor/config regression tests and commit.

### Task 8: Real rollout-batch GPU smoke and report

**Files:**
- Create: `VST-RL/scripts/run_skill_opd_smoke.sh`
- Create: `VST-RL/scripts/audit_skill_opd_smoke.py`
- Create after run: `docs/smoke/skill-opd-gpu-smoke.md`
- Create after run: ignored logs/checkpoints outside Git under `/home/bujunru/vlm-repro/`.

**Interfaces:**
- One command runs a minimum real video batch from absolute existing model/data paths, with at least two memory turns and one final turn, and performs exactly one joint optimizer update.
- Audit script reads produced metrics/artifacts and exits nonzero unless every acceptance criterion is met.

- [ ] Add failing audit tests for every required field and threshold; implement the pure audit.
- [ ] Add shell syntax/config tests and a disabled-OPD one-step regression.
- [ ] Wait for sufficient GPU memory without killing or altering existing jobs. Run reflection-only and teacher-cache probes, then the exact one-update smoke.
- [ ] If the Actor legally returns `apply_opd=false`, try another existing sample; never inject or edit a reflection.
- [ ] Require: correct T-1 boundary, no query leakage, correct reward mapping, strict Actor JSON, at least one critical transition, K=100, positive OPD token count, zero final OPD tokens, detached teacher, finite RL/OPD/total losses, completed optimizer step, and a finite change in at least one trainable Actor parameter.
- [ ] Record exact command/config/log paths, GPU peak, elapsed time, context usage, lambda-vs-RL loss scale, BF16 stability, cache memory, accepted/rejected/key/attribute metrics, and disabled regression in the smoke report.
- [ ] Run all unit/recurrent/RL regressions, compile/import checks, `bash -n`, Git isolation audit, and manifest hash audit. Commit code/report, push the branch, and verify local/remote SHA equality.
