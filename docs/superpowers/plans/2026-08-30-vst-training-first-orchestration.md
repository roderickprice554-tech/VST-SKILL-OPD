# VST Training-First Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and activate a 30-minute, idempotent controller that completes non-Ego4D VST audit and causal smoke, starts the fixed two-GPU Qwen2.5-VL-3B SFT exactly once, and runs full OVO only on physical GPUs not allocated to training.

**Architecture:** Extend the existing single orchestrator rather than add another scheduler. Independent marker-producing shell/Python stages perform audit, smoke, SFT, OVO preparation, and OVO evaluation; the pure `decide()` function enforces training-first priority and tests cover every transition before production code changes.

**Tech Stack:** Python 3.11, pytest, Bash, PyTorch/torchrun, DeepSpeed ZeRO-3, Qwen2.5-VL, existing VST and OVO evaluators.

## Global Constraints

- Experiment name: VST 3B non-Ego4D reproduction.
- Base model: `/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct`.
- SFT: two A100 40GB GPUs, one epoch, per-device batch 1, gradient accumulation 64, effective global batch 128.
- Full SFT requires valid VST media, non-Ego4D audit, seek indexes, causal smoke, at least 2TB free, and two unallocated GPUs.
- OVO formal evaluation uses all 3,035 official examples and official forward string/integer scoring.
- OVO inference uses only a whole physical GPU not allocated to SFT; spare VRAM on a training GPU is not idle capacity.
- Existing user files, logs, checkpoints, manifests, and results are preserved.

---

### Task 1: Training-first state machine and 30-minute cadence

**Files:**
- Modify: `audit/vst_download_orchestrator.py`
- Modify: `audit/run_vst_download_monitor.sh`
- Modify: `tests/test_vst_download_orchestrator.py`

**Interfaces:**
- Consumes: marker/process observations.
- Produces: `Decision(action, state)` for VST audit, smoke, SFT, OVO preparation, and OVO evaluation.

- [ ] Add failing tests proving smoke follows VST audit without waiting for OVO, SFT follows smoke, an existing SFT is not duplicated, OVO inference is blocked when both GPUs are allocated, and OVO download continues independently.
- [ ] Run `/home/bujunru/.conda/envs/vision-se/bin/python -m pytest tests/test_vst_download_orchestrator.py -q`; verify the new assertions fail because SFT/eval observations and decisions do not exist.
- [ ] Minimally extend `Observation` with `sft_pid`, `sft_complete`, `ovo_eval_pid`, `ovo_eval_complete`, `training_gpu_ids`, and `idle_gpu_ids`; order `decide()` as VST prepare → VST audit → smoke → SFT → OVO prepare/eval.
- [ ] Add allow-listed commands `run_sft` and `run_ovo_eval`; retain duplicate-process discovery and marker checks.
- [ ] Change monitor log wording and `sleep 3600` to `sleep 1800`; keep the single flock lock.
- [ ] Re-run focused tests and `bash -n audit/run_vst_download_monitor.sh`; expect PASS.
- [ ] Commit exact Task 1 files with message `feat: prioritize VST training in orchestrator`.

### Task 2: Non-Ego4D manifests, seek indexes, and blocking audit

**Files:**
- Modify: `audit/audit_vst_no_ego4d.py`
- Modify: `audit/audit_vst_no_ego4d.sh`
- Modify: `tests/test_audit_vst_no_ego4d.py`

**Interfaces:**
- Produces: filtered `*_with_seeks.jsonl`, paired `*_seeks.jsonl`, filtered RL parquet, `audit.json`, and `vst_audit.complete` only on success.

- [ ] Add failing tests for byte offsets across UTF-8 JSONL lines, nested/case-insensitive Ego4D filtering, missing media blocking, duplicate/empty records, and stale output refusal.
- [ ] Run focused tests and verify RED failures are due to absent seek/audit helpers.
- [ ] Implement `write_filtered_jsonl_and_seeks()` using binary offsets so each seek points to the start of its retained JSON line.
- [ ] Make the report record source hashes/counts, missing non-Ego4D media, uniqueness, malformed/empty fields, and split leakage summaries; return nonzero for any blocking finding.
- [ ] Ensure the shell wrapper checks `vst_prepare.complete`, writes its marker atomically only after a zero exit, and never accepts an unrelated pre-existing manifest silently.
- [ ] Run focused tests and a dry-run/temporary-fixture audit; expect PASS without touching the formal manifest directory.
- [ ] Commit exact Task 2 files with message `feat: complete non-Ego4D VST audit`.

### Task 3: Causal smoke runner

**Files:**
- Create: `audit/run_vst_smoke.py`
- Modify: `audit/run_vst_smoke.sh`
- Create: `tests/test_run_vst_smoke.py`

**Interfaces:**
- Produces: structured smoke report and `smoke.complete` only when data loading, model initialization, causal frame access, and official scoring checks pass.

- [ ] Add failing unit tests for `causal_chunks(frame_ids, chunk_size)` ensuring chunk N never includes a later frame, plus official MCQ and OVO forward scoring fixture tests.
- [ ] Verify RED failures because the runner module is absent.
- [ ] Implement the smallest isolated runner that loads a few audited manifest samples, validates seek reads, initializes the fixed 3B model in no-training mode, and checks streaming inputs with the official FPS/frame ceiling.
- [ ] Make the shell wrapper require `vst_audit.complete`, record logs/output under a new smoke directory, and atomically create the marker only after success.
- [ ] Run unit tests, then the isolated GPU smoke; expect a zero exit and a provenance-bearing report.
- [ ] Commit exact Task 3 files with message `feat: add causal VST smoke gate`.

### Task 4: Fixed 3B two-GPU SFT launcher

**Files:**
- Create: `VST-SFT/scripts/zero3-vst-2xa100.json`
- Create: `audit/run_vst_sft.sh`
- Create: `audit/validate_vst_sft_launch.py`
- Create: `tests/test_vst_sft_launch.py`

**Interfaces:**
- Consumes: audit/smoke markers and filtered manifests.
- Produces: a new timestamped checkpoint directory, provenance JSON, logs, and `sft.complete` after successful one-epoch training.

- [ ] Add failing tests that inspect the rendered command/config and require the exact 3B path, two processes, epoch 1, accumulation 64, effective batch 128, frozen visual tower behavior, new output directory, 2TB guard, and rejection when any marker/GPU is missing.
- [ ] Verify RED failures because launcher/config do not exist.
- [ ] Add a documented standard ZeRO-3 configuration compatible with the installed DeepSpeed version; validate every field against local package behavior and record its provenance rather than claiming it was published by VST.
- [ ] Implement a validator that discovers all audited train/valid JSONL files, paired seek files, GPU allocation, disk space, existing training processes, and marker provenance.
- [ ] Implement `run_vst_sft.sh` with `torchrun --nproc_per_node=2`, fixed 3B path, one epoch, accumulation 64, official optimizer/LR/scheduler/frame/text settings, isolated outputs, and atomic completion marker.
- [ ] Run tests, shell syntax check, DeepSpeed config parse, and a launch dry-run; expect PASS and no training process.
- [ ] After all previous gates pass, start the launcher exactly once and verify both GPUs, command line, log growth, and checkpoint output.
- [ ] Commit exact Task 4 files with message `feat: launch fixed VST 3B SFT` before activation.

### Task 5: Full OVO preparation and idle-GPU baseline evaluation

**Files:**
- Modify: `audit/prepare_ovobench_official.sh`
- Create: `audit/run_ovo_qwen3b_eval.sh`
- Create: `audit/validate_ovobench.py`
- Create: `tests/test_ovobench_pipeline.py`

**Interfaces:**
- Consumes: fixed 22-file snapshot, prepared media, three official annotation JSON files, idle physical GPU list.
- Produces: validated 3,035-example manifest, per-example Qwen 3B predictions, official aggregate metrics, and `ovo_eval.complete`.

- [ ] Add failing tests for 22-file completeness, 631/837/1,567 annotation counts, exact 3,035 total, official category scoring, new result directory, and refusal when the selected GPU is allocated to SFT.
- [ ] Verify RED failures because validation/launcher are absent.
- [ ] Implement validation and make OVO preparation marker-bearing only after archive/media/annotation checks pass.
- [ ] Implement the baseline wrapper around the existing evaluator using the fixed 3B model and one controller-selected idle physical GPU; preserve per-example output and official scoring.
- [ ] Run unit tests and evaluator smoke. Start full baseline only when the snapshot is complete, preparation passes, and a whole GPU is idle.
- [ ] Commit exact Task 5 files with message `feat: evaluate Qwen 3B on full OVO`.

### Task 6: Activation and end-to-end verification

**Files:**
- Modify only if verification reveals a task-scoped defect: files from Tasks 1–5.

**Interfaces:**
- Produces: active 30-minute monitor and evidence-backed status.

- [ ] Run all focused tests, full `pytest tests -q`, `bash -n` on every changed wrapper, and `python -m py_compile` on every changed Python file.
- [ ] Run controller `--dry-run` against live state and confirm it selects exactly the expected next action without duplicate processes.
- [ ] Restart only the existing monitor loop if its cadence/code must be reloaded; verify exactly one flock owner and a 1,800-second interval.
- [ ] Verify the OVO downloader remains single and active, VST preparation is not interrupted, disk remains above 2TB, and no premature GPU job starts.
- [ ] Record final commit, current state, PIDs, logs, markers, and next automatic transition in the handoff document.

