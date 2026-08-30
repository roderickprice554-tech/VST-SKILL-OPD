# VST Training-First Orchestration Design

**Date:** 2026-08-30

## Goal

Prepare and audit the non-Ego4D VST data, run a causal smoke test, and start the fixed Qwen2.5-VL-3B-Instruct SFT as soon as all gates pass. Download and validate OVO-Bench concurrently, but never let OVO evaluation delay or destabilize training.

## Fixed experiment boundaries

- The experiment is named **VST 3B non-Ego4D reproduction**; it is not a complete official-data reproduction.
- The base model is `/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct`.
- SFT runs for one epoch on two A100 40GB GPUs with per-device batch 1 and gradient accumulation 64, preserving effective global batch 128.
- Full SFT may start only after VST media preparation, non-Ego4D manifest/seek/media/leakage audit, and causal smoke all pass.
- The official complete OVO set is 3,035 examples. The 780-example cache is not a formal evaluation set.
- OVO forward tasks retain the official string/integer scoring rules and are not converted to MCQ letter accuracy.
- Every new artifact uses a new output directory; existing checkpoints, logs, manifests, and results are preserved.

## Scheduling policy

The controller checks state every 30 minutes and is idempotent: it detects existing processes and completion markers before starting any action.

GPU work is training-first. The two-GPU SFT allocation reserves both physical GPUs. OVO inference may use only an entire physical GPU that is not allocated to training; unused VRAM on a training GPU does not count as an idle GPU. Consequently, OVO download and CPU-side validation may continue during SFT, but OVO inference normally waits until SFT releases the GPUs.

The state priority is:

1. Continue the existing VST media preparation without restarting it.
2. Generate and validate non-Ego4D SFT/RL manifests and seek indexes after the media completion marker exists.
3. Run the isolated causal smoke test.
4. Start the fixed two-GPU SFT when every audit and smoke marker is valid.
5. Continue OVO download and preparation independently at reduced CPU and I/O priority.
6. Run the Qwen2.5-VL-3B-Instruct OVO baseline only when the complete OVO snapshot and 3,035-example manifest pass validation and a whole GPU is not allocated to training.

If an OVO GPU job is already running when SFT becomes ready, the controller must not start SFT into contested GPUs. It records the conflict and stops the OVO job cleanly or waits according to the evaluator's supported resume semantics; it must never kill an evaluator without preserving its outputs.

## Components

### Thirty-minute monitor

Reuse the existing orchestrator and monitor rather than create a second scheduler. Change the loop interval to 1,800 seconds and extend the state machine with explicit audit, smoke, SFT, OVO-prepare, and OVO-evaluation gates. Each transition writes timestamped status and event records.

### VST audit gate

The audit runs only after `vst_prepare.complete`. It excludes any case-insensitive path segment named `ego4d`, generates seek indexes in the filtered directory, validates JSONL/parquet structure and uniqueness, checks media existence and duration, and reports train/validation/OVO leakage. A failure produces no success marker and blocks smoke and training.

### Causal smoke gate

The smoke uses isolated samples and the official FPS/chunk/deadline behavior. Each inference step can access only current and earlier frames. It validates data loading, model initialization, memory updates, and the three official evaluator families without changing full-run parameters.

### SFT launcher

The launcher fixes the 3B model path, two GPUs, one epoch, global batch semantics, provenance, ZeRO-3 source, and a new checkpoint/log directory. It refuses to run if any required marker is absent, another training process exists, disk space is below the safety threshold, or either GPU is allocated to another workload.

### OVO pipeline

The existing fixed-revision download resumes instead of spawning a duplicate. Preparation validates all 22 files and the full 3,035 annotations before producing its completion marker. Baseline evaluation is resource-gated and writes per-example predictions plus official aggregate metrics to a new result directory.

## Failure handling and observability

- Process discovery and markers make every 30-minute pass safe to repeat.
- A stale marker is rejected when its recorded inputs, revision, or configuration do not match the current fixed inputs.
- Failed actions record the command, exit status, log path, and blocking state; automatic progression stops at that gate.
- Disk capacity is checked before extraction, training, and evaluation. Less than 2TB free blocks the next heavy stage.
- OVO download may be paused if its CPU or storage load materially reduces training throughput.

## Verification

- Unit tests cover decision priority, duplicate-process prevention, marker validation, 30-minute cadence, GPU allocation, disk guard, and failure states.
- Audit tests cover nested/case-insensitive Ego4D paths, seek offsets, missing media, malformed records, and leakage detection.
- Smoke tests demonstrate causal frame access and official scoring behavior.
- Launcher tests verify the exact 3B path, two-GPU allocation, one epoch, effective batch 128, output isolation, and refusal when a gate is missing.
- Before activation, run focused tests, the full audit/orchestrator test suite, shell syntax checks, and dry-run state transitions.

## Success criteria

- One existing OVO downloader continues without duplication.
- The monitor evaluates state every 30 minutes.
- No full SFT starts before all VST audit and causal smoke gates pass.
- Once the gates pass and resources are safe, the two-GPU Qwen2.5-VL-3B-Instruct SFT starts exactly once.
- OVO inference never uses a GPU allocated to SFT and never delays SFT priority.
- The full OVO baseline runs only against the validated 3,035-example official set and preserves official scoring.
