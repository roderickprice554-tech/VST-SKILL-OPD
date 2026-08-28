# VST Download Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deploy an hourly remote controller that finishes and verifies the pinned VST snapshot, prepares and audits all non-Ego4D media, resumes and verifies OVO, and emits a hard smoke gate without ever starting full SFT/RL.

**Architecture:** A pure decision module maps an observed snapshot to one allow-listed action. A locked hourly wrapper invokes the controller once per hour and prevents duplicate downloads or stage transitions.

**Tech Stack:** Python 3.12 standard library, pytest, Bash, ModelScope 1.39.1, Hugging Face CLI, official VST setup/audit scripts.

## Global Constraints

- Remote root is `/home/bujunru/vlm-repro/VST-full-reproduction`.
- VST commit is `aaef152ea68ffa0e9d9f7367ccf871ea2f699693`; OVO revision is `fec29e3`.
- Missing media is permitted only for normalized paths beginning exactly `Ego4D/`.
- No full SFT or RL command may be executed by this controller.
- No existing data, manifest, result, checkpoint, or log is overwritten.

---

### Task 1: Pure state decisions

**Files:**
- Create: `audit/vst_download_orchestrator.py`
- Create: `tests/test_vst_download_orchestrator.py`

**Interfaces:**
- Produces: `Observation`, `Decision`, and `decide(observation) -> Decision`.
- Actions are limited to `none`, `restart_vst`, `verify_vst`, `prepare_vst`, `audit_vst`, `resume_ovo`, `restart_ovo`, `prepare_ovo`, `run_smoke`, and `block`.

- [ ] Write failing tests for live-process deduplication, missing-process restart, non-Ego4D blocking, Ego4D-only allowance, OVO ordering, and smoke ordering.
- [ ] Run `/home/bujunru/.conda/envs/vision-se/bin/python -m pytest tests/test_vst_download_orchestrator.py -q`; expect import failure because the module does not exist.
- [ ] Implement the minimal dataclasses and priority-ordered `decide` function.
- [ ] Run the new tests plus `tests/test_run_full_inventory.py`; expect all pass.

### Task 2: Observation and atomic evidence

**Files:**
- Modify: `audit/vst_download_orchestrator.py`
- Modify: `tests/test_vst_download_orchestrator.py`
- Create at runtime: `audit/inventories/vst_modelscope_aaef152e.json`
- Create at runtime: `audit/inventories/ovo_hf_fec29e3.json`

**Interfaces:**
- Produces: `observe(paths, previous_status) -> Observation`.
- Produces atomic `logs/vst_download_orchestrator/status.json` and append-only `events.jsonl`.

- [ ] Add failing tests for wrong-size files, temporary files, three zero-growth observations, atomic status replacement, and revision mismatch.
- [ ] Run RED and confirm the assertion failures.
- [ ] Implement inventory comparison, process inspection, byte deltas, and atomic JSON writes using `os.replace`.
- [ ] Generate inventories from the pinned repository APIs and save their SHA256.
- [ ] Run GREEN.

### Task 3: Allow-listed actions and hourly daemon

**Files:**
- Modify: `audit/vst_download_orchestrator.py`
- Create: `audit/run_vst_download_monitor.sh`
- Modify: `tests/test_vst_download_orchestrator.py`

**Interfaces:**
- `execute(decision)` invokes fixed argument arrays.
- `--once` performs at most one transition.
- `--status` is read-only.

- [ ] Add failing tests that each action maps to the exact pinned command and arbitrary/full-training commands are rejected.
- [ ] Run RED.
- [ ] Implement fixed commands for ModelScope restart, official VST setup/audit, OVO resume/restart, and OVO reconstruction.
- [ ] Implement an hourly `flock` loop that exits only at `smoke_complete` or `blocked`.
- [ ] Run GREEN and `bash -n audit/run_vst_download_monitor.sh`.

### Task 4: Safe deployment

**Files:**
- Runtime: `logs/vst_download_orchestrator/monitor.log`

- [ ] Run `--once --dry-run`; while the current VST PID is live it must choose `none`.
- [ ] Stop only the obsolete VST retry watchdog after the controller passes dry-run.
- [ ] Start exactly one daemon with `nohup` and record its PID.
- [ ] Verify OVO PID 1162909 remains stopped until the VST media audit passes.
- [ ] Commit only the orchestrator, tests, wrapper, inventories, and plan.

### Task 5: Smoke runner before readiness

**Files:**
- Create: `audit/run_vst_smoke.sh`
- Create: `tests/test_smoke_contract.py`

**Interfaces:**
- Consumes frozen non-Ego4D train/validation manifests and full OVO manifest.
- Produces only `checkpoints/vst_full/smoke_sft_*` and `results/vst_full/smoke_ovo_*`.

- [ ] Add failing contract tests for maximum five optimizer steps, 2 FPS, 384 frames, causal chunk ordering, one OVO instance per category, and forbidden selected/full output paths.
- [ ] Run RED.
- [ ] Implement a one-to-five-step two-GPU SFT launch and three-instance official OVO evaluator launch using the fixed 3B base model.
- [ ] Run static contract tests, then execute only when the controller emits `run_smoke`.
- [ ] Hash smoke inputs/outputs and mark `smoke_complete`; never continue to full training.

