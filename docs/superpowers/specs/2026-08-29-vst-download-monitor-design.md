# VST Download Monitor and Smoke Gate Design

## Goal

Run an hourly, restart-safe remote state machine that completes every officially
redistributed VST training artifact, explicitly permits only missing `Ego4D/`
media, then resumes the paused full OVO-Bench download, prepares both datasets,
and finally runs an isolated smoke experiment. It must never start full SFT or
RL training.

## Fixed scope and provenance

- Remote root: `/home/bujunru/vlm-repro/VST-full-reproduction`.
- VST source: ModelScope dataset `catalan/VST-Training-Data`, commit
  `aaef152ea68ffa0e9d9f7367ccf871ea2f699693`.
- Corresponding Hugging Face revision:
  `5647583491c298aa8b2926fe9910f651fc0e692d`.
- OVO source: Hugging Face dataset `JoeLeelyf/OVO-Bench`, revision `fec29e3`,
  through `hf-mirror.com`.
- Ego4D is not downloaded or synthesized. Missing paths are allowed only when
  their normalized prefix is exactly `Ego4D/`.
- Existing data, logs, checkpoints, and manifests are never overwritten.
- Formal SFT and RL commands are outside this automation.

## Components

### `audit/vst_download_orchestrator.py`

A single-run controller. One invocation inspects current state, performs at most
one transition, writes an atomic JSON status file, and exits. It owns no
long-running download connection.

Inputs are fixed constants for repository IDs, revisions, paths, process match
patterns, and commands. The only command-line modes are `--once` and
`--status`. There is no generic arbitrary-command configuration.

### `audit/run_vst_download_monitor.sh`

A minimal hourly loop that acquires `flock`, invokes the controller with
`--once`, sleeps 3600 seconds, and repeats. `nohup` starts exactly one copy.
The lock prevents a manual run and the daemon from advancing state
simultaneously.

### `tests/test_vst_download_orchestrator.py`

Unit tests exercise state decisions using temporary directories and injected
process/command results. Tests never contact ModelScope, Hugging Face, or GPUs.

## State machine

1. `vst_downloading`
   - Record byte count, completed files, incomplete files, process state, and
     one-hour byte delta.
   - If the pinned ModelScope downloader is absent and the immutable inventory
     is incomplete, restart it with 8 file workers, 10 retries, and the same
     local directory.
   - `hdvila` completion is reported separately, but does not require a special
     launch: the snapshot downloader automatically schedules the remaining
     official files.

2. `vst_snapshot_verifying`
   - Enter only after the pinned snapshot command exits successfully.
   - Compare every official path and byte size with a saved immutable inventory.
   - Reject zero-byte, missing, wrong-size, or still-temporary files.

3. `vst_preparing`
   - Run the official `vst_video/setup_dataset.py` in a separate background job
     to join split archives, verify shipped SHA256 files, extract media, and
     rebuild seek indexes.
   - Write output to the new `data/vst-training-media-no-ego4d` directory.

4. `vst_media_auditing`
   - Build fresh SFT/RL manifests from the pinned annotations.
   - The gate passes only when every missing media row has prefix `Ego4D/`.
   - Record total and excluded Ego4D rows per source. Never edit the official
     annotation files.

5. `ovo_downloading`
   - Resume the existing stopped OVO process with `SIGCONT` if it still exists.
   - If it no longer exists and the snapshot is incomplete, start the existing
     retrying downloader script at the same revision and directory.

6. `ovo_preparing`
   - Verify all official archive parts and hashes, reconstruct/extract the
     official media layout, and freeze a manifest with hashes.

7. `smoke_running`
   - Run only after both preparation gates pass.
   - Use representative existing samples, official FPS/chunk/causal rules, and
     at most five SFT optimizer steps plus one fixed instance from each OVO task.
   - Write only under `checkpoints/vst_full/smoke_*` and
     `results/vst_full/smoke_*`; never write `sft_selected` or `rl_selected`.

8. `smoke_complete` or `blocked`
   - Save exit codes, command lines, environment versions, Git commit, seeds,
     manifest hashes, and artifact paths.
   - Stop the hourly loop after `smoke_complete`.
   - A blocker is explicit and does not trigger full training.

## Completion evidence

Atomic status file:
`logs/vst_download_orchestrator/status.json`.

Append-only event log:
`logs/vst_download_orchestrator/events.jsonl`.

Each event contains timestamp, state before/after, command decision, VST/OVO
bytes and deltas, process IDs, inventory revision, and error summary. A separate
`artifacts.sha256` covers final manifests and smoke outputs.

## Error handling

- Network errors leave partial files untouched and restart the same pinned
  command on the next hourly tick.
- Stalled means zero byte growth for three consecutive hourly checks while a
  downloader is alive; it is logged as `stalled` and restarted once.
- After three consecutive failed restarts, state becomes `blocked` and no later
  stages run automatically.
- Dataset verification, extraction, audit, or smoke failures become `blocked`
  immediately because retrying them can hide deterministic corruption.
- Every subprocess uses an argument list without `shell=True`.

## Test acceptance criteria

- An incomplete snapshot with a live matching downloader does not start a
  duplicate.
- An incomplete snapshot without a downloader requests exactly one restart.
- Three zero-growth samples request one safe restart.
- A process exit alone never marks a snapshot complete.
- Wrong-size and temporary files block verification.
- Missing non-Ego4D media blocks OVO resume and smoke.
- Missing only `Ego4D/` media permits OVO resume.
- OVO is resumed only after the VST media audit passes.
- Smoke is requested only after both immutable manifests pass.
- No transition can produce a full SFT or RL command.

