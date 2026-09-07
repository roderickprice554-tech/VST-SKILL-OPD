# VST OPD Trajectory Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce query-independent VST memory transitions, aligned rollout metadata, UID-checked final reward propagation, and complete trajectory aggregation without changing the original VST-RL objective.

**Architecture:** `VideoMemoryAgent` owns turn semantics and token snapshots; `LLMGenerationManager` owns per-row alignment and the frozen policy version; pure helpers in `recurrent.interface` validate, aggregate, and propagate final rewards. The recurrent trainer continues scoring final rows only and calls the pure reward helper before actor optimization.

**Tech Stack:** Python 3.12 (`/home/bujunru/.conda/envs/vision-se` for tests), PyTorch, NumPy, TensorDict, VERL `DataProto`, pytest.

## Global Constraints

- Work only in `/home/bujunru/vlm-repro/VST-skill-opd` on `codex/vst-skill-opd`.
- The source baseline is exactly `26f31d36eb8bcc0b480b43eaaed1277b33a10488`.
- Do not copy data, model weights, checkpoints, caches, or untracked files from any existing VST checkout.
- Memory turns consume only previous memory, current non-final chunk, and query-independent instructions.
- Only final turns may read question or option tokens.
- A trajectory contains exactly `T - 1` memory transitions and one final row.
- `policy_version` is the integer `global_steps` value frozen before rollout begins.
- Reflection, teacher top-100 distributions, LOPD, and joint OPD optimization are out of scope.

---

### Task 1: Provenance manifest and test harness

**Files:**
- Create: `VST-RL/recurrent/test/test_opd_trajectory.py`
- Create: `manifests/opd-foundation.json`

**Interfaces:**
- Consumes: Git commit, selected Python executable, externally stored asset paths.
- Produces: a deterministic JSON provenance record and a pytest module for later tasks.

- [ ] **Step 1: Write the failing manifest test**

Add a test that loads `manifests/opd-foundation.json` and asserts exact source commit, absolute environment path, `assets_copied == false`, and SHA-256-shaped hashes for every declared config/data manifest. Initially the file is absent.

- [ ] **Step 2: Verify RED**

Run: `/home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q recurrent/test/test_opd_trajectory.py::test_provenance_manifest`

Expected: FAIL because `manifests/opd-foundation.json` does not exist.

- [ ] **Step 3: Add the minimal manifest**

Create JSON with these concrete keys:

```json
{
  "source_commit": "26f31d36eb8bcc0b480b43eaaed1277b33a10488",
  "python": "/home/bujunru/.conda/envs/vision-se/bin/python",
  "assets_copied": false,
  "models": [],
  "datasets": [],
  "file_sha256": {}
}
```

Populate model/dataset absolute paths only when an existing training config identifies them. Hash only small manifests/configuration files, never the assets themselves.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused test, then commit `test: define OPD provenance contract`.

### Task 2: Pure metadata contract, aggregation, and reward propagation

**Files:**
- Modify: `VST-RL/recurrent/interface.py`
- Modify: `VST-RL/recurrent/test/test_opd_trajectory.py`

**Interfaces:**
- Produces: `validate_recurrent_turns(output, final_mask, sample_index) -> None`, `aggregate_trajectories(output, final_mask, sample_index) -> dict[str, list[int]]`, and `propagate_trajectory_reward(final_reward, output, final_mask, sample_index) -> torch.Tensor`.
- Consumes: aligned `DataProto`, Boolean final mask, and original-batch sample indices.

- [ ] **Step 1: Write failing tests**

Use a synthetic `DataProto` containing two trajectories with interleaved memory rows and one final row each. Assert ordered aggregation, rejection of missing/duplicate finals and gapped indices, constant policy version, and exact reward expansion such as:

```python
expanded = propagate_trajectory_reward(
    torch.tensor([[1.0], [3.0]]), output, final_mask, sample_index
)
assert expanded.squeeze(-1).tolist() == [1.0, 3.0, 1.0, 1.0, 3.0]
```

Add a mismatched-UID case that must raise `ValueError` instead of propagating reward across trajectories.

- [ ] **Step 2: Verify RED**

Run the new helper tests and confirm import failures name the missing functions.

- [ ] **Step 3: Implement the minimum pure helpers**

Use the required metadata keys:

```python
RECURRENT_TURN_METADATA_KEYS = (
    "trajectory_uid", "sample_index", "transition_index",
    "previous_memory_tokens", "current_chunk_boundary",
    "generated_y_t_tokens", "updated_memory_tokens", "policy_version",
)
```

Validation must compare every metadata array length with `len(output)`, compare metadata `sample_index` with the tensor argument, require one final per UID, require final transition fields to be `None`, require memory indices to be `range(n)`, and require one policy version per trajectory. Aggregation returns row indices, not copied tensors. Reward propagation first reorders final UIDs by `sample_index`, validates each row UID against its final UID, then returns `final_reward[sample_index]`.

- [ ] **Step 4: Verify GREEN and commit**

Run all tests in `test_opd_trajectory.py`, then commit `feat: define recurrent trajectory metadata`.

### Task 3: Query-independent VideoMemory turns and per-turn snapshots

**Files:**
- Modify: `VST-RL/recurrent/impls/video_memory.py`
- Modify: `VST-RL/recurrent/test/test_opd_trajectory.py`

**Interfaces:**
- Consumes: existing `uid`, chunk tensors, previous memory, generated responses.
- Produces: per-row non-tensor metadata except `policy_version`; final rows carry `None` for transition-only fields.

- [ ] **Step 1: Write failing prompt-boundary tests**

Assert `TEMPLATE_TYPE_2` has no `{prompt}` or `<problem>`. Build a non-final agent fixture whose `prompt_ids` and `question_ids` mapping entries raise on access; `action()` must still succeed. Assert its visual slice ends before the final chunk. Build a final-turn fixture and assert the rendered tokens include prior memory and the question/options.

- [ ] **Step 2: Verify RED**

Run only the prompt-boundary tests. Confirm failures show current `TEMPLATE_TYPE_2` query leakage and non-final question access.

- [ ] **Step 3: Make memory prompt construction query-independent**

Delete the `<problem>{prompt}</problem>` block from `TEMPLATE_TYPE_2`. Move reads of `prompt_ids` and `question_ids` inside `if is_final_turn:`; do not add fallback reads. Preserve both final templates.

- [ ] **Step 4: Verify prompt tests GREEN**

Run the same prompt tests and confirm they pass.

- [ ] **Step 5: Write failing transition snapshot tests**

For a synthetic three-chunk trajectory, assert two non-final metadata rows and a final row. Check previous memory, `[start_frame, end_frame)` and seconds, unpadded `Y_t`, updated memory, zero-based transition index, UID, sample index, and `None` transition fields on final.

- [ ] **Step 6: Implement minimal pending metadata lifecycle**

In `action()`, capture immutable per-row UID/index/boundary/previous-memory data. In `update()`, derive unpadded response tokens, update memory only for non-final rows, complete metadata arrays, and attach them to `gen_output.non_tensor_batch`. Copy token lists so later memory mutation cannot alter prior transitions.

- [ ] **Step 7: Verify GREEN and commit**

Run `test_opd_trajectory.py`, then commit `feat: capture query-independent memory transitions`.

### Task 4: Manager alignment and frozen policy version

**Files:**
- Modify: `VST-RL/recurrent/generation_manager.py`
- Modify: `VST-RL/recurrent/test/test_opd_trajectory.py`

**Interfaces:**
- Changes: `run_llm_loop(gen_batch, timing_raw, policy_version: int)`; return remains `(DataProto, final_mask, sample_index)`.
- Produces: aligned `response_mask`, `policy_version`, and `final_mask` rows in concatenated output.

- [ ] **Step 1: Write failing manager tests**

Use a small fake agent and rollout worker to return differently sized turns. Assert the manager freezes one integer version, creates `response_mask` from the response portion of `attention_mask`, concatenates every metadata key in row order, and rejects missing/misaligned metadata.

- [ ] **Step 2: Verify RED**

Run manager tests and confirm failure because the signature and alignment checks are missing.

- [ ] **Step 3: Implement minimal manager integration**

Require an integer `policy_version`. After each `agent.update()`, add an object array of that version and a Boolean response mask tensor. After `agent.end()`, concatenate outputs, set/check final masks, call `validate_recurrent_turns`, and return the original three values.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused suite and commit `feat: align recurrent transition batches`.

### Task 5: Trainer reward contract

**Files:**
- Modify: `VST-RL/verl/trainer/ppo/ray_trainer.py`
- Modify: `VST-RL/recurrent/test/test_opd_trajectory.py`

**Interfaces:**
- Consumes: `self.global_steps`, final-only environment reward, aligned transition output.
- Produces: UID-validated `trajectory_reward` for every recurrent row.

- [ ] **Step 1: Write failing integration/source-contract test**

Assert both recurrent generation call sites pass `policy_version=self.global_steps`. Exercise `propagate_trajectory_reward` through the recurrent reward fixture and assert wrong UID mapping fails before actor update.

- [ ] **Step 2: Verify RED**

Run the focused tests and confirm the current calls lack policy version and explicit reward validation.

- [ ] **Step 3: Implement the minimal trainer changes**

Pass `self.global_steps` at validation and training rollout entry. Keep `final_batch(...)` as the only environment-reward input. Replace direct recurrent `reward_tensor[sample_index]` uses with one computed `trajectory_reward = propagate_trajectory_reward(...)`; use it for token-level scores and preserve the existing GRPO advantage behavior.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused suite and commit `feat: validate recurrent reward propagation`.

### Task 6: Full verification and handoff

**Files:**
- Update: `manifests/opd-foundation.json` only if verified existing asset/config paths are available.

**Interfaces:**
- Produces: a clean, locally committed branch ready for later push.

- [ ] **Step 1: Run focused and baseline tests**

Run:

```bash
/home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q recurrent/test/test_opd_trajectory.py
/home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q recurrent/test/test_pad_tensor_list_to_length.py
/home/bujunru/.conda/envs/vision-se/bin/python -m py_compile recurrent/interface.py recurrent/generation_manager.py recurrent/impls/video_memory.py verl/trainer/ppo/ray_trainer.py
```

Expected: every test passes and compilation exits zero.

- [ ] **Step 2: Verify repository isolation and provenance**

Check exact worktree path, branch, merge-base ancestry from the source commit, tracked file sizes, untracked paths, and `git diff --check`. Confirm no data/model/checkpoint/cache asset entered Git.

- [ ] **Step 3: Review diff and commit final manifest update**

Every changed line must trace to the approved project-one scope. Commit only if the manifest required a verified update. Do not push; report the local commit chain and wait for explicit push authorization.
