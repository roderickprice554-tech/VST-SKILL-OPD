# VST OPD Trajectory Foundation Design

## Goal

Create an isolated implementation branch at commit
`26f31d36eb8bcc0b480b43eaaed1277b33a10488` and make recurrent video
rollouts expose query-independent memory transitions, one query-bearing final
turn, aligned transition metadata, final-answer reward propagation, and
trajectory aggregation. Reflection generation, teacher distributions, LOPD,
and joint actor optimization are explicitly out of scope.

## Isolation and provenance

- Worktree: `/home/bujunru/vlm-repro/VST-skill-opd`
- Branch: `codex/vst-skill-opd`
- Source repository: `/home/bujunru/vlm-repro/VST`
- Source commit: `26f31d36eb8bcc0b480b43eaaed1277b33a10488`
- Existing VST, VST-official, VST-SE, and full-reproduction working trees are
  read-only inputs and must not be modified or copied into the new worktree.
- Models, datasets, and checkpoints remain outside Git. Training configuration
  refers to them through absolute read-only paths.
- A manifest records the source commit, Python environment, model paths, data
  paths, and deterministic hashes for configuration and dataset manifest files.

## Turn semantics

For a trajectory with chunks `C_1, ..., C_T`, the rollout contains exactly
`T - 1` memory transitions followed by one final turn.

Memory transition `t`, for `1 <= t < T`:

```text
input  = M_{t-1} + C_t + query-independent instruction
output = Y_t
update = M_t
final_mask = False
```

The memory input must not contain or access question, option, answer, final
chunk, `prompt_ids`, or `question_ids` data. Both prompt types implement this
same contract. In particular, `TEMPLATE_TYPE_2` removes its `<problem>` block.

Final turn:

```text
input  = M_{T-1} + C_T + question/options
output = final answer
final_mask = True
```

Only the final branch may read `prompt_ids` and `question_ids`. Final-turn rows
remain part of the recurrent VST-RL rollout and reward path but are not OPD
memory transitions.

## Metadata contract

Each generated row remains aligned with the corresponding `DataProto` tensor
row after `DataProto.concat()`. The non-tensor metadata fields are:

- `trajectory_uid`: the rollout `uid`; permanent identity across turns.
- `sample_index`: position of the trajectory in the current repeated batch.
- `transition_index`: zero-based memory transition index; `None` on final rows.
- `previous_memory_tokens`: snapshot of `M_{t-1}`; `None` on final rows.
- `current_chunk_boundary`: half-open frame interval `[start_frame, end_frame)`
  plus its half-open time interval in seconds.
- `generated_y_t_tokens`: unpadded response without EOS; `None` on final rows.
- `updated_memory_tokens`: snapshot of `M_t`; `None` on final rows.
- `policy_version`: the trainer `global_steps` value frozen before the batch
  starts; identical for every row in that batch.
- `final_mask`: `False` for memory transitions and `True` for final turns.

`response_mask` remains a boolean tensor in `DataProto.batch`, aligned with the
padded response tensor. It is referenced by the transition row rather than
duplicated into object metadata.

Final rows retain chunk boundary, response mask, policy version, trajectory UID,
and sample index for alignment and auditing. They do not pretend to be memory
transitions: their transition-only token fields and index are `None`.

## Component changes

### `VST-RL/recurrent/interface.py`

Define the names and validation rules for recurrent turn metadata. Validation
checks required fields, row counts, unique final rows, monotonic zero-based
transition indices per trajectory, and the distinction between memory and final
rows. The interface does not perform file I/O.

### `VST-RL/recurrent/impls/video_memory.py`

Make both memory templates query-independent. During `action()`, snapshot the
previous memory and exact chunk boundary without touching prompt or question
tokens on non-final turns. During `update()`, capture unpadded `Y_t`, construct
the response mask from the actual generated response, update memory, and attach
the completed metadata to the generated rows. The final branch alone reads the
question-bearing fields and emits final-row metadata.

### `VST-RL/recurrent/generation_manager.py`

Freeze `policy_version` once for `run_llm_loop()`, collect each turn's metadata,
and validate alignment before and after `DataProto.concat()`. Preserve the
existing three-value return shape so current recurrent callers continue to
receive `(outputs, final_mask, sample_index)`.

### `VST-RL/verl/trainer/ppo/ray_trainer.py`

Pass the pre-rollout `self.global_steps` value into the generation manager.
Continue computing environment reward only from the final batch. Propagate
trajectory reward with `trajectory_reward = final_reward[sample_index]`, and
assert that every propagated row maps to the same `trajectory_uid` as its final
row before applying the reward.

### Trajectory aggregation

Add the smallest pure helper beside the recurrent metadata contract. It groups
rows by `trajectory_uid`, orders memory rows by `transition_index`, appends the
single final row, and rejects missing/duplicate finals, duplicate/gapped
transition indices, policy-version changes within a trajectory, and UID/index
mismatches. It performs no model calls and no disk writes.

## Error handling

Structural inconsistencies fail immediately with descriptive `ValueError` or
assertion messages before actor optimization. No fallback UID, inferred policy
version, or silent row dropping is allowed. Invalid trajectories cannot enter
later OPD stages.

## Tests and success criteria

Tests are written first and observed failing before implementation. Lightweight
unit fixtures use synthetic `DataProto` rows and a minimal tokenizer/agent
fixture; no model, video dataset, GPU, or checkpoint is required.

The focused suite proves:

1. A `T`-chunk trajectory produces exactly `T - 1` memory transitions and one
   final row.
2. `C_T` is absent from every memory-turn visual input.
3. Both memory prompt types contain no question, option, or answer tokens and
   the non-final branch never reads `prompt_ids` or `question_ids`.
4. The final input contains `M_{T-1}`, `C_T`, and the question/options.
5. Tensor rows, response masks, metadata rows, `final_mask`, and `sample_index`
   remain aligned through concatenation.
6. Final reward is propagated to all and only rows of the matching
   `trajectory_uid`.
7. Aggregation returns complete ordered trajectories and rejects cross-UID
   leakage or malformed turn sequences.
8. The new worktree remains clean at its baseline, starts from the exact commit,
   and contains none of the source working tree's untracked experiment assets.

## Deferred work

Reflection generation and leakage validation, teacher top-100 distributions,
LOPD masking/loss, and joint VST-RL plus LOPD actor updates require separate
designs and are not implemented in this project.
