# Reflection SFT and Global-Episode Skill OPD Design

Date: 2026-09-08

## Objective

Add a minimal offline Reflection SFT stage before Skill OPD training. An
external teacher prepares structured post-hoc reflection targets without seeing
the ground-truth answer. The SFT checkpoint then initializes the Actor used for
streaming rollout. During OPD training, a frozen snapshot of that same current
Actor acts as both the Analyzer and the skill-conditioned top-100 teacher.

Correct the current localized-only mask so that an accepted `episode_skill`
supervises every non-final memory transition. Analyzer-selected key transitions
add a `step_skill`; final-answer tokens never receive OPD.

## Explicit constraints

- Neither the offline external teacher nor the online Analyzer receives the
  ground-truth answer, correct option text, or correct option letter.
- Both receive only a boolean `is_correct`, not the raw environment reward.
- The external API is not called in the current smoke. The calling interface and
  an optional OpenAI-compatible implementation are present, but smoke data uses
  a response explicitly labelled as a fixture.
- The online Analyzer and OPD teacher are not external models. Both use the
  current Actor snapshot before any optimizer step for the rollout batch.
- Keep VST's append-only streaming memory and its existing PPO/GRPO, entropy,
  reference-KL, reward, and advantage behavior.
- Do not copy models, video data, or checkpoints into the worktree.
- Current acceptance is CPU code smoke; real model/GPU behavior is outside this
  change's completion gate.

## Non-goals

- Selecting a final external teacher provider or model.
- Calling a paid external API during tests or smoke.
- Full Reflection SFT training on the Qwen checkpoint.
- Full-scale RL training or hyperparameter selection.
- Free-form chain-of-thought targets.
- A second OPD teacher pass or a separately weighted step-level OPD loss.

## Terminology

- **External teacher**: an offline model that creates Reflection SFT labels.
- **Analyzer**: the frozen current Actor generating reflection JSON online.
- **OPD teacher**: the same frozen current Actor producing skill-conditioned
  top-100 distributions before the Actor update.
- **Student**: the trainable Actor evaluated on the original, unaugmented memory
  prompt during the update.
- **Episode skill**: one query-independent skill applicable to every memory turn
  in an accepted trajectory.
- **Step skill**: an additional query-independent skill for an Analyzer-selected
  key transition.

## End-to-end flow

### Offline Reflection SFT preparation

1. Read previously generated, reward-bearing rollout trajectories.
2. Convert the final reward to `is_correct: bool` and discard the raw value from
   the external-teacher and student-facing request.
3. Build an Analyzer request from an allowlist: observed video, all memory
   transitions, question/options, Actor prediction, `is_correct`,
   `trajectory_uid`, and `policy_version`.
4. Either export requests without calling a provider, or pass them to a
   configured external-teacher client.
5. Parse each response as one strict reflection JSON object and apply the same
   schema, transition-boundary, metadata, and query-leakage validator used
   online.
6. Convert accepted request/response pairs into VST-SFT conversations. The user
   turn contains video plus the Analyzer prompt; the assistant turn contains
   only the exact reflection JSON.
7. Train assistant JSON tokens with ordinary causal-language-model SFT loss.
8. Use the resulting checkpoint as `actor_rollout_ref.model.path` for RL.

### Online Skill OPD training

1. The Actor produces `T-1` memory transitions and one final answer turn.
2. The environment scores final rows. Code converts each trajectory result to
   `is_correct` and does not expose the raw reward or label to the Analyzer.
3. Complete trajectories are assembled by unique `trajectory_uid`.
4. Before any optimizer step, the current Actor greedily generates reflection
   JSON with `do_sample=false`, `temperature=0`, `top_p=1`, and at most 256 new
   tokens.
5. Invalid or leaking reflections receive zero OPD masks while the original
   VST-RL path remains active.
6. For each accepted trajectory, attach `episode_skill` to all non-final memory
   rows. Attach `step_skill` only to selected key rows.
7. The frozen current Actor produces teacher-selected top-100 distributions on
   the skill-augmented rows.
8. The trainable Actor produces student logits on the original rows. The Actor
   update combines unchanged VST-RL loss with one global-episode OPD loss.

## Analyzer request contract

The serialized request contains exactly:

```json
{
  "trajectory_uid": "sample-id:rollout-0",
  "policy_version": 12,
  "observed_video": "provider-neutral video reference",
  "transitions": [
    {
      "transition_index": 0,
      "previous_memory_tokens": [],
      "current_chunk_boundary": {},
      "generated_y_t_tokens": [],
      "updated_memory_tokens": []
    }
  ],
  "query": "question and options",
  "prediction": "Actor prediction",
  "is_correct": false
}
```

No generic dictionary from the training dataset is forwarded. This allowlist
prevents fields such as `ground_truth`, `answer`, `label`, `solution`, or raw
reward from entering the request accidentally.

## Reflection target contract

Apply form:

```json
{
  "apply_opd": true,
  "episode_skill": "query-independent episode skill",
  "key_transitions": [
    {
      "transition_index": 1,
      "kind": "preserve",
      "memory_attribute": "temporal_order",
      "step_skill": "query-independent transition skill"
    }
  ]
}
```

Skip form:

```json
{
  "apply_opd": false,
  "episode_skill": null,
  "key_transitions": [],
  "skip_reason": "memory_cause_uncertain"
}
```

The existing strict schema remains authoritative. Free-form reasoning and
undeclared fields are rejected. The program adds schema version, trajectory ID,
and policy version outside the model payload.

## External-teacher boundary

Introduce a small provider-neutral client protocol accepting one serialized
Analyzer request and returning one raw text response. The data-building command
supports three explicit modes:

- `export`: write requests only and make no network call;
- `import`: join previously obtained responses and validate them;
- `call`: use a configured client implementation.

The optional OpenAI-compatible client reads endpoint, model, and API key only
from explicit command arguments/environment variables. It is never selected by
default. Tests inject a fake client, and the CPU smoke uses `import` with a
fixture response. Reports must record `teacher_source=fixture` or
`teacher_source=external`; a fixture cannot satisfy an external-call claim.

## Reflection SFT representation

Reuse the existing `VST-SFT` Hugging Face Trainer and multimodal conversation
dataset instead of introducing a second trainer. Add a focused converter that
writes the existing JSONL conversation shape and its seek index. The video path
remains an absolute read-only reference. Assistant-token masking already present
in `VST-SFT` ensures that only reflection JSON is supervised.

The SFT launch configuration takes an explicit input JSONL, base model path, and
output checkpoint path. It does not automatically start RL. The RL smoke/config
accepts an explicit SFT checkpoint path so the stage boundary is auditable.

## Global-episode and key-step OPD

For every accepted trajectory:

- non-key memory row teacher instruction: `Episode skill: ...`;
- key memory row teacher instruction: `Episode skill: ...` plus
  `Step skill: ...`;
- final row teacher instruction: none.

Add an `opd_episode_mask` covering response tokens on every non-final memory row
of an accepted, metadata-valid reflection. Keep `opd_key_mask` for metrics and
for deciding where to append `step_skill`, but do not require it for the loss.

The single effective mask is:

\[
m_{t,j}=m^{response}_{t,j}m^{memory}_t m^{episode}_t
        m^{reflection}_t m^{metadata}_t.
\]

Teacher and student are normalized on the same teacher-selected top-100 support:

\[
\mathcal L_{OPD}=
\frac{\sum_{t,j}m_{t,j}
D_{KL}(\tilde p_{t,j}\parallel\tilde q_{t,j})}
{\max(1,\sum_{t,j}m_{t,j})}.
\]

There is no double-counted second loss for key rows. Their target differs because
the teacher sees both episode and step skills. The joint loss stays:

\[
\mathcal L_{total}=\mathcal L_{VST-RL}
+\lambda_{OPD}\mathcal L_{OPD},\qquad \lambda_{OPD}=0.01
\]

for smoke. With `skill_opd.enable=false`, the original loss object and update
path remain unchanged.

## Failure handling

- Missing or non-boolean correctness: reject the SFT/Analyzer request.
- Missing reward before boolean conversion: never default to incorrect.
- External client absent in `export`/`import` mode: no error and no network call.
- External client absent in `call` mode: fail before processing samples.
- Invalid JSON, schema, IDs, transition indices, or leakage: reject the label or
  zero the online trajectory's OPD masks.
- Empty accepted SFT set: fail instead of launching training.
- Empty online OPD token set: return differentiable zero OPD and retain VST-RL.
- Context above the shared 32K budget: reject before generation/teacher forward.

## Metrics

Retain existing metrics and add:

- SFT request, accepted, rejected, and fixture/external counts;
- SFT target token count and finite loss;
- episode-supervised row/token counts;
- key-step row/token counts;
- non-key episode-only token count;
- final-answer OPD token count, required to be zero;
- reflection acceptance and skip-reason counts;
- teacher retained mass, cache bytes, dtype, context length;
- original VST-RL, OPD, weighted OPD, and total losses.

## Test strategy

Tests are written before implementation and cover:

1. External/SFT request allowlist excludes every answer/label/raw-reward alias.
2. External and online Analyzer prompts share the same semantic fields.
3. Export/import modes perform no external call; call mode uses the injected
   client and validates responses.
4. SFT conversion produces video-plus-prompt input and exact JSON target, with
   only assistant tokens labelled.
5. Reflection schema and leakage rejection remain unchanged.
6. Accepted episode skill marks all memory rows, including non-key rows.
7. Only key rows receive step skill; final rows receive neither skill nor OPD.
8. Teacher cache is produced before optimizer update and remains detached.
9. The global-episode top-100 KL has gradients only through the student.
10. Disabled mode returns the original VST-RL loss path.

## CPU smoke acceptance

One CPU smoke runs the following sequence without an external network call:

1. Construct a two-memory-turn plus one-final-turn trajectory.
2. Export an answer-free external-teacher request.
3. Import a strict fixture reflection and record `teacher_source=fixture`.
4. Build one Reflection SFT conversation and perform a finite toy SFT backward
   and parameter update over reflection target tokens.
5. Assemble online Analyzer input with only `is_correct`.
6. Apply episode skill to both memory turns and step skill to exactly one key
   turn.
7. Build detached top-100 teacher cache and calculate finite global-episode OPD.
8. Verify positive OPD token counts on both key and non-key memory rows, zero on
   the final row, and complete one combined optimizer update.
9. Verify disabled-path equivalence.

The smoke report states explicitly that it validates CPU code flow with fixture
external output. It does not claim an external API call, real Reflection SFT
checkpoint, real Actor-generated JSON, or GPU execution.

## Expected implementation surface

- `VST-RL/recurrent/reflection.py`: correctness-only Analyzer contract.
- `VST-RL/recurrent/reflection_sft.py`: request/export/import/client boundary and
  SFT conversion helpers.
- `VST-RL/recurrent/skill_opd.py`: episode-wide annotations and skill attachment.
- `VST-RL/verl/trainer/ppo/ray_trainer.py`: boolean Analyzer input and metrics.
- `VST-RL/verl/workers/actor/dp_actor.py`: consume episode-wide OPD mask.
- `VST-RL/scripts/build_reflection_sft_data.py`: offline data command.
- `VST-SFT`: minimal launch/config support for generated reflection JSONL.
- Focused unit tests and an updated CPU smoke runner/auditor/report.
