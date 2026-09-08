# Reflection SFT and Global-Episode Skill OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a correctness-only offline reflection-SFT data path and change Skill OPD so a frozen current Actor supervises every accepted memory turn with an episode skill, while key turns additionally receive a step skill and final-answer tokens receive no OPD.

**Architecture:** Keep one strict reflection schema and one allowlisted Analyzer payload shared by the offline external-teacher path and online frozen-Actor path. Reuse the existing VST-SFT causal-LM trainer for assistant-only reflection JSON supervision, then reuse the current pre-update Actor snapshot for both online analysis and top-100 teacher logits. Preserve one OPD loss: non-key rows use the episode skill, key rows use episode plus step skill, and the effective mask covers all accepted non-final memory rows.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers/Trainer, OpenAI-compatible asynchronous client already used by VST-RL, pytest, Bash, Git.

## Global Constraints

- Neither offline teacher nor online Analyzer receives a ground-truth answer, correct option text/letter, label, solution, or raw reward.
- Both Analyzer paths receive exactly one outcome signal: `is_correct: bool`.
- Convert a finite final environment reward to correctness at the RL boundary with `reward > 0`; reject a missing or non-finite reward instead of defaulting to incorrect.
- External API code must exist, but no test or CPU smoke may select `call` mode or make a network request.
- Online Analyzer and OPD teacher must use the same current Actor snapshot captured before the rollout batch optimizer step.
- Every accepted non-final memory row receives `episode_skill`; only selected key rows additionally receive `step_skill`; final rows receive neither.
- `opd_key_mask` remains available for skill attachment and metrics, but must not gate OPD loss.
- Preserve the existing VST-RL policy-gradient, entropy, reference-KL, reward, advantage, append-only memory, and final-turn behavior.
- Do not copy models, videos, datasets, or checkpoints into the Git worktree; use explicit absolute read-only asset paths.
- Keep `skill_opd.enable: false` as the default and preserve the disabled update path exactly.
- Completion requires CPU unit tests and a real toy backward/optimizer smoke; full Qwen SFT and GPU RL runs are explicitly outside scope.

## File map

- `VST-RL/recurrent/skill_opd.py`: trajectory outcome contract, trajectory assembly, row-aligned episode/key masks, and teacher prompt augmentation.
- `VST-RL/recurrent/reflection.py`: the single Analyzer prompt and strict reflection parser/validator used offline and online.
- `VST-RL/recurrent/reflection_sft.py`: provider-neutral teacher request/client boundary and validated VST-SFT record conversion.
- `VST-RL/scripts/build_reflection_sft_data.py`: explicit `export`, `import`, and `call` CLI modes.
- `VST-SFT/run_reflection_sft.sh`: narrow launcher that takes explicit JSONL, base-model, and checkpoint paths.
- `VST-RL/verl/trainer/ppo/skill_opd_loss.py`: episode-wide top-100 KL and unchanged joint-loss composition.
- `VST-RL/verl/trainer/ppo/ray_trainer.py`: reward-to-correctness boundary, Analyzer invocation, episode/key skill construction, cache generation, and metrics.
- `VST-RL/verl/workers/actor/dp_actor.py`: consumes `opd_episode_mask` and combines one OPD term with the existing VST-RL loss.
- `VST-RL/recurrent/test_skill_opd_trajectory.py`, `VST-RL/recurrent/test_reflection.py`, `VST-RL/tests/test_reflection_sft.py`, `VST-RL/tests/test_skill_opd_loss.py`, `VST-RL/recurrent/test_skill_opd_actor.py`: focused unit and source-integration tests.
- `VST-RL/scripts/run_reflection_sft_opd_code_smoke.py`, `VST-RL/scripts/audit_reflection_sft_opd_smoke.py`, `VST-RL/tests/test_reflection_sft_opd_smoke.py`: fixture-only CPU end-to-end smoke and truthful audit.
- `docs/smoke/reflection-sft-global-episode-opd-cpu-smoke.md`: reproducible evidence, limitations, hashes, and commands.

---

### Task 1: Replace Analyzer raw reward with boolean correctness

**Files:**
- Modify: `VST-RL/recurrent/skill_opd.py`
- Modify: `VST-RL/recurrent/reflection.py`
- Modify: `VST-RL/recurrent/test_skill_opd_trajectory.py`
- Modify: `VST-RL/recurrent/test_reflection.py`

**Interfaces:**
- Produces: `reward_to_is_correct(reward: float) -> bool`.
- Produces: `ReflectionTrajectory.is_correct: bool` and `assemble_reflection_trajectories(*, output: DataProto, final_mask: torch.Tensor, sample_index: torch.Tensor, correctness_by_trajectory: Mapping[str, bool], query_tokens_by_sample: Mapping[int, list[int]], query_text_by_sample: Mapping[int, str], prediction_text_by_final_row: Mapping[int, str], observed_video_by_sample: Mapping[int, Any]) -> list[ReflectionTrajectory]`.
- Produces: `ReflectionTrajectory.to_analyzer_input() -> dict[str, Any]` containing only the documented allowlist.
- Consumed by: Tasks 2 and 5.

- [ ] **Step 1: Write failing outcome-boundary and allowlist tests**

```python
def test_reward_to_is_correct_is_explicit_and_finite():
    assert reward_to_is_correct(1.0) is True
    assert reward_to_is_correct(0.0) is False
    assert reward_to_is_correct(-1.0) is False
    with pytest.raises(ValueError, match="finite"):
        reward_to_is_correct(float("nan"))


def test_analyzer_payload_contains_boolean_correctness_without_label_aliases():
    trajectory = _trajectory(is_correct=False)
    payload = trajectory.to_analyzer_input()
    assert payload["is_correct"] is False
    assert set(payload) == {
        "trajectory_uid", "policy_version", "observed_video", "transitions",
        "query", "prediction", "is_correct",
    }
    serialized = json.dumps(payload).lower()
    for forbidden in ("ground_truth", "correct_answer", "label", "solution", '"reward"'):
        assert forbidden not in serialized


def test_assembly_rejects_non_boolean_correctness():
    with pytest.raises(TypeError, match="boolean correctness"):
        _assemble(correctness_by_trajectory={"traj-0": 1})
```

- [ ] **Step 2: Run the focused tests and confirm the old contract fails**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q recurrent/test_skill_opd_trajectory.py recurrent/test_reflection.py`

Expected: FAIL because `reward_to_is_correct` and `is_correct` do not exist and the prompt still contains `Reward`.

- [ ] **Step 3: Implement the minimal correctness-only contract**

```python
def reward_to_is_correct(reward: float) -> bool:
    value = float(reward)
    if not math.isfinite(value):
        raise ValueError("final reward must be finite before correctness conversion")
    return value > 0.0


@dataclass(frozen=True)
class ReflectionTrajectory:
    group_uid: str
    trajectory_uid: str
    policy_version: int
    sample_index: int
    transitions: tuple[MemoryTransition, ...]
    transition_rows: tuple[int, ...]
    final_row: int
    query_tokens: tuple[int, ...]
    query_text: str
    prediction_tokens: tuple[int, ...]
    prediction_text: str
    is_correct: bool
    observed_video: Any

    def to_analyzer_input(self) -> dict[str, Any]:
        if type(self.is_correct) is not bool:
            raise TypeError("Analyzer requires boolean correctness")
        return {
            "trajectory_uid": self.trajectory_uid,
            "policy_version": self.policy_version,
            "observed_video": self.observed_video,
            "transitions": [
                {
                    "transition_index": transition.transition_index,
                    "previous_memory_tokens": list(transition.previous_memory_tokens),
                    "current_chunk_boundary": transition.current_chunk_boundary,
                    "generated_y_t_tokens": list(transition.generated_y_t_tokens),
                    "updated_memory_tokens": list(transition.updated_memory_tokens),
                }
                for transition in self.transitions
            ],
            "query": self.query_text,
            "prediction": self.prediction_text,
            "is_correct": self.is_correct,
        }
```

Change the reflection prompt line from `Reward: ...` to `Prediction correct: true|false`; do not add an answer field.

- [ ] **Step 4: Run focused tests**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q recurrent/test_skill_opd_trajectory.py recurrent/test_reflection.py`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the correctness boundary**

```bash
git add VST-RL/recurrent/skill_opd.py VST-RL/recurrent/reflection.py VST-RL/recurrent/test_skill_opd_trajectory.py VST-RL/recurrent/test_reflection.py
git diff --cached --check
git commit -m "feat: hide raw rewards from reflection analyzers"
```

---

### Task 2: Add the provider-neutral offline reflection-SFT boundary

**Files:**
- Create: `VST-RL/recurrent/reflection_sft.py`
- Create: `VST-RL/tests/test_reflection_sft.py`

**Interfaces:**
- Consumes: `ReflectionTrajectory.to_analyzer_input()` and the existing `build_reflection_prompt(...)`, `parse_and_validate_reflection(text, trajectory, policy_version, max_key_transitions=3)` validation path.
- Produces: `TeacherRequest`, `ReflectionTeacherClient.generate(request: TeacherRequest) -> str`, `OpenAICompatibleReflectionTeacher`, `offline_record_to_trajectory`, `build_teacher_request`, `validate_teacher_response`, `to_vst_sft_record`, and `write_jsonl_with_seeks`.
- Consumed by: Tasks 3 and 6.

- [ ] **Step 1: Write failing request, fake-client, validation, and SFT-record tests**

```python
class FailIfCalledClient:
    async def generate(self, request):
        raise AssertionError("external client must not be called")


def test_teacher_request_is_the_online_analyzer_allowlist():
    request = build_teacher_request(_trajectory(is_correct=False))
    assert request.payload == _trajectory(is_correct=False).to_analyzer_input()
    assert type(request.payload["is_correct"]) is bool


def test_offline_record_converts_reward_before_building_request():
    trajectory = offline_record_to_trajectory({
        "trajectory_uid": "sample-0:rollout-0",
        "policy_version": 12,
        "observed_video": "/readonly/video.mp4",
        "transitions": [_transition_payload(0), _transition_payload(1)],
        "query": "What changed?\nA. left\nB. right",
        "prediction": "B",
        "final_reward": 0.0,
    })
    assert trajectory.is_correct is False
    assert "final_reward" not in build_teacher_request(trajectory).payload


def test_fixture_response_becomes_exact_assistant_json():
    raw = json.dumps(_valid_apply_payload(), separators=(",", ":"))
    accepted = validate_teacher_response(_trajectory(False), raw, source="fixture")
    record = to_vst_sft_record(_trajectory(False), accepted)
    assert record[0]["role"] == "user"
    assert record[0]["content"][0]["type"] == "video"
    assert record[1] == {
        "role": "assistant", "content": [{"type": "text", "text": raw}],
    }


def test_response_with_leaked_query_is_rejected():
    raw = json.dumps(_valid_apply_payload(step_skill="choose option A"))
    with pytest.raises(ValueError, match="leak"):
        validate_teacher_response(_trajectory(False), raw, source="fixture")
```

- [ ] **Step 2: Run the new test module and confirm import failure**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q tests/test_reflection_sft.py`

Expected: FAIL with `ModuleNotFoundError: recurrent.reflection_sft`.

- [ ] **Step 3: Implement focused dataclasses and protocol**

```python
@dataclass(frozen=True)
class TeacherRequest:
    payload: dict[str, Any]
    analyzer_prompt: str


class ReflectionTeacherClient(Protocol):
    async def generate(self, request: TeacherRequest) -> str:
        raise NotImplementedError


class OpenAICompatibleReflectionTeacher:
    def __init__(self, *, model: str, api_key: str, base_url: str | None = None):
        if not model or not api_key:
            raise ValueError("call mode requires model and API key")
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, request: TeacherRequest) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {
                        "url": str(request.payload["observed_video"]),
                    }},
                    {"type": "text", "text": request.analyzer_prompt},
                ],
            }],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("external teacher returned an empty response")
        return content


def build_teacher_request(trajectory: ReflectionTrajectory) -> TeacherRequest:
    payload = trajectory.to_analyzer_input()
    return TeacherRequest(payload=payload, analyzer_prompt=build_reflection_prompt(trajectory))


def offline_record_to_trajectory(record: Mapping[str, Any]) -> ReflectionTrajectory:
    required = {
        "trajectory_uid", "policy_version", "observed_video", "transitions",
        "query", "prediction", "final_reward",
    }
    if set(record) != required:
        raise ValueError("offline trajectory fields must exactly match the input schema")
    transitions = tuple(
        MemoryTransition(
            row_index=index,
            transition_index=item["transition_index"],
            previous_memory_tokens=tuple(item["previous_memory_tokens"]),
            current_chunk_boundary=item["current_chunk_boundary"],
            generated_y_t_tokens=tuple(item["generated_y_t_tokens"]),
            updated_memory_tokens=tuple(item["updated_memory_tokens"]),
        )
        for index, item in enumerate(record["transitions"])
    )
    return ReflectionTrajectory(
        group_uid=record["trajectory_uid"],
        trajectory_uid=record["trajectory_uid"],
        policy_version=int(record["policy_version"]),
        sample_index=0,
        transitions=transitions,
        transition_rows=tuple(range(len(transitions))),
        final_row=len(transitions),
        query_tokens=(), query_text=str(record["query"]),
        prediction_tokens=(), prediction_text=str(record["prediction"]),
        is_correct=reward_to_is_correct(record["final_reward"]),
        observed_video=record["observed_video"],
    )


@dataclass(frozen=True)
class AcceptedTeacherResponse:
    raw_response: str
    parsed: ReflectionEnvelope
    source: str


def validate_teacher_response(trajectory, raw_response: str, *, source: str):
    if source not in {"fixture", "external"}:
        raise ValueError("teacher source must be fixture or external")
    parsed = parse_and_validate_reflection(
        raw_response, trajectory, trajectory.policy_version,
    )
    if not parsed.reflection_valid:
        raise ValueError(parsed.rejection_reason or "invalid teacher reflection")
    return AcceptedTeacherResponse(raw_response=raw_response, parsed=parsed, source=source)
```

The optional client constructs `AsyncOpenAI` only inside its constructor and sends the shared Analyzer prompt plus the explicit `observed_video` URL. `call` mode therefore rejects local filesystem video paths and requires an externally accessible URL; `export`/`import` remain the supported local-asset workflow. The client never reads or forwards dataset dictionaries.

- [ ] **Step 4: Implement VST-SFT record and byte-offset writer**

```python
def to_vst_sft_record(trajectory, accepted):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": os.fspath(trajectory.observed_video)},
                {"type": "text", "text": build_reflection_prompt(trajectory)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": accepted.raw_response}],
        },
    ]


def write_jsonl_with_seeks(records, output_path: Path) -> Path:
    seeks = []
    with output_path.open("wb") as stream:
        for record in records:
            seeks.append(stream.tell())
            stream.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
    seeks_path = output_path.with_name(output_path.stem + "_seeks.jsonl")
    seeks_path.write_text(json.dumps(seeks) + "\n", encoding="utf-8")
    return seeks_path
```

- [ ] **Step 5: Run tests and commit**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q tests/test_reflection_sft.py recurrent/test_reflection.py`

Expected: PASS, including the fake client never being called.

```bash
git add VST-RL/recurrent/reflection_sft.py VST-RL/tests/test_reflection_sft.py
git diff --cached --check
git commit -m "feat: add reflection SFT teacher boundary"
```

---

### Task 3: Add explicit export/import/call data CLI and narrow SFT launcher

**Files:**
- Create: `VST-RL/scripts/build_reflection_sft_data.py`
- Modify: `VST-RL/tests/test_reflection_sft.py`
- Create: `VST-SFT/run_reflection_sft.sh`

**Interfaces:**
- Consumes: Task 2 request, client, validation, conversion, and writer functions.
- Produces: CLI subcommands `export`, `import`, and `call`; `call` is the only mode allowed to instantiate a client.
- Produces: SFT launcher arguments `TRAIN_JSONL`, `BASE_MODEL`, and `OUTPUT_DIR` with absolute-path checks.

- [ ] **Step 1: Write failing CLI isolation tests**

```python
def test_export_and_import_never_construct_external_client(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection_sft, "OpenAICompatibleReflectionTeacher", FailOnInit)
    assert main(["export", "--trajectory-jsonl", str(_rollouts(tmp_path)),
                 "--requests-jsonl", str(tmp_path / "requests.jsonl")]) == 0
    assert main(["import", "--trajectory-jsonl", str(_rollouts(tmp_path)),
                 "--responses-jsonl", str(_responses(tmp_path)),
                 "--output-jsonl", str(tmp_path / "train.jsonl")]) == 0


def test_call_requires_explicit_model_and_api_key(tmp_path):
    with pytest.raises(SystemExit):
        main(["call", "--trajectory-jsonl", str(_rollouts(tmp_path)),
              "--output-jsonl", str(tmp_path / "train.jsonl")])


def test_call_uses_injected_client_and_validates_response(tmp_path):
    client = RecordingClient(json.dumps(_valid_apply_payload()))
    summary = asyncio.run(call_teacher(_call_args(tmp_path), client=client))
    assert len(client.requests) == 1
    assert summary == {
        "request_count": 1, "accepted_count": 1, "rejected_count": 0,
        "fixture_count": 0, "external_count": 1,
    }
```

- [ ] **Step 2: Run tests and confirm the script import fails**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q tests/test_reflection_sft.py`

Expected: FAIL because `scripts.build_reflection_sft_data` does not exist.

- [ ] **Step 3: Implement the three-mode CLI without a default network path**

```python
def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--trajectory-jsonl", required=True)
    export.add_argument("--requests-jsonl", required=True)
    import_ = subparsers.add_parser("import")
    import_.add_argument("--trajectory-jsonl", required=True)
    import_.add_argument("--responses-jsonl", required=True)
    import_.add_argument("--output-jsonl", required=True)
    call = subparsers.add_parser("call")
    call.add_argument("--trajectory-jsonl", required=True)
    call.add_argument("--output-jsonl", required=True)
    call.add_argument("--model", required=True)
    call.add_argument("--api-key", required=True)
    call.add_argument("--base-url")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "export":
        return export_requests(args)
    if args.mode == "import":
        return import_responses(args)
    client = OpenAICompatibleReflectionTeacher(
        model=args.model, api_key=args.api_key, base_url=args.base_url,
    )
    return asyncio.run(call_teacher(args, client=client))
```

Each output summary must include integer `request_count`, `accepted_count`, `rejected_count`, `fixture_count`, and `external_count`. Fail if `accepted_count == 0`.

- [ ] **Step 4: Add the minimal SFT launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${TRAIN_JSONL:?set an absolute reflection SFT JSONL path}"
: "${BASE_MODEL:?set an absolute base model path}"
: "${OUTPUT_DIR:?set an absolute checkpoint output path}"
for path in "$TRAIN_JSONL" "$BASE_MODEL" "$OUTPUT_DIR"; do
  case "$path" in /*) ;; *) echo "all paths must be absolute" >&2; exit 2 ;; esac
done
torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" train.py \
  --deepspeed ./scripts/zero3.json \
  --do_train true \
  --overwrite_output_dir true \
  --output_dir "$OUTPUT_DIR" \
  --run_name reflection-sft \
  --pretrained_model_name_or_path "$BASE_MODEL" \
  --train_annotation_paths "$TRAIN_JSONL" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 25 \
  --logging_steps 1 \
  --report_to tensorboard \
  --text_sink 512 \
  --text_sliding_window 32768
```

Do not refactor `VST-SFT/run.sh`; the focused launcher above is the only launcher change.

The launcher completion message must print the checkpoint directory and the exact RL handoff form `actor_rollout_ref.model.path=<absolute OUTPUT_DIR>`. It must not start RL automatically.

- [ ] **Step 5: Verify import mode and shell syntax, then commit**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q VST-RL/tests/test_reflection_sft.py && bash -n VST-SFT/run_reflection_sft.sh`

Expected: tests PASS and `bash -n` exits 0; no network call occurs.

```bash
git add VST-RL/scripts/build_reflection_sft_data.py VST-RL/tests/test_reflection_sft.py VST-SFT/run_reflection_sft.sh
git diff --cached --check
git commit -m "feat: add offline reflection SFT data workflow"
```

---

### Task 4: Expand annotations from key-only to every accepted memory row

**Files:**
- Modify: `VST-RL/recurrent/skill_opd.py`
- Modify: `VST-RL/recurrent/test_skill_opd_trajectory.py`

**Interfaces:**
- Produces: row-aligned `opd_episode_mask`, `opd_key_mask`, `opd_episode_skill`, `opd_step_skill`, and `opd_valid_token_mask`.
- Consumed by: Task 5 trainer and Actor changes.

- [ ] **Step 1: Write the failing episode/key/final mask test**

```python
def test_episode_skill_supervises_non_key_memory_rows_but_never_final_row():
    annotations = build_opd_annotations(output, [_trajectory(False)], [_valid_reflection(key=1)])
    assert annotations["opd_episode_mask"].any(dim=-1).tolist() == [True, True, False]
    assert annotations["opd_key_mask"].any(dim=-1).tolist() == [False, True, False]
    assert annotations["opd_valid_token_mask"].any(dim=-1).tolist() == [True, True, False]
    assert annotations["opd_episode_skill"].tolist() == ["track temporal state", "track temporal state", None]
    assert annotations["opd_step_skill"].tolist() == [None, "preserve the transition", None]
```

Also retain a rejected-reflection test asserting all five OPD masks are false and all skill strings are `None`.

- [ ] **Step 2: Run the focused test and confirm the non-key row fails**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q recurrent/test_skill_opd_trajectory.py`

Expected: FAIL because only the key row currently receives reflection/metadata/skill masks.

- [ ] **Step 3: Implement episode-wide annotation once per accepted trajectory**

```python
episode_mask = torch.zeros(response_shape, dtype=torch.bool, device=device)
for trajectory, reflection in zip(trajectories, reflections):
    metadata_valid = (
        reflection.trajectory_uid == trajectory.trajectory_uid
        and reflection.policy_version == trajectory.policy_version
    )
    if not (reflection.reflection_valid and reflection.apply_opd and metadata_valid):
        continue
    for row in trajectory.transition_rows:
        episode_mask[row] = output.batch["response_mask"][row].bool()
        reflection_mask[row] = True
        metadata_mask[row] = True
        episode_skills[row] = reflection.episode_skill
    for key_transition in reflection.key_transitions:
        row = rows_by_transition[key_transition.transition_index]
        key_mask[row] = output.batch["response_mask"][row].bool()
        step_skills[row] = key_transition.step_skill

valid_token_mask = (
    output.batch["response_mask"].bool()
    & memory_mask & episode_mask & reflection_mask & metadata_mask
)
```

- [ ] **Step 4: Run trajectory tests and commit**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q recurrent/test_skill_opd_trajectory.py`

Expected: PASS; the non-key memory row is OPD-valid and the final row remains zero.

```bash
git add VST-RL/recurrent/skill_opd.py VST-RL/recurrent/test_skill_opd_trajectory.py
git diff --cached --check
git commit -m "feat: supervise all memory turns with episode skills"
```

---

### Task 5: Integrate episode-wide top-100 OPD into the existing Actor update

**Files:**
- Modify: `VST-RL/verl/trainer/ppo/skill_opd_loss.py`
- Modify: `VST-RL/tests/test_skill_opd_loss.py`
- Modify: `VST-RL/verl/trainer/ppo/ray_trainer.py`
- Modify: `VST-RL/verl/workers/actor/dp_actor.py`
- Modify: `VST-RL/recurrent/test_skill_opd_actor.py`

**Interfaces:**
- Consumes: `reward_to_is_correct`, `correctness_by_trajectory`, and Task 4 annotations.
- Produces: `skill_conditioned_topk_opd_loss(..., episode_mask, reflection_mask, metadata_mask)`.
- Preserves: `combine_vst_rl_and_opd_loss(vst_rl_loss, opd_loss, enabled, lambda_opd)`.

- [ ] **Step 1: Replace the key-gated loss test with episode-wide assertions**

```python
def test_opd_loss_includes_non_key_episode_tokens_and_excludes_final_tokens():
    episode_mask = torch.tensor([[1, 1], [1, 1], [0, 0]], dtype=torch.bool)
    key_mask = torch.tensor([[0, 0], [1, 1], [0, 0]], dtype=torch.bool)
    loss, metrics = skill_conditioned_topk_opd_loss(
        student_logits, indices, teacher_log_probs,
        response_mask=torch.ones(3, 2, dtype=torch.bool),
        memory_mask=torch.tensor([[1, 1], [1, 1], [0, 0]], dtype=torch.bool),
        episode_mask=episode_mask,
        reflection_mask=episode_mask,
        metadata_mask=episode_mask,
    )
    assert torch.isfinite(loss)
    assert metrics["opd/valid_token_count"] == 4
    assert int((episode_mask & ~key_mask).sum()) == 2


def test_disabled_joint_loss_is_the_same_tensor_object():
    original = torch.tensor(3.0, requires_grad=True)
    combined = combine_vst_rl_and_opd_loss(
        original, None, enabled=False, lambda_opd=0.01,
    )
    assert combined is original
```

Retain tests for detached teacher cache, top-100 normalization, BF16 student logits, empty-mask differentiable zero, and disabled joint-loss identity.

- [ ] **Step 2: Run loss and source-integration tests and confirm old key gate fails**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q tests/test_skill_opd_loss.py recurrent/test_skill_opd_actor.py`

Expected: FAIL because `skill_conditioned_topk_opd_loss` and `opd_episode_mask` are absent.

- [ ] **Step 3: Implement the single episode-gated KL**

```python
def skill_conditioned_topk_opd_loss(
    student_logits, teacher_topk_indices, teacher_topk_log_probs,
    response_mask, memory_mask, episode_mask, reflection_mask, metadata_mask,
):
    masks = (response_mask, memory_mask, episode_mask, reflection_mask, metadata_mask)
    if teacher_topk_indices.shape != teacher_topk_log_probs.shape:
        raise ValueError("teacher top-k cache shapes must match")
    if student_logits.shape[:-1] != teacher_topk_indices.shape[:-1]:
        raise ValueError("student and teacher leading dimensions must match")
    if any(mask.shape != student_logits.shape[:-1] for mask in masks):
        raise ValueError("all OPD masks must match response positions")
    indices = teacher_topk_indices.detach().to(student_logits.device)
    teacher_log_probs = teacher_topk_log_probs.detach().to(
        device=student_logits.device, dtype=torch.float32,
    )
    student_log_probs = torch.log_softmax(
        torch.gather(student_logits.float(), -1, indices), dim=-1,
    )
    token_kl = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    valid_mask = torch.ones_like(response_mask, dtype=torch.bool)
    for mask in masks:
        valid_mask &= mask.to(student_logits.device).bool()
    valid_count = int(valid_mask.sum().item())
    loss = (token_kl * valid_mask).sum() / max(1, valid_count)
    return loss, {
        "opd/valid_token_count": valid_count,
        "opd/response_token_count": int(response_mask.bool().sum().item()),
        "opd/token_kl_mean": float(loss.detach().item()),
    }
```

Remove the old `localized_topk_opd_loss` import and call rather than keeping a compatibility wrapper that could accidentally reintroduce key-only semantics.

- [ ] **Step 4: Build one skill instruction per row in the trainer**

```python
teacher_skills = np.empty(len(batch), dtype=object)
teacher_skills[:] = None
for row, episode_skill in enumerate(annotations["opd_episode_skill"]):
    if episode_skill is None:
        continue
    text = f"Episode skill: {episode_skill}"
    step_skill = annotations["opd_step_skill"][row]
    if step_skill is not None:
        text += f"\nStep skill: {step_skill}"
    teacher_skills[row] = text
```

Pass `teacher_skills` to the existing prompt augmenter. In `ray_trainer.py`, convert final rewards to a `dict[str, bool]` before trajectory assembly, set `correctness_mapped=True`, and ensure no raw reward argument reaches the Analyzer manager.

- [ ] **Step 5: Change Actor mask selection and metrics**

```python
required_opd_keys = (
    "opd_topk_indices", "opd_teacher_topk_log_probs",
    "opd_response_mask", "opd_memory_mask", "opd_episode_mask",
    "opd_reflection_mask", "opd_metadata_mask",
)
opd_loss, opd_metrics = skill_conditioned_topk_opd_loss(
    student_logits, data["opd_topk_indices"], data["opd_teacher_topk_log_probs"],
    data["opd_response_mask"], data["opd_memory_mask"],
    data["opd_episode_mask"], data["opd_reflection_mask"],
    data["opd_metadata_mask"],
)
loss = combine_vst_rl_and_opd_loss(
    vst_rl_loss, opd_loss, enabled=True,
    lambda_opd=skill_opd_config.get("lambda_opd", 0.01),
)
```

Add metrics for episode rows/tokens, key rows/tokens, non-key episode-only tokens, and final OPD tokens. Assert/report final OPD tokens as zero; do not add a second key loss.

Extend `recurrent/test_skill_opd_actor.py` with source-order assertions that `generate_reflections` and `generate_opd_teacher_topk` occur before `update_actor`, and that the same batch-level `policy_version` is passed through trajectory assembly and reflection validation.

- [ ] **Step 6: Run focused tests and commit**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q tests/test_skill_opd_loss.py tests/test_skill_opd_cache.py recurrent/test_skill_opd_actor.py recurrent/test_skill_opd_trajectory.py`

Expected: PASS, with non-key memory tokens included in the valid count and final tokens excluded.

```bash
git add VST-RL/verl/trainer/ppo/skill_opd_loss.py VST-RL/tests/test_skill_opd_loss.py VST-RL/verl/trainer/ppo/ray_trainer.py VST-RL/verl/workers/actor/dp_actor.py VST-RL/recurrent/test_skill_opd_actor.py
git diff --cached --check
git commit -m "feat: train episode-wide skill OPD"
```

---

### Task 6: Add fixture-only CPU SFT-to-OPD smoke and audit

**Files:**
- Create: `VST-RL/scripts/run_reflection_sft_opd_code_smoke.py`
- Create: `VST-RL/scripts/audit_reflection_sft_opd_smoke.py`
- Create: `VST-RL/tests/test_reflection_sft_opd_smoke.py`
- Create: `docs/smoke/reflection-sft-global-episode-opd-cpu-smoke.md`

**Interfaces:**
- Consumes: Tasks 1-5 public functions only.
- Produces: a JSON metrics artifact proving fixture source, assistant-only SFT update, episode/non-key/key/final mask counts, top-100 cache, joint Actor update, and no GPU/API use.

- [ ] **Step 1: Write the failing audit contract**

```python
def test_cpu_smoke_report_is_truthful(tmp_path):
    metrics = run_smoke(tmp_path)
    assert metrics["device"] == "cpu"
    assert metrics["external_api_called"] is False
    assert metrics["teacher_source"] == "fixture"
    assert metrics["sft_target_token_count"] > 0
    assert math.isfinite(metrics["sft_loss"])
    assert metrics["sft_trainable_param_max_change"] > 0
    assert metrics["episode_memory_row_count"] == 2
    assert metrics["key_memory_row_count"] == 1
    assert metrics["non_key_episode_row_count"] == 1
    assert metrics["final_opd_token_count"] == 0
    assert metrics["top_k"] == 100
    assert metrics["actor_trainable_param_max_change"] > 0
    assert metrics["disabled_path_equal"] is True
```

- [ ] **Step 2: Run the new smoke test and confirm missing-module failure**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q tests/test_reflection_sft_opd_smoke.py`

Expected: FAIL because the new smoke runner does not exist.

- [ ] **Step 3: Implement a real toy SFT optimizer step**

```python
torch.manual_seed(0)
sft_model = torch.nn.Embedding(128, 16)
optimizer = torch.optim.SGD(sft_model.parameters(), lr=0.1)
labels = assistant_only_labels(input_ids, assistant_start)
logits = sft_model(input_ids)
sft_loss = torch.nn.functional.cross_entropy(
    logits[:-1], labels[1:], ignore_index=-100,
)
before = [parameter.detach().clone() for parameter in sft_model.parameters()]
sft_loss.backward()
optimizer.step()
```

Use a valid fixed reflection JSON and `source="fixture"`; do not instantiate `OpenAICompatibleReflectionTeacher`. The toy vocabulary may be smaller than production, but the mask must supervise assistant reflection tokens only.

- [ ] **Step 4: Continue the same smoke through episode-wide OPD**

```python
annotations = build_opd_annotations(output, trajectories, reflections)
teacher_logits = frozen_actor(skill_augmented_inputs).detach()
indices, teacher_log_probs, retained_mass = build_teacher_topk(
    teacher_logits, top_k=100, temperature=1.0,
)
opd_loss, opd_metrics = skill_conditioned_topk_opd_loss(
    trainable_actor(original_inputs), indices, teacher_log_probs,
    response_mask, annotations["opd_memory_mask"],
    annotations["opd_episode_mask"], annotations["opd_reflection_mask"],
    annotations["opd_metadata_mask"],
)
total_loss = combine_vst_rl_and_opd_loss(
    vst_rl_loss, opd_loss, enabled=True, lambda_opd=0.01,
)
total_loss.backward()
actor_optimizer.step()
```

The frozen snapshot must have `requires_grad=False`, teacher cache tensors must be detached, and the metrics file must report separate `vst_rl_loss`, `opd_loss`, `weighted_opd_loss`, and `total_loss`.

- [ ] **Step 5: Implement strict audit and execute smoke**

The audit exits non-zero unless all Step 1 assertions hold, the context is within 32768 tokens, all losses/cache values are finite, the teacher retained mass is within `(0, 1]`, and the report records `gpu_used=false`.

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python scripts/run_reflection_sft_opd_code_smoke.py --output /tmp/reflection-sft-opd-smoke.json && /home/bujunru/.conda/envs/vision-se/bin/python scripts/audit_reflection_sft_opd_smoke.py /tmp/reflection-sft-opd-smoke.json`

Expected: both commands exit 0 and audit prints `PASS`; report explicitly says this is fixture-only CPU code smoke, not external-teacher, Qwen-SFT, or GPU-RL evidence.

- [ ] **Step 6: Run the smoke test and commit**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q tests/test_reflection_sft_opd_smoke.py`

Expected: PASS.

```bash
git add VST-RL/scripts/run_reflection_sft_opd_code_smoke.py VST-RL/scripts/audit_reflection_sft_opd_smoke.py VST-RL/tests/test_reflection_sft_opd_smoke.py docs/smoke/reflection-sft-global-episode-opd-cpu-smoke.md
git diff --cached --check
git commit -m "test: add reflection SFT and episode OPD CPU smoke"
```

---

### Task 7: Run full regression, provenance checks, and push the isolated branch

**Files:**
- Modify only if evidence changed: `docs/smoke/reflection-sft-global-episode-opd-cpu-smoke.md`
- Verify: `manifests/`, Git state, remote branch.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: clean, pushed `codex/vst-skill-opd` branch with reproducible CPU evidence.

- [ ] **Step 1: Run the complete CPU regression suite**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd/VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python -m pytest -q`

Expected: all CPU tests PASS; CUDA-only tests may remain skipped with explicit skip reasons.

- [ ] **Step 2: Run syntax and smoke validation**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd && bash -n VST-SFT/run_reflection_sft.sh && /home/bujunru/.conda/envs/vision-se/bin/python -m py_compile VST-RL/recurrent/reflection_sft.py VST-RL/scripts/build_reflection_sft_data.py VST-RL/scripts/run_reflection_sft_opd_code_smoke.py VST-RL/scripts/audit_reflection_sft_opd_smoke.py && cd VST-RL && /home/bujunru/.conda/envs/vision-se/bin/python scripts/run_reflection_sft_opd_code_smoke.py --output /tmp/reflection-sft-opd-smoke.json && /home/bujunru/.conda/envs/vision-se/bin/python scripts/audit_reflection_sft_opd_smoke.py /tmp/reflection-sft-opd-smoke.json`

Expected: syntax checks exit 0 and smoke audit prints `PASS`.

- [ ] **Step 3: Check leakage strings and final/key gating mechanically**

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd && rg -n 'ground_truth|correct_answer|"reward"|Reward:' VST-RL/recurrent/reflection.py VST-RL/recurrent/reflection_sft.py VST-RL/scripts/build_reflection_sft_data.py`

Expected: no matches in teacher/Analyzer payload construction; validator denylist strings are allowed only when tests or validation code clearly reject them.

Run: `cd /home/bujunru/vlm-repro/VST-skill-opd && rg -n 'opd_episode_mask|opd_key_mask|final_opd_token_count|skill_conditioned_topk_opd_loss' VST-RL/recurrent/skill_opd.py VST-RL/verl/trainer/ppo/ray_trainer.py VST-RL/verl/workers/actor/dp_actor.py VST-RL/verl/trainer/ppo/skill_opd_loss.py`

Expected: episode mask gates the loss; key mask is used only for step-skill attachment/metrics; final token metric is present.

- [ ] **Step 4: Record immutable evidence and commit only if the report changed**

Update the smoke document with the exact test counts, smoke JSON SHA-256, current commit parent, Python/Torch versions, absolute asset-reference policy, `teacher_source=fixture`, `external_api_called=false`, and GPU/Qwen limitations.

```bash
git add docs/smoke/reflection-sft-global-episode-opd-cpu-smoke.md
git diff --cached --check
git diff --cached --quiet || git commit -m "docs: record reflection SFT OPD smoke evidence"
```

- [ ] **Step 5: Verify isolation and push only the feature branch**

Run: `git -C /home/bujunru/vlm-repro/VST-skill-opd status --short --branch && git -C /home/bujunru/vlm-repro/VST-skill-opd rev-parse HEAD && git -C /home/bujunru/vlm-repro/VST-skill-opd merge-base --is-ancestor 26f31d3 HEAD`

Expected: worktree clean, branch is `codex/vst-skill-opd`, and baseline ancestry check exits 0. Confirm `git status --porcelain` is empty before pushing.

Run: `git -C /home/bujunru/vlm-repro/VST-skill-opd push git@github.com:roderickprice554-tech/VST-SKILL-OPD.git HEAD:refs/heads/codex/vst-skill-opd`

Expected: only `codex/vst-skill-opd` advances in `roderickprice554-tech/VST-SKILL-OPD`; no existing VST repository or worktree is modified.

- [ ] **Step 6: Verify local and remote heads match**

Run: `git -C /home/bujunru/vlm-repro/VST-skill-opd rev-parse HEAD && git -C /home/bujunru/vlm-repro/VST-skill-opd ls-remote git@github.com:roderickprice554-tech/VST-SKILL-OPD.git refs/heads/codex/vst-skill-opd`

Expected: the two commit hashes are identical.
