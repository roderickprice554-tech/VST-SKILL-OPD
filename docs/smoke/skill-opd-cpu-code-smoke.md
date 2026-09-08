# Skill OPD CPU code smoke

Recorded on 2026-09-08 after the user explicitly changed the current acceptance
gate from a real GPU/model smoke to a CPU code smoke.

## Scope

- No GPU was used (`CUDA_VISIBLE_DEVICES=""`).
- The reflection is a schema-valid fixture and is reported as
  `reflection_source=fixture`; it is not presented as Actor-generated evidence.
- The smoke exercises the formal trajectory assembler, reflection validator,
  OPD annotation builder, teacher top-100 cache, localized OPD loss, combined
  VST-RL + LOPD loss, backward pass, and one CPU SGD update.
- This verifies the code path and invariants. It does not establish real-model
  JSON reliability, BF16/GPU stability, or production-scale resource usage.

## Commands

From `/home/bujunru/vlm-repro/VST-skill-opd/VST-RL`:

```bash
CUDA_VISIBLE_DEVICES="" /home/bujunru/.conda/envs/vision-se/bin/python \
  scripts/run_skill_opd_code_smoke.py \
  /home/bujunru/vlm-repro/skill-opd-smoke/cpu-code-smoke.json

CUDA_VISIBLE_DEVICES="" /home/bujunru/.conda/envs/vision-se/bin/python \
  scripts/audit_skill_opd_smoke.py --cpu-code \
  /home/bujunru/vlm-repro/skill-opd-smoke/cpu-code-smoke.json
```

The audit printed `Skill OPD smoke audit passed`.

## Results

| Metric | Value |
| --- | ---: |
| Memory transitions | 2 |
| Final turns | 1 |
| Trajectories | 1 |
| Reward mapped | true |
| Reflection valid/applied | 1 / 1 |
| Key transitions | 1 |
| Query leakage count | 0 |
| Teacher top-k | 100 |
| Valid LOPD tokens | 2 |
| Final-answer LOPD tokens | 0 |
| Teacher detached | true |
| Teacher retained mass | 0.9624049663543701 |
| Teacher cache bytes | 7230 |
| VST-RL loss | 1.0238683223724365 |
| LOPD loss | 0.791822075843811 |
| Total loss | 1.031786561012268 |
| Optimizer completed | true |
| Maximum parameter change | 0.00042128562927246094 |
| Disabled path equivalent | true |

## Supporting checks

- Related CPU tests: `95 passed, 2 skipped` (the skipped cases require CUDA).
- Smoke audit tests alone: `31 passed`.
- Python compile and shell syntax checks passed.
- Hydra smoke configuration composed successfully with the existing absolute,
  read-only model and dataset paths and `trainer.total_training_steps=1`.
- The raw synthetic report is stored outside Git at
  `/home/bujunru/vlm-repro/skill-opd-smoke/cpu-code-smoke.json`.

GPU peak memory is not applicable to this CPU-only acceptance run.
