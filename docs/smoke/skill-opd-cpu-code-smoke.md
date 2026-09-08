# Reflection SFT and global-episode Skill OPD CPU smoke

Recorded on 2026-09-08 after the acceptance gate was explicitly limited to CPU
code smoke.

## Scope

- `CUDA_VISIBLE_DEVICES` was empty and the report records `device=cpu` and
  `gpu_used=false`.
- The external-teacher response is a strict fixture. The report records
  `teacher_source=fixture` and `external_api_called=false`.
- The smoke converts the raw final reward to `is_correct`, builds the same
  answer-free Analyzer request used online, creates the existing VST-SFT
  conversation shape, and performs a real toy causal-LM backward/update on six
  assistant reflection tokens.
- It then freezes a copy of the current toy Actor, creates skill-conditioned
  top-100 teacher distributions, evaluates the trainable Actor on original
  inputs, combines VST-RL and OPD losses, and performs one optimizer update.
- This does not claim an external API call, a real Qwen Reflection-SFT
  checkpoint, Actor-generated reflection reliability, or GPU RL execution.

## Commands

From `/home/bujunru/vlm-repro/VST-skill-opd/VST-RL`:

```bash
CUDA_VISIBLE_DEVICES="" /home/bujunru/.conda/envs/vision-se/bin/python \
  scripts/run_skill_opd_code_smoke.py \
  /home/bujunru/vlm-repro/skill-opd-smoke/reflection-sft-global-episode-opd-cpu.json

CUDA_VISIBLE_DEVICES="" /home/bujunru/.conda/envs/vision-se/bin/python \
  scripts/audit_skill_opd_smoke.py --cpu-code \
  /home/bujunru/vlm-repro/skill-opd-smoke/reflection-sft-global-episode-opd-cpu.json
```

The audit printed `Skill OPD smoke audit passed`.

## Results

| Metric | Value |
| --- | ---: |
| Teacher source / external API | fixture / false |
| SFT assistant target tokens | 6 |
| Toy SFT loss | 5.294656753540039 |
| Toy SFT maximum parameter change | 0.02961796522140503 |
| Memory transitions / final turns | 2 / 1 |
| Episode-supervised memory rows | 2 |
| Key memory rows | 1 |
| Non-key episode-only rows | 1 |
| Valid OPD tokens / final OPD tokens | 3 / 0 |
| Teacher top-k / detached | 100 / true |
| Teacher retained mass | 0.9191030859947205 |
| VST-RL loss | 0.3437940180301666 |
| OPD loss | 0.034291822463274 |
| Weighted OPD loss (`lambda=0.01`) | 0.00034291822463274003 |
| Total loss | 0.34413692355155945 |
| Actor maximum parameter change | 0.0008015930652618408 |
| Disabled path returns original loss object | true |

The JSON report SHA-256 is
`d6b9ba9ef2a868922594cb38ab83f55953d4b5d7bf7f1a15ebece369990a5b33`.

## Environment and supporting checks

- Python: 3.12.13
- PyTorch: 2.8.0+cu128, executed with CUDA hidden
- Full pytest with CUDA hidden: 138 passed, 2 CUDA-only tests skipped
- Python compile and both Bash syntax checks passed
- Model, dataset, video, and checkpoint assets remain external absolute paths;
  none are copied into this worktree
