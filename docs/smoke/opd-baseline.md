# Skill OPD baseline

Recorded on 2026-09-07 before the real Skill OPD GPU smoke.

## Git isolation

- Isolated worktree: `/home/bujunru/vlm-repro/VST-skill-opd`
- Branch: `codex/vst-skill-opd`
- Required ancestor: `26f31d36eb8bcc0b480b43eaaed1277b33a10488`
- Shared Git repository: `/home/bujunru/vlm-repro/VST/.git`
- Protected sibling worktrees were not copied or modified.
- Model, dataset, checkpoint, and cache contents were not copied into this worktree.

## Runtime

- Python: `3.12.13` at `/home/bujunru/.conda/envs/vision-se/bin/python`
- PyTorch: `2.8.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- NVIDIA driver: `580.173.02`
- GPUs: two `NVIDIA A100-PCIE-40GB`, 40960 MiB each
- Observation-time usage: GPU 0 used 38143 MiB; GPU 1 used 39051 MiB. Existing jobs will not be interrupted; the smoke waits for capacity.

## Read-only external assets

- Model: `/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct`
- Dataset: `/home/bujunru/vlm-repro/VST-full-reproduction/data/VST-Training-Data-official-5647583491c2/vst_rl_data/train.parquet`
- Environment manifest: `/home/bujunru/vlm-repro/VST-skill-opd/VST-RL/requirements.txt`

Stable manifest and tracked-file SHA-256 values are stored in `manifests/opd-foundation.json`.
