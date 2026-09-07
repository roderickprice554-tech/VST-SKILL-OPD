import numpy as np
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from recurrent.skill_opd import augment_teacher_inputs, validate_opd_teacher_cache
from verl.protocol import DataProto


REPO_ROOT = Path(__file__).resolve().parents[2]


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [90, 91]


def test_skill_tokens_are_inserted_between_memory_prompt_and_original_response():
    input_ids = torch.tensor([[0, 1, 2, 11, 0], [0, 3, 4, 21, 22]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 1, 1, 1]])
    responses = torch.tensor([[11, 0], [21, 22]])

    augmented_ids, augmented_attention = augment_teacher_inputs(
        input_ids,
        attention_mask,
        responses,
        np.array(["skill", None], dtype=object),
        _Tokenizer(),
        max_context_tokens=16,
    )

    assert augmented_ids[0].tolist() == [1, 2, 90, 91, 11, 0]
    assert augmented_ids[1].tolist() == [0, 0, 3, 4, 21, 22]
    assert augmented_attention.tolist() == [[1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1]]


def _rollout_and_cache():
    rollout = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[5, 0], [6, 7]]),
            "response_mask": torch.tensor([[True, False], [True, True]]),
        },
        non_tensors={
            "trajectory_uid": np.array(["a", "b"], dtype=object),
            "policy_version": np.array([4, 4], dtype=np.int64),
            "transition_index": np.array([0, None], dtype=object),
        },
    )
    log_probs = torch.log_softmax(torch.randn(2, 2, 100), dim=-1)
    cache = DataProto.from_dict(
        tensors={
            "opd_topk_indices": torch.zeros(2, 2, 100, dtype=torch.long),
            "opd_teacher_topk_log_probs": log_probs,
            "opd_retained_mass": torch.ones(2, 2),
            "opd_valid_token_mask": torch.tensor([[True, False], [False, False]]),
        },
        non_tensors={
            "trajectory_uid": np.array(["a", "b"], dtype=object),
            "policy_version": np.array([4, 4], dtype=np.int64),
            "transition_index": np.array([0, None], dtype=object),
        },
    )
    return rollout, cache


def test_teacher_cache_shape_metadata_detachment_and_normalization_are_validated():
    rollout, cache = _rollout_and_cache()

    validate_opd_teacher_cache(rollout, cache, top_k=100)

    assert cache.batch["opd_topk_indices"].shape == (2, 2, 100)
    assert all(not value.requires_grad for value in cache.batch.values())


@pytest.mark.parametrize("field", ["trajectory_uid", "policy_version", "transition_index"])
def test_teacher_cache_rejects_metadata_mismatch(field):
    rollout, cache = _rollout_and_cache()
    cache.non_tensor_batch[field] = cache.non_tensor_batch[field].copy()
    cache.non_tensor_batch[field][0] = "wrong"

    with pytest.raises(ValueError, match=field):
        validate_opd_teacher_cache(rollout, cache, top_k=100)


def test_teacher_cache_rejects_wrong_k_or_nonfinite_distribution():
    rollout, cache = _rollout_and_cache()
    cache.batch["opd_teacher_topk_log_probs"][0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        validate_opd_teacher_cache(rollout, cache, top_k=100)

    _, cache = _rollout_and_cache()
    cache.batch["opd_topk_indices"] = cache.batch["opd_topk_indices"][..., :99]
    cache.batch["opd_teacher_topk_log_probs"] = cache.batch[
        "opd_teacher_topk_log_probs"
    ][..., :99]
    with pytest.raises(ValueError, match="shape|top_k"):
        validate_opd_teacher_cache(rollout, cache, top_k=100)


def test_teacher_augmentation_enforces_32k_total_limit():
    with pytest.raises(ValueError, match="context"):
        augment_teacher_inputs(
            torch.tensor([[1, 2, 3]]),
            torch.ones(1, 3, dtype=torch.long),
            torch.tensor([[3]]),
            np.array(["skill"], dtype=object),
            _Tokenizer(),
            max_context_tokens=3,
        )


def test_worker_cache_uses_current_actor_under_no_grad_before_update():
    actor_source = (
        REPO_ROOT / "VST-RL" / "verl" / "workers" / "actor" / "dp_actor.py"
    ).read_text(encoding="utf-8")
    worker_source = (
        REPO_ROOT / "VST-RL" / "verl" / "workers" / "fsdp_workers.py"
    ).read_text(encoding="utf-8")
    trainer_source = (
        REPO_ROOT / "VST-RL" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert "torch.no_grad()" in actor_source
    assert "self.actor.compute_opd_teacher_cache(" in worker_source
    assert "self.ref_policy.compute_opd_teacher_cache" not in worker_source
    assert trainer_source.index("compute_opd_teacher_cache(") < trainer_source.index(
        "update_actor(batch)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_current_actor_teacher_is_detached_while_student_path_is_trainable():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    class _TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.arange(8, dtype=torch.float32, device="cuda"))

        def forward(self, input_ids, **kwargs):
            batch, sequence = input_ids.shape
            return SimpleNamespace(logits=self.logits.expand(batch, sequence, -1))

    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = _TinyActor()
    actor.use_remove_padding = False
    actor.use_ulysses_sp = False
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3, 4]], device="cuda"),
            "attention_mask": torch.ones(1, 4, dtype=torch.long, device="cuda"),
            "position_ids": torch.arange(4, device="cuda").unsqueeze(0),
            "responses": torch.tensor([[3, 4]], device="cuda"),
            "opd_valid_token_mask": torch.tensor([[True, False]], device="cuda"),
        }
    )

    cache = actor.compute_opd_teacher_cache(data, top_k=3, temperature=1.0)
    selected = actor._forward_opd_student_selected(data.batch, cache["opd_topk_indices"])
    selected.sum().backward()

    assert all(not value.requires_grad for value in cache.values())
    assert cache["opd_topk_indices"].shape == (1, 2, 3)
    assert actor.actor_module.logits.grad is not None
