import pytest
import torch
import torch.nn.functional as F

from verl.trainer.ppo.skill_opd_loss import build_teacher_topk, localized_topk_opd_loss


def _all_masks(batch, response):
    mask = torch.ones(batch, response, dtype=torch.bool)
    return [mask.clone() for _ in range(5)]


def test_teacher_selects_support_and_normalizes_only_on_it():
    logits = torch.tensor([[[0.0, 9.0, 1.0, 8.0]]])

    indices, log_probs, retained_mass = build_teacher_topk(logits, top_k=2, temperature=1.0)

    assert indices.tolist() == [[[1, 3]]]
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(1, 1))
    expected_mass = F.softmax(logits, dim=-1)[..., [1, 3]].sum(-1)
    assert torch.allclose(retained_mass, expected_mass)


def test_teacher_cache_is_detached_and_only_student_receives_gradient():
    teacher = torch.randn(1, 2, 5, requires_grad=True)
    student = torch.randn(1, 2, 5, requires_grad=True)
    indices, log_probs, _ = build_teacher_topk(teacher, top_k=3, temperature=1.0)

    loss, _ = localized_topk_opd_loss(student, indices, log_probs, *_all_masks(1, 2))
    loss.backward()

    assert indices.requires_grad is False
    assert log_probs.requires_grad is False
    assert teacher.grad is None
    assert student.grad is not None


def test_mask_product_selects_only_tokens_valid_in_every_mask():
    teacher = torch.tensor([[[4.0, 1.0], [1.0, 4.0]]])
    student = torch.tensor([[[1.0, 4.0], [4.0, 1.0]]], requires_grad=True)
    indices, log_probs, _ = build_teacher_topk(teacher, top_k=2, temperature=1.0)
    masks = _all_masks(1, 2)
    masks[2][0, 1] = False

    loss, metrics = localized_topk_opd_loss(student, indices, log_probs, *masks)
    first_only, _ = localized_topk_opd_loss(
        student[:, :1], indices[:, :1], log_probs[:, :1], *_all_masks(1, 1)
    )

    assert torch.allclose(loss, first_only)
    assert metrics["opd/valid_token_count"] == 1


def test_no_valid_tokens_returns_differentiable_zero():
    student = torch.randn(1, 2, 4, requires_grad=True)
    indices, log_probs, _ = build_teacher_topk(torch.randn(1, 2, 4), 3, 1.0)
    masks = _all_masks(1, 2)
    masks[3].zero_()

    loss, metrics = localized_topk_opd_loss(student, indices, log_probs, *masks)
    loss.backward()

    assert loss.item() == 0.0
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0
    assert metrics["opd/valid_token_count"] == 0


def test_final_answer_tokens_have_zero_opd_mask():
    teacher = torch.randn(2, 3, 7)
    student = torch.randn(2, 3, 7, requires_grad=True)
    indices, log_probs, _ = build_teacher_topk(teacher, 5, 1.0)
    masks = _all_masks(2, 3)
    masks[1][1].zero_()  # memory_mask: second row is the final answer row

    _, metrics = localized_topk_opd_loss(student, indices, log_probs, *masks)

    assert metrics["opd/valid_token_count"] == 3


def test_top_k_is_exactly_100_when_vocabulary_permits():
    indices, log_probs, _ = build_teacher_topk(torch.randn(2, 3, 101), 100, 1.0)

    assert indices.shape == log_probs.shape == (2, 3, 100)


def test_localized_loss_equals_full_forward_kl_when_k_covers_vocabulary():
    teacher = torch.randn(2, 3, 7)
    student = torch.randn(2, 3, 7, requires_grad=True)
    indices, teacher_log_probs, _ = build_teacher_topk(teacher, 100, 1.0)

    actual, _ = localized_topk_opd_loss(
        student, indices, teacher_log_probs, *_all_masks(2, 3)
    )
    full_teacher_log_probs = F.log_softmax(teacher, dim=-1)
    expected = (
        full_teacher_log_probs.exp()
        * (full_teacher_log_probs - F.log_softmax(student, dim=-1))
    ).sum(-1).mean()

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.isfinite(actual)


def test_bfloat16_teacher_input_produces_finite_normalized_cache():
    indices, log_probs, retained_mass = build_teacher_topk(
        torch.randn(1, 2, 128, dtype=torch.bfloat16), 100, 1.0
    )

    assert indices.shape[-1] == 100
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(1, 2), atol=1e-5)
    assert torch.isfinite(retained_mass).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_loss_is_finite():
    teacher = torch.randn(1, 2, 128, device="cuda", dtype=torch.bfloat16)
    student = torch.randn(1, 2, 128, device="cuda", requires_grad=True)
    indices, log_probs, _ = build_teacher_topk(teacher, 100, 1.0)
    masks = [mask.cuda() for mask in _all_masks(1, 2)]

    loss, _ = localized_topk_opd_loss(student, indices, log_probs, *masks)
    loss.backward()

    assert torch.isfinite(loss)
