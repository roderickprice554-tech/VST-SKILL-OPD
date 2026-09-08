import torch


def combine_vst_rl_and_opd_loss(
    vst_rl_loss: torch.Tensor,
    opd_loss: torch.Tensor | None,
    *,
    enabled: bool,
    lambda_opd: float,
) -> torch.Tensor:
    if not enabled:
        return vst_rl_loss
    if opd_loss is None:
        raise ValueError("enabled Skill OPD requires an OPD loss")
    return vst_rl_loss + lambda_opd * opd_loss


def build_teacher_topk(
    teacher_logits: torch.Tensor,
    top_k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a detached teacher-selected support and distribution."""
    if teacher_logits.ndim < 2:
        raise ValueError("teacher_logits must include response and vocabulary dimensions")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled_logits = teacher_logits.detach().float() / temperature
    support_size = min(top_k, scaled_logits.shape[-1])
    top_values, top_indices = torch.topk(scaled_logits, support_size, dim=-1)
    top_log_probs = torch.log_softmax(top_values, dim=-1)
    retained_mass = torch.gather(
        torch.softmax(scaled_logits, dim=-1), -1, top_indices
    ).sum(dim=-1)
    return top_indices.detach(), top_log_probs.detach(), retained_mass.detach()


def skill_conditioned_topk_opd_loss(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    memory_mask: torch.Tensor,
    episode_mask: torch.Tensor,
    reflection_mask: torch.Tensor,
    metadata_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compute forward KL on teacher support at tokens allowed by every mask."""
    if teacher_topk_indices.shape != teacher_topk_log_probs.shape:
        raise ValueError("teacher top-k indices and log-probabilities must have equal shapes")
    if student_logits.shape[:-1] != teacher_topk_indices.shape[:-1]:
        raise ValueError("student and teacher cache leading dimensions must match")

    expected_mask_shape = student_logits.shape[:-1]
    masks = (response_mask, memory_mask, episode_mask, reflection_mask, metadata_mask)
    if any(mask.shape != expected_mask_shape for mask in masks):
        raise ValueError("all OPD masks must match student response positions")

    indices = teacher_topk_indices.detach().to(student_logits.device)
    teacher_log_probs = teacher_topk_log_probs.detach().to(
        device=student_logits.device, dtype=torch.float32
    )
    selected_student_logits = torch.gather(student_logits.float(), -1, indices)
    student_log_probs = torch.log_softmax(selected_student_logits, dim=-1)
    token_kl = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)

    valid_mask = torch.ones(expected_mask_shape, dtype=torch.bool, device=student_logits.device)
    for mask in masks:
        valid_mask &= mask.to(student_logits.device).bool()
    valid_count = int(valid_mask.sum().item())
    loss = (token_kl * valid_mask).sum() / max(1, valid_count)
    metrics = {
        "opd/valid_token_count": valid_count,
        "opd/response_token_count": int(response_mask.bool().sum().item()),
        "opd/token_kl_mean": float(loss.detach().item()),
    }
    return loss, metrics
