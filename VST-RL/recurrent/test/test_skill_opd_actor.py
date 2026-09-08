from pathlib import Path

import torch

from verl.trainer.ppo.skill_opd_loss import combine_vst_rl_and_opd_loss


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_disabled_path_returns_original_vst_rl_loss_without_opd_input():
    rl_loss = torch.tensor(2.5, requires_grad=True)

    total = combine_vst_rl_and_opd_loss(rl_loss, None, enabled=False, lambda_opd=0.01)

    assert total is rl_loss


def test_enabled_path_adds_weighted_opd_and_updates_parameter():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    rl_loss = parameter.square()
    opd_loss = (parameter - 3).square()

    total = combine_vst_rl_and_opd_loss(rl_loss, opd_loss, enabled=True, lambda_opd=0.01)
    total.backward()
    optimizer.step()

    assert torch.allclose(total.detach(), torch.tensor(1.04))
    assert torch.isfinite(total)
    assert parameter.detach().item() != 1.0


def test_actor_only_requires_cache_and_second_forward_when_opd_is_enabled():
    actor_source = (
        REPO_ROOT / "VST-RL" / "verl" / "workers" / "actor" / "dp_actor.py"
    ).read_text(encoding="utf-8")

    assert 'skill_opd_enabled = data.meta_info.get("skill_opd", {}).get("enable", False)' in actor_source
    assert 'if skill_opd_enabled:' in actor_source
    assert '"opd_topk_indices"' in actor_source
    assert '"opd_episode_mask"' in actor_source
    assert "skill_conditioned_topk_opd_loss(" in actor_source
    assert "localized_topk_opd_loss(" not in actor_source


def test_trainer_converts_reward_to_correctness_and_builds_episode_only_skills():
    trainer_source = (
        REPO_ROOT / "VST-RL" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert "reward_to_is_correct" in trainer_source
    assert "correctness_by_trajectory=" in trainer_source
    assert 'text = f"Episode skill: {episode_skill}"' in trainer_source
    assert 'text += f"\\nStep skill: {step_skill}"' in trainer_source
    assert 'metrics["skill_opd/non_key_episode_row_count"]' in trainer_source
    assert 'metrics["skill_opd/final_opd_token_count"]' in trainer_source
