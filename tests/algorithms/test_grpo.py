import pytest
import torch
from torch import nn
from torch.optim import Adam

from orbit.algorithms.grpo import (
    GRPOLoss,
    GRPOTrainer,
    compute_group_advantages,
)


def test_compute_group_advantages():
    # 2 prompts, group size 4 = 8 samples total
    # Group 1: [0, 0, 1, 1] -> mean 0.5
    # Group 2: [2, 2, 2, 2] -> std 0 -> normalized 0
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    adv = compute_group_advantages(rewards, group_size=4)

    assert adv.shape == rewards.shape

    # Group 1 mean should be 0, std should be 1
    g1_adv = adv[:4]
    assert pytest.approx(g1_adv.mean().item(), abs=1e-5) == 0.0
    assert pytest.approx(g1_adv.std().item(), abs=1e-3) == 1.0

    # Group 2 all same reward -> 0 advantage
    g2_adv = adv[4:]
    assert torch.allclose(g2_adv, torch.zeros(4), atol=1e-4)


def test_grpo_loss_clipping():
    loss_fn = GRPOLoss(clip_range=0.2, kl_coeff=0.0)

    # 1 sample, 3 tokens
    # old_logprobs = [-1.0, -1.0, -1.0]
    old_logprobs = torch.tensor([[-1.0, -1.0, -1.0]])
    ref_logprobs = old_logprobs.clone()
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    advantages = torch.tensor([[1.0, 1.0, 1.0]])

    # When ratio = 1.0 (logprobs == old_logprobs), loss = -1.0
    out_identity = loss_fn(old_logprobs, old_logprobs, ref_logprobs, advantages, mask)
    assert pytest.approx(out_identity.policy_loss.item()) == -1.0
    assert out_identity.clip_fraction.item() == 0.0

    # When ratio is very high (e.g., 2.0), clipped ratio is 1.2
    high_logprobs = old_logprobs + torch.log(torch.tensor(2.0))
    out_high = loss_fn(high_logprobs, old_logprobs, ref_logprobs, advantages, mask)
    # Clipped loss should be -1.2
    assert pytest.approx(out_high.policy_loss.item()) == -1.2
    assert out_high.clip_fraction.item() == 1.0


def test_grpo_trainer_optimization_step():
    # Simple linear model predicting logits for 2 classes
    model = nn.Linear(4, 2)
    optimizer = Adam(model.parameters(), lr=1e-2)
    trainer = GRPOTrainer(model=model, optimizer=optimizer)

    # Mock inputs
    x = torch.randn(4, 4)
    logits = model(x)
    logprobs = torch.log_softmax(logits, dim=-1)[:, 0:1]  # 4 samples, 1 token
    old_logprobs = logprobs.detach().clone()
    ref_logprobs = old_logprobs.clone()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4, 1)

    metrics = trainer.train_step(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=rewards,
        group_size=4,
        mask=mask,
    )

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "grad_norm" in metrics
    assert metrics["mean_reward"] == 0.5
