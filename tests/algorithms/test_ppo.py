import pytest
import torch
from torch import nn
from torch.optim import Adam

from orbit.algorithms.ppo import PPOLoss, PPOTrainer


def test_ppo_loss_clipping_and_value_function():
    loss_fn = PPOLoss(clip_range=0.2, value_clip_range=0.2, vf_coeff=0.5, entropy_coeff=0.01)

    logprobs = torch.tensor([[-0.5, -0.5]])
    old_logprobs = torch.tensor([[-0.5, -0.5]])
    values = torch.tensor([[1.0, 1.0]])
    old_values = torch.tensor([[1.0, 1.0]])
    advantages = torch.tensor([[1.0, 1.0]])
    returns = torch.tensor([[2.0, 2.0]])
    entropy = torch.tensor([[0.5, 0.5]])

    # When ratio = 1.0:
    # policy_obj = 1.0 * 1.0 = 1.0 -> policy_loss = -1.0
    # value_loss = 0.5 * (1.0 - 2.0)^2 = 0.5
    # entropy_loss = -0.5
    # total_loss = -1.0 + 0.5 * 0.5 + 0.01 * (-0.5) = -1.0 + 0.25 - 0.005 = -0.755
    out = loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        values=values,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        entropy=entropy,
    )

    assert pytest.approx(out.policy_loss.item()) == -1.0
    assert pytest.approx(out.value_loss.item()) == 0.5
    assert pytest.approx(out.entropy_loss.item()) == -0.5
    assert pytest.approx(out.loss.item()) == -0.755
    assert out.clip_fraction.item() == 0.0


def test_ppo_trainer_step():
    model = nn.Linear(4, 2)
    optimizer = Adam(model.parameters(), lr=1e-2)
    trainer = PPOTrainer(model=model, optimizer=optimizer)

    x = torch.randn(2, 4)
    logits = model(x)
    logprobs = torch.log_softmax(logits, dim=-1)[:, 0:1]
    old_logprobs = logprobs.detach().clone()
    values = torch.tensor([[0.5], [0.5]])
    old_values = torch.tensor([[0.5], [0.5]])
    advantages = torch.tensor([[1.0], [-1.0]])
    returns = torch.tensor([[1.5], [-0.5]])
    mask = torch.ones(2, 1)

    metrics = trainer.train_step(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        values=values,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        mask=mask,
    )

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "grad_norm" in metrics
