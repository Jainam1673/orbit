import pytest
import torch

from orbit.algorithms.common import (
    compute_gae,
    compute_kl_divergence,
    masked_mean,
    masked_sum,
)


def test_masked_reductions():
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    # Masked sum: (1+2) + (5+6) = 14.0
    assert pytest.approx(masked_sum(tensor, mask).item()) == 14.0

    # Masked mean: 14.0 / 4 elements = 3.5
    assert pytest.approx(masked_mean(tensor, mask).item()) == 3.5

    # Along dimension 1: [ (1+2)/2, (5+6)/2 ] = [1.5, 5.5]
    row_means = masked_mean(tensor, mask, dim=-1)
    assert pytest.approx(row_means[0].item()) == 1.5
    assert pytest.approx(row_means[1].item()) == 5.5


def test_compute_gae_exact_values():
    # 1 episode of 3 steps
    # rewards: [0, 0, 1], values: [0.5, 0.5, 0.5], dones: [0, 0, 1]
    # gamma = 1.0, gae_lambda = 1.0 (standard Monte Carlo returns)
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    dones = torch.tensor([[0.0, 0.0, 1.0]])

    advantages, returns = compute_gae(
        rewards, values, dones, gamma=1.0, gae_lambda=1.0
    )

    # Return at step 0 = 1.0, step 1 = 1.0, step 2 = 1.0
    assert torch.allclose(returns, torch.tensor([[1.0, 1.0, 1.0]]))

    # Advantage at step 0 = 1.0 - 0.5 = 0.5
    assert torch.allclose(advantages, torch.tensor([[0.5, 0.5, 0.5]]))


def test_compute_kl_divergence():
    logprobs = torch.tensor([-0.2, -0.5, -1.0])
    ref_logprobs = torch.tensor([-0.2, -0.4, -0.8])

    # Standard k1 estimator: logp - ref_logp
    k1 = compute_kl_divergence(logprobs, ref_logprobs, estimator="k1")
    assert torch.allclose(k1, logprobs - ref_logprobs)

    # k3 estimator: should be non-negative in expectation and 0 when identical
    k3_identical = compute_kl_divergence(logprobs, logprobs, estimator="k3")
    assert torch.allclose(k3_identical, torch.zeros_like(logprobs))

    k3 = compute_kl_divergence(logprobs, ref_logprobs, estimator="k3")
    assert (k3 >= 0.0).all()
