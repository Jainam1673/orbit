"""Common mathematical utilities and reduction primitives for RL algorithms in ORBIT."""

from __future__ import annotations

import torch


def masked_sum(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Computes sum over elements where mask is True (or 1)."""
    masked = tensor * mask
    if dim is not None:
        return masked.sum(dim=dim)
    return masked.sum()


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes mean over elements where mask is True (or 1)."""
    total = masked_sum(tensor, mask, dim=dim)
    count = mask.sum(dim=dim) if dim is not None else mask.sum()
    return total / (count.clamp(min=1.0) + eps)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Generalized Advantage Estimation (GAE) and returns targets.

    Args:
        rewards: Tensor of shape [batch_size, seq_len]
        values: Tensor of shape [batch_size, seq_len]
        dones: Tensor of shape [batch_size, seq_len] (1 if terminal, 0 otherwise)
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter

    Returns:
        Tuple of (advantages, returns) both of shape [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_non_terminal = 1.0 - dones[:, t]
            next_values = torch.zeros(batch_size, device=rewards.device)
        else:
            next_non_terminal = 1.0 - dones[:, t]
            next_values = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_values * next_non_terminal - values[:, t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_kl_divergence(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    estimator: str = "k1",
) -> torch.Tensor:
    """Computes per-token KL divergence between policy logprobs and reference logprobs.

    Estimators:
    - 'k1': Standard log_p - log_ref
    - 'k3': Unbiased low-variance estimator (exp(log_ref - log_p) - (log_ref - log_p) - 1)
           from Schulman (2020) 'Approximating KL Divergence'.
    """
    ratio = ref_logprobs - logprobs
    if estimator == "k3":
        return torch.exp(ratio) - ratio - 1.0
    return logprobs - ref_logprobs
