"""Group Relative Policy Optimization (GRPO) loss and advantage calculation."""

from dataclasses import dataclass

import torch

from orbit.algorithms.common import compute_kl_divergence, masked_mean


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes group-normalized relative advantages for GRPO.

    Args:
        rewards: 1D Tensor of shape [B * G] or 2D Tensor of shape [B, G]
        group_size: Number of candidate completions sampled per prompt (G)
        eps: Small constant for numerical stability

    Returns:
        Tensor of same shape as input containing normalized advantages.
    """
    orig_shape = rewards.shape
    flat_rewards = rewards.view(-1, group_size)

    mean = flat_rewards.mean(dim=-1, keepdim=True)
    std = flat_rewards.std(dim=-1, keepdim=True)

    # Standardize per group
    normalized = (flat_rewards - mean) / (std + eps)
    return normalized.view(orig_shape)


@dataclass(frozen=True)
class GRPOLossOutput:
    """Detailed telemetry and loss terms produced during a GRPO update step."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


class GRPOLoss:
    """Computes the token-level clipped surrogate loss and KL penalty for GRPO."""

    def __init__(
        self,
        clip_range: float = 0.2,
        kl_coeff: float = 0.05,
        kl_estimator: str = "k3",
    ):
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.kl_estimator = kl_estimator

    def __call__(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> GRPOLossOutput:
        """Computes GRPO loss given token log-probabilities and normalized advantages.

        Args:
            logprobs: Current policy log-probabilities [batch_size, seq_len]
            old_logprobs: Rollout policy log-probabilities [batch_size, seq_len]
            ref_logprobs: Reference frozen policy log-probabilities [batch_size, seq_len]
            advantages: Group-normalized advantages [batch_size] or [batch_size, seq_len]
            mask: Binary attention/loss mask [batch_size, seq_len] (1 for completion tokens)

        Returns:
            GRPOLossOutput containing scalar losses and tracking metrics.
        """
        # Ensure advantages broadcast across sequence length
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)

        # 1. Probability ratio: rho_t = exp(log_pi - log_old)
        log_ratio = logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        # 2. Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_obj = torch.min(surr1, surr2)

        # Negative objective -> minimization loss
        policy_loss = -masked_mean(policy_obj, mask)

        # 3. Reference KL penalty
        token_kl = compute_kl_divergence(logprobs, ref_logprobs, estimator=self.kl_estimator)
        kl_loss = masked_mean(token_kl, mask)

        # 4. Total loss
        total_loss = policy_loss + self.kl_coeff * kl_loss

        # 5. Diagnostics
        with torch.no_grad():
            clipped = (ratio < (1.0 - self.clip_range)) | (ratio > (1.0 + self.clip_range))
            clip_fraction = masked_mean(clipped.float(), mask)
            approx_kl = masked_mean(old_logprobs - logprobs, mask)

        return GRPOLossOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )
