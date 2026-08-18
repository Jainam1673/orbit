"""Proximal Policy Optimization (PPO) loss with clipped surrogate and value objectives."""

from dataclasses import dataclass

import torch

from orbit.algorithms.common import masked_mean


@dataclass(frozen=True)
class PPOLossOutput:
    """Telemetry and loss terms produced during a PPO update step."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


class PPOLoss:
    """Computes clipped surrogate policy loss, value function loss, and entropy regularization for PPO."""

    def __init__(
        self,
        clip_range: float = 0.2,
        value_clip_range: float | None = 0.2,
        vf_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
    ):
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff

    def __call__(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
    ) -> PPOLossOutput:
        """Computes combined PPO loss.

        Args:
            logprobs: Current policy log-probabilities [batch_size, seq_len]
            old_logprobs: Rollout policy log-probabilities [batch_size, seq_len]
            values: Current critic state-value predictions [batch_size, seq_len]
            old_values: Rollout critic predictions [batch_size, seq_len]
            advantages: Estimated advantages [batch_size, seq_len]
            returns: Target returns [batch_size, seq_len]
            mask: Optional token mask [batch_size, seq_len]
            entropy: Optional token entropy tensor [batch_size, seq_len]
        """
        if mask is None:
            mask = torch.ones_like(logprobs)

        # 1. Clipped Surrogate Policy Loss
        log_ratio = logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_obj = torch.min(surr1, surr2)
        policy_loss = -masked_mean(policy_obj, mask)

        # 2. Value Function Loss (with optional clipping)
        vf_loss1 = (values - returns) ** 2
        if self.value_clip_range is not None:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.value_clip_range,
                self.value_clip_range,
            )
            vf_loss2 = (values_clipped - returns) ** 2
            vf_loss_elem = torch.max(vf_loss1, vf_loss2)
        else:
            vf_loss_elem = vf_loss1

        value_loss = 0.5 * masked_mean(vf_loss_elem, mask)

        # 3. Entropy regularization
        if entropy is not None:
            entropy_loss = -masked_mean(entropy, mask)
        else:
            entropy_loss = torch.tensor(0.0, device=logprobs.device)

        # Total combined loss
        total_loss = policy_loss + self.vf_coeff * value_loss + self.entropy_coeff * entropy_loss

        # Diagnostics
        with torch.no_grad():
            clipped = (ratio < (1.0 - self.clip_range)) | (ratio > (1.0 + self.clip_range))
            clip_fraction = masked_mean(clipped.float(), mask)
            approx_kl = masked_mean(old_logprobs - logprobs, mask)

        return PPOLossOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )
