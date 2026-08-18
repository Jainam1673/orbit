"""GRPO Trainer coordinating policy optimization steps and telemetry."""

import torch
from torch import nn
from torch.optim import Optimizer

from orbit.algorithms.grpo.loss import GRPOLoss, compute_group_advantages


class GRPOTrainer:
    """Trainer orchestrating policy optimization using GRPO."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: GRPOLoss | None = None,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or GRPOLoss()
        self.max_grad_norm = max_grad_norm

    def train_step(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Executes a single optimization step using GRPO.

        Args:
            logprobs: Differentiable policy log-probabilities [batch_size, seq_len]
            old_logprobs: Log-probabilities from rollout [batch_size, seq_len]
            ref_logprobs: Frozen reference model log-probabilities [batch_size, seq_len]
            rewards: Scalar rewards [batch_size]
            group_size: Group size G
            mask: Token loss mask [batch_size, seq_len]

        Returns:
            Dictionary of metrics for monitoring and telemetry.
        """
        self.optimizer.zero_grad()

        # Compute group-relative normalized advantages
        advantages = compute_group_advantages(rewards, group_size=group_size)

        loss_out = self.loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            mask=mask,
        )

        loss_out.loss.backward()

        grad_norm = 0.0
        if self.max_grad_norm > 0.0:
            grad_norm = float(
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            )

        self.optimizer.step()

        metrics: dict[str, float] = {
            "loss": float(loss_out.loss.item()),
            "policy_loss": float(loss_out.policy_loss.item()),
            "kl_loss": float(loss_out.kl_loss.item()),
            "clip_fraction": float(loss_out.clip_fraction.item()),
            "approx_kl": float(loss_out.approx_kl.item()),
            "grad_norm": grad_norm,
            "mean_reward": float(rewards.mean().item()),
        }
        return metrics
