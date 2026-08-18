"""PPO Trainer coordinating actor-critic policy optimization steps."""

import torch
from torch import nn
from torch.optim import Optimizer

from orbit.algorithms.ppo.loss import PPOLoss


class PPOTrainer:
    """Trainer orchestrating policy and value updates using PPO."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: PPOLoss | None = None,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or PPOLoss()
        self.max_grad_norm = max_grad_norm

    def train_step(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Executes a single optimization step using PPO.

        Returns:
            Dictionary of loss terms and gradient norms.
        """
        self.optimizer.zero_grad()

        loss_out = self.loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            values=values,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
            entropy=entropy,
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
            "value_loss": float(loss_out.value_loss.item()),
            "entropy_loss": float(loss_out.entropy_loss.item()),
            "clip_fraction": float(loss_out.clip_fraction.item()),
            "approx_kl": float(loss_out.approx_kl.item()),
            "grad_norm": grad_norm,
        }
        return metrics
