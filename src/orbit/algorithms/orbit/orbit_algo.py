"""ORBIT Research Algorithm: Unifying Adaptive Curricula with Group Relative Policy Optimization."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from orbit.algorithms.grpo.loss import (
    GRPOLoss,
    GRPOLossOutput,
    compute_group_advantages,
)
from orbit.curriculum.strategies import AdaptiveFrontierCurriculum
from orbit.data.trajectory import Trajectory
from orbit.environments.base import TaskSpec


@dataclass(frozen=True)
class OrbitStepResult:
    """Output metrics from an ORBIT algorithm training step."""

    loss_output: GRPOLossOutput
    frontier_difficulty: float
    recent_success_rate: float
    mean_reward: float
    grad_norm: float


class OrbitAlgorithm:
    """The ORBIT research algorithm integrating adaptive learning frontier curricula with GRPO."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        curriculum: AdaptiveFrontierCurriculum | None = None,
        loss_fn: GRPOLoss | None = None,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.curriculum = curriculum or AdaptiveFrontierCurriculum()
        self.loss_fn = loss_fn or GRPOLoss()
        self.max_grad_norm = max_grad_norm

    def sample_task(self) -> TaskSpec:
        """Samples the next task from the adaptive curriculum."""
        return self.curriculum.sample_task()

    def update_with_trajectory(self, trajectory: Trajectory) -> None:
        """Updates the adaptive curriculum state with empirical outcome."""
        self.curriculum.update(trajectory)

    def train_step(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int,
        mask: torch.Tensor,
    ) -> OrbitStepResult:
        """Performs a policy optimization step and returns diagnostic metrics."""
        self.optimizer.zero_grad()

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

        curriculum_metrics = self.curriculum.get_metrics()
        frontier = curriculum_metrics.get("current_frontier_difficulty", 0.5)
        success_rate = curriculum_metrics.get("recent_success_rate", 0.0)

        return OrbitStepResult(
            loss_output=loss_out,
            frontier_difficulty=frontier,
            recent_success_rate=success_rate,
            mean_reward=float(rewards.mean().item()),
            grad_norm=grad_norm,
        )
