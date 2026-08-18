"""Defensive reward wrapper applying safety penalties and anomaly mitigation."""

from orbit.data.trajectory import RewardBreakdown, Step, Trajectory
from orbit.environments.base import TaskSpec
from orbit.rewards.base import BaseRewardFunction
from orbit.rewards.safety import RewardAnomalyDetector


class SafetyGuardedRewardFunction(BaseRewardFunction):
    """Wraps a reward function and applies penalty accounting for detected safety anomalies."""

    def __init__(
        self,
        base_reward_fn: BaseRewardFunction,
        detector: RewardAnomalyDetector | None = None,
        length_penalty_weight: float = 0.0001,
    ):
        super().__init__()
        self.base_reward_fn = base_reward_fn
        self.detector = detector or RewardAnomalyDetector()
        self.length_penalty_weight = length_penalty_weight

    def compute_reward(
        self,
        step: Step,
        task: TaskSpec,
        trajectory: Trajectory | None = None,
    ) -> RewardBreakdown:
        # Base reward decomposition
        base_breakdown = self.base_reward_fn.compute_reward(
            step=step,
            task=task,
            trajectory=trajectory,
        )

        # Anomaly and safety penalty analysis
        report = self.detector.analyze(step.action.raw_text)
        length_penalty = len(step.action.raw_text) * self.length_penalty_weight
        total_penalties = (
            base_breakdown.penalties + report.suggested_penalty + length_penalty
        )

        return RewardBreakdown(
            env_reward=base_breakdown.env_reward,
            verifier_reward=base_breakdown.verifier_reward,
            shaping_reward=base_breakdown.shaping_reward,
            critic_reward=base_breakdown.critic_reward,
            penalties=total_penalties,
        )
