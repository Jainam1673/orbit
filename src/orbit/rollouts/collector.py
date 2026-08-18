"""Rollout collection engine for executing agent-environment interaction episodes."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from orbit.agents.base import BaseAgent
from orbit.data.trajectory import RewardBreakdown, Step, Trajectory
from orbit.environments.base import BaseEnvironment, TaskSpec
from orbit.utils.provenance import capture_provenance


@dataclass(frozen=True)
class RolloutStats:
    """Aggregate statistics computed over a batch of collected trajectories."""

    total_episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    total_steps: int
    reward_breakdown_mean: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RolloutCollector:
    """Orchestrates agent-environment interaction loops and captures audited trajectories."""

    def __init__(self, default_run_id: str = "run_default"):
        self.default_run_id = default_run_id

    def collect_episode(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        task: TaskSpec | None = None,
        run_id: str | None = None,
        episode_id: str | None = None,
        seed: int = 0,
    ) -> Trajectory:
        """Executes a single complete episodic interaction between agent and environment."""
        r_id = run_id or self.default_run_id
        ep_id = episode_id or f"ep_{uuid.uuid4().hex[:8]}"

        agent.reset()
        obs, env_info = env.reset(task=task, seed=seed)
        task_id = env_info.get("task_id", "unknown_task")

        model_version = getattr(agent, "model_client", None)
        model_id = getattr(model_version, "model_id", "unknown")

        provenance = capture_provenance(
            model_version=model_id,
            env_version=getattr(env, "version", "0.1.0"),
            seed=seed,
        )

        traj = Trajectory(
            run_id=r_id,
            episode_id=ep_id,
            task_id=task_id,
            provenance=provenance,
            metadata={"env_id": env.env_id, "initial_info": env_info},
        )

        while True:
            action = agent.act(obs, traj)
            (
                next_obs,
                reward_breakdown,
                terminated,
                truncated,
                step_info,
            ) = env.step(action)
            done = terminated or truncated

            step = Step(
                step_index=traj.num_steps,
                observation=obs,
                action=action,
                reward=reward_breakdown,
                done=done,
                truncated=truncated,
                info=step_info,
            )
            traj.add_step(step)

            if done:
                traj.success = bool(step_info.get("is_correct", False))
                break

            obs = next_obs

        return traj

    def collect_batch(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        tasks: list[TaskSpec] | int = 5,
        run_id: str | None = None,
        base_seed: int = 0,
    ) -> tuple[list[Trajectory], RolloutStats]:
        """Collects a batch of trajectories and computes aggregate metrics."""
        task_list: list[TaskSpec | None]
        if isinstance(tasks, int):
            task_list = [None] * tasks
        else:
            task_list = list(tasks)

        trajectories: list[Trajectory] = []
        for i, t in enumerate(task_list):
            ep_seed = base_seed + i
            traj = self.collect_episode(
                agent=agent,
                env=env,
                task=t,
                run_id=run_id,
                seed=ep_seed,
            )
            trajectories.append(traj)

        # Compute aggregate metrics
        n = len(trajectories)
        if n == 0:
            return [], RolloutStats(0, 0.0, 0.0, 0.0, 0)

        success_count = sum(1 for t in trajectories if t.success)
        total_steps = sum(t.num_steps for t in trajectories)
        total_reward = sum(t.total_reward for t in trajectories)

        sum_rb = RewardBreakdown(
            env_reward=sum(t.reward_breakdown_sum.env_reward for t in trajectories),
            verifier_reward=sum(t.reward_breakdown_sum.verifier_reward for t in trajectories),
            shaping_reward=sum(t.reward_breakdown_sum.shaping_reward for t in trajectories),
            critic_reward=sum(t.reward_breakdown_sum.critic_reward for t in trajectories),
            penalties=sum(t.reward_breakdown_sum.penalties for t in trajectories),
        )

        rb_mean = {
            "env_reward": sum_rb.env_reward / n,
            "verifier_reward": sum_rb.verifier_reward / n,
            "shaping_reward": sum_rb.shaping_reward / n,
            "critic_reward": sum_rb.critic_reward / n,
            "penalties": sum_rb.penalties / n,
            "total": sum_rb.total / n,
        }

        stats = RolloutStats(
            total_episodes=n,
            success_rate=success_count / n,
            mean_reward=total_reward / n,
            mean_steps=total_steps / n,
            total_steps=total_steps,
            reward_breakdown_mean=rb_mean,
        )

        return trajectories, stats
