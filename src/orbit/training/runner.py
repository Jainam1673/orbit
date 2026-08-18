"""Experiment execution runner and manifest serialization for ORBIT."""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from orbit.agents.base import AgentConfig
from orbit.agents.reasoning import ReasoningAgent
from orbit.config import ExperimentConfig
from orbit.curriculum.manager import CurriculumManager
from orbit.data.trajectory import Provenance
from orbit.environments.math.generator import MathTaskGenerator
from orbit.environments.registry import make_environment
from orbit.models.factory import get_model_client
from orbit.training.trainer import TrainingLoop
from orbit.utils.provenance import capture_provenance
from orbit.utils.seed import set_seed


@dataclass(frozen=True)
class ExperimentResult:
    """Complete summary and output artifacts of an executed experiment."""

    experiment_id: str
    config: dict[str, Any]
    provenance: Provenance
    summary: dict[str, Any]
    output_dir: str
    duration_sec: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "provenance": self.provenance.to_dict(),
            "summary": self.summary,
            "output_dir": self.output_dir,
            "duration_sec": self.duration_sec,
            "timestamp": self.timestamp,
        }


def run_experiment(
    config: ExperimentConfig | None = None,
    num_steps: int = 10,
    eval_interval: int = 5,
) -> ExperimentResult:
    """Runs an end-to-end reproducible experiment from a structured configuration."""
    cfg = config or ExperimentConfig()
    start_time = time.perf_counter()
    exp_id = f"exp_{cfg.name}_{uuid.uuid4().hex[:8]}"

    # 1. Deterministic Seeding
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    # 2. Capture Provenance
    provenance = capture_provenance(
        model_version=cfg.model.name,
        env_version=cfg.environment.version,
        seed=cfg.seed,
    )

    # 3. Output directory setup
    run_output_dir = os.path.join(cfg.output_dir, exp_id)
    os.makedirs(run_output_dir, exist_ok=True)

    # 4. Instantiate Components
    env = make_environment(cfg.environment.env_id, max_steps=cfg.environment.max_episode_steps)
    model_client = get_model_client(cfg.model.provider, model_id=cfg.model.name)

    agent_cfg = AgentConfig(
        agent_id=cfg.agent.agent_id,
        system_prompt=cfg.agent.system_prompt,
        max_steps=cfg.agent.max_steps,
    )
    agent = ReasoningAgent(config=agent_cfg, model_client=model_client)

    curriculum_mgr = CurriculumManager(
        strategy=cfg.curriculum.strategy,
        seed=cfg.seed,
        target_success_rate=cfg.curriculum.frontier_target_success_rate,
        learning_rate=cfg.curriculum.learning_rate,
        min_difficulty=cfg.curriculum.min_difficulty,
        max_difficulty=cfg.curriculum.max_difficulty,
        window_size=cfg.curriculum.window_size,
    )

    # 5. Prepare held-out evaluation tasks (separate deterministic seed)
    eval_gen = MathTaskGenerator(seed=cfg.seed + 9999)
    eval_tasks = [
        eval_gen.generate_task(difficulty=d, task_id=f"eval_{i}")
        for i, d in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]

    # 6. Execute Training Loop
    loop = TrainingLoop(
        agent=agent,
        env=env,
        curriculum=curriculum_mgr.curriculum,
        eval_tasks=eval_tasks,
        run_id=exp_id,
    )

    summary = loop.run(
        num_steps=num_steps,
        eval_interval=eval_interval,
        base_seed=cfg.seed,
    )

    duration = time.perf_counter() - start_time

    # 7. Write Manifest and Results
    result = ExperimentResult(
        experiment_id=exp_id,
        config=asdict(cfg),
        provenance=provenance,
        summary=summary,
        output_dir=run_output_dir,
        duration_sec=duration,
        timestamp=provenance.timestamp,
    )

    manifest_path = os.path.join(run_output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result
