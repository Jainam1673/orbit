"""Configuration schemas and Hydra structured configurations for ORBIT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Model backend configuration."""

    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    provider: str = "huggingface"  # huggingface, vllm, mock
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    device: str = "auto"
    dtype: str = "bfloat16"
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class AgentConfigSchema:
    """Agent runtime configuration."""

    agent_id: str = "orbit_agent"
    max_steps: int = 10
    system_prompt: str = (
        "You are an expert reasoning agent. Solve tasks step by step."
    )


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    env_id: str = "math_reasoning"
    version: str = "0.1.0"
    max_episode_steps: int = 10
    difficulty_min: float = 0.1
    difficulty_max: float = 1.0


@dataclass
class AlgorithmConfig:
    """RL algorithm training configuration."""

    name: str = "grpo"  # ppo, grpo, orbit
    lr: float = 1e-5
    clip_range: float = 0.2
    kl_coeff: float = 0.05
    gamma: float = 1.0
    gae_lambda: float = 0.95
    batch_size: int = 16
    mini_batch_size: int = 4
    num_epochs: int = 1
    gradient_accumulation_steps: int = 2
    group_size: int = 4  # For GRPO


@dataclass
class CurriculumConfig:
    """Curriculum engine configuration."""

    strategy: str = "adaptive"  # fixed, static, adaptive, self_generated
    frontier_target_success_rate: float = 0.6
    learning_rate: float = 0.1
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    window_size: int = 20


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "phase0_foundation"
    seed: int = 42
    deterministic: bool = True
    output_dir: str = "experiments/outputs"
    wandb_project: str = "orbit"
    wandb_enabled: bool = False

    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfigSchema = field(default_factory=AgentConfigSchema)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
