from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from orbit.config import ExperimentConfig


def test_hydra_config_loading():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="config")
        assert cfg.name == "phase0_foundation"
        assert cfg.seed == 42
        assert cfg.model.lora_enabled is True
        assert cfg.algorithm.name == "grpo"
        assert cfg.curriculum.strategy == "adaptive"


def test_experiment_config_dataclass_defaults():
    exp = ExperimentConfig()
    assert exp.name == "phase0_foundation"
    assert exp.seed == 42
    assert exp.model.provider == "huggingface"
    assert exp.algorithm.clip_range == 0.2
