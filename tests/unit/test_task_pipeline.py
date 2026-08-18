from orbit.curriculum.self_generated import SelfGeneratedCurriculum
from orbit.curriculum.task_generator import LLMTaskGenerator
from orbit.curriculum.validator import TaskPipelineValidator
from orbit.data.trajectory import Trajectory
from orbit.environments.base import TaskSpec
from orbit.models.mock import MockModelClient


def test_llm_task_generator_json_parsing():
    json_response = (
        '```json\n{"prompt": "Solve 3x + 1 = 10", "ground_truth": "3", "family": "math_algebra", "difficulty": 0.35}\n```'
    )
    client = MockModelClient(default_response=json_response)
    generator = LLMTaskGenerator(model_client=client)

    candidate = generator.generate_candidate(target_difficulty=0.35)
    assert candidate.prompt == "Solve 3x + 1 = 10"
    assert candidate.ground_truth == "3"
    assert candidate.family == "math_algebra"
    assert candidate.difficulty == 0.35
    assert candidate.task_id.startswith("gen_")


def test_task_pipeline_validator_structure_and_deduplication():
    validator = TaskPipelineValidator()

    # 1. Malformed structure (empty prompt)
    bad_task = TaskSpec(task_id="bad_1", family="math", prompt="", ground_truth="42")
    res_bad = validator.process_candidate(bad_task)
    assert res_bad.is_admitted is False
    assert "malformed_structure" in validator.rejection_counts

    # 2. Valid task
    valid_task = TaskSpec(
        task_id="valid_1",
        family="math",
        prompt="Calculate 12 * 12",
        ground_truth="144",
    )
    res_valid = validator.process_candidate(valid_task)
    assert res_valid.is_admitted is True
    assert validator.total_admitted == 1

    # 3. Duplicate task (same prompt)
    res_dup = validator.process_candidate(valid_task)
    assert res_dup.is_admitted is False
    assert "Duplicate" in res_dup.reason


def test_task_pipeline_validator_reward_leakage():
    validator = TaskPipelineValidator()

    leaked_task = TaskSpec(
        task_id="leak_1",
        family="math",
        prompt="The answer is 42. What is the answer?",
        ground_truth="42",
    )
    res = validator.process_candidate(leaked_task)
    assert res.is_admitted is False
    assert "Reward leakage" in res.reason


def test_self_generated_curriculum_lifecycle():
    # Setup LLM client that generates valid math tasks
    valid_json = '{"prompt": "Compute 7 * 8", "ground_truth": "56", "family": "math", "difficulty": 0.2}'
    client = MockModelClient(default_response=valid_json)

    curriculum = SelfGeneratedCurriculum(
        model_client=client,
        target_success_rate=0.6,
        learning_rate=0.1,
        seed=42,
    )

    # First task sampled should be from LLM generator
    task1 = curriculum.sample_task()
    assert task1.prompt == "Compute 7 * 8"
    assert task1.ground_truth == "56"

    # Next task will trigger deduplication since mock returns same prompt, so fallback will admit
    task2 = curriculum.sample_task()
    assert task2 is not None
    assert task2.prompt != ""

    # Update with success
    curriculum.update(Trajectory("r1", "e1", task1.task_id, success=True))

    metrics = curriculum.get_metrics()
    assert metrics["total_admitted_tasks"] >= 2
    assert metrics["admission_rate"] > 0.0
    assert "rejection_distribution" in metrics
