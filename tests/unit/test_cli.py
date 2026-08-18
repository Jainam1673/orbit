import json
import os

from orbit.cards import generate_dataset_card, generate_model_card
from orbit.cli import main


def test_cli_help_and_empty():
    assert main([]) == 0


def test_cli_train_command(tmp_path):
    out_dir = str(tmp_path / "cli_train_out")
    code = main(["train", "--steps", "2", "--output-dir", out_dir, "--provider", "mock"])
    assert code == 0
    assert os.path.exists(out_dir)


def test_cli_eval_command():
    code = main(["eval", "--num-tasks", "2", "--provider", "mock"])
    assert code == 0


def test_cli_ablate_command():
    code = main(["ablate", "--num-tasks", "2"])
    assert code == 0


def test_cli_generate_tasks_command(tmp_path):
    tasks_file = str(tmp_path / "test_tasks.json")
    code = main(["generate-tasks", "--count", "3", "--difficulty", "0.4", "--output-file", tasks_file])
    assert code == 0
    assert os.path.exists(tasks_file)
    with open(tasks_file, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 3


def test_cli_report_and_bundle(tmp_path):
    run_dir = tmp_path / "sample_run"
    run_dir.mkdir()
    manifest_file = run_dir / "manifest.json"

    manifest_data = {
        "experiment_id": "exp_sample_123",
        "timestamp": "2026-08-18T12:00:00Z",
        "duration_sec": 4.5,
        "config": {"seed": 42, "curriculum": {"strategy": "adaptive"}},
        "provenance": {"git_commit": "abcdef1", "platform": "Linux"},
        "summary": {
            "total_steps": 5,
            "mean_reward": 0.8,
            "overall_success_rate": 0.8,
            "eval_history": [{"step": 5, "eval_total_episodes": 2, "eval_success_rate": 1.0, "eval_mean_reward": 1.0, "eval_mean_steps": 1.0}],
        },
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # 1. Report command
    report_file = str(tmp_path / "REPORT.md")
    code_rep = main(["report", "--manifest", str(manifest_file), "--output-file", report_file])
    assert code_rep == 0
    assert os.path.exists(report_file)

    # 2. Bundle command
    bundle_zip = str(tmp_path / "bundle.zip")
    code_bun = main(["bundle", "--run-dir", str(run_dir), "--output-zip", bundle_zip])
    assert code_bun == 0
    assert os.path.exists(bundle_zip)


def test_model_and_dataset_cards():
    model_card = generate_model_card(
        model_name="orbit-math-reasoner",
        run_id="run_42",
        provenance={"seed": 42, "git_commit": "deadbeef", "gpu_name": "NVIDIA A100"},
        eval_metrics={"pass_at_1": 0.85, "mean_reward": 0.88, "ci_95": "[0.82, 0.91]"},
    )
    assert "# Model Card: orbit-math-reasoner" in model_card
    assert "NVIDIA A100" in model_card

    dataset_card = generate_dataset_card(
        dataset_name="orbit-math-eval-v1",
        total_tasks=100,
        categories=["arithmetic", "algebra", "quadratic"],
    )
    assert "# Dataset Card: orbit-math-eval-v1" in dataset_card
    assert "Total Tasks**: 100" in dataset_card
