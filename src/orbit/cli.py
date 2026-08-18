"""Unified research CLI entrypoint for ORBIT."""

import argparse
import json
import sys

from orbit.agents.base import AgentConfig
from orbit.agents.reasoning import ReasoningAgent
from orbit.config import ExperimentConfig
from orbit.environments.math.environment import MathEnvironment
from orbit.environments.math.generator import MathTaskGenerator
from orbit.evaluation.ablations import AblationCondition, AblationRunner
from orbit.evaluation.evaluator import StandardEvaluator
from orbit.models.factory import get_model_client
from orbit.reporting import generate_research_report, package_release_bundle
from orbit.training.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Constructs the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="orbit",
        description="ORBIT — Language-Agent RL Research Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. train
    p_train = subparsers.add_parser("train", help="Run reproducible training experiment")
    p_train.add_argument("--steps", type=int, default=10, help="Total training steps")
    p_train.add_argument("--strategy", type=str, default="adaptive", help="Curriculum strategy")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--provider", type=str, default="mock", help="Model provider (mock/hf)")
    p_train.add_argument("--output-dir", type=str, default="experiments", help="Output directory")

    # 2. eval
    p_eval = subparsers.add_parser("eval", help="Evaluate model on benchmark tasks")
    p_eval.add_argument("--num-tasks", type=int, default=10, help="Number of evaluation tasks")
    p_eval.add_argument("--provider", type=str, default="mock", help="Model provider")
    p_eval.add_argument("--seed", type=int, default=42, help="Evaluation seed")

    # 3. ablate
    p_ablate = subparsers.add_parser("ablate", help="Run ablation matrix comparison")
    p_ablate.add_argument("--num-tasks", type=int, default=5, help="Tasks per condition")
    p_ablate.add_argument("--seed", type=int, default=42, help="Seed")

    # 4. generate-tasks
    p_gen = subparsers.add_parser("generate-tasks", help="Procedurally generate verified task sets")
    p_gen.add_argument("--count", type=int, default=10, help="Number of tasks to generate")
    p_gen.add_argument("--difficulty", type=float, default=0.5, help="Difficulty level [0.0-1.0]")
    p_gen.add_argument("--output-file", type=str, default="tasks.json", help="Output JSON file path")
    p_gen.add_argument("--seed", type=int, default=42, help="Generator seed")

    # 5. bundle
    p_bundle = subparsers.add_parser("bundle", help="Package experiment into a release bundle")
    p_bundle.add_argument("--run-dir", type=str, required=True, help="Directory of completed run")
    p_bundle.add_argument("--output-zip", type=str, default="bundle.zip", help="Output ZIP file")

    # 6. report
    p_report = subparsers.add_parser("report", help="Generate research Markdown report from manifest")
    p_report.add_argument("--manifest", type=str, required=True, help="Path to manifest.json")
    p_report.add_argument("--output-file", type=str, default="REPORT.md", help="Output Markdown path")

    # 7. server
    p_server = subparsers.add_parser("server", help="Launch research console FastAPI server")
    p_server.add_argument("--host", type=str, default="127.0.0.1", help="Server host interface")
    p_server.add_argument("--port", type=int, default=8000, help="Server HTTP port")

    return parser


def main(args: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    if parsed.command == "train":
        cfg = ExperimentConfig(
            name=f"cli_{parsed.strategy}",
            seed=parsed.seed,
            output_dir=parsed.output_dir,
        )
        cfg.model.provider = parsed.provider
        cfg.curriculum.strategy = parsed.strategy

        result = run_experiment(config=cfg, num_steps=parsed.steps)
        print(f"Training completed successfully. Run ID: {result.experiment_id}")
        print(f"Artifacts saved to: {result.output_dir}")
        return 0

    if parsed.command == "eval":
        client = get_model_client(parsed.provider)
        agent = ReasoningAgent(AgentConfig(), model_client=client)
        env = MathEnvironment()
        gen = MathTaskGenerator(seed=parsed.seed)
        tasks = [
            gen.generate_task(difficulty=min(1.0, (i + 1) / parsed.num_tasks))
            for i in range(parsed.num_tasks)
        ]
        evaluator = StandardEvaluator(run_id="cli_eval")
        res = evaluator.evaluate_agent(agent=agent, env=env, tasks=tasks)
        print(f"Evaluation Complete: Pass@1 = {res.pass_at_1 * 100:.1f}%, Mean Reward = {res.mean_reward:.3f}")
        return 0

    if parsed.command == "ablate":
        client = get_model_client("mock")
        env = MathEnvironment()
        gen = MathTaskGenerator(seed=parsed.seed)
        tasks = [gen.generate_task(difficulty=0.5) for _ in range(parsed.num_tasks)]

        conditions = [
            AblationCondition("Baseline_Standard", "Full system prompt", {}),
            AblationCondition("Ablation_Minimal", "Minimal prompt", {"agent_config": {"system_prompt": "Answer directly."}}),
        ]
        runner = AblationRunner()
        comparisons = runner.run_ablation_matrix(conditions, client, env, tasks)
        table = runner.format_markdown_table(comparisons)
        print(table)
        return 0

    if parsed.command == "generate-tasks":
        gen = MathTaskGenerator(seed=parsed.seed)
        tasks = [
            gen.generate_task(difficulty=parsed.difficulty, task_id=f"cli_task_{i}").to_dict()
            for i in range(parsed.count)
        ]
        with open(parsed.output_file, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        print(f"Generated {parsed.count} tasks at difficulty {parsed.difficulty:.2f} -> {parsed.output_file}")
        return 0

    if parsed.command == "bundle":
        out = package_release_bundle(parsed.run_dir, parsed.output_zip)
        print(f"Release bundle created: {out}")
        return 0

    if parsed.command == "report":
        with open(parsed.manifest, encoding="utf-8") as f:
            manifest_data = json.load(f)
        report_md = generate_research_report(manifest_data)
        with open(parsed.output_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Report generated: {parsed.output_file}")
        return 0

    if parsed.command == "server":
        import uvicorn

        print(f"Starting ORBIT Research Console on http://{parsed.host}:{parsed.port}/dashboard")
        uvicorn.run("orbit.server.app:app", host=parsed.host, port=parsed.port, log_level="info")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
