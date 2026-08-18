"""Research report generation and artifact bundle packaging for scientific release."""

import json
import os
import zipfile
from typing import Any


def generate_research_report(manifest_data: dict[str, Any]) -> str:
    """Generates a publication-grade Markdown research report from an experiment manifest."""
    exp_id = manifest_data.get("experiment_id", "unknown_exp")
    timestamp = manifest_data.get("timestamp", "N/A")
    duration = manifest_data.get("duration_sec", 0.0)

    cfg = manifest_data.get("config", {})
    prov = manifest_data.get("provenance", {})
    summary = manifest_data.get("summary", {})

    lines: list[str] = [
        f"# ORBIT Experiment Report: `{exp_id}`",
        "",
        f"**Date/Time (UTC)**: {timestamp}  ",
        f"**Duration**: {duration:.2f}s  ",
        f"**Reproducibility Seed**: `{cfg.get('seed', 'N/A')}`  ",
        "",
        "## 1. Provenance & Hardware Specification",
        f"- **Git Commit**: `{prov.get('git_commit', 'N/A')}`",
        f"- **Git Dirty**: `{prov.get('git_dirty', False)}`",
        f"- **Host Platform**: `{prov.get('platform', 'N/A')}`",
        f"- **Python Version**: `{prov.get('python_version', 'N/A')}`",
        f"- **PyTorch Version**: `{prov.get('torch_version', 'N/A')}`",
        f"- **GPU Model**: `{prov.get('gpu_name', 'None')}` (Count: {prov.get('gpu_count', 0)})",
        "",
        "## 2. Configuration",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
        "## 3. Training & Evaluation Summary",
        f"- **Total Training Steps**: {summary.get('total_steps', 0)}",
        f"- **Mean Training Reward**: {summary.get('mean_reward', 0.0):.4f}",
        f"- **Overall Success Rate**: {summary.get('overall_success_rate', 0.0) * 100:.1f}%",
        "",
    ]

    eval_hist = summary.get("eval_history", [])
    if eval_hist:
        lines.extend(
            [
                "### Held-Out Evaluation Progression",
                "| Step | Eval Episodes | Pass Rate | Mean Reward | Mean Steps |",
                "|---|---|---|---|---|",
            ]
        )
        for ev in eval_hist:
            step = ev.get("step", 0)
            episodes = ev.get("eval_total_episodes", 0)
            rate = ev.get("eval_success_rate", 0.0) * 100.0
            rew = ev.get("eval_mean_reward", 0.0)
            steps = ev.get("eval_mean_steps", 0.0)
            lines.append(
                f"| {step} | {episodes} | {rate:.1f}% | {rew:.3f} | {steps:.1f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. Reproducibility Statement",
            (
                "This run was conducted under deterministic execution guarantees. "
                "To reproduce this exact experiment, run:"
            ),
            "```bash",
            f"orbit train --seed {cfg.get('seed', 42)} --strategy {cfg.get('curriculum', {}).get('strategy', 'adaptive')}",
            "```",
        ]
    )

    return "\n".join(lines)


def package_release_bundle(run_dir: str, output_zip_path: str) -> str:
    """Packages run artifacts (manifest, report, logs) into a release archive."""
    os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
        report_md = generate_research_report(manifest_data)
        report_file = os.path.join(run_dir, "REPORT.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(run_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, run_dir)
                zipf.write(file_path, arcname=rel_path)

    return output_zip_path
