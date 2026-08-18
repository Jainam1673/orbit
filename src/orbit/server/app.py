"""FastAPI backend and research console endpoints for ORBIT."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from orbit.config import ExperimentConfig
from orbit.reporting import generate_research_report
from orbit.training.runner import run_experiment
from orbit.utils.provenance import capture_provenance

app = FastAPI(
    title="ORBIT Research Console",
    description="Scientific experiment dashboard, trajectory visualizer, and curriculum monitor.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPERIMENTS_DIR = os.path.abspath("experiments")


class TrainRequest(BaseModel):
    steps: int = 10
    strategy: str = "adaptive"
    seed: int = 42
    provider: str = "mock"
    output_dir: str = "experiments"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "orbit-research-console", "version": "0.1.0"}


@app.get("/api/system")
def get_system_info() -> dict[str, Any]:
    prov = capture_provenance()
    return prov.to_dict()


@app.get("/api/experiments")
def list_experiments() -> list[dict[str, Any]]:
    if not os.path.exists(EXPERIMENTS_DIR):
        return []

    experiments: list[dict[str, Any]] = []
    for entry in os.listdir(EXPERIMENTS_DIR):
        exp_path = os.path.join(EXPERIMENTS_DIR, entry)
        manifest_path = os.path.join(exp_path, "manifest.json")
        if os.path.isdir(exp_path) and os.path.exists(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    data = json.load(f)
                experiments.append(
                    {
                        "experiment_id": data.get("experiment_id", entry),
                        "timestamp": data.get("timestamp", ""),
                        "duration_sec": data.get("duration_sec", 0.0),
                        "seed": data.get("config", {}).get("seed", 42),
                        "strategy": data.get("config", {}).get("curriculum", {}).get("strategy", "adaptive"),
                        "total_steps": data.get("summary", {}).get("total_steps", 0),
                        "mean_reward": data.get("summary", {}).get("mean_reward", 0.0),
                        "overall_success_rate": data.get("summary", {}).get("overall_success_rate", 0.0),
                    }
                )
            except (json.JSONDecodeError, OSError, KeyError):
                pass

    experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return experiments


@app.get("/api/experiments/{exp_id}")
def get_experiment_details(exp_id: str) -> dict[str, Any]:
    exp_path = os.path.join(EXPERIMENTS_DIR, exp_id)
    manifest_path = os.path.join(exp_path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found.")

    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/experiments/{exp_id}/report")
def get_experiment_report(exp_id: str) -> dict[str, str]:
    exp_data = get_experiment_details(exp_id)
    report_md = generate_research_report(exp_data)
    return {"experiment_id": exp_id, "report_markdown": report_md}


@app.post("/api/train")
def trigger_training(req: TrainRequest) -> dict[str, Any]:
    cfg = ExperimentConfig(
        name=f"api_{req.strategy}",
        seed=req.seed,
        output_dir=req.output_dir,
    )
    cfg.model.provider = req.provider
    cfg.curriculum.strategy = req.strategy

    result = run_experiment(config=cfg, num_steps=req.steps)
    return result.to_dict()


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard_html() -> str:
    """Embedded interactive research console UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ORBIT — Research Console</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-8 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <div class="h-4 w-4 rounded-full bg-indigo-500 animate-pulse"></div>
            <h1 class="text-xl font-bold tracking-wider text-slate-50">ORBIT <span class="text-xs font-mono text-indigo-400 bg-indigo-950 px-2 py-0.5 rounded border border-indigo-800">RESEARCH CONSOLE</span></h1>
        </div>
        <div id="system-badge" class="text-xs font-mono text-slate-400">Loading system status...</div>
    </header>

    <main class="max-w-7xl mx-auto px-8 py-8 space-y-8">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Experiments</div>
                <div id="metric-total-exp" class="text-3xl font-bold mt-2 text-indigo-400">--</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Active Policy</div>
                <div class="text-3xl font-bold mt-2 text-emerald-400">GRPO</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Curriculum Mechanism</div>
                <div class="text-3xl font-bold mt-2 text-sky-400">Adaptive Frontier</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Verifier Invariant</div>
                <div class="text-3xl font-bold mt-2 text-purple-400">Exact Symbolic</div>
            </div>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-lg font-semibold text-slate-100">Experiment Execution History</h2>
                <button onclick="loadExperiments()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded transition">Refresh</button>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm text-slate-300">
                    <thead class="bg-slate-950/60 text-xs uppercase text-slate-400 border-b border-slate-800">
                        <tr>
                            <th class="px-4 py-3">Run ID</th>
                            <th class="px-4 py-3">Strategy</th>
                            <th class="px-4 py-3">Seed</th>
                            <th class="px-4 py-3">Steps</th>
                            <th class="px-4 py-3">Success Rate</th>
                            <th class="px-4 py-3">Mean Reward</th>
                            <th class="px-4 py-3">Duration</th>
                        </tr>
                    </thead>
                    <tbody id="experiments-table" class="divide-y divide-slate-800/60 font-mono text-xs">
                        <tr><td colspan="7" class="px-4 py-8 text-center text-slate-500">No experiments loaded yet.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <script>
        async function loadSystemInfo() {
            try {
                const res = await fetch('/api/system');
                const data = await res.json();
                document.getElementById('system-badge').innerText = `Git: ${data.git_commit.substring(0, 7)} | Platform: ${data.platform}`;
            } catch (e) {
                document.getElementById('system-badge').innerText = 'Backend connected';
            }
        }

        async function loadExperiments() {
            try {
                const res = await fetch('/api/experiments');
                const data = await res.json();
                document.getElementById('metric-total-exp').innerText = data.length;
                const tbody = document.getElementById('experiments-table');
                if (data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" class="px-4 py-8 text-center text-slate-500">No recorded experiment runs found in ./experiments</td></tr>';
                    return;
                }
                tbody.innerHTML = data.map(exp => `
                    <tr class="hover:bg-slate-800/40 transition">
                        <td class="px-4 py-3 font-semibold text-slate-100">${exp.experiment_id}</td>
                        <td class="px-4 py-3"><span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 border border-slate-700 text-indigo-300">${exp.strategy}</span></td>
                        <td class="px-4 py-3 text-slate-400">${exp.seed}</td>
                        <td class="px-4 py-3 text-slate-300">${exp.total_steps}</td>
                        <td class="px-4 py-3 text-emerald-400 font-bold">${(exp.overall_success_rate * 100).toFixed(1)}%</td>
                        <td class="px-4 py-3 text-slate-200">${exp.mean_reward.toFixed(3)}</td>
                        <td class="px-4 py-3 text-slate-400">${exp.duration_sec.toFixed(2)}s</td>
                    </tr>
                `).join('');
            } catch (e) {
                console.error(e);
            }
        }

        loadSystemInfo();
        loadExperiments();
    </script>
</body>
</html>"""
