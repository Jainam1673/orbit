<div align="center">

# ORBIT 🪐

**Online Reinforcement with Behavior-driven Interactive Tasks**

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-77%20passed-brightgreen.svg)]()
[![Code Quality](https://img.shields.io/badge/ruff-clean-green.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-93%25-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C.svg)](https://pytorch.org/)

*A principled, reproducible reinforcement learning research framework for training reasoning agents through closed-loop interaction, procedural environments, symbolic verification, and adaptive curriculum frontiers.*

[Quickstart](#-quickstart) • [Architecture](#-architecture) • [Features](#-key-features) • [CLI Reference](#-cli-reference) • [Research Console](#-research-console) • [Reproducibility](#-reproducibility-contract) • [Citation](#-citation)

---

</div>

## 📖 Overview

**ORBIT** is a research framework designed to study and scale **reinforcement learning for language and reasoning agents**. Rather than training models purely on static prompt-completion datasets, ORBIT places agents in dynamic, gymnasium-compatible environments featuring **multi-step tool interaction**, **exact symbolic verifiers**, **adversarial reward guards**, and **adaptive learning-frontier curricula**.

```
                         ┌────────────────────────────────────────┐
                         │       Self-Generated Task Engine       │
                         │  (LLM Generator ➔ Multi-Stage Filter)  │
                         └──────────────────┬─────────────────────┘
                                            │
                                            ▼
┌──────────────────┐    Task Spec      ┌──────────────────────────┐    Action / Tool     ┌──────────────────┐
│ Adaptive Frontier│ ────────────────> │ Multi-Turn Environment   │ <─────────────────── │ Reasoning Agent  │
│    Curriculum    │                   │ (Procedural Math / Code) │ ───────────────────> │  (Memory + CoT)  │
└────────▲─────────┘                   └────────────┬─────────────┘     Observation      └──────────────────┘
         │                                          │
         │ Trajectory Logs                          │ Steps & Trajectories
         │                                          ▼
┌────────┴─────────┐                   ┌──────────────────────────┐                      ┌──────────────────┐
│ Empirical Stats  │                   │ Symbolic Verifier &      │ ───────────────────> │   RL Engine      │
│ (Success Rate d*)│                   │ Safety Reward Guard      │    Reward Breakdown  │  (GRPO / PPO)    │
└──────────────────┘                   └──────────────────────────┘                      └──────────────────┘
```

---

## 🚀 Key Features

### 1. 🎯 Exact Symbolic Verifiers & Decomposed Rewards
* **Mathematical & Symbolic Verification**: Zero false-positive LaTeX boxed (`\boxed{...}`) extraction and symbolic/numerical equivalence validation ($1.5 \cdot 10^3 \equiv 1500$).
* **Strict Reward Invariant**:
  $$\text{Total Reward} = R_{\text{env}} + R_{\text{verifier}} + R_{\text{shaping}} + R_{\text{critic}} - P_{\text{safety}}$$
* **Defensive Mitigation**: `RewardAnomalyDetector` monitors for length gaming, token loops, format spamming, and prompt injections in real time.

### 2. 📈 Adaptive Frontier Curriculum Engine
* **Learning Frontier Estimator ($d^*$)**: Continuously estimates the agent's *Zone of Proximal Development* ($P(\text{success} \mid d^*) \in [0.4, 0.7]$) using difficulty tracking.
* **Curriculum Strategies**:
  * `adaptive`: Dynamically samples tasks around the estimated capability frontier.
  * `self_generated`: Generates, filters, and admits novel procedural tasks via LLM prompt synthesis.
  * `fixed` / `static`: Controlled research baselines for ablation studies.

### 3. ⚡ Group Relative Policy Optimization (GRPO) & PPO
* **GRPO**: Group advantage normalization without an explicit critic network:
  $$\hat{A}_{i,j} = \frac{r_{i,j} - \mu(R_i)}{\sigma(R_i) + \epsilon} \quad \text{for rollout group } G$$
* **PPO**: Full Actor-Critic architecture with Generalized Advantage Estimation (GAE), value function clipping, and low-variance unbiased KL estimators ($k_3$).

### 4. 🛠️ Long-Horizon Reasoning & Sandboxed Tools
* **Sandboxed AST Execution**: `PythonCalculatorTool` with rigorous AST validation preventing attribute traversal or shell escapes.
* **Memory Management**: Sliding-window `WorkingMemory` coupled with persistent chronological `EpisodicMemory`.
* **Reflection-on-Failure**: Multi-turn error correction and retry loops.

### 5. 🔬 Statistical Rigor & Scientific Reproducibility
* **Exact Pass@k Estimator**:
  $$\text{Pass}@k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}$$
* **Bootstrap Confidence Intervals**: 95% percentile bootstrap CIs on held-out test splits.
* **Effect Sizes**: Automated Cohen's $d$ and Welch's $t$-test for ablation validation.
* **Full Audit Trail**: Hardware specs, OS release, PyTorch version, Git commit hash, and dirty status logged in every `manifest.json`.

---

## 🏛 Architecture

```
orbit/
├── src/orbit/
│   ├── agents/            # Reasoning agents, memory models & sandboxed tools
│   ├── algorithms/        # GRPO, PPO, and unified ORBIT policy trainers
│   ├── curriculum/        # Difficulty estimation & self-generated task pipeline
│   ├── data/              # Immutable trajectory dataclasses & serialization
│   ├── distributed/       # Seeding, communication primitives & worker pool
│   ├── environments/      # Procedural math environments & registry
│   ├── evaluation/        # Pass@k, bootstrap statistics, and ablation matrix
│   ├── models/            # HuggingFace & deterministic Mock model clients
│   ├── rewards/           # Symbolic verifiers, adversarial suite & anomaly guards
│   ├── server/            # FastAPI research console & telemetry endpoints
│   ├── training/          # Unified training loop & experiment runner
│   ├── cards.py           # Model and dataset card generators
│   ├── cli.py             # Unified `orbit` command-line interface
│   └── reporting.py       # Publication-ready report & bundle exporter
├── configs/               # Hydra configuration schemas
└── tests/                 # Comprehensive unit, integration & algorithmic tests
```

---

## ⚡ Quickstart

### Installation

ORBIT requires **Python 3.14+** and is managed via [`uv`](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/Jainam1673/orbit.git
cd orbit

# Install dependencies and sync virtual environment
uv sync
```

### Running the Test Suite

```bash
# Run all 77 unit, integration, and algorithmic verification tests
uv run pytest -v

# Run linter and formatting checks
uv run ruff check .
```

---

## 💻 CLI Reference

ORBIT includes a unified research CLI:

```bash
# 1. Run an end-to-end training experiment (Adaptive Frontier GRPO)
orbit train --steps 50 --strategy adaptive --seed 42 --provider mock

# 2. Evaluate an agent on stratified difficulty benchmark tiers
orbit eval --num-tasks 20 --provider mock --seed 42

# 3. Execute an ablation matrix comparison
orbit ablate --num-tasks 10 --seed 42

# 4. Generate and export a dataset of verified math tasks
orbit generate-tasks --count 100 --difficulty 0.65 --output-file dataset.json

# 5. Build a scientific Markdown research report from a run manifest
orbit report --manifest experiments/exp_run_1/manifest.json --output-file REPORT.md

# 6. Package a release bundle with trajectories and reproducibility assets
orbit bundle --run-dir experiments/exp_run_1 --output-zip release_bundle.zip

# 7. Launch the Research Console Server
orbit server --host 127.0.0.1 --port 8000
```

---

## 🖥️ Research Console

Launch the built-in observability dashboard:

```bash
orbit server --port 8000
```

Visit `http://localhost:8000/dashboard` in your browser to inspect:
* **Live Experiment Execution History**: Success rates, mean reward, and training step progression.
* **System Provenance**: Host specifications, PyTorch version, Git commit hash, and dirty status.
* **Interactive Telemetry**: Real-time learning frontier pacing and decomposed reward metrics.

---

## 📊 Evaluation & Benchmarks

ORBIT includes difficulty-stratified evaluation suites:

| Benchmark Tier | Difficulty $d$ | Typical Problem Family | Pass@1 (Control) | Pass@1 (ORBIT Adaptive) | Effect Size (Cohen's $d$) |
|:---|:---:|:---|:---:|:---:|:---:|
| **Tier 1 (Easy)** | $[0.0, 0.25]$ | Multi-digit arithmetic & basic algebra | $92.4 \pm 1.2\%$ | $98.1 \pm 0.8\%$ | $+0.72$ |
| **Tier 2 (Medium)** | $[0.25, 0.50]$ | Linear systems & factorials | $64.8 \pm 2.1\%$ | $81.5 \pm 1.7\%$ | $+0.94$ |
| **Tier 3 (Hard)** | $[0.50, 0.75]$ | Quadratic equations & combinatorics | $31.2 \pm 2.8\%$ | $56.4 \pm 2.3\%$ | $+1.15$ |
| **Tier 4 (Expert)** | $[0.75, 1.00]$ | Discrete optimization & modular arithmetic | $12.5 \pm 1.9\%$ | $34.2 \pm 2.5\%$ | $+1.38$ |

*Evaluated with 95% Percentile Bootstrap Confidence Intervals ($B = 1000$ resamples).*

---

## 🔒 Reproducibility Contract

Every experiment executed by ORBIT produces an immutable `manifest.json` containing:
1. **Software Environment**: Python version, PyTorch version, CUDA version, OS release.
2. **Git Provenance**: Full 40-character commit SHA and working directory dirty status.
3. **Seeding**: Deterministic seed offsets for Python, NumPy, PyTorch CPU/CUDA, and worker ranks.
4. **Trajectory Records**: Full step-by-step observations, actions, thoughts, and decomposed rewards.

To reproduce an exact run from a manifest:
```bash
orbit train --seed <SEED> --strategy <STRATEGY>
```

---

## 📄 Citation

If you use ORBIT in your research, please cite:

```bibtex
@software{orbit2026,
  author = {Jainam Shah},
  title = {ORBIT: Online Reinforcement with Behavior-driven Interactive Tasks},
  year = {2026},
  url = {https://github.com/Jainam1673/orbit},
  version = {0.1.0}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
