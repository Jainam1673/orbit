# ORBIT — MASTER PROJECT SPECIFICATION

**Status:** Living project constitution  
**Audience:** Human engineers, research engineers, AI coding agents, AI research agents, reviewers  
**Authority:** Highest-level project specification in this repository

---

## 0. Purpose of This Document

This document is the authoritative engineering and research contract for ORBIT.

Every human or AI agent working on the repository MUST read this document before making changes.

ORBIT is intended to become a serious research-grade platform for investigating adaptive reinforcement learning for language agents, with a strong emphasis on:

- scientific validity
- reproducibility
- measurable progress
- adaptive curricula
- self-generated tasks
- long-horizon agent behavior
- efficient training
- generalization
- evaluation
- reward-hacking analysis
- scalable experimentation

ORBIT is NOT intended to be:

- a collection of disconnected AI demos
- a generic SaaS dashboard
- a "feature showcase"
- a fake SOTA implementation
- a benchmark-number generator
- an unnecessarily complicated microservice platform
- a system that prioritizes visual polish over research correctness

The primary objective is to build a system capable of producing trustworthy experimental evidence.

---

# 1. NON-NEGOTIABLE RULES

## 1.1 Read Before Coding

Before modifying the repository, an agent MUST:

1. Read `MASTER.md` completely.
2. Inspect the current repository.
3. Inspect `pyproject.toml`.
4. Inspect `uv.lock`.
5. Inspect `mise.toml` or the current mise configuration.
6. Inspect relevant source files.
7. Identify the current implementation phase.
8. Determine what is actually implemented versus merely planned.

Never assume documentation means implementation exists.

---

## 1.2 Do Not Rewrite Working Code Without Reason

Agents MUST NOT:

- rewrite the repository unnecessarily
- replace working libraries without justification
- delete working features without understanding their purpose
- introduce architectural changes merely because an alternative looks cleaner
- create duplicate implementations
- silently change public interfaces

Existing code must be understood before modification.

---

## 1.3 Scientific Honesty

ORBIT is a research project.

Agents MUST NOT fabricate:

- benchmark results
- performance numbers
- research findings
- SOTA claims
- statistical significance
- scalability claims
- reproducibility claims
- experiment outcomes

If an experiment fails, record the failure.

If a hypothesis is unsupported, say so.

If results are inconclusive, label them inconclusive.

A negative result is scientifically valuable.

---

# 2. PROJECT VISION

ORBIT investigates the following broad research question:

> Can adaptive, self-generated curricula improve the sample efficiency, robustness, and generalization of reinforcement learning for long-horizon language agents?

The project should investigate the interaction between:

```text
Agent
   ↓
Environment
   ↓
Trajectory
   ↓
Reward / Verification
   ↓
RL Update
   ↓
Updated Policy
   ↓
Curriculum
   ↓
New / Selected Tasks
   ↓
Agent
```

The central system should make this loop observable, reproducible, and experimentally controllable.

---

# 3. RESEARCH PRINCIPLES

The project follows these principles:

1. Establish strong baselines before claiming improvement.
2. Separate training and evaluation data.
3. Use held-out evaluation tasks.
4. Use multiple seeds for important experiments when feasible.
5. Record configuration and environment versions.
6. Measure compute and sample efficiency.
7. Perform ablations for claimed contributions.
8. Explicitly investigate reward hacking.
9. Prefer objective verification over subjective scoring.
10. Never optimize solely for attractive charts.
11. Make experiments reproducible.
12. Document limitations.

---

# 4. TECHNOLOGY CONTRACT

The following technology choices are project defaults.

## Runtime and Language Management

Use:

- `mise`

Do not introduce:

- pyenv
- Conda
- Poetry
- another runtime manager

unless explicitly justified and documented.

---

## Python Package Management

Use:

- `uv`

All Python dependencies belong in:

```text
pyproject.toml
uv.lock
```

Use:

```bash
uv add PACKAGE
uv add --dev PACKAGE
uv run COMMAND
```

Do not use `pip install` as the project dependency-management mechanism.

---

## Python

Target:

```text
Python 3.12+
```

The exact supported version must be defined in project configuration.

---

# 5. MACHINE LEARNING STACK

Core:

- PyTorch
- NumPy
- SciPy
- Pandas where justified

Foundation models:

- Hugging Face Transformers

Parameter-efficient training:

- PEFT
- LoRA where appropriate

RL:

- custom PPO implementation where research transparency is required
- custom GRPO implementation where research transparency is required
- Hugging Face TRL where useful for reference implementations or infrastructure

Inference:

- vLLM

Experiment configuration:

- Hydra

Experiment tracking:

- Weights & Biases

Distributed training:

- PyTorch Distributed initially

Distributed orchestration:

- Ray only when justified by actual workloads

---

# 6. BACKEND STACK

The control plane should use:

- FastAPI
- Pydantic
- PostgreSQL
- SQLAlchemy where appropriate
- Alembic for migrations

Object storage:

- S3-compatible storage
- MinIO for local development when appropriate

Communication:

- HTTP APIs
- WebSockets or SSE for live experiment events where appropriate

The backend is a control and observability plane.

It must not contain the core research algorithms merely because it is convenient.

---

# 7. FRONTEND STACK

The frontend is a research console.

Use:

- Next.js
- React
- TypeScript
- Bun
- Tailwind CSS
- shadcn/ui
- TanStack Query
- Zod

The frontend must communicate with the backend through typed APIs.

The frontend MUST NOT directly import:

- PyTorch research code
- training internals
- environment internals
- model implementation internals

The architectural boundary is:

```text
Next.js
   ↓
FastAPI
   ↓
Research Services
   ↓
Research Engine
```

---

# 8. FRONTEND PRODUCT VISION

The UI should eventually provide:

- Research Overview
- Experiments
- Experiment Details
- Training Runs
- Agent Runs
- Trajectories
- Environments
- Curriculum
- Evaluations
- Benchmarks
- Models
- Ablations
- Safety / Reward-Hacking Analysis
- System Metrics
- GPU Metrics

The interface should feel like a research laboratory console rather than a generic SaaS dashboard.

Do not build fake dashboards.

If data is unavailable, display:

- empty state
- loading state
- error state
- unavailable state

Never fabricate metrics for UI purposes.

---

# 9. CORE REPOSITORY STRUCTURE

The repository should evolve toward:

```text
orbit/
├── MASTER.md
├── README.md
├── pyproject.toml
├── uv.lock
├── mise.toml
├── .gitignore
│
├── src/
│   └── orbit/
│       ├── agents/
│       ├── models/
│       ├── algorithms/
│       │   ├── ppo/
│       │   ├── grpo/
│       │   └── orbit/
│       ├── environments/
│       ├── curriculum/
│       ├── rewards/
│       ├── rollouts/
│       ├── training/
│       ├── evaluation/
│       ├── benchmarks/
│       ├── distributed/
│       ├── data/
│       └── safety/
│
├── backend/
├── frontend/
│
├── configs/
├── experiments/
├── tests/
│   ├── unit/
│   ├── algorithms/
│   ├── environments/
│   ├── integration/
│   ├── distributed/
│   └── reproducibility/
│
├── scripts/
│
├── docs/
│   ├── architecture/
│   ├── research/
│   ├── engineering/
│   └── decisions/
│
├── papers/
└── data/
```

Do not create directories simply because they look impressive.

Create them when they have an actual responsibility.

---

# 10. ARCHITECTURAL LAYERS

ORBIT must maintain clear boundaries between:

1. Research Engine
2. Agent Runtime
3. Environment System
4. Curriculum System
5. Reward / Verification System
6. Rollout System
7. Training System
8. Evaluation System
9. Control Plane
10. Frontend
11. Infrastructure

Dependencies should generally flow from higher-level orchestration toward lower-level implementations without creating circular dependencies.

---

# 11. AGENT ARCHITECTURE

Agents must be modular.

Core abstractions should include concepts such as:

- `Agent`
- `Policy`
- `Planner`
- `Memory`
- `Tool`
- `Observation`
- `Action`
- `Trajectory`

Model implementations must be replaceable.

Do not couple the agent interface to one model provider.

Agent configuration must support:

- model identity
- generation parameters
- seed
- tool configuration
- memory configuration
- environment configuration

Trajectories must be recordable.

---

# 12. ENVIRONMENT ARCHITECTURE

Environment APIs should be Gymnasium-inspired.

Each environment should define:

- task specification
- observation
- action
- reset
- step
- termination
- reward
- verification
- metadata
- difficulty

Initial environment priorities:

1. Mathematics
2. Coding
3. Planning
4. Scientific reasoning

Do not implement a large number of environments before at least one environment is rigorous and experimentally useful.

---

# 13. CODING ENVIRONMENT SECURITY

Model-generated code is untrusted.

Never execute generated code directly on the host.

Use isolated execution.

Initial implementation may use Docker-based isolation.

The execution pipeline should resemble:

```text
Task
 ↓
Agent
 ↓
Generated Code
 ↓
Sandbox
 ↓
Execution
 ↓
Tests / Verifier
 ↓
Reward
```

The sandbox must restrict:

- filesystem access
- network access where unnecessary
- credentials
- host process access
- arbitrary device access

Security-sensitive code must receive explicit review.

---

# 14. TRAJECTORY SYSTEM

A trajectory should contain enough information to reconstruct an agent interaction.

At minimum, where applicable:

- run ID
- episode ID
- task ID
- timestamp
- observation
- action
- tool call
- tool result
- reward
- termination
- policy/model version
- environment version
- seed

Trajectory storage must be versionable.

Do not store sensitive information unnecessarily.

---

# 15. REWARD ARCHITECTURE

Rewards must be explicit.

Separate, where applicable:

- environment reward
- verifier reward
- shaping reward
- critic reward
- penalties

The final reward should be traceable to its components.

Reward computation must be inspectable.

Never hide critical reward logic inside opaque callbacks.

---

# 16. VERIFICATION

Objective verification is preferred whenever possible.

Examples:

Coding:

```text
generated solution
 ↓
compile
 ↓
tests
 ↓
result
```

Mathematics:

```text
generated answer
 ↓
symbolic/numeric verifier
 ↓
correctness
```

Planning:

```text
plan
 ↓
constraint checker
 ↓
validity
```

The evaluator must be separated from the agent whenever practical.

---

# 17. PPO

PPO must be implemented in a transparent and testable way if it is used as a research baseline.

Document:

- objective
- clipping
- advantages
- returns
- value loss
- entropy
- KL
- batching
- normalization
- gradient accumulation

Critical algorithmic logic must remain understandable.

Write unit tests for mathematical invariants and simple synthetic cases.

---

# 18. GRPO

GRPO must be treated as an explicit algorithmic component rather than an unexplained training wrapper.

Document:

- group sampling
- reward computation
- normalization
- relative advantages
- policy objective
- KL handling
- batching

Use third-party implementations as references where appropriate, but do not hide research-critical behavior.

---

# 19. ORBIT ALGORITHM

Do not prematurely declare ORBIT to be a novel algorithm.

The initial hypothesis is:

> Adaptive self-generated curricula can improve sample efficiency and generalization for long-horizon language-agent reinforcement learning.

The research process is:

```text
Hypothesis
 ↓
Baseline
 ↓
Implementation
 ↓
Experiment
 ↓
Measurement
 ↓
Analysis
 ↓
Modification
 ↓
Ablation
 ↓
Conclusion
```

Only experimentally supported mechanisms should become formal ORBIT contributions.

---

# 20. CURRICULUM ENGINE

The curriculum engine is a central research component.

It should eventually reason about:

- task difficulty
- success probability
- novelty
- diversity
- learning frontier
- transferability
- stagnation

It may:

- select tasks
- generate tasks
- increase difficulty
- decrease difficulty
- reject tasks
- detect collapse
- balance exploration and exploitation

Every curriculum decision should be observable.

---

# 21. SELF-GENERATED TASK PIPELINE

Generated tasks must not automatically enter training.

Use:

```text
Generate
 ↓
Validate
 ↓
Deduplicate
 ↓
Estimate Difficulty
 ↓
Verify
 ↓
Admit
 ↓
Train
```

Protect against:

- malformed tasks
- trivial tasks
- duplicated tasks
- reward leakage
- evaluator exploitation
- curriculum collapse
- distribution collapse

---

# 22. EVALUATION SYSTEM

Training and evaluation must be separated.

Evaluation should include:

- training performance
- validation performance
- held-out performance
- OOD performance
- robustness
- sample efficiency
- compute efficiency
- long-horizon completion
- recovery from failure
- transfer

Do not evaluate only on tasks used to generate the curriculum.

---

# 23. BASELINES

Before claiming improvement, implement appropriate baselines.

Minimum intended comparison:

```text
Baseline A
Fixed task distribution

Baseline B
RL without adaptive curriculum

Baseline C
Static / predefined curriculum

Treatment
ORBIT adaptive curriculum
```

Additional baselines should be added based on relevant research literature.

---

# 24. ABLATIONS

Any claimed component should be ablatable.

Potential ablations include:

- adaptive curriculum
- self-generated tasks
- difficulty estimator
- novelty mechanism
- reward shaping
- memory
- critic
- curriculum selection
- task filtering

A component should not be described as essential without evidence.

---

# 25. EXPERIMENT PROTOCOL

Every meaningful experiment must record:

- experiment ID
- git commit
- model version
- environment version
- config
- dataset version
- seed
- hardware
- software versions
- training steps
- token count where applicable
- GPU hours where measurable
- checkpoints
- metrics
- evaluation results

Experiments should be reproducible from documented commands.

---

# 26. STATISTICS

Do not treat a single successful run as proof.

Where feasible:

- use multiple seeds
- report variance
- use confidence intervals
- calculate effect sizes
- use appropriate statistical tests

Never manufacture significance.

---

# 27. OBSERVABILITY

Training should expose useful metrics such as:

- reward
- success rate
- episode length
- policy loss
- value loss
- entropy
- KL divergence
- learning rate
- throughput
- tokens/sec
- GPU utilization
- GPU memory
- curriculum difficulty
- task distribution

Metrics must have a defined meaning.

Do not create meaningless metrics just for dashboards.

---

# 28. EXPERIMENT TRACKING

W&B should be used where appropriate.

Each run should be linked to:

- configuration
- source commit
- model
- environment
- seed
- artifacts
- metrics

Do not make W&B a hard runtime requirement for basic local tests unless necessary.

---

# 29. REPRODUCIBILITY

Reproducibility is a first-class feature.

Use:

- locked dependencies
- explicit configurations
- seeds
- versioned environments
- model identifiers
- dataset versions
- experiment manifests
- reproducible commands

A researcher should be able to determine how an experiment was produced.

---

# 30. SAFETY AND REWARD HACKING

ORBIT must explicitly investigate:

- reward hacking
- specification gaming
- evaluator exploitation
- unsafe tool usage
- prompt injection where applicable
- distribution shift
- curriculum exploitation
- reward-model exploitation

Never claim safety based on a small test set.

Document limitations.

---

# 31. PERFORMANCE PHILOSOPHY

Do not optimize prematurely.

Use:

```text
Correctness
 ↓
Profiling
 ↓
Identify bottleneck
 ↓
Optimization
 ↓
Benchmark
 ↓
Correctness verification
```

Never claim:

- "10x faster"
- "production scale"
- "SOTA performance"
- "linear scaling"

without measurements.

---

# 32. DEPENDENCY POLICY

Before adding a dependency:

1. Check whether the standard library solves the problem.
2. Check whether an existing dependency already solves it.
3. Check maintenance status.
4. Check compatibility.
5. Check whether the dependency materially improves the system.

Avoid dependency bloat.

---

# 33. TESTING POLICY

Every new feature requires appropriate tests.

Tests should cover:

- normal behavior
- edge cases
- invalid inputs
- failure modes
- interface contracts
- algorithmic invariants
- reproducibility where relevant

Common validation:

```bash
uv run pytest
uv run ruff check .
```

Run relevant frontend checks with Bun when frontend code changes.

---

# 34. FRONTEND TESTING

Frontend changes must preserve:

- TypeScript correctness
- API contract correctness
- accessibility
- responsive behavior
- loading states
- error states
- empty states

Use the existing project tooling rather than introducing unnecessary test frameworks.

---

# 35. DOCUMENTATION

Major systems require documentation.

Document:

- purpose
- architecture
- public interfaces
- configuration
- examples
- tests
- limitations

Research algorithms require mathematical documentation.

Architecture changes should use ADRs when appropriate.

---

# 36. GIT POLICY

Use small, logical commits.

Examples:

```text
feat: add adaptive curriculum interface
feat: implement PPO baseline
feat: add trajectory persistence
fix: correct advantage normalization
test: validate curriculum sampling
docs: document experiment protocol
```

Never commit:

- secrets
- API keys
- credentials
- unnecessary model weights
- huge datasets
- local virtual environments
- temporary files

---

# 37. SECRET MANAGEMENT

Never hardcode:

- API keys
- W&B credentials
- cloud credentials
- database passwords
- tokens

Use environment variables.

Provide `.env.example` where necessary.

Never print secrets.

---

# 38. IMPLEMENTATION PHASES

The intended progression is:

## Phase 0 — Foundation

- repository structure
- configuration
- logging
- reproducibility
- testing
- base interfaces

## Phase 1 — Environment Framework

- environment API
- first rigorous environment
- verification
- trajectory representation

## Phase 2 — Baseline Agents

- model interface
- agent runtime
- baseline policy
- rollout collection

## Phase 3 — RL Baselines

- PPO
- GRPO
- baseline experiments

## Phase 4 — Adaptive Curriculum

- task difficulty
- task selection
- curriculum state
- curriculum metrics

## Phase 5 — Self-Generated Tasks

- task generation
- validation
- deduplication
- verification
- admission

## Phase 6 — ORBIT Research Mechanism

- candidate mechanism
- controlled experiment
- comparison
- ablation

## Phase 7 — Long-Horizon Agents

- memory
- tools
- multi-step planning
- recovery

## Phase 8 — Distributed Training

- multi-GPU
- distributed rollout
- scalable experiment execution

## Phase 9 — Evaluation

- benchmarks
- OOD evaluation
- robustness
- ablations
- statistical analysis

## Phase 10 — Safety Research

- reward hacking
- specification gaming
- evaluator exploitation
- adversarial curriculum behavior

## Phase 11 — Research Release

- reproducibility package
- documentation
- benchmark suite
- paper artifacts

## Phase 12 — Research Console

- Next.js UI
- experiment explorer
- live training
- trajectories
- evaluations
- curriculum visualization
- model registry
- system metrics

Agents MUST NOT skip directly to later phases because they are visually impressive.

---

# 39. PHASE COMPLETION

A phase is complete only when appropriate:

- implementation exists
- tests exist
- documentation exists
- validation passes
- reproducible example/experiment exists
- limitations are documented

"Code exists" does not mean "phase complete."

---

# 40. AI CODING AGENT PROTOCOL

Every coding agent MUST follow:

```text
READ
 ↓
INSPECT
 ↓
UNDERSTAND
 ↓
PLAN
 ↓
IMPLEMENT
 ↓
TEST
 ↓
REVIEW
 ↓
DOCUMENT
 ↓
REPORT
```

Before modifying code, identify:

- target subsystem
- current behavior
- desired behavior
- architectural constraints
- relevant tests
- potential regressions

---

# 41. AI RESEARCH AGENT PROTOCOL

Research agents must distinguish between:

- established fact
- repository fact
- experimental observation
- hypothesis
- inference
- speculation

Never convert a hypothesis into a fact.

When proposing a research direction, provide:

1. Hypothesis
2. Motivation
3. Mechanism
4. Baseline
5. Experiment
6. Metrics
7. Expected outcomes
8. Failure criteria
9. Ablations
10. Interpretation

---

# 42. TASK EXECUTION CONTRACT

When an agent receives a coding task:

### Step 1

Read `MASTER.md`.

### Step 2

Inspect relevant files.

### Step 3

Identify the smallest correct implementation.

### Step 4

Plan the changes.

### Step 5

Implement.

### Step 6

Add or update tests.

### Step 7

Run validation.

### Step 8

Inspect the final diff.

### Step 9

Check architectural boundaries.

### Step 10

Update documentation if behavior changed.

### Step 11

Report:

```text
Implemented:
- ...

Files changed:
- ...

Tests:
- ...

Validation:
- ...

Known limitations:
- ...

Next recommended step:
- ...
```

---

# 43. FAILURE HANDLING

If an implementation fails:

1. Preserve the failure information.
2. Determine the root cause.
3. Avoid masking the problem.
4. Fix the underlying issue where appropriate.
5. Add a regression test.
6. Document the lesson if it affects architecture or research.

Do not simply weaken tests to make them pass.

---

# 44. ARCHITECTURAL DECISIONS

Significant architectural decisions should be recorded in:

```text
docs/decisions/
```

Use ADR-style documents when a decision affects:

- system boundaries
- dependencies
- data formats
- training architecture
- experiment methodology
- infrastructure
- security

An ADR should explain:

- context
- decision
- alternatives
- rationale
- consequences

---

# 45. SECURITY PRINCIPLES

Treat:

- model output
- generated code
- uploaded files
- external tool output
- task definitions
- user-provided prompts

as untrusted input.

Validate boundaries.

Sandbox code execution.

Do not expose credentials to agents.

Do not allow arbitrary host-level execution.

---

# 46. DATA PROVENANCE

Important datasets and generated task sets should have:

- identifiers
- versions
- provenance
- creation metadata
- validation status

Avoid silently modifying datasets used by experiments.

---

# 47. MODEL PROVENANCE

Record:

- model identifier
- model version/revision
- tokenizer version
- configuration
- quantization settings if applicable
- adapter configuration
- training checkpoint

Model artifacts must be traceable to experiments.

---

# 48. NO FAKE COMPLETENESS

Agents must not create placeholder implementations and label them production-ready.

Acceptable:

```python
raise NotImplementedError("ORBIT curriculum admission is not implemented yet")
```

when documented and appropriate.

Unacceptable:

```python
return {"success": True}
```

when the real behavior has not been implemented.

---

# 49. NO FAKE BENCHMARKS

Do not create benchmark numbers manually.

Benchmark output must originate from actual execution.

Store:

- benchmark configuration
- hardware
- software
- dataset
- repetitions
- raw measurements

---

# 50. NO FAKE RESEARCH

Never write:

> ORBIT improves generalization by 27%.

unless an actual experiment established it.

Use:

> Preliminary experiment: ...

when appropriate.

Research claims must trace back to experiment artifacts.

---

# 51. MINIMALITY PRINCIPLE

The best implementation is not the largest implementation.

Prefer:

```text
small correct system
       ↓
validated experiment
       ↓
measured bottleneck
       ↓
targeted improvement
```

over:

```text
huge architecture
       ↓
untested complexity
       ↓
unclear results
```

---

# 52. CURRENT PRIORITY

Unless another explicit project phase is already active, prioritize:

```text
1. Establish repository correctness.
2. Establish reproducibility.
3. Establish environment abstraction.
4. Build one rigorous environment.
5. Build baseline agent.
6. Build baseline rollout system.
7. Establish evaluation.
8. Implement PPO/GRPO baseline.
9. Run controlled experiments.
10. Only then expand the adaptive curriculum research.
```

Do not start with the dashboard.

Do not start with distributed infrastructure.

Do not start with Kubernetes.

Do not start with elaborate model serving.

The research loop comes first.

---

# 53. DEFINITION OF SUCCESS

ORBIT succeeds if it eventually produces:

1. A technically sound language-agent RL framework.
2. Reproducible experiments.
3. Strong baselines.
4. A defensible adaptive curriculum mechanism.
5. Evidence for or against the research hypotheses.
6. Robust evaluation.
7. Meaningful ablations.
8. Scalable training experiments.
9. Reward-hacking and safety analysis.
10. A reproducible public artifact.
11. A technically credible research paper.

The goal is not merely to build software.

The goal is to produce evidence.

---

# 54. FINAL PRINCIPLE

When choosing between:

### Impressive complexity

and

### Scientifically useful simplicity

Choose scientifically useful simplicity.

When choosing between:

### Convenient shortcuts

and

### Reproducibility

Choose reproducibility.

When choosing between:

### Hiding failure

and

### Documenting failure

Choose documentation.

When choosing between:

### Another feature

and

### A better experiment

Choose the better experiment.

Build ORBIT as if every line of code will be reviewed by another researcher, every experiment will be reproduced, every result will be challenged, and every claim will be tested for falsifiability.

That is the standard.
