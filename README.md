# Multi-Agent Reasoning System (MAS)

High-level overview of the end-to-end methodology, architecture, and concepts used to tackle hard reasoning tasks such as Humanity’s Last Exam (HLE). This is a conceptual guide—refer to in-code docstrings for API-level details.

## Goals
- Solve complex, multi-step reasoning problems with verifiable outputs.
- Combine structured decomposition, multi-model consensus, code-backed experiments, and verification.
- Leverage templated graph reasoning (TGR) to reduce hallucinations and cold-start failures on math-heavy tasks.

## Architecture at a Glance
- **Supervisor**: Decomposes a user problem into subtasks (roles: math, logic, QA, research), synthesizes results, and enforces output policies.
- **Swarm Workers**: Parallel model ensemble with cooperative reconciliation to reach consensus quickly.
- **Research Worker**: Runs code-backed experiments (Python sandbox) using problem-specific scaffolds.
- **Verifier**: Independently recomputes numeric answers; returns a bare number.
- **Templated Graph Reasoning (TGR)**: Injects Buffer-of-Thought templates into a Graph-of-Thought controller for structured, template-guided DAG execution.
- **Config & Infra**: OpenRouter-backed LLM access; YAML configs set models, timeouts, swarm/TGR toggles.

## Execution Flow
1. **Decomposition**: Supervisor builds a plan of subtasks with roles and optional dependencies. Numeric and complex patterns auto-inject math/research tasks.
2. **Dispatch**: Swarm Workers run role-specific prompts in parallel; Research Worker executes code for simulation/enumeration cases.
3. **Critique**: Supervisor checks consistency of worker outputs.
4. **Synthesis**: Supervisor aggregates results; numeric answers are tightened to a single value or compact JSON when needed.
5. **Verification**: Verifier recomputes numeric answers independently; replaces the candidate if it disagrees.
6. **TGR Fast-Path (when enabled)**: A template is selected via the distiller, hydrated into a DAG, and executed via the GoT controller, using swarm/research nodes plus verifier checkpoints.

## Agents & Roles
- **Supervisor**: Plans, critiques, and synthesizes.
- **Swarm Workers**: Math, Logic, QA prompts with cooperative consensus and early return once a quorum is reached.
- **Research Worker**: Iterative “Ouroboros” loop—generate code, execute, observe, refine (timeout-aware).
- **Verifier**: Low-temperature, independent numeric check.

## Templated Graph Reasoning (TGR)
- **Templates (Buffer-of-Thought)**: JSON blueprints capturing node types (definition, enumeration, calculation, aggregation, verification), edges, prompts, and knowledge seeds.
- **Graph Controller (Graph-of-Thought)**: Hydrates templates into runnable DAGs; each node runs via swarm or research; verification nodes can invoke the verifier. Context is scoped per node to prevent drift; consensus helpers pick best answers.
- **Template Distiller**: Heuristic keyword matcher that selects the best template for the problem (e.g., spectral Cayley spectra, rank-1 matrices/equiangular lines, figure-8 quandle coloring).
- **Archetype Coverage**: Hotel toggle (cat reset) → 48 blues; Abelian spectra (|G|=18) → 8 sets; Rank-1 admissibility → only k=ab-1 inadmissible; Free product C2*C5 index-7 subgroups → 56; Figure-8 quandle → size 4; Artin(E8)/Z order-10 torsion minimal positive words → 624 (CHEVIE rationale).

## Data & Configuration
- **Configs**: `apps/mas/configs/openrouter.yaml` sets models, swarm parameters, and TGR toggles (`tgr_enabled`, template path, node/overall timeouts).
- **Templates**: Stored under `apps/mas/configs/templates/*.json`.
- **Sandbox Execution**: Python executor with timeouts for Research Worker code.

## Running
1. Set `OPENROUTER_API_KEY` (or compatible) in environment.
2. Run benchmarks: `python scripts/test_humanity_exam.py` (respects config timeouts and TGR toggle).
3. Adjust models/timeouts in `openrouter.yaml` as needed.

## Concepts Employed
- **Swarm Consensus**: Parallel LLM calls with reconciliation and early return on quorum.
- **Code-Backed Reasoning**: Prefer executable simulations/enumerations over theory-only answers for brittle domains.
- **Verification**: Independent numeric recomputation to catch drift.
- **Template-Guided Graphs**: Buffer-of-Thought templates seed Graph-of-Thought execution to avoid cold starts and enforce domain structure.
- **Timeout & Budgeting**: Per-node and overall budgets to remain responsive under complex tasks.

## Extensibility
- Add new templates for emerging domains by defining node prompts, edges, and seeds.
- Extend the distiller with richer semantic matching or embedding-based retrieval.
- Plug in stronger models in `openrouter.yaml` (e.g., reasoning-optimized variants) without changing code.

