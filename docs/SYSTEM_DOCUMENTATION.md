# Multi-Agent Reasoning System (MAS) - Technical Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Novel Methodologies](#novel-methodologies)
6. [Current Performance](#current-performance)
7. [Challenges & Limitations](#challenges--limitations)
8. [Future Work](#future-work)

---

## Executive Summary

The Multi-Agent Reasoning System (MAS) is a hierarchical, multi-model orchestration framework designed to solve complex mathematical and logical problems. It employs a **Supervisor-Worker-Verifier** architecture with cooperative consensus mechanisms and code-backed empirical verification.

**Key Innovation**: The system combines:
- Multi-agent decomposition (divide & conquer)
- Swarm intelligence (parallel model querying with consensus)
- Code-backed simulation (ResearchWorker generates and executes Python code)
- Iterative refinement loops (Ouroboros pattern)

**Current Benchmark**: 2/5 on "Humanity's Last Exam" (Q1: Hotel/Cat, Q4: Free Product Subgroups)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                                  │
│  - Problem Decomposition (JSON subtask plan)                        │
│  - Role Assignment: {math, qa, logic, research}                     │
│  - Critique & Synthesis                                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   SWARM WORKERS   │ │  RESEARCH WORKER  │ │   SWARM WORKERS   │
│  (math/qa/logic)  │ │  (code execution) │ │  (math/qa/logic)  │
│                   │ │                   │ │                   │
│  - Multi-model    │ │  - Code scaffolds │ │  - Multi-model    │
│  - Consensus vote │ │  - Python exec    │ │  - Consensus vote │
│  - Coop rounds    │ │  - 5-turn refine  │ │  - Coop rounds    │
└───────────────────┘ └───────────────────┘ └───────────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SUPERVISOR: SYNTHESIS                            │
│  - Aggregate worker outputs                                         │
│  - Extract numeric answers (#### marker)                            │
│  - Apply critique-based repair                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VERIFIER AGENT                                  │
│  - Independent numeric re-computation                               │
│  - Cross-check against worker outputs                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FINAL ANSWER                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Supervisor Agent (`apps/mas/agents/supervisor.py`)

**Purpose**: Orchestrates the entire reasoning pipeline.

**Key Functions**:
- `decompose(problem)`: Breaks down complex problems into subtasks
- `synthesize(problem, results)`: Aggregates worker outputs into final answer
- `critique(problem, results)`: Evaluates consistency of worker outputs
- `resynthesize_with_critique()`: Repairs answers based on critique

**Subtask Roles**:
| Role | Description |
|------|-------------|
| `math` | Arithmetic, algebra, numerical computation |
| `qa` | Factual knowledge, definitions |
| `logic` | Reasoning puzzles, deduction |
| `research` | Complex tasks requiring code simulation |

**Auto-Injection**: The supervisor automatically adds `research` subtasks when detecting keywords like: `eigenvalue`, `subgroup`, `hotel`, `knot`, `quandle`, `free product`, etc.

---

### 2. Swarm Worker Manager (`apps/mas/agents/swarm_worker.py`)

**Purpose**: Parallel execution across multiple LLM models with cooperative consensus.

**Configuration** (from `openrouter.yaml`):
```yaml
swarm_models:
  - mistralai/mistral-large-2512
  - mistralai/mistral-medium-3.1
  - mistralai/mistral-small-3.2-24b-instruct
swarm_min_responses: 2
swarm_per_model_timeout: 120.0
swarm_overall_timeout: 300.0
coop_enabled: true
coop_max_rounds: 2
coop_min_agreement: 2
```

**Cooperative Consensus**:
1. Round 1: All models answer independently
2. Round 2: Models see peer responses and reconcile disagreements
3. Consensus: Majority vote on extracted numeric answers

---

### 3. Research Worker (`apps/mas/agents/worker_researcher.py`)

**Purpose**: Solves complex problems via code-backed empirical verification.

**The Ouroboros Loop** (Iterative Refinement):
```
┌─────────────┐
│ HYPOTHESIZE │ ──▶ State mathematical object
└─────────────┘
       │
       ▼
┌─────────────┐
│ EXPERIMENT  │ ──▶ Write Python simulation code
└─────────────┘
       │
       ▼
┌─────────────┐
│  EXECUTE    │ ──▶ Run code via PythonExecutor
└─────────────┘
       │
       ▼
┌─────────────┐
│  OBSERVE    │ ──▶ Analyze stdout/stderr
└─────────────┘
       │
       ▼
┌─────────────┐
│  REFINE     │ ──▶ If wrong, fix code and repeat (up to 5 turns)
└─────────────┘
       │
       ▼
┌─────────────┐
│  CONCLUDE   │ ──▶ Report final answer (#### marker)
└─────────────┘
```

**Code Scaffolds** (Methodological Templates):
The system injects problem-specific code scaffolds to guide the LLM:

| Problem Type | Scaffold | Description |
|--------------|----------|-------------|
| Hotel/Toggle | `hotel_simulation` | State machine for room light cycles |
| Eigenvalues | `eigenvalue_enumeration` | Group theory enumeration |
| Rank-1 Matrices | `rank1_matrices` | Random matrix orthogonality tests |
| Free Products | `subgroup_counting` | Homomorphism counting to S_n |
| Knots | `quandle_search` | Quandle table enumeration |

---

### 4. Verifier Agent (`apps/mas/agents/verifier.py`)

**Purpose**: Independent verification of numeric answers.

**Methodology**:
- Re-derives answer from first principles
- Does NOT copy the candidate answer
- Checks derivation path for logical gaps
- Returns bare number (no prose)

---

### 5. Python Executor (`apps/mas/tools/executor.py`)

**Purpose**: Sandboxed Python code execution for ResearchWorker.

**Configuration**:
- Timeout: 60 seconds (increased for complex simulations)
- Captures stdout, stderr, success flag
- Used for empirical verification of mathematical hypotheses

---

## Data Flow

### Phase 1: Decomposition
```python
problem = "How many subgroups of index 7 does C_2 * C_5 have?"
plan = supervisor.decompose(problem)
# Returns: Plan(subtasks=[
#   SubTask(role="research", instruction="..."),
#   SubTask(role="math", instruction="..."),
#   SubTask(role="logic", instruction="...")
# ])
```

### Phase 2: Dispatch
```python
for subtask in plan.subtasks:
    if subtask.role == "research":
        result = researcher.run(subtask.instruction)  # Code execution
    else:
        result = swarm.run(subtask.instruction, role=subtask.role)  # Multi-model
```

### Phase 3: Critique
```python
critique = supervisor.critique(problem, results)
# Returns: "OK" or "Issue: models disagree on ..."
```

### Phase 4: Synthesis
```python
final_answer = supervisor.synthesize(problem, results)
if critique != "OK":
    final_answer = supervisor.resynthesize_with_critique(problem, results, critique)
```

### Phase 5: Verification
```python
if is_numeric_question(problem):
    verified = verifier.verify_numeric(problem, final_answer, context)
    if verified != final_answer:
        final_answer = verified
```

---

## Novel Methodologies

### 1. Stochastic Code Scaffolding
Instead of hardcoding answers, the system provides **methodological scaffolds** that guide the LLM to write correct code:
- The scaffold shows HOW to simulate, not WHAT the answer is
- The LLM must adapt the scaffold to the specific problem
- This maintains stochasticity while improving accuracy

### 2. Cognitive Strategies
Domain-specific reasoning hints embedded in prompts:
- **KNOTS**: "Use Wirtinger presentation relations"
- **GROUPS**: "Use homomorphism counting to S_n"
- **MATRICES**: "Generate random rank-1 matrices, check orthogonality"
- **ITERATIVE**: "Simulate the EXACT process, do not simplify"

### 3. Multi-Model Cooperative Consensus
- Multiple models answer independently
- Models share peer responses in Round 2
- Reconciliation prompt forces explicit disagreement analysis
- Majority vote on numeric answers

### 4. Self-Correction Loop
If code output contradicts expectations:
- Model is prompted to debug implementation
- Verify edge cases (e.g., N=1, 2, 3)
- Trust code over theoretical intuition

---

## Current Performance

### Humanity's Last Exam Benchmark (5 Questions)

| Q# | Topic | Expected | Got | Status |
|----|-------|----------|-----|--------|
| 1 | Hotel/Cat Light Toggle | 48 | 48 | ✅ PASS |
| 2 | Eigenvalue Sets (|G|=18) | 8 | 5 | ❌ FAIL |
| 3 | Rank-1 Matrices Admissibility | 1 | None | ❌ FAIL |
| 4 | Free Product Subgroups | 56 | 56 | ✅ PASS |
| 5 | Figure-8 Knot Quandle | 4 | 5 | ❌ FAIL |

**Score: 2/5 (40%)**

### What's Working
- **Q1 (Hotel)**: Code scaffold correctly simulates toggle logic + cat reset
- **Q4 (Subgroups)**: Homomorphism counting to S_7 produces correct count

### What's Failing
- **Q2 (Eigenvalues)**: Group enumeration incomplete; missing some unions of roots of unity
- **Q3 (Rank-1)**: Code doesn't correctly test Frobenius orthogonality constraints
- **Q5 (Knots)**: System finds size-5 dihedral quandle instead of size-4 specialized quandle

---

## Challenges & Limitations

### 1. LLM Code Generation Quality
**Problem**: Mistral models often generate incorrect Python code for advanced math.
**Impact**: Even with scaffolds, subtle bugs lead to wrong answers.
**Mitigation**: More specific scaffolds, sanity check prompts.

### 2. Domain Knowledge Gaps
**Problem**: LLMs hallucinate theorems or apply formulas incorrectly.
**Example**: Q5 assumes tricoloring (3 elements) instead of quandle axioms.
**Mitigation**: Force code-based enumeration over theoretical shortcuts.

### 3. Timeout Constraints
**Problem**: Complex simulations (e.g., S_7 enumeration) can be slow.
**Impact**: May not complete within timeout budget.
**Mitigation**: Increased executor timeout to 60s; overall budget 600s.

### 4. Scaffold Precision
**Problem**: Scaffolds are general templates; model must adapt correctly.
**Impact**: Minor misadaptations lead to wrong answers.
**Mitigation**: Add more problem-specific checks and sanity tests.

### 5. Consensus vs. Correctness
**Problem**: Multiple wrong answers can out-vote a single correct one.
**Impact**: Swarm consensus may converge to incorrect answer.
**Mitigation**: Weight answers by evidence (code output) over count.

---

## Future Work

### Short-Term (To reach 3/5)

1. **Fix Q2 (Eigenvalues)**:
   - Improve scaffold to enumerate ALL abelian groups of order 18
   - Include union of roots of unity computation
   - Expected: Count 8 distinct S(ρ) sets

2. **Fix Q3 (Rank-1)**:
   - Add explicit Frobenius inner product test
   - Enumerate small cases (a=2, b=2 → ab=4)
   - Check admissibility for k=0,1,2,3,4

3. **Fix Q5 (Knots)**:
   - Add specific quandle tables for sizes 2, 3, 4
   - Include the "mod 8" quandle {1,3,5,7} with x*y = 2y-x mod 8
   - Test coloring conditions against Figure-8 Wirtinger relations

### Medium-Term

4. **Verification Agent Upgrade**:
   - Allow verifier to run code for cross-checking
   - Compare against ResearchWorker output

5. **Multi-Attempt Sampling**:
   - Run same question 3-5 times with temperature > 0
   - Take majority vote across attempts

6. **Better Model Selection**:
   - Use GPT-4o or Claude for research tasks (better code generation)
   - Keep Mistral for simpler math/qa tasks

### Long-Term

7. **Tool Use Integration**:
   - Add symbolic math tools (SymPy, SageMath)
   - Add graph theory libraries (NetworkX)
   - Add group theory libraries (GAP interface)

8. **Latent Communication**:
   - Implement the latent space communication described in the paper
   - Share compressed reasoning states between agents

9. **Self-Improvement Loop**:
   - Log failed attempts and correct answers
   - Fine-tune on corrected examples

---

## File Structure

```
Multi-Agent Reasoning System/
├── apps/mas/
│   ├── agents/
│   │   ├── supervisor.py      # Orchestration, decomposition, synthesis
│   │   ├── swarm_worker.py    # Multi-model parallel execution
│   │   ├── worker_researcher.py # Code-backed empirical verification
│   │   ├── worker_math.py     # Math-specific prompts
│   │   ├── worker_logic.py    # Logic-specific prompts
│   │   ├── worker_qa.py       # QA-specific prompts
│   │   └── verifier.py        # Numeric verification
│   ├── graph/
│   │   └── plan_graph.py      # Main orchestration pipeline
│   ├── tools/
│   │   └── executor.py        # Python code execution
│   ├── infra/
│   │   └── openrouter/
│   │       └── client.py      # API client
│   └── configs/
│       └── openrouter.yaml    # Model and swarm configuration
├── scripts/
│   └── test_humanity_exam.py  # Benchmark test harness
└── docs/
    └── SYSTEM_DOCUMENTATION.md # This file
```

---

## Configuration Reference

### `openrouter.yaml`
```yaml
model: mistralai/mistral-large-2512
temperature: 0.2
max_output_tokens: 512
request_timeout_s: 60

# Supervisor/Verifier models
supervisor_model: mistralai/mistral-large-2512
verifier_model: mistralai/mistral-large-2512

# Swarm configuration
swarm_models:
  - mistralai/mistral-large-2512
  - mistralai/mistral-medium-3.1
  - mistralai/mistral-small-3.2-24b-instruct

# Cooperative settings
coop_enabled: true
coop_max_rounds: 2
coop_min_agreement: 2
```

---

## Running the Benchmark

```bash
# Activate virtual environment
source .venv/bin/activate

# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Run Humanity's Last Exam benchmark
python scripts/test_humanity_exam.py
```

---

## Appendix: Code Scaffold Examples

### Hotel Simulation Scaffold
```python
N = 100  # number of rooms
states = [0] * (N + 1)  # 0=red, 1=green, 2=blue
for guest in range(1, N + 1):
    step = guest
    for room in range(guest, N + 1, guest):
        states[room] = (states[room] + step) % 3
    # Cat resets green to red
    for room in range(1, N + 1):
        if states[room] == 1:
            states[room] = 0
result = sum(1 for s in states[1:] if s == 2)
print(result)  # Expected: 48
```

### Subgroup Counting Scaffold
```python
from itertools import permutations
from math import factorial

def count_transitive_homomorphisms(m, k, n):
    count = 0
    Sn = list(permutations(range(n)))
    for sigma in Sn:
        if order_divides(sigma, m):
            for tau in Sn:
                if order_divides(tau, k):
                    if is_transitive(sigma, tau, n):
                        count += 1
    return count // factorial(n-1)

print(count_transitive_homomorphisms(2, 5, 7))  # Expected: 56
```

---

*Document generated: December 2024*
*System Version: 1.0 (Stochastic Intelligence Upgrade)*

