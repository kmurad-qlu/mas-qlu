# MAS-qlu

Scalable, hierarchical multi-agent reasoning system with strict output policies, per-role model control, and an optional latent communication pipeline.

- Orchestrator: LangGraph state machine (supervisor → math → QA → logic → critic → synthesize)
- Agents: Supervisor (decompose/synthesize/critique), Workers (math/qa/logic), Verifier (numeric check)
- LLM IO: OpenAI-compatible client via OpenRouter (per-role model overrides supported)
- UI: Gradio chat
- Benchmarks: GSM8K, HotpotQA, BBH boolean
- Latent pipeline: HF local model flow with hidden→embedding projection and KV-transfer

## Contents

- apps/mas/graph/plan_graph.py: Orchestrator graph and state
- apps/mas/agents/: Supervisor, workers, verifier
- apps/mas/infra/: OpenRouter client, env helpers, HF runner
- apps/mas/web/chat_ui.py: Gradio interface
- apps/mas/benchmarks/: GSM8K, HotpotQA, BBH runners
- apps/mas/latent/: Latent comms (alignment, projection, KV transfer)
- apps/mas/configs/: OpenRouter YAMLs (model choices, per-role overrides)

## Methodology

- Hierarchical decomposition: Supervisor converts a user problem into a JSON Plan of SubTasks (roles: math, qa, logic).
- Specialized workers: Each role uses a tailored system prompt and decoding settings.
- Critique loop: A critic pass evaluates consistency; a repair synthesis may run if issues are found.
- Strict output policy:
  - Single-number questions → return exactly the number (no prose).
  - Multi-quantity questions → compact JSON (snake_case keys, numeric values).
  - Yes/No → “yes” or “no”; boolean expressions → “T” or “F”.
- Numeric extraction and guarding:
  - Prefer ‘#### <number>’ from MathWorker; otherwise safely extract a single unique number; avoid guessing when multiple numbers exist.
- Latent communication (optional):
  - Learn a linear map W to project last hidden states to input embeddings; continue generation using KV cache and projected “latent seed” embeddings (HF local models).

## Architecture

### Orchestrator (LangGraph)
- State: `GraphState(problem, plan, results, critique_note, final_answer[, thinking_log])`
- Nodes/flow:
  1) supervisor.decompose → Plan(JSON)
  2) dispatch math/qa/logic → collect `(SubTask, result)`
  3) supervisor.critique → short note or “OK”
  4) supervisor.synthesize → final answer
  5) Enforcers:
     - If single-number question, coerce to bare number (only if unambiguous)
     - If multi-answer, accept JSON and render as lines
  6) Optional: verifier.verify_numeric to correct numeric outputs

### Agents
- Supervisor: `decompose`, `synthesize`, `critique`, `resynthesize_with_critique`; supports per-call model overrides and fallbacks.
- Workers:
  - MathWorker: emits “#### <number>”
  - QAWorker: concise factual answers
  - LogicWorker: terse logical conclusions
- Verifier: second-opinion numeric checker.

### Infra
- OpenRouterClient: OpenAI chat-completions with retries, timeouts, usage tracking.
- Env helpers: `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`), `OPENAI_BASE_URL`, referer/title headers.
- HF runner: tokenize, hidden states, KV caching, sampled/greedy continuation.

## Models and configuration

Central config (choose one):
- `apps/mas/configs/openrouter.yaml` (default)
- `apps/mas/configs/openrouter_grok.yaml`
- `apps/mas/configs/openrouter_gpt4o.yaml`
- `apps/mas/configs/openrouter_claude35.yaml`

Typical keys:
model: openai/gpt-5.1
temperature: 0.0
top_p: 0.9
max_output_tokens: 1024
request_timeout_s: 120
# Optional per-role overrides and fallbacks
worker_model: openai/o3-mini
supervisor_fallback_model: x-ai/grok-4
worker_fallback_model: x-ai/grok-4.1-fast
supervisor_secondary_fallback_model: openai/gpt-4o
worker_secondary_fallback_model: openai/gpt-4oNotes:
- All agents share the same OpenRouter client; each call can override `model` at runtime.
- If no model is provided in YAML, a safe default (e.g., Mixtral) is used.

## Getting started

### Requirements
- Python 3.10+
- A valid OpenRouter/OpenAI-style API key

### Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment
cp apps/mas/env.example .env   # then edit with your values
# or export vars directly:
# export OPENROUTER_API_KEY=...
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1### Run the chat UI
python -m apps.mas.web.chat_ui \
  --config apps/mas/configs/openrouter.yaml \
  --server-name 127.0.0.1 \
  --server-port 7860Open `http://127.0.0.1:7860`.

## Benchmarks

GSM8K (OpenRouter baseline):
python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter.yamlHotpotQA:
python -m apps.mas.scripts.run_hotpotqa_baseline --n 5 --config apps/mas/configs/openrouter.yamlBBH Boolean:
python -m apps.mas.scripts.run_bbh_baseline --n 5 --config apps/mas/configs/openrouter.yaml## Latent pipeline (HF local)

Quick demo on GSM8K with latent seeding:
python -m apps.mas.scripts.run_gsm8k_latent \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --alignment apps/mas/latent/W_centered.npz \
  --n 10 --latent_steps 8 --max_new_tokens 64Core steps:
1) Encode prompt and run forward to collect last hidden state + KV
2) Project hidden → input embeddings with W (mean-centered, normalized)
3) Step embeddings through model using `past_key_values`
4) Continue generation (sample first tokens then greedy)

## Multi-answer behavior

- If the prompt requests multiple quantities, the supervisor returns a compact JSON (e.g., `{"total_now": 18, "per_friend": 4, "remainder": 2}`).
- The graph renders that JSON as lines for readability.
- Single-number prompts return a bare number line only.

## Extending

- New worker:
  - Add `apps/mas/agents/worker_<name>.py` with a concise persona and `run()`
  - Instantiate and wire a dispatch node in `plan_graph.py`
  - Update supervisor decomposition prompt to emit the new role
- Per-role models:
  - Set `worker_model` / `supervisor_*_model` in YAML
  - or pass `model=...` on agent calls

## Security

- Do not commit real `.env` or API keys.
- `.env` is loaded from project root or app-level `.env`; falls back to `env.example`.
- Consider Git LFS for large artifacts (e.g., `.npy`, `.npz`, `.pt`, `.bin`, `.pdf`).

## License

Add a license (MIT/Apache-2.0) to clarify usage.
