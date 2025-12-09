## GSM8K (n=10) — Orchestrated Pipeline Comparison (OpenRouter)

- Date: 2025-12-05
- Pipeline: `apps/mas/graph/plan_graph.py` (supervisor+workers), numeric-only post-processing for single-number tasks
- Eval script: `apps/mas/scripts/run_gsm8k_baseline.py`

### Models and Configs

- Meta Llama 3.1 70B Instruct — `apps/mas/configs/openrouter.yaml` (model: `meta-llama/llama-3.1-70b-instruct`)
- GPT‑4o — `apps/mas/configs/openrouter_gpt4o.yaml` (model: `openai/gpt-4o`)
- Claude 3.5 Sonnet — `apps/mas/configs/openrouter_claude35.yaml` (model: `anthropic/claude-3.5-sonnet`)
- Grok — `apps/mas/configs/openrouter_grok.yaml` (model: attempted `x-ai/grok-2`, `x-ai/grok-2-mini`)

### Results (n=10)

| Model | Accuracy | Command |
|---|---:|---|
| meta‑llama/llama‑3.1‑70b‑instruct | 9/10 = 90.00% | `python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter.yaml` |
| openai/gpt‑4o | 4/10 = 40.00% | `python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter_gpt4o.yaml` |
| anthropic/claude‑3.5‑sonnet | 5/10 = 50.00% | `python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter_claude35.yaml` |
| x‑ai/grok (2 / 2‑mini) | Not evaluated (API 404: no endpoints) | `python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter_grok.yaml` |

Notes:
- Grok runs failed with OpenRouter errors: “No endpoints found for x-ai/grok‑2/‑mini”; likely account/model access issue.
- All runs used temperature=0.0 and the same orchestrator prompts.

### Observations

- On this 10‑sample slice, the Llama 3.1 70B orchestrated pipeline performed best (90%). GPT‑4o and Claude 3.5 Sonnet underperformed relative to this configuration/sample (40–50%). Differences can stem from small sample variance, OpenRouter routing/provider, or model alignment with our system prompts.
- Numeric enforcement ensured clean single‑number outputs for GSM8K‑style questions.

### Reproduction

```bash
source .venv/bin/activate
python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter.yaml
python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter_gpt4o.yaml
python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter_claude35.yaml
python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter_grok.yaml   # may 404 with current API access
```

