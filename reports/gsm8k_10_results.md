## GSM8K (n=10) Benchmarks - Current Framework

- Date: 2025-12-05
- Config: `apps/mas/configs/openrouter.yaml` (model: `meta-llama/llama-3.1-70b-instruct`), numeric-only output enforced

### Results

- Orchestrator (OpenRouter) baseline:
  - Accuracy: 8/10 = 80.00%
  - Command:
    - `python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter.yaml`

- Latent hybrid (local HF) baseline:
  - Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - Alignment: `apps/latent/W.npy` (fallback, W only; zero-centered means; default embed_norm=1.0)
  - Max new tokens: 32
  - Accuracy: 0/10 = 0.00%
  - Command:
    - `python -m apps.mas.scripts.run_gsm8k_latent --n 10 --alignment apps/latent/W.npy --max_new_tokens 32`
  - Note: This alignment file is not model‑specific (W only). A proper centered alignment (`W_centered.npz` with mean_H/mean_E and calibrated embed_norm for the exact model) is typically required; without it, generations degraded to incoherent text.

### Observations

- The orchestrator pipeline with OpenRouter produced strong short‑form numeric answers (single number outputs enforced), achieving 80% on a small sample.
- The latent hybrid baseline requires a correctly trained alignment for the chosen HF model; using a generic `W.npy` yielded non‑informative generations and 0% accuracy.

### Next Steps

- Train and use a model‑matched alignment (`W_centered.npz`) for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (or another local model) to get a meaningful latent baseline.
- Optionally compare against a single‑agent prompt (no decomposition) over OpenRouter to measure the effect of orchestration vs. direct prompting.

