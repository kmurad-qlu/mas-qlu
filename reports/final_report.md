# MAS Baselines vs Latent (Initial Snapshot)

- Environment: OpenRouter (Mixtral 8x7B Instruct) for baseline; TinyLlama 1.1B for latent experiments.
- Config: `apps/mas/configs/openrouter.yaml`
- Scripts:
  - GSM8K baseline: `python -m apps.mas.scripts.run_gsm8k_baseline --n 10 --config apps/mas/configs/openrouter.yaml`
  - GSM8K latent: `python -m apps.mas.scripts.run_gsm8k_latent --n 10`
  - HotpotQA baseline: `python -m apps.mas.scripts.run_hotpotqa_baseline --n 3 --config apps/mas/configs/openrouter.yaml`
  - BBH baseline: `python -m apps.mas.scripts.run_bbh_baseline --n 3 --config apps/mas/configs/openrouter.yaml`

## Results

- GSM8K (baseline / OpenRouter, Mixtral 8x7B): 3 / 10 = 30.00%
- GSM8K (baseline / OpenRouter, Llama 3.1 70B + strict prompts): 4 / 10 = 40.00%
- GSM8K (latent-hybrid / TinyLlama): 0 / 10 = 0.00%
- HotpotQA (baseline / OpenRouter, Llama 3.1 70B): 1 / 3 = 33.33%
- BBH Boolean (baseline / OpenRouter, Llama 3.1 70B): 3 / 3 = 100.00%

## Notes

- Baseline is functional with Supervisor v2 (runtime critique) and text-only messaging via LangGraph.
- Switching the OpenRouter route to Llama 3.1 70B and enforcing '#### <answer>' for math improved GSM8K to 40% on a quick 10-sample run. BBH boolean hit 100% on 3 samples.
- Latent prototype demonstrates W alignment training and KV transfer, but accuracy is low with a tiny local model; expect improvements using a stronger local model and more alignment data.
- Next steps:
  - Improve supervisor prompts and worker personas.
  - Increase alignment data and evaluate alternative projection heads (e.g., MLP).
  - Explore using a 7B local model for latent experiments if compute allows.

