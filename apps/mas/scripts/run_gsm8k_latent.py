from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
import torch

from ..infra.hf_runner import HFRunner, HFRunnerConfig
from ..latent.kv_transfer import export_pkv_from_prompt, continue_with_pkv
from ..latent.wa_train import load_alignment
from ..latent.latent_io import project_hidden_to_inputs_embeds
from ..eval.metrics import exact_or_numeric_match


PROMPT_TEMPLATE = "You are a helpful math solver. Solve concisely.\n\nQuestion:\n{q}\n\nAnswer:"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM8K with hybrid latent pipeline (local HF)")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--alignment", type=str, default=str(Path(__file__).resolve().parents[2] / "latent" / "W_centered.npz"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--latent_steps", type=int, default=8, help="How many latent embeddings to inject")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    cfg = HFRunnerConfig(model_name_or_path=args.model, load_in_4bit=False, device_map="auto", torch_dtype="auto")
    runner = HFRunner(cfg)
    W, mean_H, mean_E, embed_norm = load_alignment(args.alignment)

    ds = load_dataset("gsm8k", "main", split="test")
    correct = 0

    for i in range(min(args.n, len(ds))):
        q = ds[i]["question"]
        gold = ds[i]["answer"]
        prompt = PROMPT_TEMPLATE.format(q=q)

        # Export last hidden and PKV from the prompt
        enc = runner.encode(prompt)
        out = runner.forward_with_hidden(enc["input_ids"], enc.get("attention_mask"))
        last_hidden = out["last_hidden"]  # [1, seq, hidden]
        pkv = out["past_key_values"]

        # Inject K latent steps using the last K hidden vectors projected to input embeddings
        seq_len = last_hidden.shape[1]
        k = min(args.latent_steps, seq_len)
        latent_chunk = last_hidden[:, seq_len - k : seq_len, :]  # [1, k, hidden]
        latent_embeds = project_hidden_to_inputs_embeds(latent_chunk, W, mean_H, mean_E, embed_norm)  # [1, k, embed]
        for t in range(k):
            step_embed = latent_embeds[:, t : t + 1, :]  # [1,1,embed]
            step_out = runner.step_inputs_embeds(step_embed, pkv)
            pkv = step_out["past_key_values"]

        # Continue generation using sampling for stability
        seed_id = int(enc["input_ids"][0, -1].item())
        eos_id = runner.tokenizer.eos_token_id
        pred_text = runner.generate_sampled_from_pkv(
            past_key_values=pkv,
            seed_token_id=seed_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            eos_token_id=eos_id,
            greedy_after=8,
        )
        ok = exact_or_numeric_match(pred_text, gold)
        correct += int(ok)
        print(f"[{i+1}] OK={ok} | pred={pred_text!r} | gold={gold!r}")

    print(f"Accuracy (latent hybrid): {correct}/{min(args.n, len(ds))} = {correct/min(args.n, len(ds)):.2%}")


if __name__ == "__main__":
    main()

