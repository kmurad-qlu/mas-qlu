from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from ..infra.hf_runner import HFRunner, HFRunnerConfig
from ..latent.wa_dataset import collect_pairs
from ..latent.wa_train import train_ridge, RidgeConfig, save_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train alignment matrix W (H^L -> E)")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[2] / "latent" / "W_centered.npz"))
    parser.add_argument("--max_tokens", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--lambda_reg", type=float, default=1e-2)
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    cfg = HFRunnerConfig(model_name_or_path=args.model, load_in_4bit=False, device_map="auto", torch_dtype="auto")
    runner = HFRunner(cfg)

    # Use GSM8K questions as text sources
    ds = load_dataset("gsm8k", "main", split="train")
    texts = [ds[i]["question"] for i in range(min(args.n_samples, len(ds)))]

    H, E = collect_pairs(runner, texts, max_tokens=args.max_tokens, stride=args.stride)
    if H.size == 0 or E.size == 0:
        raise RuntimeError("No pairs collected; adjust max_tokens/stride or dataset.")

    W, mean_H, mean_E = train_ridge(H, E, RidgeConfig(lambda_reg=args.lambda_reg))
    # Compute target embedding norm from model's embedding layer
    with torch.inference_mode():
        emb_weight = runner.model.get_input_embeddings().weight.detach().float()
        embed_norm = float(emb_weight.norm(dim=-1).mean().item())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_alignment(out_path.as_posix(), W, mean_H, mean_E, embed_norm)
    print(f"Saved centered alignment to {out_path} (embed_norm={embed_norm:.3f})")


if __name__ == "__main__":
    main()

