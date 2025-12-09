from __future__ import annotations

import argparse
from pathlib import Path

from ..infra.env import load_env
from ..benchmarks.gsm8k import evaluate_gsm8k


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM8K baseline over OpenRouter")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "configs" / "openrouter.yaml"),
        help="Path to OpenRouter config YAML",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of problems to evaluate")
    args = parser.parse_args()

    load_env(None)
    evaluate_gsm8k(config_path=args.config, n=args.n)


if __name__ == "__main__":
    main()

