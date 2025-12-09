from __future__ import annotations

import argparse
from pathlib import Path

from ..infra.env import load_env
from ..benchmarks.hotpotqa import evaluate_hotpotqa


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA baseline over OpenRouter")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "configs" / "openrouter.yaml"),
        help="Path to OpenRouter config YAML",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of problems to evaluate")
    args = parser.parse_args()

    load_env(None)
    evaluate_hotpotqa(config_path=args.config, n=args.n)


if __name__ == "__main__":
    main()

