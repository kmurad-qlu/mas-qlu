from __future__ import annotations

from typing import List, Tuple

from datasets import load_dataset

from ..graph.plan_graph import build_graph, GraphState
from ..eval.metrics import exact_or_numeric_match


def load_gsm8k(n: int = 10) -> List[Tuple[str, str]]:
    ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    for i in range(min(n, len(ds))):
        rows.append((ds[i]["question"], ds[i]["answer"]))
    return rows


def evaluate_gsm8k(config_path: str, n: int = 10) -> None:
    g = build_graph(config_path)
    app = g.compile()
    items = load_gsm8k(n)
    correct = 0
    for idx, (q, gold) in enumerate(items, start=1):
        state = GraphState(problem=q)
        out = app.invoke(state)
        if hasattr(out, "final_answer"):
            pred = (out.final_answer or "").strip()  # pydantic object path
        elif isinstance(out, dict):
            pred = (out.get("final_answer", "") or "").strip()
        else:
            pred = ""
        ok = exact_or_numeric_match(pred, gold)
        correct += int(ok)
        print(f"[{idx}] OK={ok} | pred={pred!r} | gold={gold!r}")
    print(f"Accuracy: {correct}/{len(items)} = {correct/len(items):.2%}")

