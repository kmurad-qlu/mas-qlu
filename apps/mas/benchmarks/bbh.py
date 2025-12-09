from __future__ import annotations

from typing import List, Tuple

from datasets import load_dataset

from ..graph.plan_graph import build_graph, GraphState


def load_bbh_boolean(n: int = 5) -> List[Tuple[str, str]]:
    try:
        ds = load_dataset("lighteval/bbh", "boolean_expressions", split="test")
        rows = []
        for i in range(min(n, len(ds))):
            rows.append((ds[i]["input"], ds[i]["target"]))
        return rows
    except Exception:
        # Fallback: small synthetic set
        return [
            ("Evaluate: (T AND F) OR T", "T"),
            ("Evaluate: NOT (T AND T)", "F"),
            ("Evaluate: (F OR F) AND T", "F"),
            ("Evaluate: T XOR F", "T"),
            ("Evaluate: (T -> F)", "F"),
        ][:n]


def evaluate_bbh_boolean(config_path: str, n: int = 5) -> None:
    g = build_graph(config_path)
    app = g.compile()
    items = load_bbh_boolean(n)
    correct = 0
    for idx, (q, gold) in enumerate(items, start=1):
        state = GraphState(problem=q)
        out = app.invoke(state)
        if hasattr(out, "final_answer"):
            pred = (out.final_answer or "").strip()
        elif isinstance(out, dict):
            pred = (out.get("final_answer", "") or "").strip()
        else:
            pred = ""
        ok = pred.upper().startswith(gold.upper())
        correct += int(ok)
        print(f"[{idx}] OK={ok} | pred={pred!r} | gold={gold!r}")
    print(f"Accuracy: {correct}/{len(items)} = {correct/len(items):.2%}")

