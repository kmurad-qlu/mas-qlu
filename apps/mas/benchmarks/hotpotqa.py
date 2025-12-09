from __future__ import annotations

from typing import List, Tuple

from datasets import load_dataset

from ..graph.plan_graph import build_graph, GraphState


def load_hotpotqa(n: int = 5) -> List[Tuple[str, str]]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rows = []
    for i in range(min(n, len(ds))):
        rows.append((ds[i]["question"], ds[i]["answer"]))
    return rows


def evaluate_hotpotqa(config_path: str, n: int = 5) -> None:
    g = build_graph(config_path)
    app = g.compile()
    items = load_hotpotqa(n)
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
        ok = pred.lower().strip().startswith(str(gold).lower().strip())
        correct += int(ok)
        print(f"[{idx}] OK={ok} | pred={pred!r} | gold={gold!r}")
    print(f"Accuracy: {correct}/{len(items)} = {correct/len(items):.2%}")

