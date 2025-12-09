from __future__ import annotations

from typing import Any, List, Tuple

import torch

from ..infra.hf_runner import HFRunner


def export_pkv_from_prompt(runner: HFRunner, prompt: str) -> Tuple[Any, List[int]]:
    enc = runner.encode(prompt)
    out = runner.forward_with_hidden(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
    past_key_values = out["past_key_values"]
    return past_key_values, enc["input_ids"][0].tolist()


def continue_with_pkv(runner: HFRunner, token_ids: List[int], past_key_values: Any, max_new_tokens: int = 64) -> str:
    text, _ = runner.continue_with_past(token_ids=token_ids, past_key_values=past_key_values, max_new_tokens=max_new_tokens)
    return text

