from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch

from ..infra.hf_runner import HFRunner


def collect_pairs(
    runner: HFRunner,
    texts: Iterable[str],
    max_tokens: int = 5000,
    stride: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (H^L, E) pairs where H^L are last-layer hidden states and E are input embeddings.
    We sub-sample positions with a stride to control memory.
    Returns:
      H: [N, hidden_size]
      E: [N, embed_size]
    """
    H_list: List[np.ndarray] = []
    E_list: List[np.ndarray] = []
    total = 0
    for text in texts:
        enc = runner.encode(text)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        with torch.inference_mode():
            inputs_embeds = runner.get_input_embeds(input_ids)  # [1, seq, embed]
            out = runner.forward_with_hidden(input_ids=input_ids, attention_mask=attn)
            last_hidden = out["last_hidden"]  # [1, seq, hidden]

        seq_len = input_ids.shape[1]
        positions = list(range(0, seq_len, stride))
        h = last_hidden[0, positions, :].detach().cpu().float().numpy()
        e = inputs_embeds[0, positions, :].detach().cpu().float().numpy()
        H_list.append(h)
        E_list.append(e)
        total += len(positions)
        if total >= max_tokens:
            break
    H = np.concatenate(H_list, axis=0) if H_list else np.zeros((0, 0), dtype=np.float32)
    E = np.concatenate(E_list, axis=0) if E_list else np.zeros((0, 0), dtype=np.float32)
    return H, E

