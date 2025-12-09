from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from ..infra.hf_runner import HFRunner
from .wa_train import load_alignment


def project_hidden_to_inputs_embeds(
    hidden: torch.Tensor,
    W: np.ndarray,
    mean_H: np.ndarray,
    mean_E: np.ndarray,
    target_embed_norm: float = 1.0,
) -> torch.Tensor:
    """
    hidden: [batch, seq, hidden_size]
    W: [hidden_size, embed_size]
    mean_H: [hidden_size]
    mean_E: [embed_size]
    returns inputs_embeds: [batch, seq, embed_size]
    """
    b, s, h = hidden.shape
    H = hidden.detach().cpu().float().numpy().reshape(-1, h)  # [b*s, h]
    Hc = H - mean_H.reshape(1, -1)
    E = Hc @ W + mean_E.reshape(1, -1)  # [b*s, embed]
    # Normalize to target norm
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-6
    E = E * (target_embed_norm / norms)
    embed_size = E.shape[1]
    E_tensor = torch.from_numpy(E).to(hidden.device).reshape(b, s, embed_size)
    return E_tensor


def nearest_token_ids(runner: HFRunner, inputs_embeds: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """
    Map inputs_embeds back to nearest token ids using the model's embedding matrix.
    """
    emb_matrix = runner.model.get_input_embeddings().weight  # [vocab, embed]
    # cosine similarity (promote to float32 for numerical stability)
    x = torch.nn.functional.normalize(inputs_embeds.float(), dim=-1)
    y = torch.nn.functional.normalize(emb_matrix.float(), dim=-1)
    scores = torch.einsum("bse,ve->bsv", x, y)  # [batch, seq, vocab]
    top = torch.topk(scores, k=top_k, dim=-1).indices
    return top[..., 0]

