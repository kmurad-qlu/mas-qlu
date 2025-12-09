from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RidgeConfig:
    lambda_reg: float = 1e-2


def train_ridge(H: np.ndarray, E: np.ndarray, cfg: RidgeConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train linear map W with mean-centering such that (H - mean_H) W ≈ (E - mean_E).
    Returns (W, mean_H, mean_E).
    Shapes:
      H: [N, hidden_size]
      E: [N, embed_size]
      W: [hidden_size, embed_size]
    """
    assert H.ndim == 2 and E.ndim == 2
    mean_H = H.mean(axis=0, keepdims=True)
    mean_E = E.mean(axis=0, keepdims=True)
    Hc = H - mean_H
    Ec = E - mean_E
    hidden = H.shape[1]
    A = Hc.T @ Hc + cfg.lambda_reg * np.eye(hidden, dtype=H.dtype)
    B = Hc.T @ Ec
    W = np.linalg.solve(A, B)
    return W, mean_H.squeeze(0), mean_E.squeeze(0)


def save_alignment(path: str, W: np.ndarray, mean_H: np.ndarray, mean_E: np.ndarray, embed_norm: float) -> None:
    np.savez(path, W=W, mean_H=mean_H, mean_E=mean_E, embed_norm=np.array([embed_norm], dtype=np.float32))


def load_alignment(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = np.load(path)
    if isinstance(data, np.ndarray):
        # Backward compatibility with .npy (W only)
        W = data
        return W, np.zeros((W.shape[0],), dtype=W.dtype), np.zeros((W.shape[1],), dtype=W.dtype), 1.0
    W = data["W"]
    mean_H = data["mean_H"]
    mean_E = data["mean_E"]
    embed_norm = float(data["embed_norm"][0]) if "embed_norm" in data.files else 1.0
    return W, mean_H, mean_E, embed_norm

