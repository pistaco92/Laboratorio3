"""Utilidades de reproducibilidad, conteo de parámetros y métricas."""

from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int) -> None:
    """Fija semillas en Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Devuelve cuda si está disponible, si no cpu."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model) -> int:
    """Cuenta parámetros entrenables de un módulo PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_macro_f1_and_acc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Calcula macro-F1 y accuracy (usa sklearn si está disponible, si no NumPy)."""
    acc = float(np.mean(y_true == y_pred))
    try:
        from sklearn.metrics import f1_score

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except ImportError:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        macro_f1 = float(np.mean(f1s))
    return macro_f1, acc
