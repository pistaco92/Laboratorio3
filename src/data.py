"""Carga de bases (Olivetti y LFW) y particiones estratificadas reproducibles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import ROOT_DIR, SEARCH_SEED, SPLITS_DIR


def load_olivetti(npy_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Carga olivetti_faces.npy (400 imágenes 64x64 de 40 identidades, 10 fotos c/u)."""
    path = npy_path or (ROOT_DIR / "olivetti_faces.npy")
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")

    images = np.load(path).astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0

    # Olivetti en scikit-learn viene ordenado en bloques de 10 fotos por identidad (0..39)
    n_identities = len(images) // 10
    labels = np.repeat(np.arange(n_identities, dtype=np.int64), 10)
    class_names = [f"Persona_{i:02d}" for i in range(n_identities)]
    return images, labels, class_names


def load_lfw(
    data_dir: Path | None = None,
    min_faces_per_person: int = 20,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Carga el subconjunto LFW del laboratorio anterior o vía scikit-learn.

    Busca primero archivos locales ``lfw_faces.npy`` y ``lfw_labels.npy`` (y opcionalmente
    ``lfw_names.json``) en la carpeta del proyecto o en ``data_dir``. Si no existen,
    intenta descargar/cargar ``fetch_lfw_people`` de scikit-learn y redimensionar a 64x64.
    """
    base = data_dir or ROOT_DIR
    imgs_file = base / "lfw_faces.npy"
    lbls_file = base / "lfw_labels.npy"
    names_file = base / "lfw_names.json"

    if imgs_file.exists() and lbls_file.exists():
        images = np.load(imgs_file).astype(np.float32)
        labels = np.load(lbls_file).astype(np.int64)
        if images.max() > 1.0:
            images = images / 255.0
        if names_file.exists():
            class_names = json.loads(names_file.read_text(encoding="utf-8"))
        else:
            unique_ids = sorted(np.unique(labels).tolist())
            class_names = [f"LFW_{i:03d}" for i in unique_ids]
        return images, labels, class_names

    try:
        from sklearn.datasets import fetch_lfw_people
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise FileNotFoundError(
            "No se encontraron 'lfw_faces.npy'/'lfw_labels.npy' y falta scikit-learn/scipy "
            "para cargar fetch_lfw_people."
        ) from exc

    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=0.5, color=False)
    raw = lfw.images.astype(np.float32)
    if raw.max() > 1.0:
        raw = raw / 255.0

    # Asegurar resolución exacta 64x64
    if raw.shape[1:] != (64, 64):
        resized = np.empty((len(raw), 64, 64), dtype=np.float32)
        zy = 64.0 / raw.shape[1]
        zx = 64.0 / raw.shape[2]
        for i in range(len(raw)):
            resized[i] = np.clip(zoom(raw[i], (zy, zx), order=1), 0.0, 1.0)
        raw = resized

    labels = lfw.target.astype(np.int64)
    class_names = list(lfw.target_names)
    return raw, labels, class_names


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Despacha la carga según el nombre de la base ('olivetti' o 'lfw')."""
    name = name.lower()
    if name == "olivetti":
        return load_olivetti()
    if name == "lfw":
        return load_lfw()
    raise ValueError(f"Base desconocida: {name}. Usa 'olivetti' o 'lfw'.")


def _stratified_two_way(
    indices: np.ndarray,
    labels: np.ndarray,
    holdout_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide ``indices`` en (resto, holdout) de forma estratificada exacta por clase."""
    n_total = len(indices)
    n_holdout = int(round(n_total * holdout_fraction))

    classes, inv = np.unique(labels[indices], return_inverse=True)
    per_class = [indices[inv == c].copy() for c in range(len(classes))]
    for arr in per_class:
        rng.shuffle(arr)

    counts = np.array([len(arr) for arr in per_class], dtype=np.int64)
    exact = counts * holdout_fraction
    take = np.floor(exact).astype(np.int64)

    # Garantizar al menos 1 muestra por clase en ambos lados cuando haya >= 2 muestras
    for c, m in enumerate(counts):
        if m >= 2 and take[c] == 0:
            take[c] = 1
        if m >= 2 and take[c] == m:
            take[c] = m - 1

    # Ajustar hasta alcanzar exactamente n_holdout
    diff = n_holdout - int(take.sum())
    remainders = exact - np.floor(exact)
    order = np.argsort(-(remainders + rng.uniform(0, 1e-6, size=len(classes))))
    if diff > 0:
        for c in order:
            if diff == 0:
                break
            if take[c] < counts[c] - 1:
                take[c] += 1
                diff -= 1
    elif diff < 0:
        for c in reversed(order):
            if diff == 0:
                break
            if take[c] > 1:
                take[c] -= 1
                diff += 1

    holdout_parts, rest_parts = [], []
    for c, arr in enumerate(per_class):
        k = int(take[c])
        holdout_parts.append(arr[:k])
        rest_parts.append(arr[k:])

    rest = np.sort(np.concatenate(rest_parts))
    holdout = np.sort(np.concatenate(holdout_parts))
    return rest, holdout


def get_or_create_splits(
    dataset_name: str,
    labels: np.ndarray,
    seed: int = SEARCH_SEED,
    overwrite: bool = False,
) -> Dict[str, np.ndarray]:
    """Genera o carga los índices fijos TRAIN (60%), VAL (15%) y TEST (25%).

    Protocolo del laboratorio (lámina 13):
    1) Reservar 25 % de imágenes para TEST estratificado.
    2) Dentro del 75 % restante, reservar 20 % para VAL estratificado.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_file = SPLITS_DIR / f"{dataset_name}_splits.npz"

    if split_file.exists() and not overwrite:
        data = np.load(split_file)
        return {
            "train": data["train"],
            "val": data["val"],
            "test": data["test"],
        }

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(labels), dtype=np.int64)

    train_val_idx, test_idx = _stratified_two_way(all_idx, labels, 0.25, rng)
    train_idx, val_idx = _stratified_two_way(train_val_idx, labels, 0.20, rng)

    np.savez(
        split_file,
        train=train_idx,
        val=val_idx,
        test=test_idx,
        seed=np.array([seed]),
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_image_dataloaders(
    images: np.ndarray,
    labels: np.ndarray,
    splits: Dict[str, np.ndarray],
    batch_size: int = 32,
):
    """Crea TensorDatasets y DataLoaders [B, 1, 64, 64] para TRAIN, VAL y TEST."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loaders = {}
    for split_name in ("train", "val", "test"):
        idx = splits[split_name]
        x_tensor = torch.from_numpy(images[idx]).float().unsqueeze(1)
        y_tensor = torch.as_tensor(labels[idx], dtype=torch.long)
        ds = TensorDataset(x_tensor, y_tensor)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
        )
    return loaders
