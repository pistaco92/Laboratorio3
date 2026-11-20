"""Paso 8 y visualizaciones: Repetición con semillas 42, 123, 2026 y evaluación única en TEST."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import (
    CLASSIFIERS,
    CLASSIFIERS_DIR,
    CODES_DIR,
    ENCODERS,
    ENCODERS_DIR,
    FIGURES_DIR,
    FINAL_SEEDS,
    LATENT_DIMS,
    MAX_EPOCHS_AE,
    MAX_EPOCHS_CLS,
    OPTUNA_DIR,
    PATIENCE_AE,
    PATIENCE_CLS,
    RESULTS_DIR,
    SEARCH_SEED,
)
from .data import get_or_create_splits, load_dataset
from .models_ae import build_autoencoder
from .search_optuna import _build_code_loader, train_classifier_once
from .train_ae import train_autoencoder
from .utils import compute_macro_f1_and_acc, get_device


def _eval_test_once(
    model: nn.Module, z_test: np.ndarray, y_test: np.ndarray, batch_size: int, device
) -> tuple[float, float, np.ndarray]:
    loader = _build_code_loader(z_test, y_test, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for zb, _ in loader:
            logits = model(zb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    macro_f1, acc = compute_macro_f1_and_acc(y_test, y_pred)
    return macro_f1, acc, y_pred


def _eval_ae_test_mse(
    dataset_name: str,
    encoder_kind: str,
    latent_dim: int,
    seed: int,
    device,
) -> float:
    ckpt_path = ENCODERS_DIR / f"{dataset_name}_{encoder_kind}_d{latent_dim}_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ae = build_autoencoder(encoder_kind, latent_dim).to(device)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()

    images, labels, _ = load_dataset(dataset_name)
    splits = get_or_create_splits(dataset_name, labels, seed=SEARCH_SEED)
    x_test = torch.from_numpy(images[splits["test"]]).float().unsqueeze(1).to(device)
    with torch.no_grad():
        z = ae.encode(x_test)
        x_hat = ae.decode(z)
        mse = float(nn.MSELoss()(x_hat, x_test).item())
    return mse


def evaluate_final_dataset(
    dataset_name: str,
    encoders: tuple[str, ...] = ENCODERS,
    latent_dims: tuple[int, ...] = LATENT_DIMS,
    classifiers: tuple[str, ...] = CLASSIFIERS,
    seeds: tuple[int, ...] = FINAL_SEEDS,
    max_epochs_ae: int = MAX_EPOCHS_AE,
    patience_ae: int = PATIENCE_AE,
    max_epochs_cls: int = MAX_EPOCHS_CLS,
    patience_cls: int = PATIENCE_CLS,
) -> pd.DataFrame:
    """Ejecuta la evaluación final con las 3 semillas sobre TEST y genera gráficos."""
    device = get_device()
    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []

    for enc in encoders:
        for d in latent_dims:
            # 1) Asegurar encoders + códigos z para las 3 semillas (compartidos por ambos clasificadores)
            ae_test_mse_by_seed: Dict[int, float] = {}
            ae_meta_by_seed: Dict[int, Dict[str, Any]] = {}

            for seed in seeds:
                tag = f"{dataset_name}_{enc}_d{d}_seed{seed}"
                ckpt_path = ENCODERS_DIR / f"{tag}.pt"
                codes_path = CODES_DIR / f"{tag}.npz"
                if not ckpt_path.exists() or not codes_path.exists():
                    info = train_autoencoder(
                        dataset_name=dataset_name,
                        encoder_kind=enc,
                        latent_dim=d,
                        seed=seed,
                        max_epochs=max_epochs_ae,
                        patience=patience_ae,
                    )
                else:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    info = {
                        "train_time_s": ckpt.get("train_time_s", 0.0),
                        "n_params": ckpt.get("n_params", 0),
                        "best_epoch": ckpt.get("best_epoch", 0),
                    }
                ae_meta_by_seed[seed] = info
                ae_test_mse_by_seed[seed] = _eval_ae_test_mse(dataset_name, enc, d, seed, device)

            # 2) Entrenar y evaluar cada clasificador con los best_params de Optuna
            for cls_kind in classifiers:
                study_name = f"{dataset_name}_{enc}_d{d}_{cls_kind}"
                best_file = OPTUNA_DIR / study_name / "best_params.json"
                if not best_file.exists():
                    raise FileNotFoundError(
                        f"Falta {best_file}. Ejecuta Optuna antes de evaluar TEST."
                    )
                best_info = json.loads(best_file.read_text(encoding="utf-8"))
                best_params = best_info["best_params"]

                for seed in seeds:
                    tag = f"{dataset_name}_{enc}_d{d}_seed{seed}"
                    codes = np.load(CODES_DIR / f"{tag}.npz")
                    z_train, y_train = codes["z_train"], codes["y_train"]
                    z_val, y_val = codes["z_val"], codes["y_val"]
                    z_test, y_test = codes["z_test"], codes["y_test"]
                    n_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

                    res = train_classifier_once(
                        z_train=z_train,
                        y_train=y_train,
                        z_val=z_val,
                        y_val=y_val,
                        latent_dim=d,
                        n_classes=n_classes,
                        classifier_kind=cls_kind,
                        params=best_params,
                        seed=seed,
                        max_epochs=max_epochs_cls,
                        patience=patience_cls,
                    )

                    test_f1, test_acc, y_pred = _eval_test_once(
                        res["model"],
                        z_test,
                        y_test,
                        batch_size=int(best_params["batch"]),
                        device=device,
                    )

                    cls_ckpt = CLASSIFIERS_DIR / f"{dataset_name}_{enc}_d{d}_{cls_kind}_seed{seed}.pt"
                    torch.save(
                        {
                            "state_dict": res["model"].state_dict(),
                            "params": best_params,
                            "dataset": dataset_name,
                            "encoder": enc,
                            "latent_dim": d,
                            "classifier": cls_kind,
                            "seed": seed,
                            "n_classes": n_classes,
                            "best_epoch": res["best_epoch"],
                            "val_macro_f1": res["best_val_f1"],
                            "test_macro_f1": test_f1,
                            "test_acc": test_acc,
                            "backend": res["backend"],
                        },
                        cls_ckpt,
                    )

                    run_rows.append(
                        {
                            "dataset": dataset_name,
                            "encoder": enc,
                            "latent_dim": d,
                            "classifier": cls_kind,
                            "seed": seed,
                            "test_macro_f1": test_f1,
                            "test_acc": test_acc,
                            "val_macro_f1": res["best_val_f1"],
                            "test_recon_mse": ae_test_mse_by_seed[seed],
                            "cls_best_epoch": res["best_epoch"],
                            "cls_params": res["n_params"],
                            "cls_time_s": res["train_time_s"],
                            "ae_params": ae_meta_by_seed[seed]["n_params"],
                            "ae_time_s": ae_meta_by_seed[seed]["train_time_s"],
                        }
                    )

                    if seed == seeds[0]:
                        _plot_confusion_matrix(
                            y_test,
                            y_pred,
                            title=f"{dataset_name.upper()} | {enc.upper()} d={d} + {cls_kind.upper()} (seed={seed})",
                            out_path=FIGURES_DIR / f"cm_{dataset_name}_{enc}_d{d}_{cls_kind}.png",
                        )

            # Gráficos por (enc, d) usando la primera semilla
            _plot_ae_curves(dataset_name, enc, d, seeds[0])
            _plot_reconstructions(dataset_name, enc, d, seeds[0], device)
            _plot_latent_pca_2d(dataset_name, enc, d, seeds[0])

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(RESULTS_DIR / f"{dataset_name}_runs.csv", index=False)

    summary_df = (
        runs_df.groupby(["dataset", "encoder", "latent_dim", "classifier"], as_index=False)
        .agg(
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_recon_mse_mean=("test_recon_mse", "mean"),
            cls_params=("cls_params", "first"),
            ae_params=("ae_params", "first"),
            cls_time_s_mean=("cls_time_s", "mean"),
            ae_time_s_mean=("ae_time_s", "mean"),
        )
        .sort_values("test_macro_f1_mean", ascending=False)
    )
    summary_df.to_csv(RESULTS_DIR / f"{dataset_name}_summary.csv", index=False)
    return summary_df


def _plot_ae_curves(dataset_name: str, enc: str, d: int, seed: int) -> None:
    hist_path = ENCODERS_DIR / f"{dataset_name}_{enc}_d{d}_seed{seed}_history.json"
    if not hist_path.exists():
        return
    hist = pd.DataFrame(json.loads(hist_path.read_text(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["epoch"], hist["train_mse"], label="Train MSE")
    ax.plot(hist["epoch"], hist["val_mse"], label="Val MSE")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE por píxel")
    ax.set_title(f"Reconstrucción {dataset_name.upper()} - {enc.upper()} (d={d}, seed={seed})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"curve_mse_{dataset_name}_{enc}_d{d}.png", dpi=140)
    plt.close(fig)

    if enc == "cvae" and "train_kl" in hist.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist["epoch"], hist["train_kl"], label="Train KL")
        ax.plot(hist["epoch"], hist["val_kl"], label="Val KL")
        ax.set_xlabel("Época")
        ax.set_ylabel("Divergencia KL")
        ax.set_title(f"KL {dataset_name.upper()} - CVAE (d={d}, seed={seed})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"curve_kl_{dataset_name}_cvae_d{d}.png", dpi=140)
        plt.close(fig)

    if enc == "caae" and "train_d_loss" in hist.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist["epoch"], hist["train_d_loss"], label="Pérdida Discriminador D")
        ax.plot(hist["epoch"], hist["train_adv_loss"], label="Pérdida Adversarial E")
        ax.set_xlabel("Época")
        ax.set_ylabel("BCEWithLogits")
        ax.set_title(f"Adversarial {dataset_name.upper()} - CAAE (d={d}, seed={seed})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"curve_adv_{dataset_name}_caae_d{d}.png", dpi=140)
        plt.close(fig)


def _plot_reconstructions(dataset_name: str, enc: str, d: int, seed: int, device, n_show: int = 8) -> None:
    ckpt_path = ENCODERS_DIR / f"{dataset_name}_{enc}_d{d}_seed{seed}.pt"
    if not ckpt_path.exists():
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ae = build_autoencoder(enc, d).to(device)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()

    images, labels, _ = load_dataset(dataset_name)
    splits = get_or_create_splits(dataset_name, labels, seed=SEARCH_SEED)
    idx = splits["test"][:n_show]
    x = torch.from_numpy(images[idx]).float().unsqueeze(1).to(device)
    with torch.no_grad():
        x_hat = ae.decode(ae.encode(x)).cpu().numpy()[:, 0]

    fig, axes = plt.subplots(2, len(idx), figsize=(1.6 * len(idx), 3.4))
    for i in range(len(idx)):
        axes[0, i].imshow(images[idx[i]], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon")
    fig.suptitle(f"Original vs Reconstrucción ({dataset_name.upper()}, {enc.upper()}, d={d})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"recon_{dataset_name}_{enc}_d{d}.png", dpi=140)
    plt.close(fig)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("ID Predicho")
    ax.set_ylabel("ID Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_latent_pca_2d(dataset_name: str, enc: str, d: int, seed: int) -> None:
    """Visualización exploratoria 2D por PCA sobre z_test (lámina 75)."""
    codes_path = CODES_DIR / f"{dataset_name}_{enc}_d{d}_seed{seed}.npz"
    if not codes_path.exists():
        return
    data = np.load(codes_path)
    z = data["z_test"]
    y = data["y_test"]

    z_centered = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
    proj = z_centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=y, cmap="tab20", s=24, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="ID")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA exploratorio de z (TEST) - {dataset_name.upper()} {enc.upper()} d={d}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"pca2d_{dataset_name}_{enc}_d{d}.png", dpi=140)
    plt.close(fig)
