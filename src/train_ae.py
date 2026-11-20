"""Paso 2 y Paso 3: Entrenamiento de encoders (CAE/CVAE/CAAE) y extracción de códigos z."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .config import (
    CODES_DIR,
    ENCODERS_DIR,
    MAX_EPOCHS_AE,
    PATIENCE_AE,
    SEARCH_SEED,
)
from .data import build_image_dataloaders, get_or_create_splits, load_dataset
from .models_ae import build_autoencoder, loss_cvae
from .utils import count_parameters, get_device, set_seed


def _eval_reconstruction(model: nn.Module, kind: str, loader, device, beta: float = 0.1) -> Dict[str, float]:
    model.eval()
    mse_fn = nn.MSELoss(reduction="mean")
    total_mse = 0.0
    total_kl = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            bs = xb.size(0)
            if kind == "cvae":
                mu, logvar = model.encode_params(xb)
                x_hat = model.decode(mu)  # evaluación determinista con mu
                _, _, kl_mean = loss_cvae(x_hat, xb, mu, logvar, beta=beta)
                total_kl += kl_mean.item() * bs
            else:
                x_hat = model(xb)
            total_mse += mse_fn(x_hat, xb).item() * bs
            total_n += bs

    metrics = {"mse": total_mse / max(1, total_n)}
    if kind == "cvae":
        metrics["kl"] = total_kl / max(1, total_n)
    return metrics


def train_autoencoder(
    dataset_name: str,
    encoder_kind: str,
    latent_dim: int,
    seed: int = SEARCH_SEED,
    max_epochs: int = MAX_EPOCHS_AE,
    patience: int = PATIENCE_AE,
    lr: float = 1e-3,
    beta: float = 0.1,
    lambda_adv: float = 0.01,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Entrena un autoencoder (CAE, CVAE o CAAE) con parada temprana en VAL."""
    set_seed(seed)
    device = get_device()
    images, labels, class_names = load_dataset(dataset_name)
    splits = get_or_create_splits(dataset_name, labels, seed=SEARCH_SEED)
    loaders = build_image_dataloaders(images, labels, splits, batch_size=batch_size)

    model = build_autoencoder(encoder_kind, latent_dim).to(device)
    mse_fn = nn.MSELoss()

    if encoder_kind == "caae":
        bce_fn = nn.BCEWithLogitsLoss()
        opt_rec = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr
        )
        opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4)
        opt_e = torch.optim.Adam(model.encoder.parameters(), lr=2e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[Dict[str, float]] = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_mse_sum = 0.0
        train_kl_sum = 0.0
        train_d_sum = 0.0
        train_adv_sum = 0.0
        n_samples = 0

        for xb, _ in loaders["train"]:
            xb = xb.to(device)
            bs = xb.size(0)
            n_samples += bs

            if encoder_kind == "cae":
                opt.zero_grad()
                x_hat = model(xb)
                loss = mse_fn(x_hat, xb)
                loss.backward()
                opt.step()
                train_mse_sum += loss.item() * bs

            elif encoder_kind == "cvae":
                opt.zero_grad()
                x_hat, mu, logvar = model(xb)
                loss, _, kl_mean = loss_cvae(x_hat, xb, mu, logvar, beta=beta)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    train_mse_sum += mse_fn(x_hat, xb).item() * bs
                train_kl_sum += kl_mean.item() * bs

            elif encoder_kind == "caae":
                # 1. Fase de reconstrucción (E + G)
                opt_rec.zero_grad()
                z = model.encode(xb)
                x_hat = model.decode(z)
                rec_loss = mse_fn(x_hat, xb)
                rec_loss.backward()
                opt_rec.step()
                train_mse_sum += rec_loss.item() * bs

                # 2. Fase del discriminador (D)
                model.discriminator.requires_grad_(True)
                opt_d.zero_grad()
                prior = torch.randn(bs, latent_dim, device=device)
                real_logits = model.discriminator(prior)
                fake_logits = model.discriminator(model.encode(xb).detach())
                loss_d = bce_fn(real_logits, torch.ones_like(real_logits)) + bce_fn(
                    fake_logits, torch.zeros_like(fake_logits)
                )
                loss_d.backward()
                opt_d.step()
                train_d_sum += loss_d.item() * bs

                # 3. Fase adversarial del encoder (E)
                model.discriminator.requires_grad_(False)
                opt_e.zero_grad()
                gen_logits = model.discriminator(model.encode(xb))
                loss_adv = bce_fn(gen_logits, torch.ones_like(gen_logits))
                (lambda_adv * loss_adv).backward()
                opt_e.step()
                model.discriminator.requires_grad_(True)
                train_adv_sum += loss_adv.item() * bs

        val_metrics = _eval_reconstruction(model, encoder_kind, loaders["val"], device, beta=beta)
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_mse": train_mse_sum / max(1, n_samples),
            "val_mse": val_metrics["mse"],
        }
        if encoder_kind == "cvae":
            row["train_kl"] = train_kl_sum / max(1, n_samples)
            row["val_kl"] = val_metrics["kl"]
        elif encoder_kind == "caae":
            row["train_d_loss"] = train_d_sum / max(1, n_samples)
            row["train_adv_loss"] = train_adv_sum / max(1, n_samples)

        history.append(row)

        if val_metrics["mse"] < best_val_mse - 1e-7:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    elapsed_s = time.perf_counter() - t0
    model.load_state_dict(best_state)

    # Congelar y guardar checkpoint
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset_name}_{encoder_kind}_d{latent_dim}_seed{seed}"
    ckpt_path = ENCODERS_DIR / f"{tag}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dataset": dataset_name,
            "encoder_kind": encoder_kind,
            "latent_dim": latent_dim,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "n_params": count_parameters(model),
            "train_time_s": elapsed_s,
            "class_names": class_names,
            "history": history,
        },
        ckpt_path,
    )

    history_path = ENCODERS_DIR / f"{tag}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Paso 3: Extraer y estandarizar códigos (usando solo media y std de TRAIN)
    codes_path = extract_and_save_codes(
        model=model,
        dataset_name=dataset_name,
        encoder_kind=encoder_kind,
        latent_dim=latent_dim,
        seed=seed,
        images=images,
        labels=labels,
        splits=splits,
        device=device,
    )

    return {
        "tag": tag,
        "ckpt_path": str(ckpt_path),
        "codes_path": str(codes_path),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "train_time_s": elapsed_s,
        "n_params": count_parameters(model),
    }


def extract_and_save_codes(
    model: nn.Module,
    dataset_name: str,
    encoder_kind: str,
    latent_dim: int,
    seed: int,
    images: np.ndarray,
    labels: np.ndarray,
    splits: Dict[str, np.ndarray],
    device,
) -> Path:
    """Congela el encoder, extrae z de TRAIN/VAL/TEST y normaliza solo con TRAIN."""
    model.eval()
    model.requires_grad_(False)

    raw_codes: Dict[str, np.ndarray] = {}
    split_labels: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            idx = splits[split_name]
            x_tensor = torch.from_numpy(images[idx]).float().unsqueeze(1).to(device)
            z = model.encode(x_tensor).cpu().numpy().astype(np.float32)
            raw_codes[split_name] = z
            split_labels[split_name] = labels[idx].astype(np.int64)

    mean_train = raw_codes["train"].mean(axis=0, keepdims=True)
    std_train = raw_codes["train"].std(axis=0, keepdims=True)
    std_train = np.where(std_train < 1e-6, 1.0, std_train)

    norm_codes = {
        k: ((v - mean_train) / std_train).astype(np.float32)
        for k, v in raw_codes.items()
    }

    CODES_DIR.mkdir(parents=True, exist_ok=True)
    codes_path = CODES_DIR / f"{dataset_name}_{encoder_kind}_d{latent_dim}_seed{seed}.npz"
    np.savez(
        codes_path,
        z_train=norm_codes["train"],
        y_train=split_labels["train"],
        z_val=norm_codes["val"],
        y_val=split_labels["val"],
        z_test=norm_codes["test"],
        y_test=split_labels["test"],
        z_train_raw=raw_codes["train"],
        z_val_raw=raw_codes["val"],
        z_test_raw=raw_codes["test"],
        mean_train=mean_train.squeeze(0),
        std_train=std_train.squeeze(0),
    )
    return codes_path
