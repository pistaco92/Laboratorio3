"""Pasos 4 a 7: Búsqueda de hiperparámetros con Optuna sobre códigos z congelados."""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    ARTIFACTS_DIR,
    CODES_DIR,
    MAX_EPOCHS_CLS,
    N_TRIALS_OPTUNA,
    OPTUNA_DIR,
    OPTUNA_STORAGE_URL,
    PATIENCE_CLS,
    SEARCH_SEED,
)
from .models_cls import CNN1DClassifier, build_fcn1d
from .utils import compute_macro_f1_and_acc, count_parameters, get_device, set_seed


def _build_code_loader(z: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.from_numpy(z).float().unsqueeze(1)  # [B, 1, d]
    y_t = torch.as_tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _eval_classifier(model: nn.Module, loader: DataLoader, device) -> tuple[float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for zb, yb in loader:
            zb = zb.to(device)
            logits = model(zb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return compute_macro_f1_and_acc(y_true, y_pred)


def build_classifier(
    classifier_kind: str,
    latent_dim: int,
    n_classes: int,
    params: Dict[str, Any],
) -> tuple[nn.Module, str]:
    """Construye CNN 1D propia o FCN-1D según hiperparámetros."""
    classifier_kind = classifier_kind.lower()
    if classifier_kind == "cnn1d":
        model = CNN1DClassifier(
            latent_dim=latent_dim,
            n_classes=n_classes,
            channels=int(params["channels"]),
            kernel_size=int(params["kernel"]),
            dropout=float(params["dropout"]),
        )
        return model, "pytorch_cnn1d"
    if classifier_kind == "fcn1d":
        return build_fcn1d(n_classes=n_classes, width=int(params["channels"]))
    raise ValueError(f"Clasificador desconocido: {classifier_kind}. Usa 'cnn1d' o 'fcn1d'.")


def train_classifier_once(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    latent_dim: int,
    n_classes: int,
    classifier_kind: str,
    params: Dict[str, Any],
    seed: int,
    max_epochs: int = MAX_EPOCHS_CLS,
    patience: int = PATIENCE_CLS,
    trial=None,
) -> Dict[str, Any]:
    """Entrena un clasificador 1D desde cero y selecciona mejor época por macro-F1 en VAL."""
    set_seed(seed)
    device = get_device()

    train_loader = _build_code_loader(z_train, y_train, batch_size=int(params["batch"]), shuffle=True)
    val_loader = _build_code_loader(z_val, y_val, batch_size=int(params["batch"]), shuffle=False)

    model, backend = build_classifier(classifier_kind, latent_dim, n_classes, params)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["wd"]),
    )

    best_f1 = -1.0
    best_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(zb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_f1, val_acc = _eval_classifier(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if trial is not None:
            import optuna

            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if no_improve >= patience:
            break

    elapsed_s = time.perf_counter() - t0
    model.load_state_dict(best_state)

    return {
        "model": model,
        "backend": backend,
        "best_val_f1": best_f1,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "n_params": count_parameters(model),
        "train_time_s": elapsed_s,
    }


def run_optuna_study(
    dataset_name: str,
    encoder_kind: str,
    latent_dim: int,
    classifier_kind: str,
    n_trials: int = N_TRIALS_OPTUNA,
    max_epochs: int = MAX_EPOCHS_CLS,
    patience: int = PATIENCE_CLS,
) -> Dict[str, Any]:
    """Ejecuta un study de Optuna para una combinación (base, encoder, dim, clasificador)."""
    import optuna

    codes_path = CODES_DIR / f"{dataset_name}_{encoder_kind}_d{latent_dim}_seed{SEARCH_SEED}.npz"
    if not codes_path.exists():
        raise FileNotFoundError(
            f"No existen códigos en {codes_path}. Entrena primero el autoencoder."
        )

    data = np.load(codes_path)
    z_train, y_train = data["z_train"], data["y_train"]
    z_val, y_val = data["z_val"], data["y_val"]
    n_classes = int(max(y_train.max(), y_val.max()) + 1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    study_name = f"{dataset_name}_{encoder_kind}_d{latent_dim}_{classifier_kind}"
    study_dir = OPTUNA_DIR / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=SEARCH_SEED, n_startup_trials=3)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3, n_warmup_steps=10, interval_steps=5
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=OPTUNA_STORAGE_URL,
        load_if_exists=True,
    )

    best_overall = {"f1": -1.0, "ckpt_path": study_dir / "best_trial_model.pt"}

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "batch": trial.suggest_categorical("batch", [16, 32]),
            "channels": trial.suggest_categorical("channels", [16, 32]),
        }
        if classifier_kind == "cnn1d":
            params["kernel"] = trial.suggest_categorical("kernel", [3, 5])
            params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.2, 0.4])

        res = train_classifier_once(
            z_train=z_train,
            y_train=y_train,
            z_val=z_val,
            y_val=y_val,
            latent_dim=latent_dim,
            n_classes=n_classes,
            classifier_kind=classifier_kind,
            params=params,
            seed=SEARCH_SEED + trial.number,
            max_epochs=max_epochs,
            patience=patience,
            trial=trial,
        )

        trial.set_user_attr("best_epoch", res["best_epoch"])
        trial.set_user_attr("n_params", res["n_params"])
        trial.set_user_attr("train_time_s", round(res["train_time_s"], 4))
        trial.set_user_attr("seed", SEARCH_SEED + trial.number)
        trial.set_user_attr("backend", res["backend"])

        if res["best_val_f1"] > best_overall["f1"]:
            best_overall["f1"] = res["best_val_f1"]
            torch.save(
                {
                    "state_dict": res["model"].state_dict(),
                    "params": params,
                    "best_epoch": res["best_epoch"],
                    "best_val_f1": res["best_val_f1"],
                    "latent_dim": latent_dim,
                    "n_classes": n_classes,
                    "classifier_kind": classifier_kind,
                },
                best_overall["ckpt_path"],
            )

        return res["best_val_f1"]

    pending = max(0, n_trials - len(study.trials))
    if pending > 0:
        study.optimize(objective, n_trials=pending, n_jobs=1)

    trials_csv = study_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)

    best_trial = study.best_trial
    summary = {
        "study_name": study_name,
        "dataset": dataset_name,
        "encoder": encoder_kind,
        "latent_dim": latent_dim,
        "classifier": classifier_kind,
        "best_val_macro_f1": float(best_trial.value),
        "best_params": best_trial.params,
        "best_epoch": best_trial.user_attrs.get("best_epoch"),
        "n_params": best_trial.user_attrs.get("n_params"),
        "n_trials_total": len(study.trials),
        "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "n_complete": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
    }
    (study_dir / "best_params.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
