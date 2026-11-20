"""Script maestro para ejecutar el flujo completo del Laboratorio 03."""

from __future__ import annotations

import argparse

from src.config import (
    CLASSIFIERS,
    ENCODERS,
    FINAL_SEEDS,
    LATENT_DIMS,
    MAX_EPOCHS_AE,
    MAX_EPOCHS_CLS,
    N_TRIALS_OPTUNA,
    PATIENCE_AE,
    PATIENCE_CLS,
    SEARCH_SEED,
)
from src.data import get_or_create_splits, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laboratorio 03 - Codificación de rostros")
    parser.add_argument(
        "--dataset",
        choices=["olivetti", "lfw", "both"],
        default="olivetti",
        help="Base de datos a procesar (por defecto: olivetti).",
    )
    parser.add_argument(
        "--stage",
        choices=["splits", "ae", "optuna", "final", "all"],
        default="all",
        help="Etapa a ejecutar.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido para pruebas (menos épocas y 3 trials Optuna).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["olivetti", "lfw"] if args.dataset == "both" else [args.dataset]

    max_epochs_ae = 15 if args.quick else MAX_EPOCHS_AE
    patience_ae = 5 if args.quick else PATIENCE_AE
    max_epochs_cls = 15 if args.quick else MAX_EPOCHS_CLS
    patience_cls = 5 if args.quick else PATIENCE_CLS
    n_trials = 3 if args.quick else N_TRIALS_OPTUNA

    for ds in datasets:
        print(f"\n=== Base: {ds.upper()} ===")
        images, labels, _ = load_dataset(ds)
        splits = get_or_create_splits(ds, labels, seed=SEARCH_SEED)
        print(
            f"Imágenes: {images.shape} | TRAIN={len(splits['train'])}, "
            f"VAL={len(splits['val'])}, TEST={len(splits['test'])}"
        )

        if args.stage == "splits":
            continue

        if args.stage in ("ae", "all"):
            from src.train_ae import train_autoencoder

            for enc in ENCODERS:
                for d in LATENT_DIMS:
                    print(f"[AE] Entrenando {ds} | {enc.upper()} | d={d} | seed={SEARCH_SEED}")
                    info = train_autoencoder(
                        dataset_name=ds,
                        encoder_kind=enc,
                        latent_dim=d,
                        seed=SEARCH_SEED,
                        max_epochs=max_epochs_ae,
                        patience=patience_ae,
                    )
                    print(
                        f"     -> mejor época={info['best_epoch']}, "
                        f"VAL MSE={info['best_val_mse']:.5f}, tiempo={info['train_time_s']:.1f}s"
                    )

        if args.stage in ("optuna", "all"):
            from src.search_optuna import run_optuna_study

            for enc in ENCODERS:
                for d in LATENT_DIMS:
                    for cls_kind in CLASSIFIERS:
                        print(f"[Optuna] Study {ds}_{enc}_d{d}_{cls_kind} ({n_trials} trials)")
                        summary = run_optuna_study(
                            dataset_name=ds,
                            encoder_kind=enc,
                            latent_dim=d,
                            classifier_kind=cls_kind,
                            n_trials=n_trials,
                            max_epochs=max_epochs_cls,
                            patience=patience_cls,
                        )
                        print(
                            f"     -> mejor VAL macro-F1={summary['best_val_macro_f1']:.4f} "
                            f"(podados={summary['n_pruned']}/{summary['n_trials_total']})"
                        )

        if args.stage in ("final", "all"):
            from src.evaluate_final import evaluate_final_dataset

            print(f"[Final TEST] Evaluando semillas {FINAL_SEEDS} en {ds.upper()}...")
            summary_df = evaluate_final_dataset(
                dataset_name=ds,
                encoders=ENCODERS,
                latent_dims=LATENT_DIMS,
                classifiers=CLASSIFIERS,
                seeds=FINAL_SEEDS,
                max_epochs_ae=max_epochs_ae,
                patience_ae=patience_ae,
                max_epochs_cls=max_epochs_cls,
                patience_cls=patience_cls,
            )
            print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
