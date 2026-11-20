"""Configuración centralizada del Laboratorio 03."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
CODES_DIR = ARTIFACTS_DIR / "codes"
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"
CLASSIFIERS_DIR = ARTIFACTS_DIR / "classifiers"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

OPTUNA_DB_PATH = ARTIFACTS_DIR / "experimentos.db"
OPTUNA_STORAGE_URL = f"sqlite:///{OPTUNA_DB_PATH}"

DATASETS = ("olivetti", "lfw")
ENCODERS = ("cae", "cvae", "caae")
LATENT_DIMS = (32, 64)
CLASSIFIERS = ("cnn1d", "fcn1d")

SEARCH_SEED = 42
FINAL_SEEDS = (42, 123, 2026)

N_TRIALS_OPTUNA = 10
MAX_EPOCHS_AE = 80
PATIENCE_AE = 12
MAX_EPOCHS_CLS = 50
PATIENCE_CLS = 10
