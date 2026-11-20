# Laboratorio 03 — Codificación y Reconocimiento de Rostros (UCN)

Implementación completa del protocolo del Laboratorio 03:
- **Codificadores convolucionales (`CAE`, `CVAE`, `CAAE`)** para dimensiones latentes `d = 32` y `d = 64`.
- **Clasificadores 1D sobre el código `z` congelado:** `CNN 1D` propia y `FCN-1D` (`tsai`).
- **Partición estratificada fija:** 25 % `TEST`, 20 % del restante para `VAL` (60 % `TRAIN`, 15 % `VAL`, 25 % `TEST`).
- **Búsqueda de hiperparámetros con Optuna:** `TPESampler(seed=42, n_startup_trials=3)` + `MedianPruner(n_startup_trials=3, n_warmup_steps=10, interval_steps=5)`, 10 trials por estudio en `artifacts/experimentos.db` y CSVs de trials.
- **Evaluación final en `TEST`:** repetición con semillas `42`, `123` y `2026`, compartiendo el mismo encoder por semilla entre ambos clasificadores.
- **Interfaz web (`app.py`):** permite seleccionar base, codificador, dimensión `z`, clasificador 1D y muestra identidad real, predicción, probabilidades y reconstrucción.

## Requisitos e instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

- **Olivetti:** usa `olivetti_faces.npy` (400 imágenes `64x64`, 40 identidades de 10 fotos c/u).
- **LFW:** coloca tus archivos del laboratorio anterior como `lfw_faces.npy` (`[N, 64, 64]` en `[0, 1]`) y `lfw_labels.npy` (`[N]` enteros `0..C-1`) en la raíz del proyecto (opcionalmente `lfw_names.json`). Si no existen, `src/data.py` intentará descargar `fetch_lfw_people(min_faces_per_person=20)` mediante `scikit-learn`.

## Ejecución

1. **Solo particiones estratificadas (verifica 240 TRAIN / 60 VAL / 100 TEST en Olivetti):**
   ```bash
   python3 run_all.py --dataset olivetti --stage splits
   ```

2. **Pipeline completo (encoders + Optuna + evaluación 3 semillas en TEST + gráficos):**
   ```bash
   python3 run_all.py --dataset olivetti --stage all
   # o para ambas bases:
   python3 run_all.py --dataset both --stage all
   ```
   *(Para una prueba rápida de humo puedes añadir `--quick`).*

3. **Interfaz web:**
   ```bash
   python3 app.py
   ```
   Luego abre `http://127.0.0.1:8000` en tu navegador.
