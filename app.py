"""Interfaz web (lámina 76) para probar codificadores y clasificadores 1D sobre TEST."""

from __future__ import annotations

import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASSIFIERS_DIR, CODES_DIR, ENCODERS_DIR, SEARCH_SEED
from src.data import get_or_create_splits, load_dataset


def _array_to_base64_png(img_2d: np.ndarray, title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(img_2d, cmap="gray", vmin=0.0, vmax=1.0)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")
    buf = io.BytesIO()
    fig.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_inference(
    dataset: str,
    encoder_kind: str,
    latent_dim: int,
    classifier_kind: str,
    seed: int,
    sample_pos: int,
) -> dict:
    """Pasa una foto de TEST por el encoder congelado, normaliza z y clasifica con 1D."""
    import torch
    from src.models_ae import build_autoencoder
    from src.search_optuna import build_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, class_names = load_dataset(dataset)
    splits = get_or_create_splits(dataset, labels, seed=SEARCH_SEED)
    test_idx = splits["test"]
    sample_pos = max(0, min(sample_pos, len(test_idx) - 1))
    img_idx = int(test_idx[sample_pos])

    orig_img = images[img_idx]
    true_id = int(labels[img_idx])

    enc_ckpt_path = ENCODERS_DIR / f"{dataset}_{encoder_kind}_d{latent_dim}_seed{seed}.pt"
    codes_path = CODES_DIR / f"{dataset}_{encoder_kind}_d{latent_dim}_seed{seed}.npz"
    cls_ckpt_path = CLASSIFIERS_DIR / f"{dataset}_{encoder_kind}_d{latent_dim}_{classifier_kind}_seed{seed}.pt"

    missing = [str(p) for p in (enc_ckpt_path, codes_path, cls_ckpt_path) if not p.exists()]
    if missing:
        return {
            "ok": False,
            "error": (
                "Faltan artefactos entrenados. Ejecuta primero `python3 run_all.py`. "
                f"No encontrado: {', '.join(missing)}"
            ),
            "orig_b64": _array_to_base64_png(orig_img, f"Real: {class_names[true_id]}"),
            "n_test": len(test_idx),
            "sample_pos": sample_pos,
            "true_label": class_names[true_id],
        }

    enc_ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    ae = build_autoencoder(encoder_kind, latent_dim).to(device)
    ae.load_state_dict(enc_ckpt["state_dict"])
    ae.eval()

    codes_npz = np.load(codes_path)
    mean_train = torch.from_numpy(codes_npz["mean_train"]).float().to(device)
    std_train = torch.from_numpy(codes_npz["std_train"]).float().to(device)

    cls_ckpt = torch.load(cls_ckpt_path, map_location=device, weights_only=False)
    clf, _ = build_classifier(
        classifier_kind=classifier_kind,
        latent_dim=latent_dim,
        n_classes=int(cls_ckpt["n_classes"]),
        params=cls_ckpt["params"],
    )
    clf.load_state_dict(cls_ckpt["state_dict"])
    clf.to(device).eval()

    foto = torch.from_numpy(orig_img).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = ae.encode(foto)                             # [1, d]
        recon = ae.decode(z).cpu().numpy()[0, 0]        # [64, 64]
        z_norm = (z - mean_train) / std_train           # normalización de TRAIN
        logits = clf(z_norm.unsqueeze(1))               # [1, 1, d] -> [1, C]
        probs = logits.softmax(dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    top3_idx = np.argsort(-probs)[:3]
    top3 = [
        {"label": class_names[int(i)], "prob": float(probs[int(i)])}
        for i in top3_idx
    ]
    recon_mse = float(np.mean((orig_img - recon) ** 2))

    return {
        "ok": True,
        "n_test": len(test_idx),
        "sample_pos": sample_pos,
        "true_id": true_id,
        "true_label": class_names[true_id],
        "pred_id": pred_id,
        "pred_label": class_names[pred_id],
        "correct": pred_id == true_id,
        "recon_mse": recon_mse,
        "top3": top3,
        "orig_b64": _array_to_base64_png(orig_img, "Original (TEST)"),
        "recon_b64": _array_to_base64_png(recon, f"Reconstrucción ({encoder_kind.upper()})"),
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Laboratorio 03 - Codificación de Rostros</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1e293b; }
    h1 { color: #0f172a; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; background: #f8fafc; padding: 1rem; border-radius: 8px; border: 1px solid #cbd5e1; }
    label { font-size: 0.85rem; font-weight: 600; display: block; margin-bottom: 0.25rem; }
    select, input, button { width: 100%; padding: 0.45rem; border-radius: 6px; border: 1px solid #94a3b8; box-sizing: border-box; }
    button { background: #1d4ed8; color: white; font-weight: 600; border: none; cursor: pointer; margin-top: 1.25rem; }
    button:hover { background: #1e40af; }
    .cards { display: flex; gap: 1.5rem; margin-top: 1.5rem; flex-wrap: wrap; align-items: flex-start; }
    .card { border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem; background: white; }
    .ok { color: #15803d; font-weight: 700; }
    .fail { color: #b91c1c; font-weight: 700; }
  </style>
</head>
<body>
  <h1>Laboratorio 03: Codificación y Reconocimiento Facial</h1>
  <div class="grid">
    <div><label>Base</label>
      <select id="dataset"><option value="olivetti">Olivetti</option><option value="lfw">LFW</option></select>
    </div>
    <div><label>Codificador</label>
      <select id="encoder"><option value="cae">CAE</option><option value="cvae">CVAE</option><option value="caae">CAAE</option></select>
    </div>
    <div><label>Dimensión z</label>
      <select id="dim"><option value="32">32</option><option value="64">64</option></select>
    </div>
    <div><label>Clasificador 1D</label>
      <select id="classifier"><option value="cnn1d">CNN 1D propia</option><option value="fcn1d">FCN-1D (tsai)</option></select>
    </div>
    <div><label>Semilla</label>
      <select id="seed"><option value="42">42</option><option value="123">123</option><option value="2026">2026</option></select>
    </div>
    <div><label>Índice en TEST (0..N-1)</label>
      <input id="sample" type="number" min="0" value="0" />
    </div>
    <div><button onclick="evaluar()">Evaluar rostro</button></div>
  </div>
  <div id="output" class="cards"></div>

  <script>
    async function evaluar() {
      const qs = new URLSearchParams({
        dataset: document.getElementById("dataset").value,
        encoder: document.getElementById("encoder").value,
        dim: document.getElementById("dim").value,
        classifier: document.getElementById("classifier").value,
        seed: document.getElementById("seed").value,
        sample: document.getElementById("sample").value,
      });
      const res = await fetch("/api/predict?" + qs.toString());
      const data = await res.json();
      const out = document.getElementById("output");
      if (!data.ok) {
        out.innerHTML = `<div class="card"><p class="fail">${data.error}</p><img src="data:image/png;base64,${data.orig_b64}"/></div>`;
        return;
      }
      const badge = data.correct ? `<span class="ok">Correcto</span>` : `<span class="fail">Error</span>`;
      const top3 = data.top3.map(t => `<li>${t.label}: ${(t.prob * 100).toFixed(1)}%</li>`).join("");
      out.innerHTML = `
        <div class="card"><img src="data:image/png;base64,${data.orig_b64}" /></div>
        <div class="card"><img src="data:image/png;base64,${data.recon_b64}" /></div>
        <div class="card">
          <h3>Resultado (${data.sample_pos + 1}/${data.n_test} en TEST)</h3>
          <p><b>Identidad real:</b> ${data.true_label}</p>
          <p><b>Predicción:</b> ${data.pred_label} (${badge})</p>
          <p><b>MSE reconstrucción:</b> ${data.recon_mse.toFixed(5)}</p>
          <p><b>Top-3 probabilidades:</b></p><ul>${top3}</ul>
        </div>`;
    }
    evaluar();
  </script>
</body>
</html>
"""


class LabHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/predict":
            q = parse_qs(parsed.query)
            payload = run_inference(
                dataset=q.get("dataset", ["olivetti"])[0],
                encoder_kind=q.get("encoder", ["cae"])[0],
                latent_dim=int(q.get("dim", ["32"])[0]),
                classifier_kind=q.get("classifier", ["cnn1d"])[0],
                seed=int(q.get("seed", ["42"])[0]),
                sample_pos=int(q.get("sample", ["0"])[0]),
            )
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


def main(host: str = "127.0.0.1", port: int = 8001):
    server = HTTPServer((host, port), LabHandler)
    print(f"Interfaz web disponible en http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
