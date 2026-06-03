"""Demo hastalar icin LSTM/GRU/Transformer gradient saliency uretir.

Cikti:
  backend/artifacts/demo/demo_saliency.json

macOS'ta canli backward() segfault riski nedeniyle bu script offline calistirilir;
API demo hastalar icin bu dosyadan okur.

Kullanim:
  cd backend/scripts
  ENABLE_GRADIENT_SALIENCY=1 python build_demo_saliency.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Gradient hesabi icin bayragi script basinda ac
os.environ["ENABLE_GRADIENT_SALIENCY"] = "1"

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.demo_saliency_store import _GRADIENT_MODELS  # noqa: E402
from app.services.inference_registry import InferenceRegistry  # noqa: E402
from app.services.patient_store import PatientStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_PATH = BACKEND_DIR / "artifacts" / "demo" / "demo_saliency.json"
WINDOW_HOURS = 24


def build_saliency_payload() -> dict:
    """Tum demo hastalar icin gradient saliency sozlugunu uretir."""
    store = PatientStore()
    registry = InferenceRegistry()
    patients_out: dict[str, dict[str, list[float]]] = {}

    for meta in store.list_demo_patients():
        pid = meta["patient_id"]
        window = store.get_window(pid, hours=WINDOW_HOURS)
        series = [{k: v for k, v in step.items() if k != "hour"} for step in window["series"]]
        last_snap = series[-1]
        raw = registry.predict_window(last_snap, repeat_hours=WINDOW_HOURS, series=series)
        by_id = {m["model_id"]: m for m in raw["models"]}
        patients_out[pid] = {}
        for model_id in sorted(_GRADIENT_MODELS):
            weights = by_id.get(model_id, {}).get("attention_weights")
            method = by_id.get(model_id, {}).get("importance_method")
            if method != "gradient" or not weights or len(weights) != WINDOW_HOURS:
                raise RuntimeError(
                    f"{pid}/{model_id}: gradient saliency uretilemedi "
                    f"(method={method}, len={len(weights) if weights else 0})"
                )
            patients_out[pid][model_id] = [round(float(w), 8) for w in weights]
        log.info("OK %s — lstm/gru/transformer saliency kaydedildi", pid)

    return {
        "version": 1,
        "window_hours": WINDOW_HOURS,
        "method": "gradient",
        "models": sorted(_GRADIENT_MODELS),
        "patients": patients_out,
    }


def main() -> None:
    """demo_saliency.json dosyasini yazar."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_saliency_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    log.info(
        "Kaydedildi: %s (%d hasta, modeller=%s)",
        OUT_PATH,
        len(payload["patients"]),
        payload["models"],
    )


if __name__ == "__main__":
    main()
