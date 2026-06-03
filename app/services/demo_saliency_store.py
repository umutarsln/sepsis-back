"""Demo hastalar icin onceden hesaplanmis gradient saliency deposu."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SALIENCY_PATH = _BACKEND_DIR / "artifacts" / "demo" / "demo_saliency.json"

_GRADIENT_MODELS = frozenset({"lstm", "gru", "transformer"})


class DemoSaliencyStore:
    """Offline uretilmis LSTM/GRU/Transformer timestep onem agirliklarini okur."""

    def __init__(self, path: Path | None = None) -> None:
        """Depo yolunu ayarlar; dosya yoksa bos sozluk kullanilir."""
        self._path = path or _SALIENCY_PATH
        self._data: dict[str, Any] | None = None

    def _ensure_loaded(self) -> None:
        """JSON dosyasini lazy yukler."""
        if self._data is not None:
            return
        if not self._path.exists():
            log.info("Demo saliency dosyasi yok: %s", self._path)
            self._data = {"patients": {}}
            return
        self._data = json.loads(self._path.read_text())

    def get_weights(
        self,
        patient_id: str | None,
        model_id: str,
        window_hours: int = 24,
    ) -> list[float] | None:
        """Hasta ve model icin onceden hesaplanmis 24-adim saliency dondurur.

        Args:
            patient_id: Demo hasta kimligi (or. p000939).
            model_id: lstm, gru veya transformer.
            window_hours: Beklenen pencere uzunlugu.

        Returns:
            Normalize edilmis saliency listesi veya bulunamazsa None.
        """
        if not patient_id or model_id not in _GRADIENT_MODELS:
            return None
        self._ensure_loaded()
        assert self._data is not None
        if int(self._data.get("window_hours", 24)) != window_hours:
            return None
        patient_block = self._data.get("patients", {}).get(patient_id, {})
        weights = patient_block.get(model_id)
        if not isinstance(weights, list) or len(weights) != window_hours:
            return None
        return [float(w) for w in weights]

    def is_available(self) -> bool:
        """En az bir demo hasta saliency kaydi var mi kontrol eder."""
        self._ensure_loaded()
        assert self._data is not None
        return bool(self._data.get("patients"))


demo_saliency_store = DemoSaliencyStore()
