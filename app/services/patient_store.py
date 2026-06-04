"""Demo hasta saatlik seri deposu — parquet tabanli okuma."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.artifact_paths import resolve_feature_stats_path
from app.services.inference_registry import FEATURE_ORDER, LOG_TRANSFORM_COLS

log = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DEMO_DIR = _BACKEND_DIR / "artifacts" / "demo"
_FEATURE_STATS_PATH = resolve_feature_stats_path()

SCALE_COLS = {
    "HR", "O2Sat", "Temp", "MAP", "Resp", "BUN", "Chloride",
    "Creatinine", "Glucose", "Hct", "Hgb", "WBC", "Platelets",
}


def _load_scaler_stats() -> dict[str, dict]:
    """feature_stats.json scaler istatistiklerini yukler."""
    if _FEATURE_STATS_PATH.exists():
        return json.loads(_FEATURE_STATS_PATH.read_text()).get("scaler_stats", {})
    return {}


def inverse_feature_to_api(feat: str, scaled_val: float, stats: dict[str, dict]) -> float:
    """Islenmis (scaled/log) degeri API ham snapshot degerine cevirir."""
    v = float(scaled_val)
    if feat in SCALE_COLS:
        st = stats.get(feat, {})
        mean = st.get("mean", 0.0)
        std = st.get("std", 1.0) or 1.0
        v = v * std + mean
    if feat in LOG_TRANSFORM_COLS:
        v = math.expm1(max(v, 0.0))
    return round(v, 4)


def scaled_row_to_snapshot(row: pd.Series, stats: dict[str, dict]) -> dict[str, float]:
    """Tek satirdan PatientSnapshot sozlugu uretir."""
    out: dict[str, float] = {}
    for feat in FEATURE_ORDER:
        out[feat] = inverse_feature_to_api(feat, row[feat], stats)
    return out


class PatientStore:
    """Demo hasta parquet okuyucu."""

    def __init__(self) -> None:
        """Parquet ve manifest yukler."""
        self._parquet = _DEMO_DIR / "demo_patients.parquet"
        self._manifest_path = _DEMO_DIR / "demo_manifest.json"
        self._df: pd.DataFrame | None = None
        self._manifest: dict | None = None
        self._stats = _load_scaler_stats()

    def _ensure_loaded(self) -> None:
        """Lazy parquet yukleme."""
        if self._df is None:
            if not self._parquet.exists():
                raise FileNotFoundError(f"Demo parquet bulunamadi: {self._parquet}")
            self._df = pd.read_parquet(self._parquet)
        if self._manifest is None:
            if self._manifest_path.exists():
                self._manifest = json.loads(self._manifest_path.read_text())
            else:
                self._manifest = {"patients": []}

    def list_demo_patients(self) -> list[dict[str, Any]]:
        """Manifest'teki demo hasta listesini dondurur."""
        self._ensure_loaded()
        return list(self._manifest.get("patients", []))

    def get_window(
        self,
        patient_id: str,
        hours: int = 24,
        end_hour: int | None = None,
    ) -> dict[str, Any]:
        """Hastanin son N saatlik serisini API snapshot formatinda dondurur.

        Args:
            patient_id: Hasta kimligi.
            hours: Pencere uzunlugu (saat).
            end_hour: Bitis saati; None ise max Hour.

        Returns:
            patient_id, hours, end_hour, sepsis, series listesi.
        """
        self._ensure_loaded()
        assert self._df is not None
        pdf = self._df[self._df["Patient_ID"] == patient_id].sort_values("Hour")
        if pdf.empty:
            raise KeyError(f"Hasta bulunamadi: {patient_id}")

        if end_hour is None:
            end_hour = int(pdf["Hour"].max())
        window_df = pdf[pdf["Hour"] <= end_hour].tail(hours)
        if len(window_df) < hours:
            pad_n = hours - len(window_df)
            first = window_df.iloc[0] if len(window_df) else pdf.iloc[0]
            pad_rows = pd.DataFrame([first] * pad_n)
            window_df = pd.concat([pad_rows, window_df], ignore_index=True)

        series = [
            {"hour": int(row["Hour"]), **scaled_row_to_snapshot(row, self._stats)}
            for _, row in window_df.iterrows()
        ]
        meta = next(
            (p for p in self._manifest.get("patients", []) if p["patient_id"] == patient_id),
            {},
        )
        return {
            "patient_id": patient_id,
            "hours": hours,
            "end_hour": end_hour,
            "sepsis": bool(meta.get("sepsis", pdf["SepsisLabel"].max() == 1)),
            "horizon_label_end": int(window_df.iloc[-1].get("HorizonLabel", 0)),
            "series": series,
        }

    def get_latest_snapshot(self, patient_id: str) -> dict[str, float]:
        """Hastanin son saatindeki klinik snapshot degerlerini API formatinda dondurur."""
        window = self.get_window(patient_id, hours=1)
        series = window.get("series") or []
        if not series:
            raise KeyError(f"Hasta icin snapshot yok: {patient_id}")
        last = series[-1]
        return {k: float(v) for k, v in last.items() if k != "hour"}


patient_store = PatientStore()
