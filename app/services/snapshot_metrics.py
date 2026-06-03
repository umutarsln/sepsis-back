"""Snapshot ML model metrikleri — Faz 4.6/4.7 ciktilarindan yukler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import HorizonComparisonRow, SnapshotModelMetrics

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SEPSIS_SON_DIR = Path(__file__).resolve().parents[3]

_ADM4_METRICS = _SEPSIS_SON_DIR / "adim_4_2026-05-07" / "ciktilar" / "metrics_5_models.json"
_ADM46_OPTUNA = _SEPSIS_SON_DIR / "adim_4_6_2026-05-20" / "ciktilar" / "metrics_optuna_h6.json"
_ADM47_H0 = _SEPSIS_SON_DIR / "adim_4_7_2026-05-23" / "ciktilar" / "metrics_h0.json"
_ADM47_H24 = _SEPSIS_SON_DIR / "adim_4_7_2026-05-23" / "ciktilar" / "metrics_h24.json"

_MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "gradient_boosting",
    "gaussian_nb",
]

_HORIZON_LABELS: dict[int, str] = {
    0: "Anlik sepsis tespiti (h=0, SepsisLabel)",
    6: "Erken uyari (h=6, Optuna XGB/RF + Faz 4 baseline)",
    24: "24 saat erken uyari (h=24)",
}

_HORIZON_SOURCES: dict[int, str] = {
    0: "adim_4_7/metrics_h0.json",
    6: "adim_4 + adim_4_6 Optuna (XGB, RF)",
    24: "adim_4_7/metrics_h24.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    """JSON dosyasini sozluk olarak yukler."""
    return json.loads(path.read_text(encoding="utf-8"))


def merge_h6_metrics() -> dict[str, dict[str, float]]:
    """Faz 4 baseline uzerine Faz 4.6 Optuna XGB/RF metriklerini birlestirir."""
    if not _ADM4_METRICS.exists():
        return {}
    merged: dict[str, dict[str, float]] = dict(_load_json(_ADM4_METRICS))
    if _ADM46_OPTUNA.exists():
        optuna = _load_json(_ADM46_OPTUNA).get("optuna_test") or {}
        for model_id, values in optuna.items():
            if isinstance(values, dict):
                merged[model_id] = values
    return merged


def load_horizon_metrics_raw(horizon: int) -> dict[str, dict[str, float]]:
    """Belirtilen ufuk icin ham model metrik sozlugunu dondurur."""
    if horizon == 6:
        return merge_h6_metrics()
    if horizon == 0:
        if _ADM47_H0.exists():
            return _load_json(_ADM47_H0).get("metrics") or {}
        return {}
    if horizon == 24:
        if _ADM47_H24.exists():
            return _load_json(_ADM47_H24).get("metrics") or {}
        return {}
    raise ValueError(f"Desteklenmeyen horizon: {horizon}")


def _row_to_metrics(model_id: str, row: dict[str, Any], source: str) -> SnapshotModelMetrics:
    """Ham metrik satirini SnapshotModelMetrics nesnesine cevirir."""
    return SnapshotModelMetrics(
        model_id=model_id,
        auroc=float(row.get("auroc") or 0.0),
        auprc=float(row.get("auprc") or 0.0),
        sens_at_spec85=float(row.get("sens_at_spec85") or 0.0),
        f1=float(row.get("f1") or 0.0),
        threshold=float(row.get("threshold") or 0.0),
        brier=float(row.get("brier")) if row.get("brier") is not None else None,
        source=source,
    )


def get_snapshot_metrics(horizon: int) -> list[SnapshotModelMetrics]:
    """Ufuk bazli 5 ML model test metriklerini dondurur."""
    if horizon not in (0, 6, 24):
        raise ValueError(f"Desteklenmeyen horizon: {horizon}")
    raw = load_horizon_metrics_raw(horizon)
    source = _HORIZON_SOURCES[horizon]
    return [
        _row_to_metrics(model_id, raw[model_id], source)
        for model_id in _MODEL_ORDER
        if model_id in raw
    ]


def get_horizon_label(horizon: int) -> str:
    """Ufuk aciklama metnini dondurur."""
    return _HORIZON_LABELS.get(horizon, f"h={horizon}")


def get_metrics_source_label(horizon: int) -> str:
    """Metrik kaynak aciklama metnini dondurur."""
    return _HORIZON_SOURCES.get(horizon, f"h={horizon}")


def build_horizon_comparison_rows() -> list[HorizonComparisonRow]:
    """5 ML model icin h=0/6/24 AUROC ve AUPRC karsilastirma tablosu uretir."""
    by_h: dict[int, dict[str, dict[str, float]]] = {}
    for h in (0, 6, 24):
        by_h[h] = load_horizon_metrics_raw(h)

    rows: list[HorizonComparisonRow] = []
    for model_id in _MODEL_ORDER:
        m0 = by_h[0].get(model_id, {})
        m6 = by_h[6].get(model_id, {})
        m24 = by_h[24].get(model_id, {})
        if not m0 and not m6 and not m24:
            continue
        rows.append(
            HorizonComparisonRow(
                model_id=model_id,
                model_name=_model_display_name(model_id),
                h0_auroc=float(m0.get("auroc") or 0.0),
                h6_auroc=float(m6.get("auroc") or 0.0),
                h24_auroc=float(m24.get("auroc") or 0.0),
                h0_auprc=float(m0.get("auprc") or 0.0),
                h6_auprc=float(m6.get("auprc") or 0.0),
                h24_auprc=float(m24.get("auprc") or 0.0),
            )
        )
    return rows


def _model_display_name(model_id: str) -> str:
    """Model kimliginden kisa gorunen ad uretir."""
    labels = {
        "logistic_regression": "Lojistik Reg.",
        "random_forest": "Rastgele Orman",
        "xgboost": "XGBoost",
        "gradient_boosting": "Gradyan Artirma",
        "gaussian_nb": "Gaussian NB",
    }
    return labels.get(model_id, model_id)
