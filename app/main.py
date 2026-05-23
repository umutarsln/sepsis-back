"""Sepsis-son backend uygulama giris noktasi — Faz 8 ile 13 endpoint, tam Swagger/OpenAPI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from app.schemas import (
    DatasetCohortSummary,
    DatasetLabelSummary,
    DatasetLengthSummary,
    DatasetMissingRow,
    DatasetSplitSummary,
    DatasetSummaryResponse,
    DemoPatientSummary,
    ExperimentMetrics,
    ExperimentRow,
    HealthResponse,
    LeadTimeSummary,
    ModelScore,
    PatientPreset,
    PatientWindowResponse,
    ShapContribution,
    SnapshotExplainResponse,
    SnapshotPredictionRequest,
    SnapshotPredictionResponse,
    WindowModelResult,
    WindowPredictionRequest,
    WindowPredictionResponse,
)
from app.services.inference_registry import InferenceRegistry
from app.services.patient_store import patient_store

# ---------------------------------------------------------------------------
# Uygulama tanımı — OpenAPI metadata
# ---------------------------------------------------------------------------

_DESCRIPTION = """
## Sepsis Erken Uyarı Sistemi — REST API

PhysioNet 2019 Challenge veri seti üzerinde eğitilen **5 ML + 4 DL** modeli
kullanarak yoğun bakım hastalarında sepsis riskini tahmin eder.

### Özellikler
- **18 klinik feature** (vital bulgular, lab değerleri, demografik)
- **SHAP / LIME / Attention** ile tahmin açıklaması
- **Swagger UI** üzerinden doğrudan test edilebilir
"""

_TAGS: list[dict[str, str]] = [
    {
        "name": "Saglik",
        "description": "Servis canlılık kontrolü.",
    },
    {
        "name": "Tahmin",
        "description": "ML (snapshot) ve DL (pencere) model tahminleri.",
    },
    {
        "name": "Aciklama",
        "description": "SHAP global özet, LIME örnek açıklaması, Attention ısı haritası.",
    },
    {
        "name": "Metaveri",
        "description": "Model tanımlayıcıları, öznitelik istatistikleri, hasta ön ayarları, karşılaştırma tabloları.",
    },
]

app = FastAPI(
    title="Sepsis Erken Uyarı API",
    description=_DESCRIPTION,
    version="0.8.0",
    openapi_tags=_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
)

inference_registry = InferenceRegistry()

# ---------------------------------------------------------------------------
# Dizin sabitleri
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parents[1]  # backend/
_SEPSIS_SON_DIR = Path(__file__).resolve().parents[2]  # sepsis-son/

_EXPLAINABILITY_DIR = _BACKEND_DIR / "artifacts" / "results" / "explainability"
_LEAD_TIME_JSON = _BACKEND_DIR / "artifacts" / "results" / "lead_time" / "lead_time_summary.json"

_ADM2_DIR = _SEPSIS_SON_DIR / "adim_2_2026-05-07" / "ciktilar"
_ADM3_DIR = _SEPSIS_SON_DIR / "adim_3_2026-05-07" / "ciktilar"
_ADM4_DIR = _SEPSIS_SON_DIR / "adim_4_2026-05-07" / "ciktilar"
_ADM5_DIR = _SEPSIS_SON_DIR / "adim_5_2026-05-08" / "ciktilar"
_ADM6_DIR = _SEPSIS_SON_DIR / "adim_6_2026-05-09" / "ciktilar"
_ADM7_DIR = _SEPSIS_SON_DIR / "adim_7_2026-05-09" / "ciktilar"

# Faz 4 ML varsayilan hiperparametreleri (train_5_models.py ile uyumlu).
_ML_DEFAULT_PARAMS: dict[str, dict[str, float | int | str]] = {
    "logistic_regression": {
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 1000,
    },
    "random_forest": {"n_estimators": 300, "class_weight": "balanced"},
    "gradient_boosting": {"n_estimators": 200, "random_state": 42},
    "gaussian_nb": {},
}

_ML_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting",
    "gaussian_nb": "Gaussian NB",
}

_DL_LABELS: dict[str, str] = {
    "lstm": "LSTM",
    "gru": "GRU",
    "bigru_attn": "BiGRU+Attention",
    "transformer": "Transformer",
}

# Faz 5 DL egitim tier tanimlari (train_dl_models.py ile uyumlu).
_DL_TIERS: dict[str, dict[str, float | int | str | None]] = {
    "quick": {"epochs": 10, "hidden": 64, "batch": 128, "patience": None, "lr": 0.001},
    "standard": {"epochs": 20, "hidden": 64, "batch": 128, "patience": 5, "lr": 0.001},
    "thorough": {"epochs": 30, "hidden": 128, "batch": 128, "patience": 7, "lr": 0.001},
}

_FINAL_BENCHMARK_KEYS: set[tuple[str, str, str | None]] = {
    ("Faz 4", model_id, None) for model_id in _ML_LABELS
} | {
    ("Faz 5", model_id, "thorough")
    for model_id in ("lstm", "gru", "bigru_attn")
} | {("Faz 6", "transformer", None)}

# Path traversal korumasi — izin verilen model kimlikleri
_SHAP_ALLOWED = {"xgboost", "random_forest", "logistic_regression"}
_ATTN_ALLOWED = {"bigru_attn", "transformer"}

# Simulatör slider sinirlari icin klinik referans araliklari (PhysioNet 2019 tabanli).
_CLINICAL_RANGES: dict[str, dict[str, float | str]] = {
    "HR": {"min": 30, "max": 200, "normal_low": 60, "normal_high": 100, "unit": "bpm"},
    "O2Sat": {"min": 60, "max": 100, "normal_low": 95, "normal_high": 100, "unit": "%"},
    "Temp": {"min": 33, "max": 42, "normal_low": 36.1, "normal_high": 37.5, "unit": "°C"},
    "MAP": {"min": 30, "max": 150, "normal_low": 70, "normal_high": 105, "unit": "mmHg"},
    "Resp": {"min": 5, "max": 50, "normal_low": 12, "normal_high": 20, "unit": "/dk"},
    "BUN": {"min": 1, "max": 150, "normal_low": 7, "normal_high": 20, "unit": "mg/dL"},
    "Chloride": {"min": 80, "max": 130, "normal_low": 96, "normal_high": 106, "unit": "mEq/L"},
    "Creatinine": {"min": 0.1, "max": 15, "normal_low": 0.6, "normal_high": 1.3, "unit": "mg/dL"},
    "Glucose": {"min": 30, "max": 600, "normal_low": 70, "normal_high": 140, "unit": "mg/dL"},
    "Hct": {"min": 15, "max": 60, "normal_low": 36, "normal_high": 50, "unit": "%"},
    "Hgb": {"min": 5, "max": 20, "normal_low": 12, "normal_high": 17, "unit": "g/dL"},
    "WBC": {"min": 0.5, "max": 80, "normal_low": 4, "normal_high": 11, "unit": "K/µL"},
    "Platelets": {"min": 10, "max": 800, "normal_low": 150, "normal_high": 400, "unit": "K/µL"},
    "Age": {"min": 18, "max": 100, "normal_low": 18, "normal_high": 65, "unit": "yıl"},
    "HospAdmTime": {"min": -300, "max": 0, "normal_low": -48, "normal_high": 0, "unit": "saat"},
    "ICULOS": {"min": 0, "max": 200, "normal_low": 0, "normal_high": 48, "unit": "saat"},
}

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    """JSON dosyasini okur; yoksa 404 yukseltir."""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact bulunamadi: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_optional(path: Path) -> Any | None:
    """JSON dosyasini okur; yoksa None doner (deney listesi icin)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_from_raw(raw: dict[str, Any], val_auroc: float | None = None) -> ExperimentMetrics:
    """Ham metrik sozlugunden ExperimentMetrics nesnesi uretir."""
    return ExperimentMetrics(
        test_auroc=float(raw["auroc"]) if raw.get("auroc") is not None else None,
        val_auroc=val_auroc,
        auprc=float(raw["auprc"]) if raw.get("auprc") is not None else None,
        f1=float(raw["f1"]) if raw.get("f1") is not None else None,
        sens_at_spec85=float(sens_raw) if (sens_raw := raw.get("sens_at_spec85", raw.get("sens_spec85"))) is not None else None,
        brier=float(raw["brier"]) if raw.get("brier") is not None else None,
        threshold=float(raw["threshold"]) if raw.get("threshold") is not None else None,
    )


def _append_faz4_experiments(experiments: list[ExperimentRow]) -> None:
    """Faz 4 ML egitim kosularini metrics_5_models.json'dan listeye ekler."""
    metrics = _load_json_optional(_ADM4_DIR / "metrics_5_models.json")
    if not isinstance(metrics, dict):
        return

    xgb_meta = _load_json_optional(_ADM4_DIR / "xgb_best_params.json") or {}
    xgb_val_auroc = float(xgb_meta["val_auroc"]) if xgb_meta.get("val_auroc") is not None else None

    for model_id, raw in metrics.items():
        if model_id == "xgboost":
            params: dict[str, float | int | str | None] = {
                "max_depth": xgb_meta.get("max_depth", 4),
                "learning_rate": xgb_meta.get("learning_rate", 0.05),
                "n_estimators": 300,
                "scale_pos_weight": 35.89,
            }
            val_auroc = xgb_val_auroc
        else:
            params = dict(_ML_DEFAULT_PARAMS.get(model_id, {}))
            val_auroc = None

        experiments.append(
            ExperimentRow(
                id=f"faz4-{model_id}",
                name=f"{_ML_LABELS.get(model_id, model_id)} — Faz 4 baseline",
                phase="Faz 4",
                tier=None,
                status="completed",
                model=_ML_LABELS.get(model_id, model_id),
                model_id=model_id,
                model_family="ML",
                params=params,
                metrics=_metrics_from_raw(raw, val_auroc=val_auroc),
                duration="~7 dk" if model_id == "gradient_boosting" else None,
                created_at="2026-05-07",
                is_final=("Faz 4", model_id, None) in _FINAL_BENCHMARK_KEYS,
                notes="XGB grid: max_depth×learning_rate (6 kombinasyon)"
                if model_id == "xgboost"
                else None,
            )
        )


def _append_faz5_experiments(experiments: list[ExperimentRow]) -> None:
    """Faz 5 DL tier kosularini (quick/standard/thorough) listeye ekler."""
    tier_notes = {
        "thorough": "LSTM erken durdurma (epoch 18); GRU en iyi AUROC (+0.005 vs standard)",
    }
    for tier, tier_cfg in _DL_TIERS.items():
        metrics = _load_json_optional(_ADM5_DIR / tier / "metrics_dl.json")
        if not isinstance(metrics, dict):
            continue

        for model_id, raw in metrics.items():
            params = {
                **tier_cfg,
                "num_layers": 2,
                "dropout": 0.3,
                "loss": "BCEWithLogitsLoss",
                "optimizer": "Adam",
                "window_hours": 24,
                "horizon_h": 6,
            }
            experiments.append(
                ExperimentRow(
                    id=f"faz5-{tier}-{model_id}",
                    name=f"{_DL_LABELS.get(model_id, model_id)} — {tier} tier",
                    phase="Faz 5",
                    tier=tier,
                    status="completed",
                    model=_DL_LABELS.get(model_id, model_id),
                    model_id=model_id,
                    model_family="DL",
                    params=params,
                    metrics=_metrics_from_raw(raw),
                    duration=None,
                    created_at="2026-05-08",
                    is_final=("Faz 5", model_id, tier) in _FINAL_BENCHMARK_KEYS,
                    notes=tier_notes.get(tier) if model_id == "gru" else None,
                )
            )


def _append_faz6_experiments(experiments: list[ExperimentRow]) -> None:
    """Faz 6 Transformer kosusunu transformer_metrics.json'dan listeye ekler."""
    metrics = _load_json_optional(_ADM6_DIR / "transformer_metrics.json")
    if not isinstance(metrics, dict):
        return

    raw = metrics.get("transformer")
    if not isinstance(raw, dict):
        return

    experiments.append(
        ExperimentRow(
            id="faz6-transformer",
            name="Temporal Transformer — Faz 6",
            phase="Faz 6",
            tier=None,
            status="completed",
            model="Transformer",
            model_id="transformer",
            model_family="Transformer",
            params={
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.2,
                "lr": 0.001,
                "batch": 128,
                "epochs": 20,
                "window_hours": 24,
                "horizon_h": 6,
            },
            metrics=_metrics_from_raw(raw),
            duration=None,
            created_at="2026-05-09",
            is_final=("Faz 6", "transformer", None) in _FINAL_BENCHMARK_KEYS,
            notes="Flat ML + DL ile karsilastirma tablosuna eklendi",
        )
    )


def _build_experiments() -> list[ExperimentRow]:
    """Faz 4-6 egitim artifact'lerinden birlesik deney listesi uretir."""
    experiments: list[ExperimentRow] = []
    _append_faz4_experiments(experiments)
    _append_faz5_experiments(experiments)
    _append_faz6_experiments(experiments)

    if not experiments:
        raise HTTPException(status_code=404, detail="Hic deney kaydi bulunamadi")

    experiments.sort(key=lambda row: (row.created_at, row.phase, row.tier or "", row.model_id))
    return experiments


def _missing_lookup(eda: dict[str, Any]) -> dict[str, tuple[float, str]]:
    """EDA ozetinden feature -> (missing_pct, category) sozlugu uretir."""
    lookup: dict[str, tuple[float, str]] = {}
    for category in ("vitals", "labs", "demographics", "clinical"):
        for row in eda.get("missing", {}).get(category, []):
            feature = str(row.get("feature", ""))
            if not feature:
                continue
            lookup[feature] = (float(row.get("missing_pct") or 0), category)
    return lookup


def _build_dataset_summary() -> DatasetSummaryResponse:
    """Faz 2 EDA + Faz 3 split artifact'lerinden veri analizi ozeti uretir."""
    eda = _load_json(_ADM2_DIR / "eda_summary.json")
    splits = _load_json(_ADM3_DIR / "splits.json")
    feature_stats = _load_json(_ADM3_DIR / "feature_stats.json")

    kohort = eda.get("kohort", {})
    etiket = eda.get("etiket", {})
    uzunluk = eda.get("uzunluk", {})
    missing_lookup = _missing_lookup(eda)

    feature_order = list(feature_stats.get("feature_order") or [])
    selected_missing: list[DatasetMissingRow] = []
    for feature in feature_order:
        if feature in ("Gender_0", "Gender_1"):
            pct, category = missing_lookup.get("Gender", (0.0, "demographics"))
        else:
            pct, category = missing_lookup.get(feature, (0.0, "unknown"))
        selected_missing.append(
            DatasetMissingRow(
                feature=feature,
                missing_pct=round(pct, 2),
                category=category,
            )
        )

    all_missing_rows = []
    for category in ("vitals", "labs", "demographics", "clinical"):
        all_missing_rows.extend(eda.get("missing", {}).get(category, []))
    features_above_80 = sum(
        1 for row in all_missing_rows if float(row.get("missing_pct") or 0) > 80
    )

    train_n = int(splits.get("train_n") or 0)
    val_n = int(splits.get("val_n") or 0)
    test_n = int(splits.get("test_n") or 0)
    train_rate = float(splits.get("train_sepsis_rate") or 0)
    val_rate = float(splits.get("val_sepsis_rate") or 0)
    test_rate = float(splits.get("test_sepsis_rate") or 0)

    split_chart = [
        {
            "split": "Train",
            "patients": train_n,
            "sepsis_patients": round(train_n * train_rate),
            "sepsis_rate_pct": round(train_rate * 100, 2),
        },
        {
            "split": "Val",
            "patients": val_n,
            "sepsis_patients": round(val_n * val_rate),
            "sepsis_rate_pct": round(val_rate * 100, 2),
        },
        {
            "split": "Test",
            "patients": test_n,
            "sepsis_patients": round(test_n * test_rate),
            "sepsis_rate_pct": round(test_rate * 100, 2),
        },
    ]

    icu_length_chart = [
        {"label": "P5", "hours": float(uzunluk.get("p5") or 0)},
        {"label": "P25", "hours": float(uzunluk.get("p25") or 0)},
        {"label": "Medyan", "hours": float(uzunluk.get("median") or 0)},
        {"label": "P75", "hours": float(uzunluk.get("p75") or 0)},
        {"label": "P95", "hours": float(uzunluk.get("p95") or 0)},
    ]

    return DatasetSummaryResponse(
        cohort=DatasetCohortSummary(
            total_patients=int(kohort.get("toplam_hasta") or 0),
            set_a_patients=int(kohort.get("set_A") or 0),
            set_b_patients=int(kohort.get("set_B") or 0),
            total_rows=int(kohort.get("toplam_satir") or 0),
        ),
        labels=DatasetLabelSummary(
            sepsis_positive_patients=int(etiket.get("sepsis_pozitif_hasta") or 0),
            sepsis_negative_patients=int(etiket.get("sepsis_negatif_hasta") or 0),
            sepsis_patient_rate_pct=float(etiket.get("sepsis_hasta_orani_pct") or 0),
            sepsis_positive_rows=int(etiket.get("sepsis_kayit_sayisi") or 0),
            sepsis_row_rate_pct=float(etiket.get("sepsis_kayit_orani_pct") or 0),
            onset_median_hours=float(etiket.get("onset_median_saat") or 0),
        ),
        length=DatasetLengthSummary(
            median=float(uzunluk.get("median") or 0),
            mean=float(uzunluk.get("mean") or 0),
            p5=float(uzunluk.get("p5") or 0),
            p25=float(uzunluk.get("p25") or 0),
            p75=float(uzunluk.get("p75") or 0),
            p95=float(uzunluk.get("p95") or 0),
            min=float(uzunluk.get("min") or 0),
            max=float(uzunluk.get("max") or 0),
        ),
        splits=DatasetSplitSummary(
            train_patients=train_n,
            val_patients=val_n,
            test_patients=test_n,
            train_sepsis_rate_pct=round(train_rate * 100, 2),
            val_sepsis_rate_pct=round(val_rate * 100, 2),
            test_sepsis_rate_pct=round(test_rate * 100, 2),
            train_sepsis_patients=round(train_n * train_rate),
            val_sepsis_patients=round(val_n * val_rate),
            test_sepsis_patients=round(test_n * test_rate),
            seed=int(splits.get("seed") or 0),
            frozen=bool(splits.get("frozen")),
        ),
        final_feature_count=len(feature_order),
        features_above_80pct_missing=features_above_80,
        selected_feature_missing=selected_missing,
        icu_length_chart=icu_length_chart,
        split_chart=split_chart,
        source_files=[
            "adim_2_2026-05-07/ciktilar/eda_summary.json",
            "adim_3_2026-05-07/ciktilar/splits.json",
            "adim_3_2026-05-07/ciktilar/feature_stats.json",
        ],
    )


def _csv_to_records(path: Path) -> list[dict[str, Any]]:
    """CSV dosyasini dict listesine cevirir; yoksa 404 yukseltir."""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact bulunamadi: {path.name}")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _lead_time_from_mapping(data: dict[str, Any]) -> LeadTimeSummary:
    """Ham sozlukten frontend uyumlu LeadTimeSummary nesnesi uretir."""
    return LeadTimeSummary(
        n_positive_patients=int(data.get("n_positive_patients") or 0),
        n_detected=int(data.get("n_detected") or 0),
        detection_rate=float(data.get("detection_rate") or 0),
        n_early_alarm=int(data.get("n_early_alarm") or 0),
        early_alarm_rate=float(data.get("early_alarm_rate") or 0),
        median_lead_time_hours=float(data.get("median_lead_time_hours") or 0),
        mean_lead_time_hours=float(data.get("mean_lead_time_hours") or 0),
        q25_lead_time=float(data.get("q25_lead_time") or 0),
        q75_lead_time=float(data.get("q75_lead_time") or 0),
        version=str(data.get("version") or "v5"),
        threshold_at_spec85=float(data.get("threshold_at_spec85") or 0),
    )


def _lead_time_from_faz4_csv(model_id: str = "xgboost") -> LeadTimeSummary:
    """Faz 4 lead_time_summary.csv satirindan ozet metrik uretir."""
    rows = _csv_to_records(_ADM4_DIR / "lead_time_summary.csv")
    row = next((r for r in rows if r.get("model") == model_id), None)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Lead-time kaydi bulunamadi: {model_id}",
        )

    n_total = int(float(row.get("n_sepsis_total") or 0))
    n_caught = int(float(row.get("n_caught") or 0))
    detection_rate = (n_caught / n_total) if n_total else 0.0

    return LeadTimeSummary(
        n_positive_patients=n_total,
        n_detected=n_caught,
        detection_rate=detection_rate,
        n_early_alarm=n_caught,
        early_alarm_rate=detection_rate,
        median_lead_time_hours=float(row.get("median_lead_h") or 0),
        mean_lead_time_hours=float(row.get("mean_lead_h") or 0),
        q25_lead_time=0.0,
        q75_lead_time=0.0,
        version=model_id,
        threshold_at_spec85=float(row.get("threshold") or 0),
    )


def _gender_from_snapshot(snapshot: dict[str, float]) -> str:
    """Gender_0/Gender_1 one-hot alanlarindan M/F cinsiyet kodu uretir."""
    if snapshot.get("Gender_0", 0.0) >= 0.5:
        return "F"
    if snapshot.get("Gender_1", 0.0) >= 0.5:
        return "M"
    return "M"


def _snapshot_to_features(snapshot: dict[str, float]) -> dict[str, float]:
    """Snapshot sozlugunden Gender kolonlarini cikarip frontend features uretir."""
    return {
        key: float(value)
        for key, value in snapshot.items()
        if key not in {"Gender_0", "Gender_1"} and value is not None
    }


def _preset_from_snapshot_row(row: dict[str, Any]) -> PatientPreset:
    """Ic snapshot tanimini frontend PatientPreset formatina cevirir."""
    snapshot = {k: float(v) for k, v in row["snapshot"].items()}
    risk_band_map = {
        "dusuk_risk": "low",
        "yuksek_risk": "high",
        "sinir_durum": "medium",
    }
    return PatientPreset(
        preset_id=row["preset_id"],
        label=row["label"],
        risk_band=risk_band_map.get(row["preset_id"], "medium"),
        description=row["description"],
        gender=_gender_from_snapshot(snapshot),
        features=_snapshot_to_features(snapshot),
    )


def _with_clinical_ranges(stats: dict[str, Any]) -> dict[str, Any]:
    """feature_stats yanitina eksikse klinik slider araliklarini ekler."""
    if not stats.get("clinical_ranges"):
        stats = dict(stats)
        stats["clinical_ranges"] = _CLINICAL_RANGES
    return stats


# ---------------------------------------------------------------------------
# Sağlık
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Saglik"],
    summary="Servis canlılık kontrolü",
    response_description="Servis çalışıyorsa status='ok' döner.",
)
def health() -> HealthResponse:
    """Backend servisinin ayakta olduğunu doğrular.

    Yük dengeleyici health-check probe'ları için kullanılır.
    """
    return HealthResponse()


# ---------------------------------------------------------------------------
# Tahmin endpoint'leri
# ---------------------------------------------------------------------------


@app.post(
    "/predict/snapshot",
    response_model=SnapshotPredictionResponse,
    tags=["Tahmin"],
    summary="Anlık ölçümden risk skoru (5 ML model)",
    response_description="Her ML modeli için 0–1 arası risk skoru ve alert bayrağı.",
    responses={
        422: {"description": "Giriş doğrulama hatası — alan tipi uyumsuzluğu."},
        500: {"description": "Model inference hatası."},
    },
)
def predict_snapshot(req: SnapshotPredictionRequest) -> SnapshotPredictionResponse:
    """18 klinik feature içeren anlık ölçümü 5 ML modeline gönderir (h=6 erken uyarı).

    Eksik alanlar 0.0 ile doldurulur; log dönüşümü ve StandardScaler uygulanır.
    Dönüş değerindeki **alert** alanı, model eşiği aşıldığında True olur.
    """
    raw = inference_registry.predict_snapshot(req.snapshot.model_dump())
    scores = [ModelScore(**m) for m in raw]
    return SnapshotPredictionResponse(models=scores, horizon=6)


@app.post(
    "/predict/snapshot/current",
    response_model=SnapshotPredictionResponse,
    tags=["Tahmin"],
    summary="Anlık sepsis tespiti (h=0, 5 ML model)",
    response_description="SepsisLabel hedefiyle eğitilmiş modeller; mevcut saatte sepsis riski.",
)
def predict_snapshot_current(req: SnapshotPredictionRequest) -> SnapshotPredictionResponse:
    """h=0 (anlık SepsisLabel) modelleri ile risk skoru döner."""
    raw = inference_registry.predict_snapshot_horizon(req.snapshot.model_dump(), horizon=0)
    scores = [ModelScore(**m) for m in raw]
    return SnapshotPredictionResponse(models=scores, horizon=0)


@app.post(
    "/predict/snapshot/horizon/{horizon_hours}",
    response_model=SnapshotPredictionResponse,
    tags=["Tahmin"],
    summary="Belirtilen tahmin ufkunda risk skoru (h=0, 6 veya 24)",
)
def predict_snapshot_horizon(
    horizon_hours: int,
    req: SnapshotPredictionRequest,
) -> SnapshotPredictionResponse:
    """Seçilen horizon için 5 ML model skoru döner."""
    if horizon_hours not in (0, 6, 24):
        raise HTTPException(status_code=422, detail="horizon_hours yalnizca 0, 6 veya 24 olabilir")
    try:
        raw = inference_registry.predict_snapshot_horizon(
            req.snapshot.model_dump(), horizon=horizon_hours
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    scores = [ModelScore(**m) for m in raw]
    return SnapshotPredictionResponse(models=scores, horizon=horizon_hours)


@app.post(
    "/predict/snapshot/explain",
    response_model=SnapshotExplainResponse,
    tags=["Tahmin", "Aciklama"],
    summary="Anlık ölçümden risk skoru + SHAP açıklaması",
    response_description="ML model skorları ve XGBoost SHAP top-5 feature katkısı.",
    responses={
        422: {"description": "Giriş doğrulama hatası."},
        500: {"description": "Model inference veya SHAP hesaplama hatası."},
    },
)
def predict_snapshot_explain(req: SnapshotPredictionRequest) -> SnapshotExplainResponse:
    """Snapshot skorlaması yapar ve XGBoost için SHAP değerlerini hesaplar.

    **shap_top5** listesi mutlak SHAP değerine göre azalan sırada sıralanmıştır.
    Her elemanın **pct_contribution** alanı, toplam SHAP aktivasyonuna oranı gösterir.
    """
    raw = inference_registry.predict_snapshot_explain(req.snapshot.model_dump())
    scores = [ModelScore(**m) for m in raw["models"]]
    top5 = None
    if raw.get("shap_top5") is not None:
        top5 = [ShapContribution(**s) for s in raw["shap_top5"]]
    return SnapshotExplainResponse(models=scores, shap_top5=top5, horizon=raw.get("horizon", 6))


@app.post(
    "/predict/window",
    response_model=WindowPredictionResponse,
    tags=["Tahmin"],
    summary="24 saatlik zaman penceresi tahmini (4 DL model)",
    response_description="GRU, BiGRU+Attn, LSTM, Transformer model sonuçları.",
    responses={
        422: {
            "description": "repeat_hours 1–72 aralığı dışında veya alan tipi uyumsuzluğu."
        },
        500: {"description": "Torch inference hatası."},
    },
)
def predict_window(req: WindowPredictionRequest) -> WindowPredictionResponse:
    """Snapshot veya **series** ile DL pencere tahmini yapar.

    **series** verilirse son 24 saatin gercek dizisi kullanilir; yoksa snapshot
    **repeat_hours** kez tekrarlanir (geriye uyumlu demo modu).
    """
    snap = req.snapshot.model_dump()
    series_raw = None
    if req.series:
        series_raw = [s.model_dump() for s in req.series]
    raw = inference_registry.predict_window(
        snap,
        req.repeat_hours,
        series=series_raw,
    )
    dl_results = [WindowModelResult(**m) for m in raw["models"]]
    return WindowPredictionResponse(
        models=dl_results,
        window_shape=tuple(raw["window_shape"]),
        input_mode=raw.get("input_mode", "repeat"),
    )


# ---------------------------------------------------------------------------
# Explainability artifact endpoint'leri
# ---------------------------------------------------------------------------


@app.get(
    "/artifacts/shap-summary/{model_id}",
    tags=["Aciklama"],
    summary="Model bazlı SHAP global önem sıralaması",
    response_description="Feature adı ve ortalama |SHAP| değeri içeren liste.",
    responses={
        404: {"description": "model_id geçersiz veya artifact dosyası bulunamadı."},
    },
)
def get_shap_summary(model_id: str) -> list:
    """SHAP global feature ranking listesini döner.

    **model_id** parametresi şu değerleri kabul eder:
    - `xgboost`
    - `random_forest`
    - `logistic_regression`
    """
    if model_id not in _SHAP_ALLOWED:
        raise HTTPException(
            status_code=404,
            detail=f"Gecersiz model_id: '{model_id}'. Gecerli: {sorted(_SHAP_ALLOWED)}",
        )
    return _load_json(_EXPLAINABILITY_DIR / f"shap_summary_{model_id}.json")


@app.get(
    "/artifacts/attention/{model_id}",
    tags=["Aciklama"],
    summary="DL modeli attention ağırlıkları özeti",
    response_description="Timestep başına ortalama attention ağırlığı dict'i.",
    responses={
        404: {"description": "model_id geçersiz veya attention verisi bulunamadı."},
    },
)
def get_attention(model_id: str) -> dict:
    """Attention heatmap verilerini döner.

    **model_id** parametresi şu değerleri kabul eder:
    - `bigru_attn`
    - `transformer`
    """
    if model_id not in _ATTN_ALLOWED:
        raise HTTPException(
            status_code=404,
            detail=f"Gecersiz model_id: '{model_id}'. Gecerli: {sorted(_ATTN_ALLOWED)}",
        )
    data = _load_json(_EXPLAINABILITY_DIR / "attention_summary.json")
    if model_id not in data:
        raise HTTPException(
            status_code=404, detail=f"Model attention verisi mevcut degil: {model_id}"
        )
    return data[model_id]


@app.get(
    "/artifacts/feature-ranking",
    tags=["Aciklama"],
    summary="XGBoost SHAP feature önem sıralaması (Faz 7)",
    response_description="Tüm 18 feature için SHAP önem listesi (azalan sıra).",
)
def get_feature_ranking() -> list:
    """Faz 7 XGBoost SHAP global önem sıralamasını döner.

    `/artifacts/shap-summary/xgboost` ile aynı kaynağı kullanır;
    frontend feature ranking paneli için ayrı endpoint olarak sunulur.
    """
    return _load_json(_ADM7_DIR / "shap_summary_xgboost.json")


@app.get(
    "/artifacts/lime",
    tags=["Aciklama"],
    summary="LIME örnek açıklamaları (TP / FP / FN)",
    response_description="Üç hasta tipi için LIME top-10 feature katkısı.",
)
def get_lime() -> list:
    """Faz 7 LIME açıklamalarını döner.

    Her kayıt şu alanları içerir: **patient_type** (tp/fp/fn),
    **predicted_prob**, **top10_features**.
    """
    return _load_json(_ADM7_DIR / "lime_explanations.json")


@app.get(
    "/artifacts/version-comparison",
    tags=["Metaveri"],
    summary="ML vs DL model karşılaştırma tablosu (Faz 6)",
    response_description="Tüm modeller için AUROC, AUPRC, F1, Lead Time içeren tablo.",
)
def get_version_comparison() -> list:
    """Faz 6 karşılaştırma tablosunu JSON olarak döner.

    Kaynak: `adim_6_2026-05-09/ciktilar/version_comparison_summary.csv`

    Dönen sütunlar: `family`, `model_id`, `model_name`, `auroc`, `auprc`,
    `sens_spec85`, `f1`, `median_lead_h`, `mean_lead_h`.
    """
    return _csv_to_records(_ADM6_DIR / "version_comparison_summary.csv")


@app.get(
    "/artifacts/experiments",
    response_model=list[ExperimentRow],
    tags=["Metaveri"],
    summary="Faz 4-6 egitim deney gecmisi",
    response_description="Her egitim kosusunun hiperparametreleri ve test metrikleri.",
)
def get_experiments() -> list[ExperimentRow]:
    """Faz 4 ML, Faz 5 DL tier ve Faz 6 Transformer kosularini listeler.

    Modeller sayfasindan farki: final skor tablosu degil, egitim surecindeki
    tum kosular (ornegin quick vs thorough tier karsilastirmasi).
    """
    return _build_experiments()


@app.get(
    "/artifacts/dataset-summary",
    response_model=DatasetSummaryResponse,
    tags=["Metaveri"],
    summary="PhysioNet 2019 veri seti ozeti (Faz 2-3)",
    response_description="Kohort, etiket, split ve eksiklik istatistikleri.",
)
def get_dataset_summary() -> DatasetSummaryResponse:
    """Faz 2 EDA ve Faz 3 preprocessing split artifact'lerinden analiz ozeti doner."""
    return _build_dataset_summary()


@app.get(
    "/artifacts/lead-time",
    response_model=LeadTimeSummary,
    tags=["Metaveri"],
    summary="Erken uyarı süre analizi (V5 XGBoost özeti)",
    response_description="Yakalama oranı, medyan lead time ve erken alarm metrikleri.",
)
def get_lead_time() -> LeadTimeSummary:
    """Frontend dashboard ile uyumlu lead-time ozetini doner.

    Oncelik: `artifacts/results/lead_time/lead_time_summary.json`
    Yedek: Faz 4 `lead_time_summary.csv` icinden XGBoost satiri.
    """
    if _LEAD_TIME_JSON.exists():
        data = _load_json(_LEAD_TIME_JSON)
        if isinstance(data, dict) and data.get("versions"):
            data = data["versions"][0]
        return _lead_time_from_mapping(data)

    return _lead_time_from_faz4_csv("xgboost")


@app.get(
    "/models/descriptors",
    tags=["Metaveri"],
    summary="Tüm modellerin tanımlayıcı metrikleri",
    response_description="Model ID, ailenin adı, AUROC, AUPRC, F1 içeren liste.",
)
def get_model_descriptors() -> list:
    """Sistemde kayıtlı tüm ML ve DL modellerin tanımlayıcı metriklerini döner.

    Değerler Faz 4 (ML) ve Faz 6 (DL) raporlarından alınmıştır.
    """
    return [
        {
            "model_id": "logistic_regression",
            "model_name": "Lojistik Regresyon",
            "family": "ML",
            "auroc": 0.744,
            "auprc": 0.109,
            "f1_spec85": 0.154,
            "median_lead_h": 33.5,
        },
        {
            "model_id": "random_forest",
            "model_name": "Rastgele Orman",
            "family": "ML",
            "auroc": 0.798,
            "auprc": 0.145,
            "f1_spec85": 0.171,
            "median_lead_h": 31.0,
        },
        {
            "model_id": "xgboost",
            "model_name": "XGBoost",
            "family": "ML",
            "auroc": 0.822,
            "auprc": 0.177,
            "f1_spec85": 0.176,
            "median_lead_h": 30.0,
        },
        {
            "model_id": "gradient_boosting",
            "model_name": "Gradyan Artırma",
            "family": "ML",
            "auroc": 0.822,
            "auprc": 0.164,
            "f1_spec85": 0.174,
            "median_lead_h": 29.5,
        },
        {
            "model_id": "gaussian_nb",
            "model_name": "Gaussian NB",
            "family": "ML",
            "auroc": 0.700,
            "auprc": 0.093,
            "f1_spec85": 0.120,
            "median_lead_h": None,
        },
        {
            "model_id": "gru",
            "model_name": "GRU",
            "family": "DL",
            "auroc": 0.836,
            "auprc": 0.269,
            "f1_spec85": 0.271,
            "median_lead_h": None,
        },
        {
            "model_id": "bigru_attn",
            "model_name": "BiGRU + Attention",
            "family": "DL",
            "auroc": 0.826,
            "auprc": 0.247,
            "f1_spec85": 0.248,
            "median_lead_h": None,
        },
        {
            "model_id": "lstm",
            "model_name": "LSTM",
            "family": "DL",
            "auroc": 0.819,
            "auprc": 0.240,
            "f1_spec85": 0.235,
            "median_lead_h": None,
        },
        {
            "model_id": "transformer",
            "model_name": "Temporal Transformer",
            "family": "DL",
            "auroc": 0.842,
            "auprc": 0.284,
            "f1_spec85": 0.270,
            "median_lead_h": None,
        },
    ]


@app.get(
    "/preprocessing/feature-stats",
    tags=["Metaveri"],
    summary="18 özniteliğin ön-işlem istatistikleri",
    response_description="feature_order, log_transform_cols, scaler_stats, clinical_ranges içeren dict.",
)
def get_feature_stats() -> dict:
    """Faz 3'te hesaplanan öznitelik istatistiklerini döner.

    Kaynak: `adim_3_2026-05-07/ciktilar/feature_stats.json`

    Dönen alanlar: **feature_order**, **log_transform_cols**, **scaler_stats**,
    **clinical_ranges** (simulatör slider sınırları).
    """
    return _with_clinical_ranges(_load_json(_ADM3_DIR / "feature_stats.json"))


@app.get(
    "/patients/demo",
    response_model=list[DemoPatientSummary],
    tags=["Metaveri"],
    summary="Test setinden secilmis 10 demo hasta (5 sepsis + 5 non-sepsis)",
)
def list_demo_patients() -> list[DemoPatientSummary]:
    """Saatlik seri destekli DL demo icin hasta listesi."""
    try:
        rows = patient_store.list_demo_patients()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [DemoPatientSummary(**r) for r in rows]


@app.get(
    "/patients/{patient_id}/window",
    response_model=PatientWindowResponse,
    tags=["Metaveri"],
    summary="Hastanin son N saatlik klinik serisi",
)
def get_patient_window(
    patient_id: str,
    hours: int = 24,
    end_hour: int | None = None,
) -> PatientWindowResponse:
    """Demo parquet'ten hasta saatlik penceresini dondurur."""
    if hours < 1 or hours > 72:
        raise HTTPException(status_code=422, detail="hours 1-72 arasinda olmali")
    try:
        raw = patient_store.get_window(patient_id, hours=hours, end_hour=end_hour)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PatientWindowResponse(**raw)


@app.get(
    "/patients/presets",
    response_model=list[PatientPreset],
    tags=["Metaveri"],
    summary="Ön tanımlı klinik hasta profilleri",
    response_description="Üç farklı klinik senaryoya ait hasta ölçüm seti.",
)
def get_patient_presets() -> list[PatientPreset]:
    """Frontend simülatörü için hazır hasta profillerini döner.

    - **düşük_risk**: Kararlı vital bulgular, sepsis olasılığı düşük
    - **yüksek_risk**: Yüksek HR, düşük MAP, yüksek Creatinine
    - **sınır_durum**: Eşik değerleri yakın, modeller ayrışır
    """
    preset_rows = [
        {
            "preset_id": "dusuk_risk",
            "label": "Düşük Risk",
            "description": "Kararlı vital bulgular; sepsis ihtimali düşük.",
            "snapshot": {
                "HR": 75.0,
                "O2Sat": 98.0,
                "Temp": 37.1,
                "MAP": 85.0,
                "Resp": 14.0,
                "BUN": 12.0,
                "Chloride": 101.0,
                "Creatinine": 0.8,
                "Glucose": 105.0,
                "Hct": 42.0,
                "Hgb": 13.8,
                "WBC": 8.2,
                "Platelets": 250.0,
                "Age": 52.0,
                "HospAdmTime": -4.0,
                "ICULOS": 6.0,
                "Gender_0": 1.0,
                "Gender_1": 0.0,
            },
        },
        {
            "preset_id": "yuksek_risk",
            "label": "Yüksek Risk",
            "description": "Yüksek HR, düşük MAP, artmış Creatinine — sepsis uyarısı beklenir.",
            "snapshot": {
                "HR": 118.0,
                "O2Sat": 91.0,
                "Temp": 39.2,
                "MAP": 58.0,
                "Resp": 28.0,
                "BUN": 32.0,
                "Chloride": 108.0,
                "Creatinine": 2.4,
                "Glucose": 182.0,
                "Hct": 33.0,
                "Hgb": 10.5,
                "WBC": 18.6,
                "Platelets": 128.0,
                "Age": 71.0,
                "HospAdmTime": -12.0,
                "ICULOS": 24.0,
                "Gender_0": 0.0,
                "Gender_1": 1.0,
            },
        },
        {
            "preset_id": "sinir_durum",
            "label": "Sınır Durum",
            "description": "Eşik değerlerine yakın ölçümler; modeller arasında görüş ayrılığı oluşabilir.",
            "snapshot": {
                "HR": 95.0,
                "O2Sat": 95.0,
                "Temp": 38.4,
                "MAP": 68.0,
                "Resp": 21.0,
                "BUN": 22.0,
                "Chloride": 104.0,
                "Creatinine": 1.4,
                "Glucose": 138.0,
                "Hct": 37.0,
                "Hgb": 11.9,
                "WBC": 13.1,
                "Platelets": 175.0,
                "Age": 63.0,
                "HospAdmTime": -7.0,
                "ICULOS": 15.0,
                "Gender_0": 0.0,
                "Gender_1": 1.0,
            },
        },
    ]
    return [_preset_from_snapshot_row(row) for row in preset_rows]
