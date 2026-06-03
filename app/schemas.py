"""Backend API istek/yanit semalari — Faz 8 ile Swagger ornekleri ve ConfigDict eklendi."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Saglik kontrolu yaniti."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "sepsis-son-backend",
            }
        }
    )

    status: str = "ok"
    service: str = "sepsis-son-backend"


class PatientSnapshot(BaseModel):
    """18-feature hasta anlık degerleri — tum alanlar opsiyonel, eksik deger 0.0 ile doldurulur."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "HR": 95.0,
                "O2Sat": 96.0,
                "Temp": 38.5,
                "MAP": 72.0,
                "Resp": 20.0,
                "BUN": 18.0,
                "Chloride": 102.0,
                "Creatinine": 1.1,
                "Glucose": 128.0,
                "Hct": 38.0,
                "Hgb": 12.5,
                "WBC": 11.2,
                "Platelets": 210.0,
                "Age": 65.0,
                "HospAdmTime": -8.0,
                "ICULOS": 12.0,
                "Gender_0": 0.0,
                "Gender_1": 1.0,
            }
        }
    )

    # Vital bulgular
    HR: Optional[float] = Field(None, description="Kalp Hizi (vurum/dk)")
    O2Sat: Optional[float] = Field(None, description="Oksijen Saturasyonu (%)")
    Temp: Optional[float] = Field(None, description="Vucut Sicakligi (C)")
    MAP: Optional[float] = Field(None, description="Ortalama Arteryel Basinc (mmHg)")
    Resp: Optional[float] = Field(None, description="Solunum Hizi (nefes/dk)")

    # Lab degerleri
    BUN: Optional[float] = Field(None, description="Kan Ure Azotu (mg/dL)")
    Chloride: Optional[float] = Field(None, description="Klorur (mEq/L)")
    Creatinine: Optional[float] = Field(None, description="Kreatinin (mg/dL)")
    Glucose: Optional[float] = Field(None, description="Glukoz (mg/dL)")
    Hct: Optional[float] = Field(None, description="Hematokrit (%)")
    Hgb: Optional[float] = Field(None, description="Hemoglobin (g/dL)")
    WBC: Optional[float] = Field(None, description="Beyaz Kan Hucresi (K/uL)")
    Platelets: Optional[float] = Field(None, description="Trombosit (K/uL)")

    # Demografik / zaman
    Age: Optional[float] = Field(None, description="Yas")
    HospAdmTime: Optional[float] = Field(
        None, description="Hastane Kabul Suresi (saat)"
    )
    ICULOS: Optional[float] = Field(None, description="YBU'da Gecen Sure (saat)")

    # Cinsiyet (one-hot)
    Gender_0: Optional[float] = Field(None, description="Kadin (1=evet)")
    Gender_1: Optional[float] = Field(None, description="Erkek (1=evet)")


class ShapContribution(BaseModel):
    """Tek feature'in SHAP katki degeri."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature": "ICULOS",
                "shap_value": 0.42,
                "abs_shap": 0.42,
                "pct_contribution": 18.3,
            }
        }
    )

    feature: str = Field(..., description="Feature adi")
    shap_value: float = Field(..., description="Ham SHAP degeri (negatif veya pozitif)")
    abs_shap: float = Field(..., ge=0.0, description="Mutlak SHAP degeri")
    pct_contribution: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Normalize katki yuzdesi (abs_shap/toplam*100)",
    )


class SnapshotPredictionRequest(BaseModel):
    """Snapshot tahmin istegi — tek bir hasta anlık ölçümünü 5 ML modele gönderir."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "snapshot": {
                    "HR": 95.0,
                    "O2Sat": 96.0,
                    "Temp": 38.5,
                    "MAP": 72.0,
                    "Resp": 20.0,
                    "BUN": 18.0,
                    "Chloride": 102.0,
                    "Creatinine": 1.1,
                    "Glucose": 128.0,
                    "Hct": 38.0,
                    "Hgb": 12.5,
                    "WBC": 11.2,
                    "Platelets": 210.0,
                    "Age": 65.0,
                    "HospAdmTime": -8.0,
                    "ICULOS": 12.0,
                    "Gender_0": 0.0,
                    "Gender_1": 1.0,
                }
            }
        }
    )

    snapshot: PatientSnapshot


class ModelScore(BaseModel):
    """Tek model icin risk skoru."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "xgboost",
                "model_name": "XGBoost",
                "risk_score": 0.72,
                "alert": True,
                "threshold": 0.531,
            }
        }
    )

    model_id: str
    model_name: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    alert: bool
    threshold: float


class SnapshotPredictionResponse(BaseModel):
    """Snapshot tahmin yaniti — 5 ML model risk skorlari."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "models": [
                    {
                        "model_id": "xgboost",
                        "model_name": "XGBoost",
                        "risk_score": 0.72,
                        "alert": True,
                        "threshold": 0.531,
                    }
                ]
            }
        }
    )

    models: list[ModelScore]
    horizon: int = Field(6, description="Tahmin ufku (saat): 0=anlik, 6=6s, 24=24s")


class SnapshotExplainResponse(BaseModel):
    """Snapshot tahmin + SHAP top-10 aciklama yaniti."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "models": [
                    {
                        "model_id": "xgboost",
                        "model_name": "XGBoost",
                        "risk_score": 0.72,
                        "alert": True,
                        "threshold": 0.531,
                    }
                ],
                "shap_top10": [
                    {
                        "feature": "ICULOS",
                        "shap_value": 0.42,
                        "abs_shap": 0.42,
                        "pct_contribution": 18.3,
                    }
                ],
            }
        }
    )

    models: list[ModelScore]
    shap_top10: Optional[list[ShapContribution]] = Field(
        None, description="XGBoost SHAP top-10 (geriye uyumluluk; shap_by_model.xgboost ile ayni)"
    )
    shap_by_model: Optional[dict[str, list[ShapContribution]]] = Field(
        None,
        description="Model bazli SHAP top-10 katkilari (5 ML modeli, mutlak SHAP azalan)",
    )
    horizon: int = Field(6, description="Tahmin ufku (saat)")


# ---------------------------------------------------------------------------
# Pencere (Window) tahmini — Faz 5 DL modelleri
# ---------------------------------------------------------------------------


class WindowPredictionRequest(BaseModel):
    """24 saatlik pencere tahmini istegi.

    `series` verilirse gercek saatlik veri kullanilir; yoksa snapshot tekrarlanir.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "snapshot": {
                    "HR": 95.0,
                    "O2Sat": 96.0,
                    "Temp": 38.5,
                    "MAP": 72.0,
                    "Resp": 20.0,
                    "BUN": 18.0,
                    "Chloride": 102.0,
                    "Creatinine": 1.1,
                    "Glucose": 128.0,
                    "Hct": 38.0,
                    "Hgb": 12.5,
                    "WBC": 11.2,
                    "Platelets": 210.0,
                    "Age": 65.0,
                    "HospAdmTime": -8.0,
                    "ICULOS": 12.0,
                    "Gender_0": 0.0,
                    "Gender_1": 1.0,
                },
                "repeat_hours": 24,
                "series": None,
            }
        }
    )

    snapshot: PatientSnapshot
    repeat_hours: int = Field(
        default=24, ge=1, le=72, description="Tekrar pencere saati (series yoksa)"
    )
    series: Optional[list[PatientSnapshot]] = Field(
        None,
        description="Gercek saatlik seri (T adim). Verilirse repeat atlanir.",
    )
    patient_id: Optional[str] = Field(
        None,
        description="Demo hasta kimligi; offline gradient saliency lookup icin (Faz 4.8).",
    )


class WindowModelResult(BaseModel):
    """Tek DL modeli icin pencere risk skoru."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "gru",
                "model_name": "GRU",
                "risk_score": 0.68,
                "alert": True,
                "threshold": 0.45,
                "attention_weights": None,
            }
        }
    )

    model_id: str
    model_name: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    alert: bool
    threshold: float
    attention_weights: Optional[list[float]] = Field(
        None,
        description="24 timestep onem agirligi (attention veya gradient saliency)",
    )
    importance_method: Optional[str] = Field(
        None,
        description="'attention' (BiGRU+Attn) veya 'gradient' (LSTM/GRU/Transformer)",
    )


class WindowPredictionResponse(BaseModel):
    """3 DL modeli icin pencere tahmin yaniti."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "models": [
                    {
                        "model_id": "gru",
                        "model_name": "GRU",
                        "risk_score": 0.68,
                        "alert": True,
                        "threshold": 0.45,
                        "attention_weights": None,
                    }
                ],
                "window_shape": [24, 18],
            }
        }
    )

    models: list[WindowModelResult]
    window_shape: tuple[int, int] = Field(
        default=(24, 18), description="(timestep, feature) pencere boyutu"
    )
    input_mode: str = Field(
        "repeat",
        description="'series' = gercek saatlik seri, 'repeat' = snapshot tekrari",
    )


class DemoPatientSummary(BaseModel):
    """Demo hasta ozet bilgisi."""

    patient_id: str
    sepsis: bool
    n_hours: int
    default_end_hour: int
    window_hours: int = 24


class HourlySnapshot(BaseModel):
    """Saat etiketli snapshot."""

    hour: int
    HR: Optional[float] = None
    O2Sat: Optional[float] = None
    Temp: Optional[float] = None
    MAP: Optional[float] = None
    Resp: Optional[float] = None
    BUN: Optional[float] = None
    Chloride: Optional[float] = None
    Creatinine: Optional[float] = None
    Glucose: Optional[float] = None
    Hct: Optional[float] = None
    Hgb: Optional[float] = None
    WBC: Optional[float] = None
    Platelets: Optional[float] = None
    Age: Optional[float] = None
    HospAdmTime: Optional[float] = None
    ICULOS: Optional[float] = None
    Gender_0: Optional[float] = None
    Gender_1: Optional[float] = None


class PatientWindowResponse(BaseModel):
    """Hasta saatlik pencere yaniti."""

    patient_id: str
    hours: int
    end_hour: int
    sepsis: bool
    horizon_label_end: int
    series: list[HourlySnapshot]


class SnapshotModelMetrics(BaseModel):
    """Tek ML snapshot modelinin frozen test metrikleri."""

    model_id: str
    auroc: float = Field(..., ge=0.0, le=1.0)
    auprc: float = Field(..., ge=0.0, le=1.0)
    sens_at_spec85: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    brier: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: str = Field(..., description="Metrik kaynagi (ornegin adim_4_6 Optuna)")


class HorizonMetricsResponse(BaseModel):
    """Ufuk bazli snapshot ML metrik ozeti."""

    horizon: int = Field(..., description="0, 6 veya 24")
    label: str
    metrics_source: str
    models: list[SnapshotModelMetrics]


class HorizonComparisonRow(BaseModel):
    """5 ML model icin coklu ufuk AUROC/AUPRC karsilastirmasi."""

    model_id: str
    model_name: str
    h0_auroc: float
    h6_auroc: float
    h24_auroc: float
    h0_auprc: float
    h6_auprc: float
    h24_auprc: float


class LeadTimeSummary(BaseModel):
    """Erken uyari (lead-time) ozet metrikleri — frontend dashboard ile uyumlu."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "n_positive_patients": 268,
                "n_detected": 250,
                "detection_rate": 0.933,
                "n_early_alarm": 219,
                "early_alarm_rate": 0.817,
                "median_lead_time_hours": 16.5,
                "mean_lead_time_hours": 33.852,
                "q25_lead_time": 1.0,
                "q75_lead_time": 52.0,
                "version": "v5",
                "threshold_at_spec85": 0.555,
            }
        }
    )

    n_positive_patients: int = Field(..., description="Sepsis pozitif hasta sayisi")
    n_detected: int = Field(..., description="Erken uyari ile yakalanan hasta sayisi")
    detection_rate: float = Field(..., ge=0.0, le=1.0, description="Yakalama orani")
    n_early_alarm: int = Field(..., description="Eşik ustunde erken alarm sayisi")
    early_alarm_rate: float = Field(..., ge=0.0, le=1.0, description="Erken alarm orani")
    median_lead_time_hours: float = Field(..., description="Medyan lead time (saat)")
    mean_lead_time_hours: float = Field(..., description="Ortalama lead time (saat)")
    q25_lead_time: float = Field(..., description="Lead time 25. persentil (saat)")
    q75_lead_time: float = Field(..., description="Lead time 75. persentil (saat)")
    version: str = Field(..., description="Model versiyonu (ornegin v5)")
    threshold_at_spec85: float = Field(..., description="Specificity=0.85 esigindeki threshold")


class PatientPreset(BaseModel):
    """Frontend simulatörü icin hazir hasta profili."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "preset_id": "dusuk_risk",
                "label": "Dusuk Risk",
                "risk_band": "low",
                "description": "Kararli vital bulgular; sepsis ihtimali dusuk.",
                "gender": "F",
                "features": {
                    "HR": 75.0,
                    "O2Sat": 98.0,
                    "Temp": 37.1,
                    "MAP": 85.0,
                    "Resp": 14.0,
                    "Age": 52.0,
                },
            }
        }
    )

    preset_id: str
    label: str
    risk_band: str = Field(..., description="low / medium / high")
    description: str
    gender: str = Field(..., description="M veya F")
    features: dict[str, float]
    source: str = Field(
        "synthetic",
        description="synthetic (egitim senaryosu) veya demo (Faz 4.8 gercek test hastasi)",
    )
    patient_id: Optional[str] = Field(None, description="Demo preset icin gercek Patient_ID")
    sepsis: Optional[bool] = Field(None, description="Demo hastada gercek sepsis etiketi")
    preset_group: str = Field(
        "scenario",
        description="scenario (sentetik) veya demo_real (Faz 4.8)",
    )


class ExperimentMetrics(BaseModel):
    """Tek bir egitim kosusunun performans metrikleri."""

    test_auroc: Optional[float] = Field(None, description="Test seti AUROC")
    val_auroc: Optional[float] = Field(None, description="Validation seti AUROC")
    auprc: Optional[float] = Field(None, description="Test AUPRC")
    f1: Optional[float] = Field(None, description="Test F1 (Spec=0.85 esigi)")
    sens_at_spec85: Optional[float] = Field(None, description="Sensitivity@Specificity=0.85")
    brier: Optional[float] = Field(None, description="Brier kalibrasyon skoru (ML)")
    threshold: Optional[float] = Field(None, description="Val setinde belirlenen karar esigi")


class ExperimentRow(BaseModel):
    """Faz 4-6 egitim loglarindan turetilen deney kaydi."""

    id: str = Field(..., description="Benzersiz deney kimligi")
    name: str = Field(..., description="Okunabilir deney adi")
    phase: str = Field(..., description="Proje fazı (Faz 4 / Faz 5 / Faz 6)")
    tier: Optional[str] = Field(None, description="DL egitim tier'i: quick / standard / thorough")
    status: str = Field("completed", description="Deney durumu")
    model: str = Field(..., description="Model gorunen adi")
    model_id: str = Field(..., description="Model kimligi")
    model_family: str = Field(..., description="ML / DL / Transformer")
    params: dict[str, float | int | str | None] = Field(
        default_factory=dict, description="Hiperparametreler"
    )
    metrics: ExperimentMetrics
    duration: Optional[str] = Field(None, description="Egitim suresi (varsa)")
    created_at: str = Field(..., description="Deney tarihi (YYYY-MM-DD)")
    is_final: bool = Field(
        False, description="Faz 6 final benchmark tablosuna giren kosu mu"
    )
    notes: Optional[str] = Field(None, description="Rapordan ek not")


class DatasetCohortSummary(BaseModel):
    """Ham veri seti kohort ozeti (Faz 2 EDA)."""

    total_patients: int
    set_a_patients: int
    set_b_patients: int
    total_rows: int


class DatasetLabelSummary(BaseModel):
    """Sepsis etiket istatistikleri (hasta ve satir duzeyi)."""

    sepsis_positive_patients: int
    sepsis_negative_patients: int
    sepsis_patient_rate_pct: float
    sepsis_positive_rows: int
    sepsis_row_rate_pct: float
    onset_median_hours: float


class DatasetLengthSummary(BaseModel):
    """ICU kalis suresi dagilimi (saat)."""

    median: float
    mean: float
    p5: float
    p25: float
    p75: float
    p95: float
    min: float
    max: float


class DatasetSplitSummary(BaseModel):
    """Frozen hasta bazli train/val/test bolunmesi (Faz 3)."""

    train_patients: int
    val_patients: int
    test_patients: int
    train_sepsis_rate_pct: float
    val_sepsis_rate_pct: float
    test_sepsis_rate_pct: float
    train_sepsis_patients: int
    val_sepsis_patients: int
    test_sepsis_patients: int
    seed: int
    frozen: bool


class DatasetMissingRow(BaseModel):
    """Tek bir ozelligin ham eksiklik orani."""

    feature: str
    missing_pct: float
    category: str


class DatasetSummaryResponse(BaseModel):
    """Veri analizi sayfasi icin birlesik ozet."""

    cohort: DatasetCohortSummary
    labels: DatasetLabelSummary
    length: DatasetLengthSummary
    splits: DatasetSplitSummary
    final_feature_count: int
    features_above_80pct_missing: int
    selected_feature_missing: list[DatasetMissingRow]
    icu_length_chart: list[dict[str, float | str]]
    split_chart: list[dict[str, float | int | str]]
    source_files: list[str]
