"""Deploy-safe artifact path resolution — once backend/artifacts, sonra monorepo adim."""

from __future__ import annotations

from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_SEPSIS_SON_DIR = _BACKEND_DIR.parent
_ARTIFACTS_DIR = _BACKEND_DIR / "artifacts"
_PREPROCESSING_DIR = _ARTIFACTS_DIR / "preprocessing"
_DATASET_DIR = _ARTIFACTS_DIR / "dataset"
_ADIM_BUNDLE_DIR = _ARTIFACTS_DIR / "adim"
_REPORTS_DIR = _ARTIFACTS_DIR / "reports"
_METRICS_DIR = _ARTIFACTS_DIR / "metrics"
_EXPLAINABILITY_DIR = _ARTIFACTS_DIR / "results" / "explainability"


def _first_existing(*candidates: Path) -> Path:
    """Ilk var olan yolu dondurur; hicbiri yoksa son adayi dondurur."""
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def resolve_preprocessing_file(filename: str) -> Path:
    """Faz 3 preprocessing JSON dosyasi (feature_stats, splits) yolunu cozer."""
    return _first_existing(
        _PREPROCESSING_DIR / filename,
        _SEPSIS_SON_DIR / "adim_3_2026-05-07" / "ciktilar" / filename,
    )


def resolve_feature_stats_path() -> Path:
    """feature_stats.json yolunu cozer."""
    return resolve_preprocessing_file("feature_stats.json")


def resolve_dataset_eda_path() -> Path:
    """Faz 2 eda_summary.json yolunu cozer."""
    return _first_existing(
        _DATASET_DIR / "eda_summary.json",
        _SEPSIS_SON_DIR / "adim_2_2026-05-07" / "ciktilar" / "eda_summary.json",
    )


def resolve_adim_ciktilar(adim_folder: str, filename: str) -> Path:
    """adim_X/ciktilar altindaki tek dosya icin yol cozer."""
    return _first_existing(
        _ADIM_BUNDLE_DIR / adim_folder / "ciktilar" / filename,
        _SEPSIS_SON_DIR / adim_folder / "ciktilar" / filename,
    )


def resolve_adim_tier_file(adim_folder: str, tier: str, filename: str) -> Path:
    """adim_X/ciktilar/<tier>/dosya yolunu cozer."""
    return _first_existing(
        _ADIM_BUNDLE_DIR / adim_folder / "ciktilar" / tier / filename,
        _SEPSIS_SON_DIR / adim_folder / "ciktilar" / tier / filename,
    )


def resolve_metrics_file(filename: str, adim_folder: str = "adim_4_2026-05-07") -> Path:
    """Faz 4/4.6/4.7 metrik JSON dosyasi yolunu cozer."""
    return _first_existing(
        _METRICS_DIR / filename,
        _ADIM_BUNDLE_DIR / adim_folder / "ciktilar" / filename,
        _SEPSIS_SON_DIR / adim_folder / "ciktilar" / filename,
    )


def resolve_version_comparison_csv() -> Path:
    """Faz 6 version karsilastirma CSV yolunu cozer."""
    return _first_existing(
        _REPORTS_DIR / "version_comparison_summary.csv",
        _SEPSIS_SON_DIR / "adim_6_2026-05-09" / "ciktilar" / "version_comparison_summary.csv",
    )


def resolve_shap_summary_xgboost() -> Path:
    """XGBoost SHAP global ozet JSON yolunu cozer."""
    return _first_existing(
        _EXPLAINABILITY_DIR / "shap_summary_xgboost.json",
        _ADIM_BUNDLE_DIR / "adim_7_2026-05-09" / "ciktilar" / "shap_summary_xgboost.json",
        _SEPSIS_SON_DIR / "adim_7_2026-05-09" / "ciktilar" / "shap_summary_xgboost.json",
    )
