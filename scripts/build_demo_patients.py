"""Test setinden 5 sepsis + 5 non-sepsis demo hastasi cikarir.

Cikti:
  backend/artifacts/demo/demo_patients.parquet
  backend/artifacts/demo/demo_manifest.json

Kullanim:
  python build_demo_patients.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data/processed/processed_setAB.csv"
SPLITS = ROOT / "adim_3_2026-05-07/ciktilar/splits.json"
FEATURE_STATS = ROOT / "adim_3_2026-05-07/ciktilar/feature_stats.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "demo"

FEATURE_ORDER = json.loads(FEATURE_STATS.read_text())["feature_order"]
META = ["Patient_ID", "Hour", "SepsisLabel", "HorizonLabel"]
MIN_HOURS = 24
N_EACH = 5
SEED = 42


def pick_demo_patients(df: pd.DataFrame, test_ids: set[str]) -> list[str]:
    """En az MIN_HOURS satiri olan 5 sepsis + 5 non-sepsis hasta secer."""
    test_df = df[df["Patient_ID"].isin(test_ids)].copy()
    stats = (
        test_df.groupby("Patient_ID")
        .agg(n_hours=("Hour", "count"), sepsis=("SepsisLabel", "max"))
        .reset_index()
    )
    stats = stats[stats["n_hours"] >= MIN_HOURS]
    sepsis = stats[stats["sepsis"] == 1].sample(n=N_EACH, random_state=SEED)["Patient_ID"].tolist()
    non = stats[stats["sepsis"] == 0].sample(n=N_EACH, random_state=SEED + 1)["Patient_ID"].tolist()
    return sepsis + non


def build_manifest(selected: list[str], df: pd.DataFrame) -> dict:
    """Demo manifest JSON uretir."""
    patients = []
    for pid in selected:
        pdf = df[df["Patient_ID"] == pid].sort_values("Hour")
        end_hour = int(pdf["Hour"].max())
        start_hour = int(end_hour - MIN_HOURS + 1)
        patients.append(
            {
                "patient_id": pid,
                "sepsis": bool(pdf["SepsisLabel"].max() == 1),
                "n_hours": int(len(pdf)),
                "default_end_hour": end_hour,
                "default_start_hour": max(int(pdf["Hour"].min()), start_hour),
                "window_hours": MIN_HOURS,
            }
        )
    return {"version": 1, "n_patients": len(patients), "patients": patients}


def main() -> None:
    """Parquet ve manifest dosyalarini yazar."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = json.loads(SPLITS.read_text())
    test_ids = set(splits["test_patients"])
    log.info("CSV yukleniyor (test hastalari filtreli)...")
    usecols = META + FEATURE_ORDER
    df = pd.read_csv(CSV, usecols=usecols)
    selected = pick_demo_patients(df, test_ids)
    demo_df = df[df["Patient_ID"].isin(selected)].copy()
    parquet_path = OUT_DIR / "demo_patients.parquet"
    demo_df.to_parquet(parquet_path, index=False)
    manifest = build_manifest(selected, demo_df)
    (OUT_DIR / "demo_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Kaydedildi: %s (%d satir, %d hasta)", parquet_path, len(demo_df), len(selected))


if __name__ == "__main__":
    main()
