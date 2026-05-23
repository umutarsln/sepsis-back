"""Pytest konfigürasyonu ve paylaşımlı fixture'lar."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# XGBoost macOS OpenMP segfault koruması — test sürecinde paralel thread devre dışı.
# OMP_NUM_THREADS=1 olmazsa libxgboost.dylib PyTest içinde segfault üretir.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from app.main import app

# ---------------------------------------------------------------------------
# Paylaşımlı test verisi
# ---------------------------------------------------------------------------

SAMPLE_SNAPSHOT = {
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

HIGH_RISK_SNAPSHOT = {
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
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient — modeller startup sırasında lazy yüklenir.

    scope='module' ile modül başına bir kez oluşturulur; model yükleme
    maliyeti tüm testler arasında paylaşılır.
    """
    with TestClient(app) as test_client:
        yield test_client
