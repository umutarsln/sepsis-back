"""Demo hasta endpoint testleri."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestDemoPatients:
    """GET /patients/demo ve /patients/{id}/window."""

    def test_list_demo_patients(self, client: TestClient) -> None:
        """10 demo hasta listesi döner."""
        resp = client.get("/patients/demo")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 10
        sepsis_n = sum(1 for p in body if p["sepsis"])
        assert sepsis_n == 5

    def test_patient_window_24h(self, client: TestClient) -> None:
        """Seçilen hastanın 24 saatlik serisi döner."""
        listing = client.get("/patients/demo").json()
        pid = listing[0]["patient_id"]
        resp = client.get(f"/patients/{pid}/window?hours=24")
        assert resp.status_code == 200
        body = resp.json()
        assert body["patient_id"] == pid
        assert len(body["series"]) == 24
        assert "hour" in body["series"][0]
        assert "HR" in body["series"][0]

    def test_patient_not_found(self, client: TestClient) -> None:
        """Bilinmeyen hasta 404 döner."""
        resp = client.get("/patients/p999999/window")
        assert resp.status_code == 404

    def test_demo_precomputed_gradient_saliency(self, client: TestClient) -> None:
        """Demo hasta + patient_id ile LSTM/GRU/Transformer saliency döner (offline artifact)."""
        listing = client.get("/patients/demo").json()
        pid = listing[0]["patient_id"]
        win = client.get(f"/patients/{pid}/window?hours=24").json()
        series = [{k: v for k, v in step.items() if k != "hour"} for step in win["series"]]
        resp = client.post(
            "/predict/window",
            json={
                "snapshot": series[-1],
                "series": series,
                "repeat_hours": 24,
                "patient_id": pid,
            },
        )
        assert resp.status_code == 200
        by_id = {m["model_id"]: m for m in resp.json()["models"]}
        assert by_id["bigru_attn"]["importance_method"] == "attention"
        assert len(by_id["bigru_attn"]["attention_weights"]) == 24
        for mid in ("lstm", "gru", "transformer"):
            assert by_id[mid]["importance_method"] == "gradient", mid
            assert len(by_id[mid]["attention_weights"]) == 24, mid
