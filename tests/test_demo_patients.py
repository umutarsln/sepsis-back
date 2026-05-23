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
