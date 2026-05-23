"""Tahmin endpoint testleri — POST /predict/snapshot, /predict/snapshot/explain, /predict/window."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import HIGH_RISK_SNAPSHOT, SAMPLE_SNAPSHOT


class TestSnapshotPredict:
    """POST /predict/snapshot senaryoları."""

    def test_snapshot_valid(self, client: TestClient) -> None:
        """Tam dolu hasta verisi 200 ve 5 model skoru döner."""
        resp = client.post("/predict/snapshot", json={"snapshot": SAMPLE_SNAPSHOT})
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert len(body["models"]) == 5

    def test_snapshot_model_fields(self, client: TestClient) -> None:
        """Her model kaydı zorunlu alanları içerir."""
        resp = client.post("/predict/snapshot", json={"snapshot": SAMPLE_SNAPSHOT})
        assert resp.status_code == 200
        for model in resp.json()["models"]:
            assert "model_id" in model
            assert "risk_score" in model
            assert "alert" in model
            assert 0.0 <= model["risk_score"] <= 1.0

    def test_snapshot_empty_fields(self, client: TestClient) -> None:
        """Tüm alanlar None olduğunda 0.0 impute ile yine 200 döner."""
        resp = client.post("/predict/snapshot", json={"snapshot": {}})
        assert resp.status_code == 200
        assert len(resp.json()["models"]) == 5

    def test_snapshot_invalid_type(self, client: TestClient) -> None:
        """HR alanına string girildiğinde 422 Unprocessable Entity döner."""
        payload = {**SAMPLE_SNAPSHOT, "HR": "yuzelli"}
        resp = client.post("/predict/snapshot", json={"snapshot": payload})
        assert resp.status_code == 422

    def test_snapshot_extra_field(self, client: TestClient) -> None:
        """Bilinmeyen alan gönderildiğinde Pydantic yoksayar, 200 döner."""
        payload = {**SAMPLE_SNAPSHOT, "nonexistent_field": 999.9}
        resp = client.post("/predict/snapshot", json={"snapshot": payload})
        assert resp.status_code == 200

    def test_snapshot_high_risk_alert(self, client: TestClient) -> None:
        """Yüksek riskli hasta için en az bir model alert=True döner."""
        resp = client.post("/predict/snapshot", json={"snapshot": HIGH_RISK_SNAPSHOT})
        assert resp.status_code == 200
        alerts = [m["alert"] for m in resp.json()["models"]]
        assert any(alerts), "Yüksek riskli hastada hiçbir model uyarı vermedi"

    def test_snapshot_current_h0(self, client: TestClient) -> None:
        """h=0 anlik tespit endpoint'i 5 model ve horizon=0 döner."""
        resp = client.post("/predict/snapshot/current", json={"snapshot": SAMPLE_SNAPSHOT})
        assert resp.status_code == 200
        body = resp.json()
        assert body["horizon"] == 0
        assert len(body["models"]) == 5

    def test_snapshot_horizon_24(self, client: TestClient) -> None:
        """h=24 erken uyari endpoint'i horizon=24 döner."""
        resp = client.post(
            "/predict/snapshot/horizon/24",
            json={"snapshot": SAMPLE_SNAPSHOT},
        )
        assert resp.status_code == 200
        assert resp.json()["horizon"] == 24

    def test_snapshot_invalid_horizon(self, client: TestClient) -> None:
        """Desteklenmeyen horizon 422 döner."""
        resp = client.post(
            "/predict/snapshot/horizon/12",
            json={"snapshot": SAMPLE_SNAPSHOT},
        )
        assert resp.status_code == 422


class TestSnapshotExplain:
    """POST /predict/snapshot/explain senaryoları."""

    def test_explain_returns_shap(self, client: TestClient) -> None:
        """Açıkla endpoint'i shap_top5 listesi döner (en az 1 eleman)."""
        resp = client.post(
            "/predict/snapshot/explain", json={"snapshot": SAMPLE_SNAPSHOT}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "shap_top5" in body
        if body["shap_top5"] is not None:
            assert len(body["shap_top5"]) >= 1
            first = body["shap_top5"][0]
            assert "feature" in first
            assert "pct_contribution" in first
            assert 0.0 <= first["pct_contribution"] <= 100.0

    def test_explain_also_returns_models(self, client: TestClient) -> None:
        """Açıkla endpoint'i hem models hem shap_top5 döner."""
        resp = client.post(
            "/predict/snapshot/explain", json={"snapshot": SAMPLE_SNAPSHOT}
        )
        body = resp.json()
        assert "models" in body
        assert len(body["models"]) == 5


class TestWindowPredict:
    """POST /predict/window senaryoları."""

    def test_window_valid(self, client: TestClient) -> None:
        """repeat_hours=24 ile 200 ve DL model sonuçları döner."""
        resp = client.post(
            "/predict/window",
            json={"snapshot": SAMPLE_SNAPSHOT, "repeat_hours": 24},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert len(body["models"]) >= 1
        assert body.get("input_mode") == "repeat"

    def test_window_with_series(self, client: TestClient) -> None:
        """series verildiğinde input_mode=series döner."""
        series = [SAMPLE_SNAPSHOT for _ in range(24)]
        resp = client.post(
            "/predict/window",
            json={"snapshot": SAMPLE_SNAPSHOT, "repeat_hours": 24, "series": series},
        )
        assert resp.status_code == 200
        assert resp.json()["input_mode"] == "series"

    def test_window_out_of_range_zero(self, client: TestClient) -> None:
        """repeat_hours=0 (ge=1 kısıtı ihlali) 422 döner."""
        resp = client.post(
            "/predict/window",
            json={"snapshot": SAMPLE_SNAPSHOT, "repeat_hours": 0},
        )
        assert resp.status_code == 422

    def test_window_out_of_range_large(self, client: TestClient) -> None:
        """repeat_hours=100 (le=72 kısıtı ihlali) 422 döner."""
        resp = client.post(
            "/predict/window",
            json={"snapshot": SAMPLE_SNAPSHOT, "repeat_hours": 100},
        )
        assert resp.status_code == 422

    def test_window_shape(self, client: TestClient) -> None:
        """window_shape yanıt alanı [24, 18] olmalıdır."""
        resp = client.post(
            "/predict/window",
            json={"snapshot": SAMPLE_SNAPSHOT, "repeat_hours": 24},
        )
        assert resp.status_code == 200
        shape = resp.json()["window_shape"]
        assert shape == [24, 18], f"Beklenen [24,18], gelen {shape}"
