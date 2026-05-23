"""Sağlık endpoint testleri — GET /health."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealth:
    """GET /health endpoint senaryoları."""

    def test_health_ok(self, client: TestClient) -> None:
        """Servis çalışırken 200 ve status='ok' döner."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_schema(self, client: TestClient) -> None:
        """Yanıt şeması 'status' ve 'service' alanlarını içerir."""
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert "service" in body
        assert body["service"] == "sepsis-son-backend"
