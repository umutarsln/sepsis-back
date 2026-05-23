"""Artifact ve metaveri endpoint testleri — GET /artifacts/*, /models/descriptors, vb."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestShapSummary:
    """GET /artifacts/shap-summary/{model_id} senaryoları."""

    def test_shap_xgboost(self, client: TestClient) -> None:
        """xgboost için SHAP listesi 200 ve dolu liste döner."""
        resp = client.get("/artifacts/shap-summary/xgboost")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) > 0
        assert "feature" in body[0]

    def test_shap_random_forest(self, client: TestClient) -> None:
        """random_forest için SHAP listesi 200 döner."""
        resp = client.get("/artifacts/shap-summary/random_forest")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_shap_logistic_regression(self, client: TestClient) -> None:
        """logistic_regression için SHAP listesi 200 döner."""
        resp = client.get("/artifacts/shap-summary/logistic_regression")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_shap_invalid_model(self, client: TestClient) -> None:
        """Geçersiz model_id için 404 döner."""
        resp = client.get("/artifacts/shap-summary/no_such_model")
        assert resp.status_code == 404
        assert "detail" in resp.json()


class TestAttention:
    """GET /artifacts/attention/{model_id} senaryoları."""

    def test_attention_bigru(self, client: TestClient) -> None:
        """bigru_attn için attention verisi 200 döner."""
        resp = client.get("/artifacts/attention/bigru_attn")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_attention_invalid(self, client: TestClient) -> None:
        """Geçersiz model_id için 404 döner."""
        resp = client.get("/artifacts/attention/no_such_model")
        assert resp.status_code == 404


class TestArtifactEndpoints:
    """Faz 6-7 artifact endpoint senaryoları."""

    def test_feature_ranking(self, client: TestClient) -> None:
        """GET /artifacts/feature-ranking 200 ve liste döner."""
        resp = client.get("/artifacts/feature-ranking")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) > 0

    def test_lime(self, client: TestClient) -> None:
        """GET /artifacts/lime 200 ve TP/FP/FN açıklamaları döner."""
        resp = client.get("/artifacts/lime")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert "patient_type" in body[0]

    def test_version_comparison(self, client: TestClient) -> None:
        """GET /artifacts/version-comparison 200 ve auroc sütunu içerir."""
        resp = client.get("/artifacts/version-comparison")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) > 0
        assert "auroc" in body[0]

    def test_lead_time(self, client: TestClient) -> None:
        """GET /artifacts/lead-time 200 ve frontend ozet alanlarini icerir."""
        resp = client.get("/artifacts/lead-time")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert "median_lead_time_hours" in body
        assert "detection_rate" in body
        assert body["median_lead_time_hours"] > 0


class TestMetadataEndpoints:
    """Metaveri endpoint senaryoları."""

    def test_models_descriptors(self, client: TestClient) -> None:
        """GET /models/descriptors 200 ve 9 model kaydı döner."""
        resp = client.get("/models/descriptors")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 9
        model_ids = [m["model_id"] for m in body]
        assert "xgboost" in model_ids
        assert "transformer" in model_ids

    def test_feature_stats(self, client: TestClient) -> None:
        """GET /preprocessing/feature-stats 200 ve feature_order + clinical_ranges döner."""
        resp = client.get("/preprocessing/feature-stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "feature_order" in body
        assert "clinical_ranges" in body
        assert "HR" in body["clinical_ranges"]
        assert len(body["feature_order"]) == 18

    def test_patient_presets(self, client: TestClient) -> None:
        """GET /patients/presets 200 ve 3 preset (dusuk/yuksek/sinir) döner."""
        resp = client.get("/patients/presets")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 3
        preset_ids = [p["preset_id"] for p in body]
        assert "dusuk_risk" in preset_ids
        assert "yuksek_risk" in preset_ids
        assert "sinir_durum" in preset_ids

    def test_patient_presets_snapshot_fields(self, client: TestClient) -> None:
        """Her preset'in features alani 16+ klinik feature içerir."""
        resp = client.get("/patients/presets")
        for preset in resp.json():
            features = preset["features"]
            assert "HR" in features
            assert "MAP" in features
            assert "Age" in features
            assert "Gender_0" not in features
            assert "Gender_1" not in features
            assert preset["gender"] in {"M", "F"}
            assert preset["risk_band"] in {"low", "medium", "high"}
