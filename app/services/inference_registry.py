"""Inference registry — Faz 4 (sklearn/pkl) + Faz 5 (DL/pt) modelleri."""

from __future__ import annotations

import app.bootstrap_env  # noqa: F401 — numpy/torch oncesi OMP ayarlari

import json
import logging
import math
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

FEATURE_ORDER = [
    "HR",
    "O2Sat",
    "Temp",
    "MAP",
    "Resp",
    "BUN",
    "Chloride",
    "Creatinine",
    "Glucose",
    "Hct",
    "Hgb",
    "WBC",
    "Platelets",
    "Age",
    "HospAdmTime",
    "ICULOS",
    "Gender_0",
    "Gender_1",
]

LOG_TRANSFORM_COLS = {"MAP", "BUN", "Creatinine", "Glucose", "WBC", "Platelets"}

MODEL_DISPLAY = {
    "logistic_regression": "Lojistik Regresyon",
    "random_forest": "Rastgele Orman",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradyan Artirma",
    "gaussian_nb": "Gaussian NB",
}

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "gradient_boosting",
    "gaussian_nb",
]

# parents[2] = backend/, parents[3] = sepsis-son/
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SEPSIS_SON_DIR = Path(__file__).resolve().parents[3]

_BASE = _BACKEND_DIR / "artifacts" / "models"

# Tahmin ufku (saat) -> pkl oneki ve threshold dosyasi
HORIZON_PROFILES: dict[int, dict[str, str]] = {
    6: {"prefix": "snapshot", "thresholds_file": "snapshot_thresholds.json"},
    0: {"prefix": "current", "thresholds_file": "current_thresholds.json"},
    24: {"prefix": "horizon24", "thresholds_file": "horizon24_thresholds.json"},
}

_SUPPORTED_HORIZONS = frozenset(HORIZON_PROFILES.keys())

_SNAPSHOT_MODEL_PATHS: dict[str, Path] = {
    name: _BASE / f"snapshot_{name}.pkl" for name in MODEL_ORDER
}

_THRESHOLDS_PATH = _BASE / "snapshot_thresholds.json"
_FEATURE_STATS_PATH = (
    _SEPSIS_SON_DIR / "adim_3_2026-05-07" / "ciktilar" / "feature_stats.json"
)

# ---------------------------------------------------------------------------
# DL model sabitleri — Faz 5
# ---------------------------------------------------------------------------

_WINDOW_MODELS: dict[str, str] = {
    "lstm": "lstm.pt",
    "gru": "gru.pt",
    "bigru_attn": "bigru_attn.pt",
    "transformer": "transformer.pt",
}

_DL_DISPLAY_NAMES: dict[str, str] = {
    "lstm": "LSTM",
    "gru": "GRU",
    "bigru_attn": "BiGRU+Attention",
    "transformer": "Temporal Transformer",
}

_WINDOW_THRESHOLDS_PATH = _BASE / "window_thresholds.json"

# ---------------------------------------------------------------------------
# InferenceRegistry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DL model mimarileri (kopyala-uyarla: adim_5/kod/train_dl_models.py)
# ---------------------------------------------------------------------------


class _AdditiveAttention(nn.Module):
    """Bahdanau tipi additive attention katmani."""

    def __init__(self, hidden_dim: int, attn_dim: int = 64) -> None:
        """Parametreleri baslatir."""
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention agirliklarini hesaplar."""
        scores = self.v(torch.tanh(self.W(hidden_states))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class _LSTMModel(nn.Module):
    """Backend LSTM siniflandirici."""

    def __init__(self, input_dim: int = 18, hidden: int = 128) -> None:
        """Modeli baslatir."""
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim, hidden, num_layers=2, batch_first=True, dropout=0.3
        )
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        """Ileri gecis."""
        out, _ = self.rnn(x)
        logit = self.head(self.drop(out[:, -1, :])).squeeze(-1)
        return logit, None


class _GRUModel(nn.Module):
    """Backend GRU siniflandirici."""

    def __init__(self, input_dim: int = 18, hidden: int = 128) -> None:
        """Modeli baslatir."""
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden, num_layers=2, batch_first=True, dropout=0.3
        )
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        """Ileri gecis."""
        out, _ = self.rnn(x)
        logit = self.head(self.drop(out[:, -1, :])).squeeze(-1)
        return logit, None


class _BiGRUAttention(nn.Module):
    """Backend BiGRU+Attention siniflandirici."""

    def __init__(self, input_dim: int = 18, hidden: int = 128) -> None:
        """Modeli baslatir."""
        super().__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = _AdditiveAttention(hidden * 2)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ileri gecis."""
        out, _ = self.rnn(x)
        context, attn_w = self.attention(out, mask=mask)
        logit = self.head(self.drop(context)).squeeze(-1)
        return logit, attn_w


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Embedding boyutu.
        max_len: Maksimum sekans uzunlugu.
    """

    def __init__(self, d_model: int, max_len: int = 128) -> None:
        """PE buffer'i olusturur."""
        import math

        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PE ekler."""
        return x + self.pe[:, : x.size(1)]


class _TemporalTransformerModel(nn.Module):
    """Backend Temporal Transformer siniflandirici (Faz 6).

    Args:
        input_dim: Girdi feature boyutu.
        d_model: Embedding boyutu.
        n_head: Multi-head attention kafa sayisi.
        num_layers: Encoder katman sayisi.
        dim_ff: Feedforward ara boyut.
        dropout: Dropout orani.
    """

    def __init__(
        self,
        input_dim: int = 18,
        d_model: int = 128,
        n_head: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.2,
    ) -> None:
        """Modeli baslatir."""
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pe = _PositionalEncoding(d_model, max_len=26)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Ileri gecis.

        Args:
            x: (B, T, F) float.
            mask: (B, T) bool — True gecerli timestep.

        Returns:
            (logit (B,), None) demeti.
        """
        B, _T, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pe(x)
        out = self.encoder(x)
        logit = self.head(out[:, 0, :]).squeeze(-1)
        return logit, None


def _make_dl_model(model_id: str) -> nn.Module:
    """model_id'ye gore bos DL modeli olusturur.

    Args:
        model_id: 'lstm', 'gru', 'bigru_attn' veya 'transformer'.

    Returns:
        Baslatilmis, onsiz nn.Module.
    """
    if model_id == "lstm":
        return _LSTMModel()
    if model_id == "gru":
        return _GRUModel()
    if model_id == "bigru_attn":
        return _BiGRUAttention()
    if model_id == "transformer":
        return _TemporalTransformerModel()
    raise ValueError(f"Bilinmeyen DL model: {model_id}")


# ---------------------------------------------------------------------------
# Inference Registry
# ---------------------------------------------------------------------------


class InferenceRegistry:
    """Snapshot (sklearn/pkl) + Window (DL/pt) tahmin modelleri registry."""

    def __init__(self) -> None:
        """Scaler istatistiklerini ve threshold degerlerini yukler."""
        self._models: dict[int, dict[str, Any]] = {}
        self._thresholds_by_horizon: dict[int, dict[str, float]] = {
            h: self._load_horizon_thresholds(h) for h in HORIZON_PROFILES
        }
        self._thresholds: dict[str, float] = self._thresholds_by_horizon[6]
        self._scaler_stats: dict[str, dict] = self._load_scaler_stats()
        self._dl_cache: dict[str, nn.Module] = {}
        self._dl_thresholds: dict[str, float] = self._load_dl_thresholds()
        self._shap_cache: dict[int, Any] = {}
        log.info(
            "InferenceRegistry hazir — horizonlar %s, model sayisi=%d",
            sorted(HORIZON_PROFILES),
            len(MODEL_ORDER),
        )

    # ------------------------------------------------------------------
    # Yukleme
    # ------------------------------------------------------------------

    def _load_dl_thresholds(self) -> dict[str, float]:
        """window_thresholds.json'dan DL model threshold degerlerini okur.

        Returns:
            {model_id: threshold} sozlugu.
        """
        if _WINDOW_THRESHOLDS_PATH.exists():
            raw = json.loads(_WINDOW_THRESHOLDS_PATH.read_text())
            return {k: float(v["threshold"]) for k, v in raw.items()}
        log.warning("window_thresholds.json bulunamadi, varsayilan 0.5 kullanilacak")
        return {k: 0.5 for k in _WINDOW_MODELS}

    def _load_thresholds(self) -> dict[str, float]:
        """h=6 snapshot threshold dosyasini okur (geri uyumluluk).

        Returns:
            {model_id: threshold} sozlugu.
        """
        return self._load_horizon_thresholds(6)

    def _load_horizon_thresholds(self, horizon: int) -> dict[str, float]:
        """Belirtilen ufuk icin threshold dosyasini okur.

        Args:
            horizon: Tahmin ufku (0, 6 veya 24).

        Returns:
            {model_id: threshold} sozlugu.
        """
        profile = HORIZON_PROFILES.get(horizon)
        if profile is None:
            return {name: 0.5 for name in MODEL_ORDER}
        path = _BASE / profile["thresholds_file"]
        if path.exists():
            raw = json.loads(path.read_text())
            return {k: float(v["threshold"]) for k, v in raw.items()}
        log.warning("%s bulunamadi, varsayilan 0.5", path.name)
        return {name: 0.5 for name in MODEL_ORDER}

    def _model_path(self, horizon: int, model_id: str) -> Path:
        """Horizon ve model icin pkl yolunu dondurur."""
        prefix = HORIZON_PROFILES[horizon]["prefix"]
        return _BASE / f"{prefix}_{model_id}.pkl"

    def _load_scaler_stats(self) -> dict[str, dict]:
        """feature_stats.json'dan scaler mean/std degerlerini okur.

        Returns:
            {feature: {mean, std}} sozlugu.
        """
        if _FEATURE_STATS_PATH.exists():
            fs = json.loads(_FEATURE_STATS_PATH.read_text())
            return fs.get("scaler_stats", {})
        log.warning("feature_stats.json bulunamadi, ham degerler kullanilacak")
        return {}

    def _lazy_load(self, model_id: str, horizon: int = 6) -> Any:
        """Model ilk istekte yuklenir, sonraki isteklerde cache'den doner.

        Args:
            model_id: Model kimlik anahtari.
            horizon: Tahmin ufku (varsayilan h=6).

        Returns:
            Yuklenmis sklearn/xgboost modeli.
        """
        if horizon not in self._models:
            self._models[horizon] = {}
        cache = self._models[horizon]
        if model_id not in cache:
            pkl_path = self._model_path(horizon, model_id)
            if not pkl_path.exists():
                log.error("pkl bulunamadi: %s", pkl_path)
                return None
            with open(pkl_path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Trying to unpickle estimator",
                        category=UserWarning,
                    )
                    cache[model_id] = pickle.load(f)
            log.info("Model yuklendi: h=%d %s", horizon, model_id)
        return cache[model_id]

    @torch.no_grad()
    def _lazy_load_dl(self, model_id: str) -> nn.Module | None:
        """DL model state_dict'i yükler (lazy, cache'li).

        Args:
            model_id: 'lstm', 'gru' veya 'bigru_attn'.

        Returns:
            Yuklenmis nn.Module veya None.
        """
        if model_id not in self._dl_cache:
            pt_file = _WINDOW_MODELS.get(model_id)
            if pt_file is None:
                return None
            pt_path = _BASE / pt_file
            if not pt_path.exists():
                log.error(".pt dosyasi bulunamadi: %s", pt_path)
                return None
            model = _make_dl_model(model_id)
            model.load_state_dict(
                torch.load(pt_path, map_location="cpu", weights_only=True)
            )
            model.eval()
            self._dl_cache[model_id] = model
            log.info("DL model yuklendi: %s", model_id)
        return self._dl_cache[model_id]

    def _snapshot_to_window(
        self,
        snapshot: dict[str, float | None],
        repeat_hours: int = 24,
    ) -> torch.Tensor:
        """Snapshot'tan (1, repeat_hours, 18) pencere tensoru uretir.

        Snapshot preprocess edilir (log1p + scale) ve repeat_hours kez
        tekrarlanarak sabit bir pencere olusturulur.

        Args:
            snapshot: Ham hasta degerleri.
            repeat_hours: Tekrar sayisi (pencere uzunlugu).

        Returns:
            (1, T, 18) float32 tensor.
        """
        vec = self._preprocess_snapshot(snapshot)  # (1, 18)
        window = np.repeat(vec, repeat_hours, axis=0)  # (T, 18)
        return torch.from_numpy(window).unsqueeze(0)  # (1, T, 18)

    def _series_to_window(self, series: list[dict[str, float | None]]) -> torch.Tensor:
        """Saatlik snapshot listesinden (1, T, 18) gercek pencere tensoru uretir.

        Args:
            series: Her eleman bir saatin PatientSnapshot alanlari.

        Returns:
            (1, T, 18) float32 tensor.
        """
        rows = [self._preprocess_snapshot(step)[0] for step in series]
        window = np.stack(rows, axis=0).astype(np.float32)
        return torch.from_numpy(window).unsqueeze(0)

    # ------------------------------------------------------------------
    # On-isleme
    # ------------------------------------------------------------------

    def _preprocess_snapshot(self, snapshot: dict[str, float | None]) -> np.ndarray:
        """Ham snapshot sozlugunden 18-feature vektoru uretir.

        Uygulanan adimlar (Faz 3 ile ayni sira):
        1. Eksik alanlar 0.0 ile doldurulur.
        2. log1p donusumu (log_transform_cols).
        3. StandardScaler (feature_stats.json mean/std).

        Args:
            snapshot: Ham hasta degerleri sozlugu.

        Returns:
            (1, 18) sekilli numpy float32 array.
        """
        vec = np.zeros(len(FEATURE_ORDER), dtype=np.float64)
        for i, feat in enumerate(FEATURE_ORDER):
            val = snapshot.get(feat)
            vec[i] = (
                float(val)
                if val is not None and not (isinstance(val, float) and math.isnan(val))
                else 0.0
            )

        # log1p
        for i, feat in enumerate(FEATURE_ORDER):
            if feat in LOG_TRANSFORM_COLS and vec[i] > 0:
                vec[i] = math.log1p(vec[i])

        # StandardScaler
        for i, feat in enumerate(FEATURE_ORDER):
            stats = self._scaler_stats.get(feat)
            if stats:
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0) or 1.0
                vec[i] = (vec[i] - mean) / std

        return vec.reshape(1, -1).astype(np.float32)

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _positive_class_predictor(self, model: Any):
        """Ikili sinif modelinden pozitif sinif olasiligini donduren fonksiyon uretir.

        Args:
            model: predict_proba destekleyen sklearn/xgboost modeli.

        Returns:
            (N, 18) girdiyi (N,) pozitif sinif skoruna ceviren callable.
        """

        def predict(x: np.ndarray) -> np.ndarray:
            return model.predict_proba(x)[:, 1]

        return predict

    def _get_shap_explainer(self, model_id: str = "xgboost", horizon: int = 6) -> dict[str, Any] | None:
        """XGBoost icin SHAP aciklayici olusturur; TreeExplainer basarisizsa fallback kullanir.

        Args:
            model_id: Model kimlik kodu (varsayilan: 'xgboost').
            horizon: Tahmin ufku.

        Returns:
            {'kind': 'tree'|'generic', 'explainer': ...} sozlugu veya None.
        """
        if horizon not in self._shap_cache:
            self._shap_cache[horizon] = {}
        cache = self._shap_cache[horizon]
        if model_id not in cache:
            try:
                import shap as _shap

                m = self._lazy_load(model_id, horizon=horizon)
                if m is None:
                    raise ValueError(f"Model yuklenemedi: h={horizon} {model_id}")
                try:
                    explainer = _shap.TreeExplainer(m)
                    cache[model_id] = {"kind": "tree", "explainer": explainer}
                    log.info("SHAP TreeExplainer olusturuldu: h=%d %s", horizon, model_id)
                except Exception as tree_exc:
                    background = np.zeros((32, len(FEATURE_ORDER)), dtype=np.float32)
                    predict_fn = self._positive_class_predictor(m)
                    explainer = _shap.Explainer(predict_fn, background)
                    cache[model_id] = {"kind": "generic", "explainer": explainer}
                    log.warning(
                        "SHAP TreeExplainer basarisiz, Explainer fallback: h=%d %s (%s)",
                        horizon,
                        model_id,
                        tree_exc,
                    )
            except Exception as exc:
                log.warning("SHAP explainer olusturulamadi (h=%d %s): %s", horizon, model_id, exc)
                return None
        return cache.get(model_id)

    def _extract_shap_row(self, explainer_entry: dict[str, Any], X: np.ndarray) -> np.ndarray | None:
        """SHAP aciklayicisindan tek ornek icin ozellik katki vektorunu cikarir.

        Args:
            explainer_entry: _get_shap_explainer ciktisi.
            X: (1, 18) preprocess edilmis numpy array.

        Returns:
            (18,) SHAP degerleri veya hata durumunda None.
        """
        kind = explainer_entry["kind"]
        explainer = explainer_entry["explainer"]
        if kind == "tree":
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            return np.asarray(sv[0], dtype=np.float64)
        out = explainer(X)
        values = np.asarray(out.values, dtype=np.float64)
        if values.ndim == 3:
            return values[0, :, 1]
        if values.ndim == 2:
            return values[0]
        return values.reshape(-1)

    def _compute_scores(self, X: Any, horizon: int = 6) -> list[dict]:
        """5 ML modeli ile skor hesaplar (predict_snapshot icin yardimci).

        Args:
            X: (1, 18) preprocess edilmis numpy array.
            horizon: Tahmin ufku (0, 6 veya 24).

        Returns:
            [{model_id, model_name, risk_score, alert, threshold}] listesi.
        """
        thresholds = self._thresholds_by_horizon.get(horizon, self._thresholds)
        results: list[dict] = []
        for model_id in MODEL_ORDER:
            model = self._lazy_load(model_id, horizon=horizon)
            threshold = thresholds.get(model_id, 0.5)
            if model is None:
                results.append(
                    {
                        "model_id": model_id,
                        "model_name": MODEL_DISPLAY.get(model_id, model_id),
                        "risk_score": 0.0,
                        "alert": False,
                        "threshold": threshold,
                    }
                )
                continue
            try:
                prob = float(model.predict_proba(X)[0, 1])
            except Exception as exc:
                log.error("Tahmin hatasi (%s): %s", model_id, exc)
                prob = 0.0
            results.append(
                {
                    "model_id": model_id,
                    "model_name": MODEL_DISPLAY.get(model_id, model_id),
                    "risk_score": round(prob, 6),
                    "alert": prob >= threshold,
                    "threshold": round(threshold, 6),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------

    def predict_snapshot(self, snapshot: dict[str, float | None]) -> list[dict]:
        """5 ML modeli ile h=6 snapshot tahmini yapar (geri uyumlu).

        Args:
            snapshot: PatientSnapshot alanlari sozlugu.

        Returns:
            [{model_id, model_name, risk_score, alert, threshold}, ...] listesi.
        """
        return self.predict_snapshot_horizon(snapshot, horizon=6)

    def predict_snapshot_horizon(
        self,
        snapshot: dict[str, float | None],
        horizon: int = 6,
    ) -> list[dict]:
        """Belirtilen ufuk icin 5 ML modeli ile snapshot tahmini yapar.

        Args:
            snapshot: PatientSnapshot alanlari sozlugu.
            horizon: 0 (anlik), 6 veya 24 saat.

        Returns:
            Model skor listesi.

        Raises:
            ValueError: Desteklenmeyen horizon degeri.
        """
        if horizon not in _SUPPORTED_HORIZONS:
            raise ValueError(f"Desteklenmeyen horizon: {horizon}. Izin verilen: {sorted(_SUPPORTED_HORIZONS)}")
        X = self._preprocess_snapshot(snapshot)
        return self._compute_scores(X, horizon=horizon)

    def predict_snapshot_explain(
        self,
        snapshot: dict[str, float | None],
        horizon: int = 6,
    ) -> dict:
        """5 ML modeli ile skor + XGBoost SHAP top-5 hesaplar.

        Args:
            snapshot: PatientSnapshot alanlari sozlugu.
            horizon: Tahmin ufku.

        Returns:
            {'models': [...], 'shap_top5': [...] veya None, 'horizon': int} sozlugu.
        """
        X = self._preprocess_snapshot(snapshot)
        results = self._compute_scores(X, horizon=horizon)
        shap_top5 = None
        try:
            explainer_entry = self._get_shap_explainer("xgboost", horizon=horizon)
            if explainer_entry is not None:
                shap_row = self._extract_shap_row(explainer_entry, X)
                if shap_row is not None:
                    abs_sv = np.abs(shap_row)
                    total = float(abs_sv.sum()) or 1.0
                    top5_idx = np.argsort(abs_sv)[-5:][::-1]
                    shap_top5 = [
                        {
                            "feature": FEATURE_ORDER[i],
                            "shap_value": float(shap_row[i]),
                            "abs_shap": float(abs_sv[i]),
                            "pct_contribution": float(abs_sv[i] / total * 100),
                        }
                        for i in top5_idx
                    ]
        except Exception as exc:
            log.warning("SHAP hesaplanamadi: %s", exc)
        return {"models": results, "shap_top5": shap_top5, "horizon": horizon}

    def _compute_gradient_timestep_importance(
        self, model: nn.Module, x: torch.Tensor
    ) -> list[float] | None:
        """LSTM/GRU/Transformer icin timestep bazli gradient saliency uretir.

        Risk skoruna (sigmoid logit) gore girdi timestep'lerinin mutlak gradyan
        ortalamasini normalize ederek 24 saatlik onem dagilimi dondurur.
        """
        try:
            with torch.enable_grad():
                model.eval()
                x_grad = x.clone().detach().requires_grad_(True)
                logit, _ = model(x_grad)
                score = torch.sigmoid(logit.squeeze())
                model.zero_grad(set_to_none=True)
                score.backward()
                if x_grad.grad is None:
                    return None
                step = x_grad.grad.abs().mean(dim=2).squeeze(0)
                total = float(step.sum().item())
                if total <= 0:
                    return None
                return (step / total).detach().cpu().tolist()
        except Exception as exc:
            log.warning("Gradient saliency hesaplanamadi: %s", exc)
            return None

    def _resolve_window_importance(
        self, model: nn.Module, model_id: str, x: torch.Tensor, attn_w: torch.Tensor | None
    ) -> tuple[list[float] | None, str | None]:
        """DL modeli icin timestep onem agirligi ve yontem etiketini cozer."""
        if attn_w is not None:
            weights = attn_w[0].detach().cpu().tolist()
            return weights, "attention"
        if model_id in {"lstm", "gru", "transformer"}:
            weights = self._compute_gradient_timestep_importance(model, x)
            if weights is not None:
                return weights, "gradient"
        return None, None

    @torch.no_grad()
    def predict_window(
        self,
        snapshot: dict[str, float | None],
        repeat_hours: int = 24,
        series: list[dict[str, float | None]] | None = None,
    ) -> dict:
        """DL modelleri ile pencere tahmini yapar.

        series verilirse gercek saatlik seri kullanilir; aksi halde snapshot
        repeat_hours kez tekrarlanir (demo geriye uyumluluk).

        Args:
            snapshot: Son saat veya tek snapshot (series yoksa tekrarlanir).
            repeat_hours: Pencere uzunlugu (series yokken).
            series: Opsiyonel saatlik snapshot listesi (T adim).

        Returns:
            {'models': [...], 'window_shape': (T, 18), 'input_mode': str} sozlugu.
        """
        if series and len(series) > 0:
            x = self._series_to_window(series)
            input_mode = "series"
        else:
            x = self._snapshot_to_window(snapshot, repeat_hours)
            input_mode = "repeat"
        T = x.shape[1]
        results: list[dict] = []

        for model_id in _WINDOW_MODELS:
            model = self._lazy_load_dl(model_id)
            threshold = self._dl_thresholds.get(model_id, 0.5)
            display = _DL_DISPLAY_NAMES.get(model_id, model_id)

            if model is None:
                results.append(
                    {
                        "model_id": model_id,
                        "model_name": display,
                        "risk_score": 0.0,
                        "alert": False,
                        "threshold": threshold,
                        "attention_weights": None,
                        "importance_method": None,
                    }
                )
                continue

            try:
                logit, attn_w = model(x)
                prob = float(torch.sigmoid(logit).item())
            except Exception as exc:
                log.error("DL tahmin hatasi (%s): %s", model_id, exc)
                prob = 0.0
                attn_w = None

            importance, importance_method = self._resolve_window_importance(
                model, model_id, x, attn_w
            )

            results.append(
                {
                    "model_id": model_id,
                    "model_name": display,
                    "risk_score": round(prob, 6),
                    "alert": prob >= threshold,
                    "threshold": round(threshold, 6),
                    "attention_weights": importance,
                    "importance_method": importance_method,
                }
            )

        return {"models": results, "window_shape": (T, 18), "input_mode": input_mode}
