"""Simulatör icin hazir klinik hasta profilleri."""

from __future__ import annotations

from typing import Any

from app.schemas import PatientPreset

# Tum preset satirlarinda kullanilan tam 18-feature snapshot sablonu.
_DEFAULT_SNAPSHOT: dict[str, float] = {
    "HR": 75.0,
    "O2Sat": 98.0,
    "Temp": 37.1,
    "MAP": 85.0,
    "Resp": 14.0,
    "BUN": 12.0,
    "Chloride": 101.0,
    "Creatinine": 0.8,
    "Glucose": 105.0,
    "Hct": 42.0,
    "Hgb": 13.8,
    "WBC": 8.2,
    "Platelets": 250.0,
    "Age": 52.0,
    "HospAdmTime": -4.0,
    "ICULOS": 6.0,
    "Gender_0": 1.0,
    "Gender_1": 0.0,
}

# Farkli klinik senaryolari ve beklenen risk bantlari.
_PRESET_ROWS: list[dict[str, Any]] = [
    {
        "preset_id": "dusuk_risk",
        "label": "Düşük Risk",
        "risk_band": "low",
        "description": "Kararlı vital bulgular; sepsis ihtimali düşük.",
        "snapshot": {},
    },
    {
        "preset_id": "erken_enfeksiyon",
        "label": "Erken Enfeksiyon",
        "risk_band": "medium",
        "description": "Hafif ateş ve lökositoz; erken sepsis bulguları.",
        "snapshot": {
            "Temp": 38.1,
            "WBC": 12.8,
            "HR": 92.0,
            "Resp": 18.0,
            "ICULOS": 8.0,
        },
    },
    {
        "preset_id": "postop_yeni",
        "label": "Postoperatif Yeni Yatış",
        "risk_band": "medium",
        "description": "YBÜ'ye yeni alınmış postop hasta; ateş ve lökosit artışı.",
        "snapshot": {
            "ICULOS": 3.0,
            "HospAdmTime": -0.5,
            "Temp": 38.5,
            "WBC": 13.8,
            "HR": 96.0,
            "MAP": 71.0,
            "Age": 48.0,
            "Gender_0": 0.0,
            "Gender_1": 1.0,
        },
    },
    {
        "preset_id": "hiperglisemi",
        "label": "Hiperglisemi",
        "risk_band": "low",
        "description": "Belirgin glukoz yüksekliği; diğer vitaller görece stabil.",
        "snapshot": {
            "Glucose": 380.0,
            "HR": 88.0,
            "Temp": 37.6,
            "WBC": 10.5,
            "MAP": 78.0,
            "Age": 58.0,
        },
    },
    {
        "preset_id": "sinir_durum",
        "label": "Sınır Durum",
        "risk_band": "medium",
        "description": "Eşik değerlerine yakın ölçümler; modeller arasında görüş ayrılığı oluşabilir.",
        "snapshot": {
            "HR": 95.0,
            "O2Sat": 95.0,
            "Temp": 38.4,
            "MAP": 68.0,
            "Resp": 21.0,
            "BUN": 22.0,
            "Chloride": 104.0,
            "Creatinine": 1.4,
            "Glucose": 138.0,
            "Hct": 37.0,
            "Hgb": 11.9,
            "WBC": 13.1,
            "Platelets": 175.0,
            "Age": 63.0,
            "HospAdmTime": -7.0,
            "ICULOS": 15.0,
            "Gender_0": 0.0,
            "Gender_1": 1.0,
        },
    },
    {
        "preset_id": "trombositopeni",
        "label": "Trombositopeni",
        "risk_band": "medium",
        "description": "Düşük trombosit ve lökositoz; koagülopati ve enfeksiyon paterni.",
        "snapshot": {
            "Platelets": 58.0,
            "WBC": 15.8,
            "Temp": 38.8,
            "HR": 104.0,
            "MAP": 64.0,
            "Creatinine": 1.9,
            "Age": 55.0,
            "Gender_0": 0.0,
            "Gender_1": 1.0,
        },
    },
    {
        "preset_id": "renal_yetmezlik",
        "label": "Akut Böbrek Hasarı",
        "risk_band": "high",
        "description": "Yüksek kreatinin/BUN; organ disfonksiyonu odaklı profil.",
        "snapshot": {
            "Creatinine": 4.8,
            "BUN": 72.0,
            "Platelets": 92.0,
            "MAP": 66.0,
            "WBC": 14.2,
            "Temp": 37.8,
            "Age": 76.0,
            "ICULOS": 30.0,
            "Gender_0": 0.0,
            "Gender_1": 1.0,
        },
    },
    {
        "preset_id": "respiratuvar_distres",
        "label": "Respiratuvar Distres",
        "risk_band": "high",
        "description": "Düşük SpO₂ ve yüksek solunum hızı; hipoksi paterni.",
        "snapshot": {
            "O2Sat": 84.0,
            "Resp": 34.0,
            "MAP": 62.0,
            "HR": 112.0,
            "Temp": 38.7,
            "WBC": 16.5,
            "Age": 59.0,
            "Gender_0": 1.0,
            "Gender_1": 0.0,
        },
    },
    {
        "preset_id": "yasli_komorbid",
        "label": "Yaşlı Komorbid",
        "risk_band": "high",
        "description": "İleri yaş, çoklu organ stresi ve uzun YBÜ yatışı.",
        "snapshot": {
            "Age": 86.0,
            "Creatinine": 2.2,
            "MAP": 61.0,
            "WBC": 14.5,
            "BUN": 38.0,
            "Platelets": 118.0,
            "HR": 98.0,
            "Temp": 38.3,
            "ICULOS": 42.0,
            "Gender_0": 1.0,
            "Gender_1": 0.0,
        },
    },
    {
        "preset_id": "yuksek_risk",
        "label": "Yüksek Risk",
        "risk_band": "high",
        "description": "Yüksek HR, düşük MAP, artmış kreatinin — sepsis uyarısı beklenir.",
        "snapshot": {
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
        },
    },
    {
        "preset_id": "hipotansiyon",
        "label": "Hipotansiyon",
        "risk_band": "high",
        "description": "Belirgin MAP düşüklüğü ve taşikardi; dolaşım yetmezliği paterni.",
        "snapshot": {
            "MAP": 47.0,
            "HR": 128.0,
            "O2Sat": 92.0,
            "Resp": 26.0,
            "Temp": 38.9,
            "WBC": 17.2,
            "Creatinine": 1.8,
            "Age": 64.0,
            "Gender_0": 0.0,
            "Gender_1": 1.0,
        },
    },
    {
        "preset_id": "septik_sok",
        "label": "Septik Şok",
        "risk_band": "high",
        "description": "Kritik vital bulgular: hipotansiyon, taşikardi, hipoksi ve lökositoz.",
        "snapshot": {
            "HR": 132.0,
            "O2Sat": 86.0,
            "Temp": 39.9,
            "MAP": 51.0,
            "Resp": 30.0,
            "BUN": 48.0,
            "Creatinine": 3.6,
            "Glucose": 210.0,
            "WBC": 24.5,
            "Platelets": 72.0,
            "Hgb": 9.2,
            "Hct": 28.0,
            "Age": 68.0,
            "ICULOS": 18.0,
            "Gender_0": 0.0,
            "Gender_1": 1.0,
        },
    },
]


def _gender_from_snapshot(snapshot: dict[str, float]) -> str:
    """Gender_0/Gender_1 one-hot alanlarindan M/F cinsiyet kodu uretir."""
    if snapshot.get("Gender_0", 0.0) >= 0.5:
        return "F"
    if snapshot.get("Gender_1", 0.0) >= 0.5:
        return "M"
    return "M"


def _snapshot_to_features(snapshot: dict[str, float]) -> dict[str, float]:
    """Snapshot sozlugunden Gender kolonlarini cikarip frontend features uretir."""
    return {
        key: float(value)
        for key, value in snapshot.items()
        if key not in {"Gender_0", "Gender_1"} and value is not None
    }


def _merge_snapshot(overrides: dict[str, float]) -> dict[str, float]:
    """Varsayilan snapshot uzerine preset ozel degerleri uygular."""
    merged = dict(_DEFAULT_SNAPSHOT)
    merged.update(overrides)
    return merged


def preset_from_row(row: dict[str, Any]) -> PatientPreset:
    """Ham preset satirini API PatientPreset modeline cevirir."""
    snapshot = _merge_snapshot({k: float(v) for k, v in row.get("snapshot", {}).items()})
    return PatientPreset(
        preset_id=row["preset_id"],
        label=row["label"],
        risk_band=row.get("risk_band", "medium"),
        description=row["description"],
        gender=_gender_from_snapshot(snapshot),
        features=_snapshot_to_features(snapshot),
    )


def list_patient_presets() -> list[PatientPreset]:
    """Simulatör icin tum hazir hasta profillerini risk bandina gore sirali dondurur."""
    order = {"low": 0, "medium": 1, "high": 2}
    presets = [preset_from_row(row) for row in _PRESET_ROWS]
    presets.sort(key=lambda item: (order.get(item.risk_band, 1), item.label))
    return presets
