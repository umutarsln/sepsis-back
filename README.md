# Sepsis-Son Backend

Bu klasor `sepsis-son` icin ayri ve kompakt backend kodunu icerir.

## Hedef

- `sepsis-final` klasorune runtime bagimlilik olmadan calismak
- Frontend demo icin minimum gerekli endpoint'leri saglamak
- Faz 5 ve Faz 7'de endpoint genisletmelerini bu klasor uzerinden yapmak

## Dizin Yapisi

```text
backend/
  app/
    main.py
    schemas.py
    services/
      inference_registry.py
```

## Calistirma

```bash
cd sepsis-son/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uvicorn app.main:app --reload --port 8000
```

## Onemli Not

- ML modelleri **scikit-learn 1.4.2** ile egitilip kaydedilmistir.
- `scikit-learn>=1.8` ile `GradientBoostingClassifier` pkl dosyalari yuklenemez
  (`CyHalfBinomialLoss` hatasi). Bu nedenle `requirements.txt` icinde **1.5.2** sabitlenmistir.
- SHAP aciklamasi icin `shap` paketi `requirements.txt` icinde tanimlidir.
