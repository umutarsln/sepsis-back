"""macOS ve coklu OpenMP kutuphanesi icin runtime ortam ayarlari.

numpy, scikit-learn, xgboost veya torch import edilmeden once bu modul
yuklenmelidir; aksi halde OMP mutex hatalari backend'i dusurebilir.
"""

from __future__ import annotations

import os


def configure_runtime_env() -> None:
    """OpenMP/BLAS cakismalarini azaltmak icin varsayilan thread ortamini ayarlar."""
    defaults = {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


configure_runtime_env()
