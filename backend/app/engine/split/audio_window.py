"""Splitter de audio en ventanas (chunks) de features MFCC.

En lugar de leer el audio crudo y hacer sliding window a mano, trabaja
directamente con las filas ya extraidas del CSV GTZAN (features_3_sec.csv),
donde cada fila es una ventana de 3 segundos.

`split(rows)` recibe una lista de dicts (filas de pandas) que pertenecen
a la misma pista y devuelve una lista de numpy arrays, uno por ventana.

Columnas MFCC del CSV que usamos:
    mfcc1_mean .. mfcc20_mean, mfcc1_var .. mfcc20_var   -> 40 features

Si solo tienes el CSV de 30s (una fila por pista), el split devuelve una
lista con un unico array: sigue siendo valido para el pipeline.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from app.engine.base import Splitter

# Columnas de features que forman el vector acustico (orden sincronizado con mfcc.py)
_MFCC_COLS = [
    "chroma_stft_mean", "chroma_stft_var",
    "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var",
    "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var",
    "perceptr_mean", "perceptr_var",
    "tempo",
] + (
    [f"mfcc{i}_mean" for i in range(1, 21)] +
    [f"mfcc{i}_var"  for i in range(1, 21)]
)


class AudioWindowSplitter(Splitter):
    """Convierte filas CSV (ventanas de audio) en arrays numpy de features.

    Parameters
    ----------
    feature_cols: list[str] | None
        Columnas a extraer. Por defecto los 40 MFCC del CSV GTZAN.
    """

    def __init__(self, feature_cols: list[str] | None = None) -> None:
        self.feature_cols = feature_cols or _MFCC_COLS

    def split(self, rows: list[dict] | Any) -> list[np.ndarray]:
        """rows = lista de dicts (filas de una pista) -> lista de arrays (1, dim)."""
        if not rows:
            return []
        chunks = []
        for row in rows:
            try:
                vec = np.array(
                    [float(row[col]) for col in self.feature_cols],
                    dtype=np.float32,
                )
                chunks.append(vec)
            except (KeyError, ValueError, TypeError):
                # fila invalida / columna ausente: la saltamos
                continue
        return chunks
