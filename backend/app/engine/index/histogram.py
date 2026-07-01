"""Utilidades de histograma de codewords.

Un histograma es un vector denso (len = k) que sale de `Codebook.quantize`.
Para el indice invertido interesa su forma dispersa: solo las codewords con
peso > 0. Como `quantize` ya normaliza L2, el coseno entre dos histogramas se
reduce al producto punto sobre las codewords compartidas.
"""
from __future__ import annotations

import numpy as np


def to_sparse(hist: np.ndarray) -> list[tuple[int, float]]:
    """histograma denso a [(word_idx, weight), ...] solo no-cero."""
    nz = np.nonzero(hist)[0]
    return [(int(j), float(hist[j])) for j in nz]


def fuse_color(bovw: np.ndarray, color: np.ndarray, weight: float) -> np.ndarray:
    """Pega el histograma de color despues del de visual words, en uno solo.

    Ambos vienen L2-normalizados; los pesamos por `weight` (cuanto cuenta el
    color) y los concatenamos. Con esos pesos el resultado sigue L2-normalizado,
    y el coseno termina siendo un promedio ponderado: (1-weight) de la parte
    visual + weight de la de color. weight=0 deja solo SIFT.
    """
    if weight <= 0 or color is None or color.size == 0:
        return bovw
    a = float(weight)
    return np.concatenate([np.sqrt(1.0 - a) * bovw, np.sqrt(a) * color]).astype(np.float32)
