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
