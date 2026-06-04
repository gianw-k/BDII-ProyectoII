"""Busqueda por similitud coseno sobre el indice invertido.

Solo se tocan los chunks que comparten alguna codeword con la query
(accumulator pattern), no toda la coleccion. Es lo que hace eficiente al
indice invertido frente a un scan denso.

    score[chunk] = sum_w  q_w * doc_w        (sobre codewords compartidas)
    cosine       = score / (||q|| * ||doc||)
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np

from app.engine.codebook.linguistic import LinguisticCodebook
from app.engine.index.histogram import to_sparse
from app.engine.index.inverted import InvertedIndex


def search(
    query: str,
    codebook: LinguisticCodebook,
    index: InvertedIndex,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """texto query a [(chunk_id, score), ...] top-N ordenado desc."""
    q_hist = codebook.quantize(query)
    q_sparse = to_sparse(q_hist)
    if not q_sparse:
        return []
    q_norm = float(np.linalg.norm(q_hist)) or 1.0

    scores: dict[int, float] = defaultdict(float)
    for word_idx, q_w in q_sparse:
        for chunk_id, doc_w in index.get(word_idx):
            scores[chunk_id] += q_w * doc_w

    ranked: list[tuple[int, float]] = []
    for chunk_id, dot in scores.items():
        denom = q_norm * (index.norms.get(chunk_id, 1.0) or 1.0)
        ranked.append((chunk_id, dot / denom))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
