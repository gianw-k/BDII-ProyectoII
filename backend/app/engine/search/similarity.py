"""Busqueda por similitud coseno sobre el indice invertido.

Solo se tocan los chunks que comparten alguna codeword con la query
(accumulator pattern), no toda la coleccion. Es lo que hace eficiente al
indice invertido frente a un scan denso.

    score[chunk] = sum_w  q_w * doc_w        (sobre codewords compartidas)
    cosine       = score / (||q|| * ||doc||)

`search_sparse` es el corazon de todo y no le importa la modalidad: le pasas el
histograma disperso de la query (venga de texto, imagen o audio) y listo.
`search` es solo el atajo para texto: cuantiza el string y se lo pasa.
"""
from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING

from app.engine.index.histogram import to_sparse
from app.engine.index.inverted import InvertedIndex

if TYPE_CHECKING:
    from app.engine.codebook.linguistic import LinguisticCodebook

SparseHist = list[tuple[int, float]]


def search_sparse(
    q_sparse: SparseHist,
    index: InvertedIndex,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Histograma disperso de la query -> [(chunk_id, score)] top-N de mayor a menor.

    Esto lo comparten todas las modalidades. Damos por hecho que los pesos de los
    postings ya vienen L2-normalizados; ||doc|| lo sacamos de index.norms (y si ya
    estaba normalizado, vale 1.0).
    """
    if not q_sparse:
        return []
    q_norm = sum(w * w for _, w in q_sparse) ** 0.5 or 1.0

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


def search(
    query: str,
    codebook: "LinguisticCodebook",
    index: InvertedIndex,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Atajo para texto: cuantiza el string y delega en search_sparse."""
    return search_sparse(to_sparse(codebook.quantize(query)), index, top_n=top_n)
