"""Adaptadores: de un indice en memoria a `IndexData` para la DB.

Cada modalidad guarda sus datos un poco distinto (texto tiene chunks con
contenido; imagen tiene un histograma por producto), asi que aqui los
acomodamos al formato plano que entiende `repository.persist_index`. El
repositorio no conoce las apps; el acoplamiento vive solo aqui.
"""
from __future__ import annotations

import numpy as np

from app.db.repository import IndexData


def text_index_to_data(index, app: str = "music") -> IndexData:
    """MusicTextIndex -> IndexData (modalidad texto)."""
    chunks = {
        cid: (c["item_id"], c["position"], c["content"])
        for cid, c in index.chunks.items()
    }
    # histograma denso por chunk (lo que veria pgvector); k = len(terms)
    histograms = {
        cid: index.codebook.quantize(c["content"])
        for cid, c in index.chunks.items()
    }
    return IndexData(
        app=app,
        modality="text",
        items=index.items,
        chunks=chunks,
        terms=index.codebook.terms,
        postings=index.index.postings,
        histograms=histograms,
    )


def visual_index_to_data(index, app: str = "ecommerce") -> IndexData:
    """VisualIndex -> IndexData (modalidad imagen).

    Aqui cada producto es a la vez item y chunk (un solo histograma BoVW por
    producto), asi que el chunk_id local coincide con el item_id local.
    """
    chunks = {item_id: (item_id, 0, None) for item_id in index.items}
    # hist_dim = visual words + bins de color (si se fusiono el color)
    histograms = _dense_from_postings(index.index.postings, chunks.keys(), index.hist_dim)
    return IndexData(
        app=app,
        modality="image",
        items=index.items,
        chunks=chunks,
        centroids=index.codebook.centroids,
        postings=index.index.postings,
        histograms=histograms,
    )


def _dense_from_postings(postings, chunk_ids, k: int) -> dict[int, np.ndarray]:
    """Reconstruye el histograma denso de cada chunk a partir del indice invertido."""
    hist = {cid: np.zeros(k, dtype=np.float32) for cid in chunk_ids}
    for word_idx, plist in postings.items():
        for chunk_id, weight in plist:
            if chunk_id in hist:
                hist[chunk_id][word_idx] = weight
    return hist
