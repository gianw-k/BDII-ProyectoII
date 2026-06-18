"""Comparativa para imagen: indice invertido propio vs pgvector.

Aqui no hay GIN (eso es para texto); la tecnica nativa a vencer es pgvector.
Dado un producto (por external_id) buscamos los mas parecidos por las dos vias y
comparamos resultados + latencia. El producto-consulta se excluye de su propia
lista.
"""
from __future__ import annotations

import numpy as np

from app.comparisons import timed
from app.engine.search.similarity import search_sparse


def own_search(index, external_id: str, top_n: int = 10) -> dict:
    """Motor propio: usa el histograma del producto como query sobre el indice."""
    local = _local_id(index, external_id)
    if local is None:
        return {"method": "inverted_index", "latency_ms": 0.0, "count": 0, "results": []}
    q_sparse = _item_sparse(index, local)

    def run():
        hits = search_sparse(q_sparse, index.index, top_n=top_n + 1)
        out = []
        for cid, score in hits:
            if cid == local:           # no devolver el mismo producto
                continue
            meta = index.items.get(cid, {})
            out.append({"external_id": meta.get("external_id"),
                        "score": round(float(score), 4), **meta})
        return out[:top_n]

    results, ms = timed(run)
    return {"method": "inverted_index", "latency_ms": ms, "count": len(results), "results": results}


def pgvector_search(conn, external_id: str, top_n: int = 10) -> dict:
    """pgvector: toma el histograma guardado del producto y hace KNN por coseno."""
    probe_sql = """
        SELECT h.hist
        FROM histograms h
        JOIN chunks c ON c.id = h.chunk_id
        JOIN items  i ON i.id = c.item_id
        WHERE i.app = 'ecommerce' AND i.external_id = %s AND h.modality = 'image'
        LIMIT 1
    """
    knn_sql = """
        SELECT i.external_id, (h.hist <=> %s) AS dist
        FROM histograms h
        JOIN chunks c ON c.id = h.chunk_id
        JOIN items  i ON i.id = c.item_id
        WHERE h.modality = 'image' AND i.external_id <> %s
        ORDER BY dist ASC
        LIMIT %s
    """

    def run():
        with conn.cursor() as cur:
            cur.execute(probe_sql, (external_id,))
            row = cur.fetchone()
            if row is None:
                return None
            probe = np.asarray(row[0], dtype=np.float32)
            cur.execute(knn_sql, (probe, external_id, top_n))
            return cur.fetchall()

    rows, ms = timed(run)
    if rows is None:
        return {"method": "pgvector_cosine", "latency_ms": ms, "count": 0, "results": []}
    results = [{"external_id": r[0], "score": round(1.0 - float(r[1]), 4)} for r in rows]
    return {"method": "pgvector_cosine", "latency_ms": ms, "count": len(results), "results": results}


def _local_id(index, external_id: str):
    for local, meta in index.items.items():
        if str(meta.get("external_id")) == str(external_id):
            return local
    return None


def _item_sparse(index, local_id: int) -> list[tuple[int, float]]:
    """Reconstruye el histograma disperso de un producto desde el indice invertido."""
    return [
        (word_idx, weight)
        for word_idx, plist in index.index.postings.items()
        for cid, weight in plist
        if cid == local_id
    ]
