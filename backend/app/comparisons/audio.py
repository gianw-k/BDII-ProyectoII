"""Comparativa para audio: indice invertido propio vs pgvector.

Dado el filename de una pista del dataset GTZAN, buscamos las pistas
acusticamente mas similares por dos vias y comparamos resultados + latencia.
La pista de consulta se excluye de su propia lista de resultados.

Analogia exacta con image.py pero usando la modalidad 'audio'.
"""
from __future__ import annotations
import numpy as np

from app.comparisons import timed
from app.engine.search.similarity import search_sparse


def own_search(acoustic_index, filename: str, top_n: int = 10) -> dict:
    """Motor propio: usa el histograma del indice para buscar pistas similares."""
    hist = acoustic_index.get_track_features(filename)
    if hist is None:
        return {"method": "inverted_index", "latency_ms": 0.0, "count": 0, "results": []}

    from app.engine.index.histogram import to_sparse
    q_sparse = to_sparse(hist)

    def run():
        hits = acoustic_index.index.postings  # noqa — usamos search_from_hist
        results = acoustic_index.search_from_hist(hist, top_n=top_n + 1)
        out = []
        for item in results:
            if item.get("filename") == filename:
                continue
            out.append({
                "filename": item.get("filename"),
                "label":    item.get("label"),
                "score":    item.get("score"),
            })
        return out[:top_n]

    results, ms = timed(run)
    return {
        "method": "inverted_index",
        "latency_ms": ms,
        "count": len(results),
        "results": results,
    }


def pgvector_search(conn, filename: str, top_n: int = 10) -> dict:
    """pgvector: toma el histograma guardado y hace KNN por similitud coseno.

    Requiere que los datos esten cargados en Postgres con modality='audio'
    (via ingest --persist, cuando se implemente esa parte).
    """
    probe_sql = """
        SELECT h.hist
        FROM histograms h
        JOIN chunks c ON c.id = h.chunk_id
        JOIN items  i ON i.id = c.item_id
        WHERE i.app = 'music' AND i.external_id = %s AND h.modality = 'audio'
        LIMIT 1
    """
    knn_sql = """
        SELECT i.external_id, (h.hist <=> %s) AS dist
        FROM histograms h
        JOIN chunks c ON c.id = h.chunk_id
        JOIN items  i ON i.id = c.item_id
        WHERE h.modality = 'audio' AND i.external_id <> %s
        ORDER BY dist ASC
        LIMIT %s
    """

    def run():
        with conn.cursor() as cur:
            cur.execute(probe_sql, (filename,))
            row = cur.fetchone()
            if row is None:
                return None
            probe = np.asarray(row[0], dtype=np.float32)
            cur.execute(knn_sql, (probe, filename, top_n))
            return cur.fetchall()

    rows, ms = timed(run)
    if rows is None:
        return {"method": "pgvector_cosine", "latency_ms": ms, "count": 0, "results": []}
    results = [
        {"filename": r[0], "score": round(1.0 - float(r[1]), 4)}
        for r in rows
    ]
    return {
        "method": "pgvector_cosine",
        "latency_ms": ms,
        "count": len(results),
        "results": results,
    }
