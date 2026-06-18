"""Comparativa para texto: indice invertido propio vs GIN vs pgvector.

Las tres corren la MISMA query sobre los MISMOS datos (ya persistidos), asi se
puede comparar manzana con manzana: que devuelve cada una y cuanto tarda.

- own:      tu motor (codebook lingüistico + indice invertido + coseno).
- gin:      full-text nativo de Postgres (tsvector @@ tsquery, ranking ts_rank).
- pgvector: similitud coseno sobre los histogramas guardados como vector.
"""
from __future__ import annotations

import numpy as np

from app.comparisons import timed


def own_search(index, q: str, top_n: int = 10) -> dict:
    """Motor propio: el indice invertido construido con SPIMI."""
    results, ms = timed(lambda: index.search(q, top_n=top_n))
    return {"method": "inverted_index", "latency_ms": ms, "count": len(results), "results": results}


def gin_search(conn, q: str, top_n: int = 10, ts_config: str = "spanish") -> dict:
    """GIN nativo: full-text sobre las letras, agregando al mejor chunk por cancion."""
    sql = """
        SELECT i.id, i.external_id,
               i.metadata->>'title'  AS title,
               i.metadata->>'artist' AS artist,
               max(ts_rank(c.tsv, query)) AS rank
        FROM chunks c
        JOIN items i ON i.id = c.item_id,
             plainto_tsquery(%s, %s) AS query
        WHERE c.modality = 'text' AND c.tsv @@ query
        GROUP BY i.id, title, artist
        ORDER BY rank DESC
        LIMIT %s
    """

    def run():
        with conn.cursor() as cur:
            cur.execute(sql, (ts_config, q, top_n))
            return cur.fetchall()

    rows, ms = timed(run)
    results = [
        {"item_id": r[0], "external_id": r[1], "title": r[2], "artist": r[3],
         "score": round(float(r[4]), 4)}
        for r in rows
    ]
    return {"method": "gin_fulltext", "latency_ms": ms, "count": len(results), "results": results}


def pgvector_search(conn, codebook, q: str, top_n: int = 10) -> dict:
    """pgvector: cuantiza la query con el codebook y busca por coseno (<=>)."""
    q_hist = np.asarray(codebook.quantize(q), dtype=np.float32)
    if not np.any(q_hist):     # query sin codewords conocidas -> nada que comparar
        return {"method": "pgvector_cosine", "latency_ms": 0.0, "count": 0, "results": []}

    sql = """
        SELECT i.id, i.external_id,
               i.metadata->>'title'  AS title,
               i.metadata->>'artist' AS artist,
               min(h.hist <=> %s) AS dist
        FROM histograms h
        JOIN chunks c ON c.id = h.chunk_id
        JOIN items  i ON i.id = c.item_id
        WHERE h.modality = 'text'
        GROUP BY i.id, title, artist
        ORDER BY dist ASC
        LIMIT %s
    """

    def run():
        with conn.cursor() as cur:
            cur.execute(sql, (q_hist, top_n))
            return cur.fetchall()

    rows, ms = timed(run)
    results = [
        {"item_id": r[0], "external_id": r[1], "title": r[2], "artist": r[3],
         "score": round(1.0 - float(r[4]), 4)}    # coseno = 1 - distancia
        for r in rows
    ]
    return {"method": "pgvector_cosine", "latency_ms": ms, "count": len(results), "results": results}
