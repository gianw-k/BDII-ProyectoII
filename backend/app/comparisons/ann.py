"""pgvector: búsqueda exacta (fuerza bruta) vs aproximada (HNSW).

Las dos corren la misma query coseno (`<=>`) sobre los mismos histogramas; lo
único que cambia es si el planner usa el índice HNSW o no:

    exacto  -> enable_indexscan=off -> Seq Scan, recorre todo, recall perfecto
    aprox   -> hnsw.ef_search=N      -> Index Scan HNSW, más rápido, recall < 1

El recall del aproximado se compara contra el exacto, que es el que devuelve el
vecino más cercano de verdad.

La columna `hist` es `vector` sin dimensión y HNSW necesita una fija, así que el
índice y la query castean a `vector(256)`. El índice es parcial por modalidad y
sobre esa expresión, por eso la query usa `hist::vector(256)` igual para que el
planner lo reconozca.
"""
from __future__ import annotations

import numpy as np

from app.comparisons import timed

DIM = 256


# --------------------------------------------------------------------- índice
def ensure_hnsw(conn, modality: str = "text", dim: int = DIM,
                m: int = 16, ef_construction: int = 64) -> str:
    """Crea (si no existe) el índice HNSW coseno para una modalidad. Idempotente."""
    name = f"idx_hist_hnsw_{modality}"
    sql = (
        f"CREATE INDEX IF NOT EXISTS {name} "
        f"ON histograms USING hnsw ((hist::vector({dim})) vector_cosine_ops) "
        f"WITH (m = {m}, ef_construction = {ef_construction}) "
        f"WHERE modality = %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (modality,))
    conn.commit()
    return name


def drop_hnsw(conn, modality: str = "text") -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS idx_hist_hnsw_{modality}")
    conn.commit()


# ----------------------------------------------------------------- búsquedas
_SQL = """
    SELECT c.item_id, i.external_id,
           i.metadata->>'title'  AS title,
           i.metadata->>'artist' AS artist,
           h.hist::vector(%(dim)s) <=> %(q)s::vector(%(dim)s) AS dist
    FROM histograms h
    JOIN chunks c ON c.id = h.chunk_id
    JOIN items  i ON i.id = c.item_id
    WHERE h.modality = %(mod)s
    ORDER BY h.hist::vector(%(dim)s) <=> %(q)s::vector(%(dim)s)
    LIMIT %(lim)s
"""


def _dedup_to_items(rows, top_n: int) -> list[dict]:
    """Chunk-level -> item-level: nos quedamos con el mejor chunk por item."""
    best: dict[int, dict] = {}
    for item_id, ext, title, artist, dist in rows:
        if item_id not in best or dist < best[item_id]["_dist"]:
            best[item_id] = {
                "item_id": item_id, "external_id": ext, "title": title,
                "artist": artist, "_dist": float(dist),
            }
    out = sorted(best.values(), key=lambda r: r["_dist"])[:top_n]
    for r in out:
        r["score"] = round(1.0 - r.pop("_dist"), 4)   # coseno = 1 - distancia
    return out


def _query(conn, q_hist, modality: str, top_n: int, dim: int,
           *, exact: bool, ef_search: int, fetch: int) -> list[dict]:
    q = np.asarray(q_hist, dtype=np.float32)   # register_vector lo adapta a `vector`
    params = {"dim": dim, "q": q, "mod": modality, "lim": fetch}

    with conn.cursor() as cur:
        if exact:
            # sin index scan Postgres cae a Seq Scan y recorre todo
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_bitmapscan = off")
        else:
            cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
        cur.execute(_SQL, params)
        rows = cur.fetchall()
    return _dedup_to_items(rows, top_n)


def pgvector_exact(conn, q_hist, modality: str = "text", top_n: int = 10,
                   dim: int = DIM, fetch_mult: int = 4) -> dict:
    """Búsqueda exacta (Seq Scan): recorre todos los histogramas, recall perfecto."""
    def run():
        return _query(conn, q_hist, modality, top_n, dim,
                      exact=True, ef_search=0, fetch=top_n * fetch_mult)

    results, ms = timed(run)
    conn.rollback()   # cierra la txn -> limpia los SET LOCAL para la sgte query
    return {"method": "pgvector_exact", "latency_ms": ms,
            "count": len(results), "results": results}


def pgvector_hnsw(conn, q_hist, modality: str = "text", top_n: int = 10,
                  dim: int = DIM, ef_search: int = 40, fetch_mult: int = 4) -> dict:
    """ANN aproximado vía HNSW. `ef_search` mayor = más recall, más latencia."""
    def run():
        return _query(conn, q_hist, modality, top_n, dim,
                      exact=False, ef_search=ef_search, fetch=top_n * fetch_mult)

    results, ms = timed(run)
    conn.rollback()   # cierra la txn -> limpia los SET LOCAL para la sgte query
    return {"method": "pgvector_hnsw", "latency_ms": ms, "ef_search": ef_search,
            "count": len(results), "results": results}
