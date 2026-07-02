"""Persistencia del indice en PostgreSQL.

Vuelca a las tablas de db/init/01_init.sql todo lo que pide el enunciado
(codebook, histogramas, metadatos, indice invertido). La gracia es que hay un
solo nucleo generico, `persist_index`, y tanto texto como imagen le pasan sus
datos ya desarmados; asi no duplicamos la logica de escritura por modalidad.

Los ids del indice en memoria (0..N) no son los ids de la DB (BIGSERIAL), asi
que insertamos con RETURNING id y vamos guardando el mapeo local -> db para que
chunks, histogramas e indice invertido apunten a los ids correctos.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from psycopg2.extras import Json, execute_values


@dataclass
class IndexData:
    """Lo que necesita la DB, ya desarmado y agnostico de la modalidad."""
    app: str                                   # 'music' | 'ecommerce'
    modality: str                              # 'text' | 'image' | 'audio'
    items: dict[int, dict]                     # item_id local -> metadata
    # chunk_id local -> (item_id local, position, content|None)
    chunks: dict[int, tuple[int, int, str | None]]
    terms: list[str] | None = None             # codebook de texto (top-k palabras)
    centroids: np.ndarray | None = None        # codebook de imagen/audio (k, dim)
    # word_idx -> [(chunk_id local, peso), ...]
    postings: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    # chunk_id local -> histograma denso (para pgvector). Opcional.
    histograms: dict[int, np.ndarray] = field(default_factory=dict)


def reset(conn, app: str, modality: str) -> None:
    """Borra lo de esta app/modalidad para poder re-ingestar sin duplicar."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM codebook WHERE modality = %s", (modality,))
        cur.execute("DELETE FROM inverted_index WHERE modality = %s", (modality,))
        # borrar los items arrastra (CASCADE) sus chunks e histogramas.
        # Filtra por app Y modalidad: 'music' tiene texto y audio, y sin el
        # filtro el ingest de una modalidad borraba los datos de la otra.
        cur.execute(
            "DELETE FROM items WHERE app = %s AND modality = %s",
            (app, modality),
        )


def persist_index(conn, data: IndexData, *, ts_config: str = "spanish") -> dict:
    """Escribe todo el indice en la DB. Devuelve un pequeno resumen de conteos."""
    reset(conn, data.app, data.modality)
    with conn.cursor() as cur:
        item_map = _insert_items(cur, data)
        chunk_map = _insert_chunks(cur, data, item_map, ts_config)
        _insert_codebook(cur, data)
        _insert_postings(cur, data, chunk_map)
        _insert_histograms(cur, data, chunk_map)
    return {
        "items": len(data.items),
        "chunks": len(data.chunks),
        "codewords": len(data.terms or []) or (0 if data.centroids is None else len(data.centroids)),
        "postings": sum(len(p) for p in data.postings.values()),
        "histograms": len(data.histograms),
    }


# --------------------------------------------------------------------- inserts
def _insert_items(cur, data: IndexData) -> dict[int, int]:
    """Inserta items y devuelve el mapeo id_local -> id_db."""
    locals_ = sorted(data.items)
    rows = [
        (data.modality, data.app, _ext_id(data.items[i]), Json(data.items[i]))
        for i in locals_
    ]
    returned = execute_values(
        cur,
        "INSERT INTO items (modality, app, external_id, metadata) VALUES %s RETURNING id",
        rows, fetch=True,
    )
    return {loc: r[0] for loc, r in zip(locals_, returned)}


def _insert_chunks(cur, data: IndexData, item_map, ts_config) -> dict[int, int]:
    """Inserta chunks (mapeando su item) y arma su tsvector para el GIN."""
    locals_ = sorted(data.chunks)
    rows = []
    for c in locals_:
        item_local, position, content = data.chunks[c]
        rows.append((item_map[item_local], data.modality, position, content))
    returned = execute_values(
        cur,
        "INSERT INTO chunks (item_id, modality, position, content) VALUES %s RETURNING id",
        rows, fetch=True,
    )
    chunk_map = {loc: r[0] for loc, r in zip(locals_, returned)}
    # tsvector para la comparativa GIN (solo donde hay texto)
    if data.modality == "text":
        cur.execute(
            "UPDATE chunks SET tsv = to_tsvector(%s, coalesce(content, '')) "
            "WHERE modality = 'text' AND tsv IS NULL",
            (ts_config,),
        )
    return chunk_map


def _insert_codebook(cur, data: IndexData) -> None:
    if data.terms is not None:                 # codebook de texto: palabras
        rows = [(data.modality, idx, term, None) for idx, term in enumerate(data.terms)]
    elif data.centroids is not None:           # codebook de imagen/audio: centroides
        rows = [(data.modality, idx, None, c) for idx, c in enumerate(data.centroids)]
    else:
        return
    execute_values(
        cur,
        "INSERT INTO codebook (modality, word_idx, term, centroid) VALUES %s",
        rows,
    )


def _insert_postings(cur, data: IndexData, chunk_map) -> None:
    rows = [
        (data.modality, word_idx, chunk_map[chunk_local], float(weight))
        for word_idx, plist in data.postings.items()
        for chunk_local, weight in plist
    ]
    if rows:
        execute_values(
            cur,
            "INSERT INTO inverted_index (modality, word_idx, chunk_id, freq) VALUES %s",
            rows, page_size=5000,
        )


def _insert_histograms(cur, data: IndexData, chunk_map) -> None:
    rows = [
        (chunk_map[chunk_local], data.modality, np.asarray(hist, dtype=np.float32))
        for chunk_local, hist in data.histograms.items()
    ]
    if rows:
        execute_values(
            cur,
            "INSERT INTO histograms (chunk_id, modality, hist) VALUES %s",
            rows, page_size=2000,
        )


def _ext_id(meta: dict) -> str | None:
    v = meta.get("external_id")
    return None if v is None else str(v)
