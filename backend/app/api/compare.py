"""Comparativas: tu motor vs PostgreSQL nativo (GIN/GiST, pgvector).

Misma consulta por los 3 (texto) o 2 (imagen/audio) enfoques, devolviendo
resultados + latencia de cada uno. Es lo que alimenta los experimentos de la
Fase 4.

Requiere que los datos esten persistidos en Postgres (ingest --persist) y los
artefactos del motor propio en disco (para el indice invertido propio).
"""
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.comparisons import text as cmp_text
from app.comparisons import image as cmp_image
from app.comparisons import audio as cmp_audio
from app.api.music import _text_index, _acoustic_index   # reutiliza los indices ya cargados

router = APIRouter()


@lru_cache(maxsize=1)
def _conn():
    """Conexion a Postgres reutilizada entre requests."""
    from app.db.session import connect
    return connect()


@lru_cache(maxsize=1)
def _visual_index():
    from app.apps.ecommerce.visual_index import VisualIndex
    idx_dir = Path(settings.data_dir) / "index" / "ecommerce_image"
    if not (idx_dir / "index.json").exists():
        raise HTTPException(status_code=503, detail="indice visual no construido (ingest ecommerce)")
    return VisualIndex.load(idx_dir)


@router.get("/text")
def compare_text(q: str = Query(..., min_length=1), top_n: int = settings.top_n):
    """Compara la misma busqueda de letra por: indice invertido, GIN y pgvector."""
    idx = _text_index()
    conn = _conn()
    return {
        "query": q,
        "methods": [
            cmp_text.own_search(idx, q, top_n),
            cmp_text.gin_search(conn, q, top_n),
            cmp_text.pgvector_search(conn, idx.codebook, q, top_n),
        ],
    }


@router.get("/vector")
def compare_vector(external_id: str = Query(...), top_n: int = settings.top_n):
    """Compara 'productos similares' por: indice invertido propio vs pgvector."""
    idx = _visual_index()
    conn = _conn()
    return {
        "external_id": external_id,
        "methods": [
            cmp_image.own_search(idx, external_id, top_n),
            cmp_image.pgvector_search(conn, external_id, top_n),
        ],
    }


@router.get("/audio")
def compare_audio(
    filename: str = Query(..., description="Nombre de la pista GTZAN (ej: blues.00000.wav)"),
    top_n: int = settings.top_n,
):
    """Compara similitud acustica por: indice invertido propio vs pgvector.

    Mide latencia y precision de ambas tecnicas para la misma consulta,
    generando los datos que necesitan los experimentos de la Fase 4.
    """
    idx = _acoustic_index()
    conn = _conn()
    return {
        "filename": filename,
        "methods": [
            cmp_audio.own_search(idx, filename, top_n),
            cmp_audio.pgvector_search(conn, filename, top_n),
        ],
    }
