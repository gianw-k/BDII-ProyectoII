"""App 2 - Busqueda Musical Inteligente (modalidad: audio + texto).

Dos modos de consulta:
  - por letra (full-text sobre letras), implementado con el motor propio.
  - por similitud acustica (audio, MFCC, acoustic words).
"""
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from app.core.config import settings
from app.apps.music.text_index import MusicTextIndex

router = APIRouter()


@lru_cache(maxsize=1)
def _text_index() -> MusicTextIndex:
    """Carga perezosa del indice de letras (artefactos del ingest)."""
    idx_dir = Path(settings.data_dir) / "index" / "music_text"
    if not (idx_dir / "index.json").exists():
        raise HTTPException(
            status_code=503,
            detail=f"indice no construido. Corre: python -m pipelines.ingest "
                   f"--app music --data <lyrics.json> --out {idx_dir}",
        )
    return MusicTextIndex.load(idx_dir)


@router.get("/search/lyrics")
async def search_by_lyrics(q: str = Query(..., min_length=1), top_n: int = settings.top_n):
    """Busca canciones por letra via indice invertido + coseno (motor propio)."""
    results = _text_index().search(q, top_n=top_n)
    return {"query": q, "count": len(results), "results": results}


@router.post("/search/audio")
async def search_by_audio(file: UploadFile = File(...)):
    # TODO
    raise NotImplementedError
