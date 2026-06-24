"""Busqueda Musical (modalidad: audio + texto).

Dos modos de consulta:
  - por letra (full-text sobre letras), implementado con el motor propio.
  - por similitud acustica:
      * subiendo un archivo .wav/.mp3 
      * indicando el filename de una pista del dataset (busqueda por demo rapida)
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from app.core.config import settings
from app.apps.music.text_index import MusicTextIndex
from app.apps.music.acoustic_index import AcousticIndex

router = APIRouter()


# ─────────────────────────────── indices (carga perezosa) ────────────────────

@lru_cache(maxsize=1)
def _text_index() -> MusicTextIndex:
    """Carga perezosa del indice de letras (artefactos del ingest)."""
    idx_dir = Path(settings.data_dir) / "index" / "music_text"
    if not (idx_dir / "index.json").exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"indice de letras no construido. Corre: "
                f"el script de ingesta indicado en el README.md"
            ),
        )
    return MusicTextIndex.load(idx_dir)


@lru_cache(maxsize=1)
def _acoustic_index() -> AcousticIndex:
    """Carga perezosa del indice acustico (artefactos del ingest)."""
    candidates = [
        Path(settings.data_dir) / "index" / "music_audio",   # /data/index/music_audio
        Path("/app/data/index/music_audio"),                  # si --out fue relativo dentro del container
        Path("data/index/music_audio"),                       # ultimo recurso (dir de trabajo)
    ]
    idx_dir = next((p for p in candidates if (p / "index.json").exists()), None)
    if idx_dir is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Indice acustico no construido. Corre desde el host:\n"
                "docker compose exec backend python -m pipelines.ingest "
                "--app music --modality audio "
                "--data /data/raw/music/features_30_sec.csv "
                "--out /data/index/music_audio --k 128"
            ),
        )
    return AcousticIndex.load(idx_dir)


# ─────────────────────────────── endpoints ───────────────────────────────────

@router.get("/search/lyrics")
async def search_by_lyrics(
    q: str = Query(..., min_length=1),
    top_n: int = settings.top_n,
):
    """Busca canciones por letra via indice invertido + coseno (motor propio)."""
    results = _text_index().search(q, top_n=top_n)
    return {"query": q, "count": len(results), "results": results}


@router.post("/search/audio")
async def search_by_audio(
    file: UploadFile = File(...),
    top_n: int = Query(default=10, ge=1, le=50),
):
    """Busca pistas musicalmente similares subiendo un archivo .wav o .mp3.

    El servidor extrae los MFCC del audio con librosa, cuantiza contra el
    codebook acustico y devuelve las top-N pistas mas similares por coseno.
    """

    from app.engine.extractor.mfcc import MFCCExtractor

    # Leer bytes del archivo subido
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Archivo de audio vacio.")

    # Extraer features MFCC en tiempo real
    try:
        extractor = MFCCExtractor()
        features = extractor.extract(audio_bytes)   # (1, 40)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo procesar el audio: {e}",
        )

    idx = _acoustic_index()
    results = idx.search(features, top_n=top_n)
    return {
        "mode": "audio_upload",
        "filename": file.filename,
        "count": len(results),
        "results": results,
    }


@router.get("/search/audio/by_filename")
async def search_by_filename(
    filename: str = Query(..., description="Nombre de la pista del dataset (ej: blues.00000.wav)"),
    top_n: int = Query(default=10, ge=1, le=50),
):
    """Busca pistas similares a una pista conocida del dataset (demo rapida).

    Usa el histograma ya almacenado en el indice, sin necesitar el archivo .wav.
    Util para demos y para las comparativas de rendimiento.
    """

    idx = _acoustic_index()
    hist = idx.get_track_features(filename)
    if hist is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pista '{filename}' no encontrada en el indice acustico.",
        )

    results = idx.search_from_hist(hist, top_n=top_n, exclude_filename=filename)
    return {
        "mode": "by_filename",
        "query_filename": filename,
        "count": len(results),
        "results": results,
    }


@router.get("/audio/tracks")
async def list_tracks(
    genre: str | None = Query(default=None, description="Filtrar por genero (ej: blues, rock)"),
):
    """Devuelve hasta 10 pistas aleatorias del indice acustico (por genero si se indica).

    Retorna lista vacia con mensaje si el indice no esta construido todavia,
    en lugar de lanzar un error 503 que romperia el selector del frontend.
    """
    import random
    try:
        idx = _acoustic_index()
    except HTTPException:
        return {"total": 0, "tracks": [], "message": "indice acustico no disponible"}

    tracks = list(idx.tracks.values())
    if genre:
        tracks = [t for t in tracks if t.get("label", "").lower() == genre.lower()]

    # Devolver solo 10 aleatorios para no sobrecargar el frontend
    sample = random.sample(tracks, min(10, len(tracks)))
    return {
        "total": len(tracks),
        "tracks": sample,
    }
