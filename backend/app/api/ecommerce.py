"""App 1 - Busqueda Visual E-commerce (modalidad: imagen).

Usuario sube foto y retorna top-N productos similares (visual words + busqueda).
Reutiliza el mismo motor que texto: codebook (K-Means) -> histograma BoVW ->
indice invertido -> busqueda coseno.
"""
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, File, UploadFile, Query

from app.core.config import settings
from app.apps.ecommerce.visual_index import VisualIndex
from app.engine.extractor.sift import SIFTExtractor

router = APIRouter()


@lru_cache(maxsize=1)
def _visual_index() -> VisualIndex:
    """Carga perezosa del indice visual (artefactos del ingest)."""
    idx_dir = Path(settings.data_dir) / "index" / "ecommerce_image"
    if not (idx_dir / "index.json").exists():
        raise HTTPException(
            status_code=503,
            detail=f"indice no construido. Corre: python -m pipelines.ingest "
                   f"--app ecommerce --modality image --data <dir_imagenes> --out {idx_dir}",
        )
    return VisualIndex.load(idx_dir)


@lru_cache(maxsize=1)
def _extractor() -> SIFTExtractor:
    return SIFTExtractor()


@router.post("/search")
async def search_by_image(file: UploadFile = File(...), top_n: int = Query(settings.top_n)):
    """Sube una foto y devuelve top-N productos visualmente similares."""
    data = await file.read()
    try:
        descriptors = _extractor().extract(data)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    if descriptors.shape[0] == 0:
        raise HTTPException(status_code=422, detail="no se detectaron keypoints SIFT en la imagen")
    results = _visual_index().search(descriptors, top_n=top_n)
    return {"count": len(results), "results": results}
