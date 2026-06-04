"""App 1 - Busqueda Visual E-commerce (modalidad: imagen).

Usuario sube foto y retorna top-N productos similares (visual words + busqueda).
"""
from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/search")
async def search_by_image(file: UploadFile = File(...)):
    # TODO
    raise NotImplementedError
