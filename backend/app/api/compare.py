"""Comparativas: tu motor vs PostgreSQL nativo (GIN/GiST, pgvector).

Endpoints para correr la misma consulta por los 3 enfoques y devolver
resultados + metricas (latencia, etc.) para los experimentos de Fase 4.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/text")
def compare_text(q: str):
    # TODO
    raise NotImplementedError


@router.get("/vector")
def compare_vector(item_id: int):
    # TODO
    raise NotImplementedError
