"""Endpoint del ParserSQL: recibe una consulta tipo SQL y la ejecuta.

Es la cara unificada del motor. El usuario manda una sola cadena
(`SELECT ... FROM ... WHERE ...`), aqui se parsea a un `ParsedQuery` y se
despacha al indice de la modalidad correcta (texto / audio / imagen), todos
reusando el mismo nucleo de busqueda por coseno sobre indice invertido.

Las multimedia (audio/imagen) se consultan "por similitud a" un item ya
indexado, identificado por su filename; el texto se consulta por contenido.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.engine.query.parser import parse, QueryParseError, ParsedQuery
# Reusamos los cargadores perezosos de indices ya definidos en cada app.
from app.api.music import _text_index, _acoustic_index
from app.api.ecommerce import _visual_index

router = APIRouter()


class QueryRequest(BaseModel):
    sql: str


@router.post("/")
async def run_query(req: QueryRequest):
    """Parsea y ejecuta una consulta SQL del mini-lenguaje multimodal."""
    try:
        pq = parse(req.sql)
    except QueryParseError as e:
        # 400: la culpa es de la sintaxis del usuario, no del servidor.
        raise HTTPException(status_code=400, detail=f"Error de sintaxis: {e}")

    if pq.modality == "text":
        results = _run_text(pq)
    elif pq.modality == "audio":
        results = _run_audio(pq)
    elif pq.modality == "image":
        results = _run_image(pq)
    else:  # imposible por construccion del parser, pero por si acaso
        raise HTTPException(status_code=400, detail=f"modalidad no soportada: {pq.modality}")

    return {
        "parsed": pq.to_dict(),
        "count": len(results),
        "results": _project(results, pq.fields),
    }


# ─────────────────────────── ejecutores por modalidad ─────────────────────────

def _run_text(pq: ParsedQuery) -> list[dict]:
    return _text_index().search(pq.value, top_n=pq.limit)


def _run_audio(pq: ParsedQuery) -> list[dict]:
    idx = _acoustic_index()
    hist = idx.get_track_features(pq.value)
    if hist is None:
        raise HTTPException(
            status_code=404,
            detail=f"pista '{pq.value}' no encontrada en el indice acustico",
        )
    return idx.search_from_hist(hist, top_n=pq.limit, exclude_filename=pq.value)


def _run_image(pq: ParsedQuery) -> list[dict]:
    idx = _visual_index()
    if getattr(idx, "features", None) is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "la consulta de imagen por filename no esta disponible: el indice "
                "actual (SIFT + color) no guarda descriptores por item"
            ),
        )
    # localizar el item por su filename
    item_id = next(
        (i for i, m in idx.items.items() if m.get("filename") == pq.value),
        None,
    )
    if item_id is None or item_id not in idx.features:
        raise HTTPException(
            status_code=404,
            detail=f"producto '{pq.value}' no encontrado en el indice visual",
        )
    desc, xy = idx.features[item_id]
    results = idx.search(desc, top_n=pq.limit + 1, query_keypoints=xy)
    # excluir el propio item consultado del resultado
    return [r for r in results if r.get("item_id") != item_id][: pq.limit]


# ─────────────────────────── proyeccion de columnas ───────────────────────────

def _project(results: list[dict], fields: list[str]) -> list[dict]:
    """Aplica el SELECT <campos>: si no es '*', recorta cada fila a esos campos.

    `score` siempre se conserva: es el ranking de la recuperacion.
    """
    if fields == ["*"]:
        return results
    keep = set(fields) | {"score"}
    return [{k: v for k, v in r.items() if k in keep} for r in results]
