"""App 1 - indice visual (busqueda por imagen) de punta a punta.

Es el gemelo de `MusicTextIndex` pero para imagenes, y deja claro que la
arquitectura es la misma para todo: las mismas etapas (codebook -> quantize ->
SPIMI -> indice invertido -> busqueda coseno) sirven para imagen cambiando solo
dos piezas, el codebook (K-Means) y el extractor (SIFT). Todo lo demas se
reusa tal cual del lado de texto.

Lo que indexamos es el producto. Cada producto aporta un puñado de descriptores
SIFT que se resumen en un histograma Bag-of-Visual-Words.

`build()` es offline (lo llama el ingest). Le pasamos los descriptores ya
extraidos a proposito, asi este modulo no depende de OpenCV y se puede testear
con descriptores inventados. `load()` + `search()` son lo que usa el endpoint.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.engine.codebook.kmeans import KMeansCodebook
from app.engine.index.histogram import to_sparse
from app.engine.index.inverted import InvertedIndex
from app.engine.index.spimi import SPIMIIndexer
from app.engine.search.similarity import search_sparse

CODEBOOK_FILE = "codebook"      # -> codebook.npz + codebook.json
INDEX_FILE = "index.json"
META_FILE = "meta.json"


def build(
    items: Sequence[dict],
    descriptors: Sequence[np.ndarray],
    out_dir: str | Path,
    k: int = 256,
    block_size: int = 1000,
) -> "VisualIndex":
    """items[i] = metadata del producto; descriptors[i] = (n_i, dim) SIFT."""
    if len(items) != len(descriptors):
        raise ValueError("items y descriptors deben tener la misma longitud")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta: dict[int, dict] = {i: dict(m) for i, m in enumerate(items)}

    # primero el diccionario visual: K-Means sobre todos los descriptores
    codebook = KMeansCodebook(k=k)
    codebook.build(descriptors)

    # vamos pasando (item_id, histograma) a SPIMI para armar el indice invertido
    def stream():
        for item_id, desc in enumerate(descriptors):
            yield item_id, to_sparse(codebook.quantize(desc))

    index = SPIMIIndexer(block_size=block_size).build(stream())

    # guardamos todo a disco
    codebook.save(out / CODEBOOK_FILE)
    index.save(out / INDEX_FILE)
    (out / META_FILE).write_text(
        json.dumps({"items": {str(i): m for i, m in meta.items()}}),
        encoding="utf-8",
    )
    return VisualIndex(codebook, index, meta)


@dataclass
class VisualIndex:
    codebook: KMeansCodebook
    index: InvertedIndex
    items: dict[int, dict]

    @classmethod
    def load(cls, in_dir: str | Path) -> "VisualIndex":
        d = Path(in_dir)
        codebook = KMeansCodebook.load(d / CODEBOOK_FILE)
        index = InvertedIndex.load(d / INDEX_FILE)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        items = {int(i): m for i, m in meta["items"].items()}
        return cls(codebook, index, items)

    def search(self, query_descriptors: np.ndarray, top_n: int = 10) -> list[dict]:
        """Descriptores de la foto que subio el usuario -> top-N productos parecidos."""
        q_sparse = to_sparse(self.codebook.quantize(query_descriptors))
        hits = search_sparse(q_sparse, self.index, top_n=top_n)
        out = []
        for item_id, score in hits:
            item = self.items.get(item_id, {})
            out.append({
                "item_id": item_id,
                "score": round(float(score), 4),
                **item,
            })
        return out
