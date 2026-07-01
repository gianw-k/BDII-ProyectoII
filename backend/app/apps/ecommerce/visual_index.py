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
from app.engine.index.histogram import fuse_color, to_sparse
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
    colors: Sequence[np.ndarray] | None = None,
    color_weight: float = 0.5,
) -> "VisualIndex":
    """items[i] = metadata; descriptors[i] = (n_i, dim) SIFT; colors[i] = hist HSV.

    Si pasas `colors`, cada histograma BoVW se fusiona con el de color (ver
    fuse_color): las bins de color quedan como codewords extra despues de las
    visual words. Sin `colors` (o color_weight=0) queda el comportamiento de
    antes, solo SIFT.
    """
    if len(items) != len(descriptors):
        raise ValueError("items y descriptors deben tener la misma longitud")
    if colors is not None and len(colors) != len(descriptors):
        raise ValueError("colors debe tener la misma longitud que descriptors")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta: dict[int, dict] = {i: dict(m) for i, m in enumerate(items)}
    use_color = colors is not None and color_weight > 0
    color_dim = int(colors[0].shape[0]) if use_color else 0

    # primero el diccionario visual: K-Means sobre todos los descriptores
    codebook = KMeansCodebook(k=k)
    codebook.build(descriptors)

    # vamos pasando (item_id, histograma) a SPIMI para armar el indice invertido
    def stream():
        for item_id, desc in enumerate(descriptors):
            hist = codebook.quantize(desc)
            if use_color:
                hist = fuse_color(hist, colors[item_id], color_weight)
            yield item_id, to_sparse(hist)

    index = SPIMIIndexer(block_size=block_size).build(stream())

    # guardamos todo a disco
    codebook.save(out / CODEBOOK_FILE)
    index.save(out / INDEX_FILE)
    (out / META_FILE).write_text(
        json.dumps({
            "items": {str(i): m for i, m in meta.items()},
            "color_weight": color_weight if use_color else 0.0,
            "color_dim": color_dim,
        }),
        encoding="utf-8",
    )
    return VisualIndex(codebook, index, meta,
                       color_weight=color_weight if use_color else 0.0,
                       color_dim=color_dim)


@dataclass
class VisualIndex:
    codebook: KMeansCodebook
    index: InvertedIndex
    items: dict[int, dict]
    color_weight: float = 0.0
    color_dim: int = 0

    @property
    def hist_dim(self) -> int:
        """Largo del histograma fusionado: visual words + bins de color."""
        return int(self.codebook.centroids.shape[0]) + self.color_dim

    @classmethod
    def load(cls, in_dir: str | Path) -> "VisualIndex":
        d = Path(in_dir)
        codebook = KMeansCodebook.load(d / CODEBOOK_FILE)
        index = InvertedIndex.load(d / INDEX_FILE)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        items = {int(i): m for i, m in meta["items"].items()}
        return cls(codebook, index, items,
                   color_weight=float(meta.get("color_weight", 0.0)),
                   color_dim=int(meta.get("color_dim", 0)))

    def search(self, query_descriptors: np.ndarray, top_n: int = 10,
               query_color: np.ndarray | None = None) -> list[dict]:
        """Descriptores (+ color) de la foto subida -> top-N productos parecidos."""
        hist = self.codebook.quantize(query_descriptors)
        if self.color_weight > 0 and query_color is not None:
            hist = fuse_color(hist, query_color, self.color_weight)
        q_sparse = to_sparse(hist)
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
