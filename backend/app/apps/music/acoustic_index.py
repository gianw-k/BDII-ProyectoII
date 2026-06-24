"""App 2b - índice acústico (búsqueda por similitud de audio) de punta a punta.

Gemelo exacto de `visual_index.py` pero para audio MFCC.  Las etapas son
idénticas a la de imagen: solo cambian el extractor (MFCCExtractor en lugar
de SIFTExtractor) y la fuente de datos (CSV de features GTZAN en lugar de
imágenes).

Pipeline completo (igual que el enunciado):
    Split (ventanas/filas CSV)
    → Extractor (MFCC → vector 40-dim)
    → KMeansCodebook.build  (acoustic words)
    → quantize → histograma Bag-of-Acoustic-Words (k,)
    → SPIMIIndexer  (índice invertido en disco)
    → búsqueda coseno vía search_sparse (accumulator pattern)

`build()` corre offline (pipeline de ingest).
`AcousticIndex.load()` + `search()` se usan en el endpoint online.

La búsqueda online acepta:
  - query_vec  (np.ndarray 1-D de features ya extraídas, p.ej. desde librosa)
  - Los descriptores entran directamente a codebook.quantize → histograma disperso.
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

CODEBOOK_FILE = "codebook"      # → codebook.npz + codebook.json
INDEX_FILE    = "index.json"
META_FILE     = "meta.json"


def build(
    tracks: Sequence[dict],
    descriptors: Sequence[np.ndarray],
    out_dir: str | Path,
    k: int = 128,
    block_size: int = 1000,
) -> "AcousticIndex":
    """Construye el índice acústico y lo persiste en `out_dir`.

    Parameters
    ----------
    tracks : lista de dicts con metadata de cada pista
        Campos esperados: filename, label (género), y cualquier otro extra.
    descriptors : lista de arrays (n_windows, n_features) por pista.
        Cada pista puede tener varias ventanas (3s) o una sola (30s).
    out_dir : directorio donde se guardan los artefactos.
    k : número de acoustic words (clusters K-Means).
    """
    if len(tracks) != len(descriptors):
        raise ValueError("tracks y descriptors deben tener la misma longitud")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta: dict[int, dict] = {i: dict(m) for i, m in enumerate(tracks)}

    # ── 1. Codebook: K-Means sobre TODOS los vectores MFCC de la coleccion ──
    codebook = KMeansCodebook(k=k)
    codebook.build(descriptors)

    # ── 2. Índice invertido: histograma de acoustic words por pista ──
    def stream():
        for track_id, desc in enumerate(descriptors):
            # desc puede ser (n_windows, dim) o (dim,) → normalizamos
            arr = np.atleast_2d(np.asarray(desc, dtype=np.float32))
            hist = codebook.quantize(arr)
            yield track_id, to_sparse(hist)

    index = SPIMIIndexer(block_size=block_size).build(stream())

    # ── 3. Persistencia ──
    codebook.save(out / CODEBOOK_FILE)
    index.save(out / INDEX_FILE)
    (out / META_FILE).write_text(
        json.dumps({"tracks": {str(i): m for i, m in meta.items()}}),
        encoding="utf-8",
    )
    return AcousticIndex(codebook, index, meta)


@dataclass
class AcousticIndex:
    codebook: KMeansCodebook
    index: InvertedIndex
    tracks: dict[int, dict]   # track_id → {filename, label, ...}

    @classmethod
    def load(cls, in_dir: str | Path) -> "AcousticIndex":
        d = Path(in_dir)
        codebook = KMeansCodebook.load(d / CODEBOOK_FILE)
        index    = InvertedIndex.load(d / INDEX_FILE)
        raw_meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        tracks   = {int(i): m for i, m in raw_meta["tracks"].items()}
        return cls(codebook, index, tracks)

    def search(
        self,
        query_features: np.ndarray,
        top_n: int = 10,
        exclude_filename: str | None = None,
    ) -> list[dict]:
        """Busca pistas acusticamente similares a partir de features MFCC CRUDAS.

        Parameters
        ----------
        query_features : array 1-D (n_features=40,) o 2-D (n_windows, n_features).
            Features MFCC sin cuantizar (tal como salen del MFCCExtractor).
        top_n : cuantos resultados devolver.
        exclude_filename : excluye esta pista del resultado (evita auto-match).
        """
        arr = np.atleast_2d(np.asarray(query_features, dtype=np.float32))
        q_sparse = to_sparse(self.codebook.quantize(arr))
        return self._hits_to_results(q_sparse, top_n, exclude_filename)

    def search_from_hist(
        self,
        hist: np.ndarray,
        top_n: int = 10,
        exclude_filename: str | None = None,
    ) -> list[dict]:
        """Busca pistas similares a partir de un histograma ya cuantizado (k-dim).

        Util cuando el histograma ya esta almacenado en el indice (demo rapida
        por filename) y no se quiere volver a pasar por codebook.quantize.
        """
        q_sparse = to_sparse(np.asarray(hist, dtype=np.float32))
        return self._hits_to_results(q_sparse, top_n, exclude_filename)

    def _hits_to_results(
        self,
        q_sparse: list[tuple[int, float]],
        top_n: int,
        exclude_filename: str | None,
    ) -> list[dict]:
        """Ejecuta la busqueda coseno y formatea los resultados."""
        hits = search_sparse(q_sparse, self.index, top_n=top_n + 5)
        out = []
        for track_id, score in hits:
            meta = self.tracks.get(track_id, {})
            if exclude_filename and meta.get("filename") == exclude_filename:
                continue
            out.append({
                "track_id": track_id,
                "score": round(float(score), 4),
                **meta,
            })
            if len(out) >= top_n:
                break
        return out

    def get_track_features(self, filename: str) -> np.ndarray | None:
        """Devuelve el histograma disperso de una pista por su nombre de archivo.

        Util para búsqueda por pista conocida sin necesidad de re-extraer features.
        """
        for track_id, meta in self.tracks.items():
            if meta.get("filename") == filename:
                # reconstruir histograma desde el índice invertido
                hist = np.zeros(self.codebook.centroids.shape[0], dtype=np.float32)
                for word_idx, postings in self.index.postings.items():
                    for cid, weight in postings:
                        if cid == track_id:
                            hist[word_idx] = weight
                return hist
        return None
