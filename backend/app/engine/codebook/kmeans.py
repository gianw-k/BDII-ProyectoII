"""Codebook K-Means para imagen y audio.

La idea es la misma que en texto, pero sin palabras: en lugar de "agarrar las k
palabras mas comunes", juntamos todos los descriptores de la coleccion (los SIFT
de las imagenes, los MFCC de los audios) y los agrupamos en k clusters. Cada
centroide termina siendo una "palabra visual" o "palabra acustica".

    build(todos los descriptores) -> k centroides (osea, el diccionario)
    quantize(descriptores de un item) -> su histograma Bag-of-Words (largo k)

El histograma sale L2-normalizado igual que en texto, asi que de aca para
adelante el motor (SPIMI, indice invertido, busqueda) no nota la diferencia.
Ojo: esto pierde info (cada descriptor se redondea a su centroide mas cercano),
que es justo la "compresion lossy" de la que habla el enunciado.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from app.engine.base import Codebook


class KMeansCodebook(Codebook):
    def __init__(self, k: int = 256, *, seed: int = 0, batch_size: int = 4096) -> None:
        self.k = k
        self.seed = seed
        self.batch_size = batch_size
        self.centroids: np.ndarray | None = None   # (k_eff, dim)
        self.dim: int | None = None

    # ------------------------------------------------------------------ build
    def build(self, all_features: Iterable[np.ndarray]) -> None:
        """all_features = los descriptores de todos los items.

        Cada item llega como un array (n_i, dim) con sus SIFT/MFCC (o un solo
        vector). Los apilamos todos y corremos MiniBatchKMeans, que aguanta
        millones de descriptores sin tener que meterlos todos en RAM de una.
        """
        from sklearn.cluster import MiniBatchKMeans

        stacked = _stack(all_features)
        if stacked.size == 0:
            raise ValueError("no hay descriptores para construir el codebook")
        self.dim = int(stacked.shape[1])
        # no tiene sentido pedir mas clusters que muestras
        k_eff = int(min(self.k, stacked.shape[0]))
        km = MiniBatchKMeans(
            n_clusters=k_eff,
            random_state=self.seed,
            batch_size=min(self.batch_size, stacked.shape[0]),
            n_init="auto",
        )
        km.fit(stacked)
        self.centroids = km.cluster_centers_.astype(np.float32)

    # --------------------------------------------------------------- quantize
    def quantize(self, features: np.ndarray) -> np.ndarray:
        """descriptores de un item (n, dim) -> su histograma BoW normalizado (k,)."""
        if self.centroids is None:
            raise RuntimeError("codebook vacio: corre build() primero")
        k = self.centroids.shape[0]
        hist = np.zeros(k, dtype=np.float32)

        feats = np.atleast_2d(np.asarray(features, dtype=np.float32))
        if feats.size == 0:
            return hist  # item sin descriptores: histograma en cero y listo

        labels = self._assign(feats)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        hist[: counts.shape[0]] = counts
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def _assign(self, feats: np.ndarray) -> np.ndarray:
        """A que centroide cae cada descriptor (el mas cercano en L2)."""
        # truco: en vez de calcular la distancia completa, como ||x||^2 es igual
        # para todos los centroides, basta con maximizar  x.c - ||c||^2/2
        c = self.centroids  # type: ignore[assignment]
        cn = (c * c).sum(axis=1) * 0.5
        sims = feats @ c.T - cn  # (n, k)
        return np.argmax(sims, axis=1).astype(np.int64)

    # ------------------------------------------------------------ persistencia
    def save(self, path: str | Path) -> None:
        """Guarda los centroides (.npz) + la meta (.json). `path` = nombre base."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base.with_suffix(".npz"), centroids=self.centroids)
        base.with_suffix(".json").write_text(
            json.dumps({"k": self.k, "dim": self.dim, "seed": self.seed}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "KMeansCodebook":
        base = Path(path)
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        cb = cls(k=meta["k"], seed=meta.get("seed", 0))
        cb.dim = meta.get("dim")
        cb.centroids = np.load(base.with_suffix(".npz"))["centroids"].astype(np.float32)
        return cb

    # to_dict / from_dict: comodo para tests y codebooks chicos (todo en un JSON)
    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "dim": self.dim,
            "seed": self.seed,
            "centroids": [] if self.centroids is None else self.centroids.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KMeansCodebook":
        cb = cls(k=d["k"], seed=d.get("seed", 0))
        cb.dim = d.get("dim")
        cent = d.get("centroids") or []
        cb.centroids = np.asarray(cent, dtype=np.float32) if cent else None
        return cb


def _stack(all_features: Iterable[np.ndarray]) -> np.ndarray:
    """Junta los descriptores de todos los items en un unico array (N, dim)."""
    rows: list[np.ndarray] = []
    for f in all_features:
        arr = np.atleast_2d(np.asarray(f, dtype=np.float32))
        if arr.size:
            rows.append(arr)
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(rows)
