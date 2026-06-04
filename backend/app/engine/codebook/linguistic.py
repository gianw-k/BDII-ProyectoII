"""Codebook linguistico para texto.

Construccion (sobre toda la coleccion):
    tokens, contar, tomar las k palabras (stems) mas frecuentes.

Esas k codewords forman el vocabulario. `quantize` convierte un chunk en su
histograma: vector de longitud k con la frecuencia (normalizada L2) de cada
codeword en el chunk. Las codewords fuera del top-k se ignoran (cuantizacion
lossy, igual que K-Means descarta detalle en imagen/audio).
"""
from __future__ import annotations
from collections import Counter
from typing import Iterable

import numpy as np

from app.engine.base import Codebook
from app.engine.codebook.text_norm import tokens


class LinguisticCodebook(Codebook):
    def __init__(self, k: int = 256) -> None:
        self.k = k
        self.terms: list[str] = []          # word_idx: term
        self.index: dict[str, int] = {}     # term: word_idx

    def build(self, all_texts: Iterable[str]) -> None:
        """all_texts = textos crudos de todos los chunks de la coleccion."""
        freq: Counter[str] = Counter()
        for text in all_texts:
            freq.update(tokens(text))
        self.terms = [t for t, _ in freq.most_common(self.k)]
        self.index = {t: i for i, t in enumerate(self.terms)}

    def quantize(self, content: str | np.ndarray) -> np.ndarray:
        """texto del chunk a histograma L2-normalizado (len = k efectivo)."""
        if not self.terms:
            raise RuntimeError("codebook vacio: llama build() primero")
        hist = np.zeros(len(self.terms), dtype=np.float32)
        for tok in tokens(content):  # type: ignore[arg-type]
            j = self.index.get(tok)
            if j is not None:
                hist[j] += 1.0
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def to_dict(self) -> dict:
        return {"k": self.k, "terms": self.terms}

    @classmethod
    def from_dict(cls, d: dict) -> "LinguisticCodebook":
        cb = cls(k=d["k"])
        cb.terms = list(d["terms"])
        cb.index = {t: i for i, t in enumerate(cb.terms)}
        return cb
