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


import math

class LinguisticCodebook(Codebook):
    def __init__(self, k: int = 4096) -> None:
        self.k = k
        self.terms: list[str] = []          # word_idx: term
        self.index: dict[str, int] = {}     # term: word_idx
        self.idf: np.ndarray = np.array([]) # word_idx: idf score

    def build(self, all_texts: Iterable[str]) -> None:
        """all_texts = textos crudos de todos los chunks de la coleccion."""
        # Necesitamos iterar una vez todo el texto para sacar frecuencias totales y document_frequency (DF)
        freq: Counter[str] = Counter()
        df: Counter[str] = Counter()
        N = 0
        for text in all_texts:
            N += 1
            t = list(tokens(text))
            freq.update(t)
            df.update(set(t))  # cuenta en cuantos documentos unicos aparece el termino

        # Conservar el top-k más frecuente (Codebook original de tus compañeros)
        self.terms = [t for t, _ in freq.most_common(self.k)]
        self.index = {t: i for i, t in enumerate(self.terms)}

        # Computar IDF: log(N/df)
        self.idf = np.zeros(len(self.terms), dtype=np.float32)
        for i, term in enumerate(self.terms):
            self.idf[i] = math.log10(N / (df[term] or 1))

    def quantize(self, content: str | np.ndarray) -> np.ndarray:
        """texto del chunk a histograma L2-normalizado TF-IDF (len = k efectivo)."""
        if not self.terms or self.idf.size == 0:
            raise RuntimeError("codebook vacio: llama build() primero")
        hist = np.zeros(len(self.terms), dtype=np.float32)
        for tok in tokens(content):  # type: ignore[arg-type]
            j = self.index.get(tok)
            if j is not None:
                hist[j] += 1.0  # TF crudo (conteo)

        # TF sublineal log(1+tf): que un coro repita 40 veces una palabra no
        # debe pesar 40 veces mas que decirla una vez. Despues, el peso IDF.
        hist = np.log1p(hist) * self.idf

        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def to_dict(self) -> dict:
        return {"k": self.k, "terms": self.terms, "idf": self.idf.tolist()}
    
    @classmethod
    def from_dict(cls, data: dict) -> "LinguisticCodebook":
        cb = cls(k=data.get("k", 256))
        cb.terms = data.get("terms", [])
        cb.index = {t: i for i, t in enumerate(cb.terms)}
        cb.idf = np.array(data.get("idf", []), dtype=np.float32)
        return cb

    # Eliminamos el from_dict duplicado que sobreescribia al de TF-IDF
