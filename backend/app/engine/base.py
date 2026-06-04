"""Interfaces de la arquitectura unificada multimodal.

Mismo contrato para texto / imagen / audio. Cada modalidad implementa
estas 4 etapas; el resto del sistema (apps, persistencia, busqueda) es
agnostico a la modalidad.

Pipeline: split, extract, codebook, quantize/histogram, search.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable
import numpy as np


class Splitter(ABC):
    """Parte contenido en chunks (parrafos / patches / ventanas)."""
    @abstractmethod
    def split(self, content: Any) -> list[Any]: ...


class Extractor(ABC):
    """Saca features de un chunk (TF-IDF / SIFT / MFCC)."""
    @abstractmethod
    def extract(self, chunk: Any) -> np.ndarray: ...


class Codebook(ABC):
    """Diccionario compartido: top-k palabras (texto) o k centroides (K-Means)."""
    k: int

    @abstractmethod
    def build(self, all_features: Iterable[np.ndarray]) -> None: ...

    @abstractmethod
    def quantize(self, features: np.ndarray) -> np.ndarray:
        """features a histograma de codewords (vector de frecuencias)."""
        ...
