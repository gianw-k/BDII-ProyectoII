"""Extractor SIFT para imagenes (descriptores locales de 128 dimensiones).

SIFT busca puntos "interesantes" de la imagen (esquinas, texturas) que aguantan
cambios de escala y rotacion, y describe cada uno con un vector de 128 numeros.
Esos son los descriptores que despues alimentan al codebook K-Means. Una imagen
da N descriptores, y N varia segun lo que tenga la foto.

OpenCV se importa solo cuando hace falta, asi el resto del motor arranca aunque
no este instalado (los tests, por ejemplo, usan descriptores inventados y nunca
tocan OpenCV). Si de verdad intentas extraer sin tenerlo, ahi si revienta.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from app.engine.base import Extractor

_DIM = 128


class SIFTExtractor(Extractor):
    def __init__(self, max_features: int = 0) -> None:
        # 0 = sin tope, OpenCV se queda con todos los keypoints que encuentre
        self.max_features = max_features
        self._sift = None

    def _detector(self):
        if self._sift is None:
            try:
                import cv2  # type: ignore
            except ImportError as e:  # pragma: no cover - depende del entorno
                raise RuntimeError(
                    "OpenCV no disponible. Instala 'opencv-contrib-python' "
                    "para extraer SIFT."
                ) from e
            self._sift = cv2.SIFT_create(nfeatures=self.max_features)
        return self._sift

    def extract(self, chunk) -> np.ndarray:
        """chunk = ruta, bytes o array de la imagen -> sus descriptores (n, 128)."""
        import cv2  # type: ignore

        img = self._to_gray(chunk, cv2)
        _, desc = self._detector().detectAndCompute(img, None)
        if desc is None:
            return np.empty((0, _DIM), dtype=np.float32)
        return desc.astype(np.float32)

    @staticmethod
    def _to_gray(chunk, cv2) -> np.ndarray:
        if isinstance(chunk, (str, Path)):
            img = cv2.imread(str(chunk), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"no se pudo leer la imagen: {chunk}")
            return img
        if isinstance(chunk, (bytes, bytearray)):
            arr = np.frombuffer(bytes(chunk), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("bytes no decodificables como imagen")
            return img
        arr = np.asarray(chunk)
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr
