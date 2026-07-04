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
from app.core.config import settings

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

    def color_histogram(self, chunk) -> np.ndarray:
        """Histograma de color HSV de la imagen (lo que SIFT no ve). Ver color.py."""
        from app.engine.extractor.color import hsv_histogram
        return hsv_histogram(chunk)

    @staticmethod
    def _to_gray(chunk, cv2) -> np.ndarray:
        if isinstance(chunk, (str, Path)):
            img = cv2.imread(str(chunk), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"no se pudo leer la imagen: {chunk}")
        elif isinstance(chunk, (bytes, bytearray)):
            arr = np.frombuffer(bytes(chunk), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("bytes no decodificables como imagen")
        else:
            arr = np.asarray(chunk)
            if arr.ndim == 3:
                img = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            else:
                img = arr

        target_side = settings.image_max_side
        h, w = img.shape[:2]
        
        # Estandarizar estrictamente la resolución para que el vocabulario visual (codebook) 
        # compare peras con peras. Las consultas y la ingesta deben extraer al mismo tamaño.
        if max(h, w) > target_side:
            scale = target_side / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif max(h, w) < target_side:
            scale = target_side / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return img
