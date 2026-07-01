"""Histograma de color HSV para imagenes.

SIFT trabaja en escala de grises, asi que ignora el color. Para moda el color
pesa mucho (un vestido rojo y uno azul con el mismo corte dan SIFT casi iguales),
asi que sacamos aparte un histograma de color y lo fusionamos con el de visual
words.

Usamos HSV porque separa el tono (H) de la intensidad/luz (S, V), que es lo que
queremos comparar. Sacamos un histograma marginal por canal y los concatenamos:
H con mas bins (el tono es lo que mas distingue), S y V con menos.

Las fotos del dataset tienen fondo casi blanco, que no tiene tono real (su
saturacion es ~0). Para que ese fondo no llene el bin de tono, el histograma de H
solo cuenta los pixeles con algo de saturacion (la prenda), no el fondo.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from app.core.config import settings

H_BINS, S_BINS, V_BINS = 16, 8, 8
COLOR_DIM = H_BINS + S_BINS + V_BINS   # 32
_SAT_MIN = 25                          # umbral para considerar un pixel "con color"


def hsv_histogram(chunk) -> np.ndarray:
    """ruta / bytes / array de una imagen -> histograma de color (32,) L2-normalizado."""
    import cv2  # type: ignore

    bgr = _to_bgr(chunk, cv2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    sat_mask = (hsv[:, :, 1] > _SAT_MIN).astype(np.uint8)   # pixeles con color real
    h = cv2.calcHist([hsv], [0], sat_mask, [H_BINS], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], None, [S_BINS], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [V_BINS], [0, 256]).ravel()

    vec = np.concatenate([h, s, v]).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _to_bgr(chunk, cv2) -> np.ndarray:
    """Carga la imagen en color (BGR) desde ruta, bytes o array, y la achica a ~80px."""
    if isinstance(chunk, (str, Path)):
        img = cv2.imread(str(chunk), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"no se pudo leer la imagen: {chunk}")
    elif isinstance(chunk, (bytes, bytearray)):
        arr = np.frombuffer(bytes(chunk), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("bytes no decodificables como imagen")
    else:
        img = np.asarray(chunk)
        if img.ndim == 2:                                   # gris -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    max_side = settings.image_max_side
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img
