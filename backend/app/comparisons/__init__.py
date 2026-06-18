"""Comparativas: misma consulta por los 3 enfoques.

Tu indice invertido (motor propio) vs los nativos de Postgres: GIN/GiST para
texto y pgvector para imagen/audio. Cada funcion devuelve resultados + latencia
para alimentar los experimentos de la Fase 4.
"""
from __future__ import annotations
import time
from typing import Callable


def timed(fn: Callable):
    """Corre `fn` y devuelve (resultado, milisegundos)."""
    t0 = time.perf_counter()
    out = fn()
    return out, round((time.perf_counter() - t0) * 1000, 3)
