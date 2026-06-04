"""Normalizacion linguistica para el codebook de texto (via nltk).

tokenizar, minusculas/sin puntuacion, quitar stopwords (ES+EN) y
stemming (SnowballStemmer ES). Es la unica puerta de entrada del texto al
motor: tanto el ingest como las queries pasan por `tokens()`.

nltk esta permitido por el enunciado. Los corpora se descargan una sola vez
de forma perezosa con `_ensure_corpora()`.
"""
from __future__ import annotations
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# tokens = secuencias de letras (incluye acentos y ñ) y digitos
_TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)

_ready = False
_STOPWORDS: frozenset[str] = frozenset()
_stemmer: SnowballStemmer | None = None


def _ensure_corpora() -> None:
    """Descarga perezosa de stopwords + init del stemmer. Idempotente."""
    global _ready, _STOPWORDS, _stemmer
    if _ready:
        return
    try:
        stopwords.words("spanish")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    _STOPWORDS = frozenset(stopwords.words("spanish")) | frozenset(
        stopwords.words("english")
    )
    _stemmer = SnowballStemmer("spanish")
    _ready = True


def tokens(text: str, *, stem: bool = True, drop_stop: bool = True) -> list[str]:
    """texto crudo a lista de tokens normalizados (codewords candidatas)."""
    if not text:
        return []
    _ensure_corpora()
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if drop_stop and raw in _STOPWORDS:
            continue
        tok = _stemmer.stem(raw) if stem else raw  # type: ignore[union-attr]
        if len(tok) < 2:
            continue
        out.append(tok)
    return out
