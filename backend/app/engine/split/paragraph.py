"""Split de texto en parrafos (chunks).

Para letras de canciones, cada parrafo ~ estrofa. Si el texto no trae
saltos dobles, cae a un solo chunk con todo el contenido.
"""
from __future__ import annotations
import re

from app.engine.base import Splitter

# 2+ saltos de linea = separador de parrafo
_PARA_RE = re.compile(r"\n\s*\n+")


class ParagraphSplitter(Splitter):
    def __init__(self, min_chars: int = 1) -> None:
        self.min_chars = min_chars

    def split(self, content: str) -> list[str]:
        # robusto a None / NaN / numeros (algunos datasets traen letras vacias)
        if not isinstance(content, str):
            return []
        parts = _PARA_RE.split(content.strip())
        chunks = [p.strip() for p in parts if len(p.strip()) >= self.min_chars]
        # texto sin dobles saltos: un unico chunk
        if not chunks and content.strip():
            return [content.strip()]
        return chunks
