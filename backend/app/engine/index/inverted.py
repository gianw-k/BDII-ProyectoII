"""Indice invertido propio: word_idx a postings [(chunk_id, weight)].

Estructura final que consulta la busqueda online. Se construye con SPIMI
(ver spimi.py) y se persiste a JSON. Como los pesos vienen L2-normalizados
desde el codebook, el coseno query-doc es el producto punto de los postings
sobre las codewords compartidas.
"""
from __future__ import annotations
import json
from pathlib import Path


class InvertedIndex:
    def __init__(self) -> None:
        # word_idx: list[(chunk_id, weight)]
        self.postings: dict[int, list[tuple[int, float]]] = {}
        # chunk_id: norma L2 del histograma (1.0 si ya normalizado)
        self.norms: dict[int, float] = {}

    def add_posting(self, word_idx: int, chunk_id: int, weight: float) -> None:
        self.postings.setdefault(word_idx, []).append((chunk_id, weight))

    def get(self, word_idx: int) -> list[tuple[int, float]]:
        return self.postings.get(word_idx, [])

    @property
    def num_terms(self) -> int:
        return len(self.postings)

    @property
    def num_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "postings": {str(w): p for w, p in self.postings.items()},
            "norms": {str(c): n for c, n in self.norms.items()},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls()
        idx.postings = {
            int(w): [(int(c), float(wt)) for c, wt in p]
            for w, p in data["postings"].items()
        }
        idx.norms = {int(c): float(n) for c, n in data["norms"].items()}
        return idx
