"""SPIMI - Single-Pass In-Memory Indexing (obligatorio para texto).

Construye el indice invertido sin que toda la coleccion quepa en RAM:

  1. Recorre los chunks una sola vez.
  2. Acumula postings en un diccionario en memoria.
  3. Cuando el bloque llena `block_size` chunks, lo VUELCA a disco ordenado
     por word_idx (un bloque = un fichero) y libera memoria.
  4. Al final hace un k-way merge de todos los bloques en un InvertedIndex.

Entrada: stream de (chunk_id, sparse_hist) donde sparse_hist = [(word_idx, weight)].
Esto escala a 100K chunks variando block_size.
"""
from __future__ import annotations
import json
import heapq
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from app.engine.index.inverted import InvertedIndex

SparseHist = list[tuple[int, float]]


class SPIMIIndexer:
    def __init__(self, block_size: int = 1000, block_dir: str | None = None) -> None:
        self.block_size = block_size
        self._dir = Path(block_dir) if block_dir else Path(tempfile.mkdtemp(prefix="spimi_"))
        self._owns_dir = block_dir is None
        self.block_paths: list[Path] = []

    def build(self, stream: Iterable[tuple[int, SparseHist]]) -> InvertedIndex:
        block: dict[int, list[tuple[int, float]]] = {}
        norms: dict[int, float] = {}
        seen = 0
        for chunk_id, sparse in stream:
            norms[chunk_id] = _norm(sparse)
            for word_idx, weight in sparse:
                block.setdefault(word_idx, []).append((chunk_id, weight))
            seen += 1
            if seen >= self.block_size:
                self._flush(block)
                block = {}
                seen = 0
        if block:
            self._flush(block)
        index = self._merge()
        index.norms = norms
        if self._owns_dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        return index

    def _flush(self, block: dict[int, list[tuple[int, float]]]) -> None:
        path = self._dir / f"block_{len(self.block_paths):05d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for word_idx in sorted(block):
                fh.write(json.dumps([word_idx, block[word_idx]]) + "\n")
        self.block_paths.append(path)

    def _merge(self) -> InvertedIndex:
        index = InvertedIndex()
        readers = [_read_block(p) for p in self.block_paths]
        # k-way merge por word_idx (bloques ya ordenados)
        merged: Iterator[tuple[int, list[tuple[int, float]]]] = heapq.merge(
            *readers, key=lambda item: item[0]
        )
        cur_word: int | None = None
        cur_postings: list[tuple[int, float]] = []
        for word_idx, postings in merged:
            if cur_word is None:
                cur_word = word_idx
            if word_idx != cur_word:
                index.postings[cur_word] = cur_postings
                cur_word, cur_postings = word_idx, []
            cur_postings.extend(postings)
        if cur_word is not None:
            index.postings[cur_word] = cur_postings
        return index


def _norm(sparse: SparseHist) -> float:
    return sum(w * w for _, w in sparse) ** 0.5


def _read_block(path: Path) -> Iterator[tuple[int, list[tuple[int, float]]]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            word_idx, postings = json.loads(line)
            yield word_idx, [(int(c), float(w)) for c, w in postings]
