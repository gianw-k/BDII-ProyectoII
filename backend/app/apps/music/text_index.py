"""App 2 - indice de texto (busqueda por letra) end-to-end.

Junta las etapas del motor para la modalidad texto:

    canciones, split(parrafos), codebook linguistico(top-k),
    quantize(histogramas), SPIMI(indice invertido),
    persistencia (codebook.json + index.json + meta.json)

`build()` corre offline (ingest). `MusicTextIndex.load()` + `search()` corren
online en el endpoint. La busqueda es a nivel chunk (estrofa) y se agrega a
nivel cancion tomando el mejor chunk.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from app.engine.split.paragraph import ParagraphSplitter
from app.engine.codebook.linguistic import LinguisticCodebook
from app.engine.index.histogram import to_sparse
from app.engine.index.inverted import InvertedIndex
from app.engine.index.spimi import SPIMIIndexer
from app.engine.search.similarity import search as cosine_search

CODEBOOK_FILE = "codebook.json"
INDEX_FILE = "index.json"
META_FILE = "meta.json"


def build(songs: list[dict], out_dir: str | Path, k: int = 256,
          block_size: int = 1000) -> "MusicTextIndex":
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splitter = ParagraphSplitter()

    items: dict[int, dict] = {}     # item_id: meta cancion
    chunks: dict[int, dict] = {}    # chunk_id: {item_id, position, content}
    cid = 0
    for item_id, song in enumerate(songs):
        items[item_id] = {
            "external_id": song.get("external_id"),
            "title": song.get("title"),
            "artist": song.get("artist"),
        }
        for pos, text in enumerate(splitter.split(song.get("lyrics", ""))):
            chunks[cid] = {"item_id": item_id, "position": pos, "content": text}
            cid += 1

    # codebook sobre toda la coleccion de chunks
    codebook = LinguisticCodebook(k=k)
    codebook.build(c["content"] for c in chunks.values())

    # stream (chunk_id, sparse_hist) hacia SPIMI
    def stream():
        for chunk_id, c in chunks.items():
            yield chunk_id, to_sparse(codebook.quantize(c["content"]))

    index = SPIMIIndexer(block_size=block_size).build(stream())

    # persistir
    (out / CODEBOOK_FILE).write_text(json.dumps(codebook.to_dict()), encoding="utf-8")
    index.save(out / INDEX_FILE)
    (out / META_FILE).write_text(
        json.dumps({
            "items": {str(i): m for i, m in items.items()},
            "chunks": {str(c): m for c, m in chunks.items()},
        }),
        encoding="utf-8",
    )
    return MusicTextIndex(codebook, index, items, chunks)


@dataclass
class MusicTextIndex:
    codebook: LinguisticCodebook
    index: InvertedIndex
    items: dict[int, dict]
    chunks: dict[int, dict]

    @classmethod
    def load(cls, in_dir: str | Path) -> "MusicTextIndex":
        d = Path(in_dir)
        codebook = LinguisticCodebook.from_dict(
            json.loads((d / CODEBOOK_FILE).read_text(encoding="utf-8"))
        )
        index = InvertedIndex.load(d / INDEX_FILE)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        items = {int(i): m for i, m in meta["items"].items()}
        chunks = {int(c): m for c, m in meta["chunks"].items()}
        return cls(codebook, index, items, chunks)

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        # busca por chunk, agrega a cancion con el mejor score
        hits = cosine_search(query, self.codebook, self.index, top_n=top_n * 5)
        best: dict[int, tuple[float, int]] = {}  # item_id: (score, chunk_id)
        for chunk_id, score in hits:
            item_id = self.chunks[chunk_id]["item_id"]
            if item_id not in best or score > best[item_id][0]:
                best[item_id] = (score, chunk_id)
        ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
        out = []
        for item_id, (score, chunk_id) in ranked:
            item = self.items[item_id]
            out.append({
                "item_id": item_id,
                "title": item["title"],
                "artist": item["artist"],
                "external_id": item["external_id"],
                "score": round(score, 4),
                "match": self.chunks[chunk_id]["content"],
            })
        return out
