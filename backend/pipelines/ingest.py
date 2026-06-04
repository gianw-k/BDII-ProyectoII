"""Pipeline OFFLINE de indexacion (no corre en cada request).

Recorre el dataset y construye todo lo que la busqueda online consultara:

    dataset, split, extract, codebook.build, quantize/histogram,
    SPIMI/indice invertido, persistencia

Uso:
    python -m pipelines.ingest --app music --data backend/tests/fixtures/lyrics_sample.json
    python -m pipelines.ingest --app music --data /data/spotify/lyrics.json --out /data/index/music_text --k 512
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from app.core.config import settings


def ingest_music_text(data_path: str, out_dir: str, k: int, block_size: int) -> None:
    from app.apps.music.text_index import build

    songs = json.loads(Path(data_path).read_text(encoding="utf-8"))
    idx = build(songs, out_dir, k=k, block_size=block_size)
    print(
        f"[ingest] music/text listo: {len(idx.items)} canciones, "
        f"{len(idx.chunks)} chunks, codebook k={len(idx.codebook.terms)}, "
        f"{idx.index.num_terms} terminos / {idx.index.num_postings} postings"
    )
    print(f"[ingest] artefactos en {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline de indexacion multimodal")
    p.add_argument("--app", required=True, choices=["music", "ecommerce"])
    p.add_argument("--modality", default="text", choices=["text", "audio", "image"])
    p.add_argument("--data", required=True, help="ruta al dataset (json para texto)")
    p.add_argument("--out", default=None, help="dir de salida de artefactos")
    p.add_argument("--k", type=int, default=settings.codebook_k)
    p.add_argument("--block-size", type=int, default=1000)
    args = p.parse_args()

    out = args.out or str(Path(settings.data_dir) / "index" / f"{args.app}_{args.modality}")

    if args.app == "music" and args.modality == "text":
        ingest_music_text(args.data, out, args.k, args.block_size)
    else:
        raise NotImplementedError(
            f"ingest {args.app}/{args.modality} aun no implementado "
            "(slices imagen/audio pendientes)"
        )


if __name__ == "__main__":
    main()
