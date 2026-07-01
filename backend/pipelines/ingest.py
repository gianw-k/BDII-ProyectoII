"""Pipeline OFFLINE de indexacion (no corre en cada request).

Recorre el dataset y construye todo lo que la busqueda online consultara:

    dataset, split, extract, codebook.build, quantize/histogram,
    SPIMI/indice invertido, persistencia

Uso:
    python -m pipelines.ingest --app music --data backend/tests/fixtures/lyrics_sample.json
    python -m pipelines.ingest --app music --data data/raw/spotify-lyrics/lyrics.parquet --limit 1000
    python -m pipelines.ingest --app music --data /data/spotify/lyrics.csv --out /data/index/music_text --k 512
    python -m pipelines.ingest --app ecommerce --modality image --data data/raw/fashion/images --k 256
    python -m pipelines.ingest --app music --modality audio --data data/raw/music/features_30_sec.csv --k 128
    python -m pipelines.ingest --app music --modality audio --data data/raw/music/features_3_sec.csv --k 128
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from app.core.config import settings

# alias de columnas -> campo que espera build() (text_index.build)
COLUMN_ALIASES = {
    "external_id": ("external_id", "link", "id", "track_id", "uri"),
    "title": ("title", "song", "name", "track", "track_name"),
    "artist": ("artist", "artists", "artist_name", "track_artist", "performer"),
    "lyrics": ("lyrics", "text", "content", "lyric"),
}


def _map_row(row: dict) -> dict:
    """Mapea una fila cruda al schema {external_id,title,artist,lyrics}."""
    out: dict = {}
    lower = {str(c).lower(): c for c in row}
    for field, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                out[field] = row[lower[a]]
                break
        else:
            out[field] = None
    return out


def load_songs(data_path: str, limit: int | None = None) -> list[dict]:
    """Carga canciones desde .json / .parquet / .csv y normaliza columnas."""
    path = Path(data_path)
    ext = path.suffix.lower()
    if ext == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif ext in (".parquet", ".csv"):
        import pandas as pd
        df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
        if limit:
            df = df.head(limit)
        rows = df.to_dict(orient="records")
    else:
        raise ValueError(f"formato no soportado: {ext} (usa .json/.parquet/.csv)")

    songs = [_map_row(r) for r in rows]
    # external_id por defecto = indice si falta
    for i, s in enumerate(songs):
        if not s.get("external_id"):
            s["external_id"] = str(i)
    if limit:
        songs = songs[:limit]
    return songs


def ingest_music_text(data_path: str, out_dir: str, k: int, block_size: int,
                      limit: int | None = None, persist: bool = False) -> None:
    from app.apps.music.text_index import build

    songs = load_songs(data_path, limit=limit)
    idx = build(songs, out_dir, k=k, block_size=block_size)
    print(
        f"[ingest] music/text listo: {len(idx.items)} canciones, "
        f"{len(idx.chunks)} chunks, codebook k={len(idx.codebook.terms)}, "
        f"{idx.index.num_terms} terminos / {idx.index.num_postings} postings"
    )
    print(f"[ingest] artefactos en {out_dir}")
    if persist:
        from app.db.adapters import text_index_to_data
        _persist(text_index_to_data(idx))


def _persist(data) -> None:
    """Vuelca el IndexData a Postgres (paso opcional, tras construir el indice)."""
    from app.db.session import get_conn
    from app.db.repository import persist_index

    with get_conn() as conn:
        summary = persist_index(conn, data)
    print(f"[ingest] persistido en Postgres: {summary}")


def ingest_ecommerce_image(data_path: str, out_dir: str, k: int, block_size: int,
                           limit: int | None = None, persist: bool = False,
                           color_weight: float = 0.5) -> None:
    """Indexa una carpeta de imagenes de productos (SIFT + color HSV -> histograma)."""
    from app.apps.ecommerce.visual_index import build
    from app.engine.extractor.sift import SIFTExtractor

    # juntamos todas las imagenes de la carpeta (jpg/png/...)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in Path(data_path).rglob("*") if p.suffix.lower() in exts)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no se encontraron imagenes en {data_path}")

    extractor = SIFTExtractor()
    items, descriptors, colors = [], [], []
    for p in paths:
        desc = extractor.extract(p)
        if desc.shape[0] == 0:        # imagen sin keypoints: la saltamos
            continue
        items.append({"external_id": p.stem, "filename": p.name, "path": str(p)})
        descriptors.append(desc)
        colors.append(extractor.color_histogram(p))

    idx = build(items, descriptors, out_dir, k=k, block_size=block_size,
                colors=colors, color_weight=color_weight)
    total = sum(d.shape[0] for d in descriptors)
    print(
        f"[ingest] ecommerce/image listo: {len(idx.items)} productos, "
        f"{total} descriptores SIFT, codebook k={idx.codebook.centroids.shape[0]}, "
        f"color_weight={idx.color_weight} (+{idx.color_dim} bins), "
        f"{idx.index.num_terms} visual words / {idx.index.num_postings} postings"
    )
    print(f"[ingest] artefactos en {out_dir}")
    if persist:
        from app.db.adapters import visual_index_to_data
        _persist(visual_index_to_data(idx))


def ingest_music_audio(
    data_path: str,
    out_dir: str,
    k: int,
    block_size: int,
    limit: int | None = None,
    persist: bool = False,
) -> None:
    """Indexa el dataset GTZAN (CSV de features MFCC) para busqueda acustica.

    Cada fila del CSV es una ventana de audio (3s o 30s). Las filas se agrupan
    por filename: cada pista aporta N vectores MFCC que se cuantizan juntos en
    un histograma de acoustic words. Esto es el Bag-of-Acoustic-Words.

    Compatible con:
      - features_30_sec.csv  (1 fila/pista, 1000 pistas)
      - features_3_sec.csv   (10 filas/pista, ~10000 filas)
    """
    import pandas as pd
    from app.apps.music.acoustic_index import build
    from app.engine.split.audio_window import AudioWindowSplitter, _MFCC_COLS

    df = pd.read_csv(data_path)
    if limit:
        # limitar por numero de pistas unicas, no por filas
        unique_files = df["filename"].unique()[:limit]
        df = df[df["filename"].isin(unique_files)]

    splitter = AudioWindowSplitter()

    # Derivar el nombre de la pista base a partir del filename del CSV.
    # features_3_sec.csv usa filenames como "blues.00000.4.wav" (con segmento).
    # features_30_sec.csv usa "blues.00000.wav" (sin segmento).
    # Estrategia: si hay 3+ partes separadas por ".", el penultimo es el segmento.
    def base_track(fname: str) -> str:
        """blues.00000.4.wav → blues.00000.wav  |  blues.00000.wav → blues.00000.wav"""
        parts = str(fname).rsplit(".", 2)
        # "blues.00000.4.wav"  → parts = ["blues.00000", "4", "wav"]  → len=3
        # "blues.00000.wav"    → parts = ["blues.00000", "wav"]       → len=2
        if len(parts) == 3 and parts[1].isdigit():
            return f"{parts[0]}.{parts[2]}"   # strip el segmento numerico
        return str(fname)

    # Añadir columna de pista base y agrupar por ella
    df["_base"] = df["filename"].apply(base_track)

    tracks, descriptors = [], []
    for base_filename, group in df.groupby("_base"):
        rows = group.to_dict(orient="records")
        windows = splitter.split(rows)  # lista de arrays (dim,) por ventana
        if not windows:
            continue
        # Apilar las ventanas en (n_windows, dim)
        desc = np.stack(windows, axis=0).astype("float32")  # (n_windows, 40)
        label = rows[0].get("label", "unknown")
        tracks.append({
            "filename": str(base_filename),   # nombre real del .wav en disco
            "label": str(label),
            "n_windows": len(windows),
        })
        descriptors.append(desc)

    if not tracks:
        raise ValueError(f"No se encontraron pistas validas en {data_path}")

    idx = build(tracks, descriptors, out_dir, k=k, block_size=block_size)
    print(
        f"[ingest] music/audio listo: {len(idx.tracks)} pistas, "
        f"codebook k={idx.codebook.centroids.shape[0]} acoustic words, "
        f"{idx.index.num_terms} terminos / {idx.index.num_postings} postings"
    )
    print(f"[ingest] artefactos en {out_dir}")
    if persist:
        from app.db.adapters import audio_index_to_data
        _persist(audio_index_to_data(idx))


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline de indexacion multimodal")
    p.add_argument("--app", required=True, choices=["music", "ecommerce"])
    p.add_argument("--modality", default="text", choices=["text", "audio", "image"])
    p.add_argument("--data", required=True, help="ruta al dataset (.json/.parquet/.csv)")
    p.add_argument("--out", default=None, help="dir de salida de artefactos")
    p.add_argument("--k", type=int, default=settings.codebook_k)
    p.add_argument("--block-size", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None,
                   help="limitar el numero de pistas indexadas (util para cargas 1K/10K)")
    p.add_argument("--persist", action="store_true",
                   help="ademas de los artefactos, cargar el indice en Postgres")
    p.add_argument("--color-weight", type=float, default=0.5,
                   help="peso del color HSV en imagen (0 = solo SIFT, 1 = solo color)")
    args = p.parse_args()

    out = args.out or str(Path(settings.data_dir) / "index" / f"{args.app}_{args.modality}")

    if args.app == "music" and args.modality == "text":
        ingest_music_text(args.data, out, args.k, args.block_size,
                          limit=args.limit, persist=args.persist)
    elif args.app == "music" and args.modality == "audio":
        ingest_music_audio(args.data, out, args.k, args.block_size,
                           limit=args.limit, persist=args.persist)
    elif args.app == "ecommerce" and args.modality == "image":
        ingest_ecommerce_image(args.data, out, args.k, args.block_size,
                               limit=args.limit, persist=args.persist,
                               color_weight=args.color_weight)
    else:
        raise NotImplementedError(
            f"ingest {args.app}/{args.modality} no implementado todavia."
        )


if __name__ == "__main__":
    main()
