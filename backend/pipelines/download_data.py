"""Descarga datasets crudos desde un bucket S3 publico

El repo versiona un sample chico (`data/sample/*.parquet`) para prueba
inmediata; el dataset completo se baja por URL https con este script. Asi el
repo queda liviano

Bucket publico = no hace falta boto3 ni credenciales: descarga directa por URL.

Uso:
  python -m pipelines.download_data --dataset music-lyrics
  python -m pipelines.download_data --list

Para subir un dataset al bucket (una vez, con aws cli):
  aws s3 cp data/raw/spotify-lyrics/data.parquet \\
    s3://<bucket>/music/lyrics.parquet --acl public-read
"""
from __future__ import annotations
import argparse
import sys
import urllib.request
from pathlib import Path

# base del bucket S3 publico. Completar cuando el bucket exista, ej:
#   https://mi-bucket.s3.us-east-1.amazonaws.com
S3_BASE = ""

# catalogo: nombre -> key dentro del bucket + archivo destino en data/raw/
DATASETS = {
    "music-lyrics": {
        "key": "music/lyrics.parquet",
        "subdir": "spotify-lyrics",
        "filename": "data.parquet",
    },
    # futuros: "fashion-images", "gtzan-audio" ...
}

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _progress(block_num: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(100, block_num * block_size * 100 // total)
        sys.stdout.write(f"\r[download] {pct}%")
        sys.stdout.flush()


def download(name: str) -> None:
    if name not in DATASETS:
        sys.exit(f"dataset desconocido: {name}. Opciones: {', '.join(DATASETS)}")
    if not S3_BASE:
        sys.exit("falta configurar S3_BASE en download_data.py (URL del bucket publico)")

    cfg = DATASETS[name]
    url = f"{S3_BASE.rstrip('/')}/{cfg['key']}"
    dest = RAW_DIR / cfg["subdir"]
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / cfg["filename"]

    print(f"[download] {url} -> {out}")
    urllib.request.urlretrieve(url, out, reporthook=_progress)
    print(f"\n[download] listo: {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("[download] (data/raw/ esta en .gitignore; no se sube al repo)")


def main() -> None:
    p = argparse.ArgumentParser(description="Descarga datasets desde S3 publico")
    p.add_argument("--dataset", help="nombre del dataset del catalogo")
    p.add_argument("--list", action="store_true", help="listar datasets disponibles")
    args = p.parse_args()

    if args.list or not args.dataset:
        print("Datasets disponibles:")
        for k, v in DATASETS.items():
            print(f"  {k:14s} {v['key']}")
        return

    download(args.dataset)


if __name__ == "__main__":
    main()
