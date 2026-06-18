"""Fase 4 - evaluacion experimental (texto).

Corre la MISMA bateria de consultas por los 3 enfoques (indice invertido propio,
GIN, pgvector) sobre cargas crecientes y mide:

  - latencia (media / p50 / p95, en ms)
  - throughput (consultas/seg)
  - precision (recall@k de los nativos respecto al motor propio como referencia)
  - memoria (postings del indice propio + tamano de los indices nativos en disco)

Guarda un CSV y graficos PNG en experiments/results/.

Uso:
  python -m experiments.benchmark --data tests/fixtures/lyrics_sample.json --sizes 100 500 1000
  python -m experiments.benchmark --data data/raw/spotify/lyrics.csv --sizes 1000 10000 100000

El dataset es indiferente: lo que importa es el numero de canciones (--sizes).
Si pides mas canciones de las que hay, se reciclan para alcanzar el tamano (util
para probar escalabilidad con el sample chico).
"""
from __future__ import annotations
import argparse
import csv
import statistics
import tempfile
from pathlib import Path

import numpy as np

from app.apps.music.text_index import build
from app.db.adapters import text_index_to_data
from app.db.repository import persist_index
from app.db.session import connect
from app.comparisons import text as cmp
from pipelines.ingest import load_songs

RESULTS = Path(__file__).parent / "results"


def make_corpus(base: list[dict], n: int) -> list[dict]:
    """Arma un corpus de exactamente n canciones, reciclando si hace falta."""
    if len(base) >= n:
        return base[:n]
    out = []
    for i in range(n):
        s = dict(base[i % len(base)])
        s["external_id"] = f"{s.get('external_id', i)}-{i}"   # ids unicos
        out.append(s)
    return out


def pick_queries(index, n_queries: int = 30, terms_per_query: int = 3, seed: int = 0) -> list[str]:
    """Genera consultas muestreando palabras frecuentes del codebook (sirve para cualquier dataset)."""
    rng = np.random.default_rng(seed)
    vocab = index.codebook.terms
    if not vocab:
        return []
    top = vocab[: min(len(vocab), 200)]          # de las mas frecuentes
    return [" ".join(rng.choice(top, size=min(terms_per_query, len(top)), replace=False))
            for _ in range(n_queries)]


def _latency_stats(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "mean_ms": round(statistics.mean(s), 4),
        "p50_ms": round(s[len(s) // 2], 4),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 4),
        "qps": round(1000.0 / statistics.mean(s), 1) if statistics.mean(s) > 0 else 0.0,
    }


def _recall_at_k(ref: list, other: list) -> float:
    """Fraccion de los resultados de referencia que el otro metodo tambien trajo."""
    if not ref:
        return 0.0
    a = {r.get("external_id") or r.get("item_id") for r in ref}
    b = {r.get("external_id") or r.get("item_id") for r in other}
    return round(len(a & b) / len(a), 3)


def db_index_sizes(conn) -> dict:
    """Tamano en disco de los indices nativos (pg_relation_size)."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size('idx_chunks_tsv')")
        gin = cur.fetchone()[0]
        cur.execute("SELECT pg_total_relation_size('histograms')")
        hist = cur.fetchone()[0]
    return {"gin_bytes": gin, "histograms_bytes": hist}


def run(data: str, sizes: list[int], k: int, n_queries: int) -> list[dict]:
    base = load_songs(data)
    conn = connect()
    rows = []
    for n in sizes:
        corpus = make_corpus(base, n)
        idx = build(corpus, tempfile.mkdtemp(), k=k, block_size=2000)
        persist_index(conn, text_index_to_data(idx))
        conn.commit()

        queries = pick_queries(idx, n_queries=n_queries)
        lat = {"inverted_index": [], "gin_fulltext": [], "pgvector_cosine": []}
        recall = {"gin_fulltext": [], "pgvector_cosine": []}

        for q in queries:
            own = cmp.own_search(idx, q, top_n=k)
            gin = cmp.gin_search(conn, q, top_n=k)
            vec = cmp.pgvector_search(conn, idx.codebook, q, top_n=k)
            lat["inverted_index"].append(own["latency_ms"])
            lat["gin_fulltext"].append(gin["latency_ms"])
            lat["pgvector_cosine"].append(vec["latency_ms"])
            recall["gin_fulltext"].append(_recall_at_k(own["results"], gin["results"]))
            recall["pgvector_cosine"].append(_recall_at_k(own["results"], vec["results"]))

        sizes_db = db_index_sizes(conn)
        for method, samples in lat.items():
            if not samples:
                continue
            row = {"corpus_size": n, "method": method, **_latency_stats(samples)}
            row["recall_at_k"] = (round(statistics.mean(recall[method]), 3)
                                  if method in recall else 1.0)   # propio = referencia
            row["postings"] = idx.index.num_postings if method == "inverted_index" else ""
            row["native_index_bytes"] = (sizes_db["gin_bytes"] if method == "gin_fulltext"
                                         else sizes_db["histograms_bytes"] if method == "pgvector_cosine"
                                         else "")
            rows.append(row)
        print(f"[bench] n={n}: " + " | ".join(
            f"{m} {_latency_stats(s)['mean_ms']}ms" for m, s in lat.items()))
    conn.close()
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[bench] CSV -> {path}")


def plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    sizes = sorted({r["corpus_size"] for r in rows})

    # latencia media vs tamano
    plt.figure()
    for m in methods:
        ys = [next(r["mean_ms"] for r in rows if r["method"] == m and r["corpus_size"] == n) for n in sizes]
        plt.plot(sizes, ys, marker="o", label=m)
    plt.xlabel("canciones"); plt.ylabel("latencia media (ms)")
    plt.title("Latencia vs tamano de corpus"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "latency_vs_size.png", dpi=120, bbox_inches="tight")

    # recall@k de los nativos vs el motor propio
    plt.figure()
    for m in [x for x in methods if x != "inverted_index"]:
        ys = [next(r["recall_at_k"] for r in rows if r["method"] == m and r["corpus_size"] == n) for n in sizes]
        plt.plot(sizes, ys, marker="s", label=m)
    plt.xlabel("canciones"); plt.ylabel("recall@k vs indice propio")
    plt.title("Coincidencia de resultados"); plt.ylim(0, 1.05); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "recall_vs_size.png", dpi=120, bbox_inches="tight")
    print(f"[bench] graficos -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Fase 4 (texto)")
    p.add_argument("--data", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=30)
    args = p.parse_args()

    rows = run(args.data, args.sizes, args.k, args.n_queries)
    save_csv(rows, RESULTS / "text_benchmark.csv")
    plot(rows, RESULTS)


if __name__ == "__main__":
    main()
