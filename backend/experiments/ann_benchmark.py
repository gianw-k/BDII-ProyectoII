"""pgvector: búsqueda exacta vs HNSW sobre los histogramas de texto.

Barremos `hnsw.ef_search` y por cada valor medimos latencia y recall@k del HNSW
contra la búsqueda exacta (Seq Scan), que hace de referencia. Sale la curva del
trade-off: a más ef_search, más recall pero más latencia; el exacto es el techo
de recall y el piso de velocidad.

Necesita los histogramas de música persistidos en Postgres (ingest con
--persist) y el codebook en /data/index/music_text.

Uso:
  python -m experiments.ann_benchmark
  python -m experiments.ann_benchmark --ef 10 20 40 80 160 --n-queries 50 --k 10
"""
from __future__ import annotations
import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

from app.apps.music.text_index import MusicTextIndex
from app.comparisons import ann
from app.core.config import settings
from app.db.session import connect

RESULTS = Path(__file__).parent / "results"


def pick_queries(codebook, n_queries: int, terms_per_query: int = 3, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    vocab = codebook.terms
    top = vocab[: min(len(vocab), 200)]
    return [" ".join(rng.choice(top, size=min(terms_per_query, len(top)), replace=False))
            for _ in range(n_queries)]


def _ids(results: list[dict]) -> set:
    return {r.get("external_id") or r.get("item_id") for r in results}


def _recall(ref: list[dict], other: list[dict]) -> float:
    a = _ids(ref)
    return len(a & _ids(other)) / len(a) if a else 0.0


def _stats(samples: list[float]) -> dict:
    s = sorted(samples)
    mean = statistics.mean(s)
    return {
        "mean_ms": round(mean, 4),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 4),
        "qps": round(1000.0 / mean, 1) if mean > 0 else 0.0,
    }


def run(ef_values: list[int], n_queries: int, k: int) -> list[dict]:
    idx = MusicTextIndex.load(Path(settings.data_dir) / "index" / "music_text")
    cb = idx.codebook
    conn = connect()
    ann.ensure_hnsw(conn, "text")

    queries = pick_queries(cb, n_queries)
    q_hists = [np.asarray(cb.quantize(q), dtype=np.float32) for q in queries]
    q_hists = [h for h in q_hists if np.any(h)]   # descarta queries sin codewords

    # referencia: el exacto (Seq Scan) por query, una sola vez
    exact_lat, exact_res = [], []
    for h in q_hists:
        r = ann.pgvector_exact(conn, h, "text", top_n=k)
        exact_lat.append(r["latency_ms"])
        exact_res.append(r["results"])

    rows = [{"method": "pgvector_exact", "ef_search": "", "recall_at_k": 1.000,
             **_stats(exact_lat)}]
    print(f"[ann] exact: {_stats(exact_lat)['mean_ms']}ms  recall=1.000 (ref)")

    for ef in ef_values:
        lat, rec = [], []
        for h, ref in zip(q_hists, exact_res):
            r = ann.pgvector_hnsw(conn, h, "text", top_n=k, ef_search=ef)
            lat.append(r["latency_ms"])
            rec.append(_recall(ref, r["results"]))
        st = _stats(lat)
        recall = round(statistics.mean(rec), 3)
        rows.append({"method": "pgvector_hnsw", "ef_search": ef, "recall_at_k": recall, **st})
        print(f"[ann] hnsw ef={ef}: {st['mean_ms']}ms  recall={recall}")

    conn.close()
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "ef_search", "mean_ms", "p95_ms", "qps", "recall_at_k"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[ann] CSV -> {path}")


def plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hnsw = [r for r in rows if r["method"] == "pgvector_hnsw"]
    exact = next(r for r in rows if r["method"] == "pgvector_exact")
    efs = [r["ef_search"] for r in hnsw]

    # recall vs latencia: la curva del trade-off
    plt.figure()
    plt.plot([r["mean_ms"] for r in hnsw], [r["recall_at_k"] for r in hnsw],
             marker="o", label="HNSW (aprox)")
    plt.scatter([exact["mean_ms"]], [1.0], color="red", zorder=5, label="exacto (fuerza bruta)")
    for r in hnsw:
        plt.annotate(f"ef={r['ef_search']}", (r["mean_ms"], r["recall_at_k"]),
                     textcoords="offset points", xytext=(5, -8), fontsize=8)
    plt.xlabel("latencia media (ms)"); plt.ylabel("recall@k vs exacto")
    plt.title("HNSW: trade-off recall / latencia"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "ann_recall_vs_latency.png", dpi=120, bbox_inches="tight")

    # latencia vs ef_search (con el exacto como referencia horizontal)
    plt.figure()
    plt.plot(efs, [r["mean_ms"] for r in hnsw], marker="o", label="HNSW (aprox)")
    plt.axhline(exact["mean_ms"], color="red", ls="--", label="exacto (fuerza bruta)")
    plt.xlabel("hnsw.ef_search"); plt.ylabel("latencia media (ms)")
    plt.title("Latencia vs ef_search"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "ann_latency_vs_ef.png", dpi=120, bbox_inches="tight")
    print(f"[ann] graficos -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark ANN HNSW vs fuerza bruta")
    p.add_argument("--ef", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    rows = run(args.ef, args.n_queries, args.k)
    save_csv(rows, RESULTS / "ann_benchmark.csv")
    plot(rows, RESULTS)


if __name__ == "__main__":
    main()
