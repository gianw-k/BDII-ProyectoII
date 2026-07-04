"""Benchmark de CALIDAD para audio: coherencia de genero (GTZAN).

A diferencia de los otros benchmarks (que miden velocidad), este mide si la
busqueda por similitud acustica devuelve pistas del MISMO genero que la
consulta -- una senal objetiva de calidad, porque GTZAN etiqueta cada pista.

Metrica: coherencia de genero @ k = para cada pista, que fraccion de sus k
vecinos mas cercanos (por coseno sobre el histograma de acoustic words) son de
su mismo genero. Se promedia sobre las 1000 pistas.

Ablacion: compara la configuracion VIEJA (40 MFCC, sin estandarizar) contra la
NUEVA (57 features espectrales + Z-score), para cuantificar cuanto ayudaron las
mejoras. Reconstruye el pipeline (K-Means -> cuantizar -> TF-IDF -> coseno) de
forma autocontenida para controlar exactamente las dos variables.

Uso (dentro del contenedor):
  python -m experiments.genre_benchmark --data /data/raw/music/features_3_sec.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

RESULTS = Path(__file__).parent / "results"

# --- columnas ---
_MFCC = [f"mfcc{i}_mean" for i in range(1, 21)] + [f"mfcc{i}_var" for i in range(1, 21)]
_SPECTRAL = [
    "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var",
    "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo",
]
COLS_OLD = _MFCC                 # 40, config vieja
COLS_NEW = _SPECTRAL + _MFCC     # 57, config nueva


def _base(fname: str) -> str:
    """blues.00000.4.wav -> blues.00000 (agrupa las ventanas de una pista)."""
    p = fname.split(".")
    return ".".join(p[:2]) if len(p) >= 2 else fname


def _tracks(df: pd.DataFrame, cols: list[str]):
    """Devuelve [(genero, descriptores (n_ventanas, dim)), ...] por pista."""
    df = df.copy()
    df["_base"] = df["filename"].apply(_base)
    out = []
    for _, g in df.groupby("_base"):
        X = g[cols].to_numpy(dtype=np.float32)
        out.append((str(g["label"].iloc[0]), X))
    return out


def _assign(X, C, mu, sd):
    """1NN al centroide mas cercano (con Z-score si aplica)."""
    if mu is not None:
        X = (X - mu) / sd
    cn = (C * C).sum(axis=1) * 0.5
    return np.argmax(X @ C.T - cn, axis=1)


def build_histograms(tracks, k: int, standardize: bool, seed: int = 0):
    """Reconstruye el pipeline: K-Means -> cuantizar -> log(1+tf)*idf -> L2."""
    allX = np.vstack([X for _, X in tracks])
    if standardize:
        mu = allX.mean(axis=0)
        sd = allX.std(axis=0); sd[sd == 0] = 1.0
        fit_X = (allX - mu) / sd
    else:
        mu = sd = None
        fit_X = allX

    km = MiniBatchKMeans(n_clusters=k, random_state=seed,
                         batch_size=min(4096, len(fit_X)), n_init="auto").fit(fit_X)
    C = km.cluster_centers_.astype(np.float32)

    # IDF: en cuantas pistas aparece cada acoustic word
    N = len(tracks)
    labels_per = [_assign(X, C, mu, sd) for _, X in tracks]
    dfreq = np.zeros(k)
    for lab in labels_per:
        dfreq[np.unique(lab)] += 1
    idf = np.log10(N / np.maximum(dfreq, 1.0)).astype(np.float32)

    H = np.zeros((N, k), dtype=np.float32)
    for i, lab in enumerate(labels_per):
        h = np.bincount(lab, minlength=k).astype(np.float32)
        h = np.log1p(h) * idf                       # TF sublineal * IDF
        n = np.linalg.norm(h)
        if n > 0:
            h /= n                                  # L2
        H[i] = h
    genres = np.array([g for g, _ in tracks])
    return H, genres


def genre_coherence(H, genres, topk: int = 10):
    """Fraccion de los topk vecinos (coseno) que comparten genero, por pista."""
    S = H @ H.T                 # coseno (histogramas unitarios)
    np.fill_diagonal(S, -1.0)   # excluir la propia pista
    per_track = np.empty(len(H))
    for i in range(len(H)):
        nn = np.argpartition(-S[i], topk)[:topk]
        per_track[i] = np.mean(genres[nn] == genres[i])
    return per_track


def run(data: str, k: int, topk: int) -> dict:
    df = pd.read_csv(data)
    variants = {
        "vieja (40 MFCC, sin Z-score)": (COLS_OLD, False),
        "nueva (57 feats + Z-score)":   (COLS_NEW, True),
    }
    res = {}
    for name, (cols, std) in variants.items():
        tracks = _tracks(df, cols)
        H, genres = build_histograms(tracks, k=k, standardize=std)
        coh = genre_coherence(H, genres, topk=topk)
        # por genero
        per_genre = {}
        for gname in sorted(set(genres)):
            per_genre[gname] = float(np.mean(coh[genres == gname]))
        res[name] = {"overall": float(np.mean(coh)), "per_genre": per_genre,
                     "n_tracks": len(genres)}
        print(f"[genre] {name}: coherencia@{topk} = {np.mean(coh):.3f}")
    # baseline azar (generos ~balanceados)
    n_gen = len(set(genres))
    res["_baseline_azar"] = round(1.0 / n_gen, 3)
    print(f"[genre] baseline al azar ({n_gen} generos) = {1.0/n_gen:.3f}")
    return res


def save_and_plot(res: dict, topk: int) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = [n for n in res if not n.startswith("_")]

    # CSV
    with (RESULTS / "genre_benchmark.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "coherencia_genero", "n_tracks"])
        for n in names:
            w.writerow([n, round(res[n]["overall"], 3), res[n]["n_tracks"]])
        w.writerow(["baseline_azar", res["_baseline_azar"], ""])
    print(f"[genre] CSV -> {RESULTS/'genre_benchmark.csv'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) barra: overall vieja vs nueva vs azar
    plt.figure(figsize=(6, 4))
    labels = ["azar\n(1/10)"] + ["40 MFCC\nsin Z-score", "57 feats\n+ Z-score"]
    vals = [res["_baseline_azar"], res[names[0]]["overall"], res[names[1]]["overall"]]
    bars = plt.bar(labels, vals, color=["#bbbbbb", "#e08a3c", "#3c78e0"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel(f"coherencia de genero @ {topk}"); plt.ylim(0, 1)
    plt.title("Calidad de la busqueda por audio (GTZAN)")
    plt.tight_layout(); plt.savefig(RESULTS / "genre_coherence.png", dpi=120)

    # 2) por genero (config nueva)
    pg = res[names[1]]["per_genre"]
    plt.figure(figsize=(8, 4))
    gs = sorted(pg, key=lambda x: -pg[x])
    plt.bar(gs, [pg[g] for g in gs], color="#3c78e0")
    plt.axhline(res["_baseline_azar"], color="gray", ls="--", label="azar")
    plt.ylabel(f"coherencia @ {topk}"); plt.ylim(0, 1)
    plt.title("Coherencia por genero (57 feats + Z-score)")
    plt.xticks(rotation=45, ha="right"); plt.legend()
    plt.tight_layout(); plt.savefig(RESULTS / "genre_per_genre.png", dpi=120)
    print(f"[genre] graficos -> {RESULTS}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark de coherencia de genero (audio)")
    p.add_argument("--data", required=True)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()
    res = run(args.data, args.k, args.topk)
    save_and_plot(res, args.topk)


if __name__ == "__main__":
    main()
