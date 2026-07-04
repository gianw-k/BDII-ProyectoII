"""Benchmark de CALIDAD para imagen: coherencia de categoria y de color.

Mide si la busqueda visual devuelve productos del mismo TIPO (articleType:
Shirts, Watches, Shoes...) y del mismo COLOR (baseColour) que la consulta,
usando las etiquetas de styles.csv del dataset Fashion. Es la senal objetiva de
calidad que faltaba para imagen.

Ablacion (sin reingestar): el histograma persistido es la fusion
[sqrt(1-a)*BoVW(512) | sqrt(a)*color(32)]. Tomando solo la parte SIFT (512,
renormalizada), solo la de color (32) o la fusion completa (544), comparamos:
  - SIFT solo    -> deberia acertar el TIPO (forma)
  - color solo   -> deberia acertar el COLOR
  - fusion       -> equilibra ambos
Eso demuestra con numeros por que la fusion de color tiene sentido.

Ademas genera montajes visuales (consulta + top-5 recuperados) leyendo las
imagenes reales del dataset.

Uso (dentro del contenedor):
  python -m experiments.image_quality_benchmark \
    --styles /data/raw/fashion/styles.csv --images /data/raw/fashion/images
"""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
SIFT_DIM = 512          # visual words
# el resto de la dimension es color (32); total 544


def load_histograms():
    """Trae (external_id, histograma 544-d) de todas las imagenes desde Postgres."""
    from app.db.session import connect
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT i.external_id, h.hist "
            "FROM histograms h JOIN chunks c ON c.id=h.chunk_id "
            "JOIN items i ON i.id=c.item_id WHERE h.modality='image'"
        )
        rows = cur.fetchall()
    conn.close()
    ids = [str(r[0]) for r in rows]
    H = np.asarray([np.asarray(r[1], dtype=np.float32) for r in rows], dtype=np.float32)
    return ids, H


def load_styles(path: str) -> dict:
    """id -> {'type':articleType, 'color':baseColour}."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            out[str(row["id"])] = {"type": row.get("articleType", ""),
                                   "color": row.get("baseColour", "")}
    return out


def _l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def coherence(H, labels, q_idx, topk=10):
    """Fraccion de los topk vecinos (coseno) que comparten etiqueta, promediada."""
    labels = np.asarray(labels)
    vals = []
    # por lotes para no armar la matriz 44k x 44k completa
    for i in q_idx:
        sims = H @ H[i]              # (N,) coseno (H ya unitario)
        sims[i] = -1.0
        nn = np.argpartition(-sims, topk)[:topk]
        if labels[i]:
            vals.append(np.mean(labels[nn] == labels[i]))
    return float(np.mean(vals))


def run(styles_path: str, topk: int, n_queries: int, seed: int = 0):
    ids, H = load_histograms()
    styles = load_styles(styles_path)
    # nos quedamos con los que tienen etiqueta conocida
    keep = [j for j, x in enumerate(ids) if x in styles]
    ids = [ids[j] for j in keep]
    H = H[keep]
    types = [styles[x]["type"] for x in ids]
    colors = [styles[x]["color"] for x in ids]
    print(f"[img] {len(ids)} productos con etiqueta; dim={H.shape[1]}")

    views = {
        "SIFT solo (512)":  _l2(H[:, :SIFT_DIM].copy()),
        "color solo (32)":  _l2(H[:, SIFT_DIM:].copy()),
        "fusion (544)":     _l2(H.copy()),
    }
    rng = random.Random(seed)
    q_idx = rng.sample(range(len(ids)), min(n_queries, len(ids)))

    res = {}
    for name, V in views.items():
        c_type = coherence(V, types, q_idx, topk)
        c_col = coherence(V, colors, q_idx, topk)
        res[name] = {"tipo": c_type, "color": c_col}
        print(f"[img] {name:16s} -> coherencia tipo@{topk}={c_type:.3f}  color@{topk}={c_col:.3f}")

    # baselines al azar
    res["_baseline"] = {"tipo": round(_freq_baseline(types), 3),
                        "color": round(_freq_baseline(colors), 3)}
    print(f"[img] baseline azar -> tipo={res['_baseline']['tipo']}  color={res['_baseline']['color']}")
    return res, ids, _l2(H.copy())


def _freq_baseline(labels):
    """Prob. de acertar etiqueta al azar = suma de p_i^2 (dos items al azar)."""
    from collections import Counter
    c = Counter(l for l in labels if l)
    tot = sum(c.values())
    return sum((n / tot) ** 2 for n in c.values())


def save_and_plot(res: dict, topk: int):
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = [n for n in res if not n.startswith("_")]
    with (RESULTS / "image_quality.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["vista", "coherencia_tipo", "coherencia_color"])
        for n in names:
            w.writerow([n, round(res[n]["tipo"], 3), round(res[n]["color"], 3)])
        w.writerow(["baseline_azar", res["_baseline"]["tipo"], res["_baseline"]["color"]])
    print(f"[img] CSV -> {RESULTS/'image_quality.csv'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(names)); wbar = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - wbar/2, [res[n]["tipo"] for n in names], wbar, label="tipo (articleType)", color="#3c78e0")
    plt.bar(x + wbar/2, [res[n]["color"] for n in names], wbar, label="color (baseColour)", color="#e08a3c")
    plt.axhline(res["_baseline"]["tipo"], color="#3c78e0", ls=":", lw=1)
    plt.axhline(res["_baseline"]["color"], color="#e08a3c", ls=":", lw=1)
    plt.xticks(x, names); plt.ylim(0, 1); plt.ylabel(f"coherencia @ {topk}")
    plt.title("Calidad de la busqueda visual (Fashion)"); plt.legend()
    plt.tight_layout(); plt.savefig(RESULTS / "image_quality.png", dpi=120)
    print(f"[img] grafico -> {RESULTS/'image_quality.png'}")


def montages(ids, H, images_dir: str, styles_path: str, n=4, topk=5, seed=7):
    """Collage: por cada consulta, la imagen consulta + sus top-k vecinos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    styles = load_styles(styles_path)
    idpos = {x: j for j, x in enumerate(ids)}
    rng = random.Random(seed)
    # elegir consultas cuyos vecinos existan en disco
    queries = rng.sample(range(len(ids)), n)

    fig, axes = plt.subplots(n, topk + 1, figsize=((topk + 1) * 1.7, n * 1.9))
    if n == 1:
        axes = axes[None, :]
    for r, qi in enumerate(queries):
        sims = H @ H[qi]; sims[qi] = -1.0
        nn = np.argsort(-sims)[:topk]
        cells = [(qi, "CONSULTA", None)] + [(j, f"{sims[j]:.2f}", None) for j in nn]
        for cpos, (j, cap, _) in enumerate(cells):
            ax = axes[r][cpos]; ax.axis("off")
            p = Path(images_dir) / f"{ids[j]}.jpg"
            try:
                ax.imshow(mpimg.imread(str(p)))
            except Exception:
                ax.text(0.5, 0.5, "?", ha="center")
            t = styles.get(ids[j], {}).get("type", "")
            ax.set_title(f"{cap}\n{t[:14]}", fontsize=7,
                         color="crimson" if cpos == 0 else "black")
    fig.suptitle("Busqueda visual: consulta (rojo) + top-5 productos similares", y=1.0)
    fig.tight_layout()
    fig.savefig(RESULTS / "image_montage.png", dpi=130, bbox_inches="tight")
    print(f"[img] montaje -> {RESULTS/'image_montage.png'}")


def main():
    p = argparse.ArgumentParser(description="Benchmark de calidad de imagen + montajes")
    p.add_argument("--styles", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=1500)
    args = p.parse_args()
    res, ids, H = run(args.styles, args.topk, args.n_queries)
    save_and_plot(res, args.topk)
    montages(ids, H, args.images, args.styles)


if __name__ == "__main__":
    main()
