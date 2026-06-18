# Sistema Multimodal de Recuperación y Búsqueda

Proyecto 2 — Base de Datos 2 (2026-1). Motor unificado de recuperación que
funciona sobre **texto, imagen y audio** con la misma arquitectura, y que se
compara contra las técnicas nativas de PostgreSQL (GIN full-text y pgvector).

**Apps implementadas:** Búsqueda Musical por letra (texto) y Búsqueda Visual
E-commerce (imagen).

## Arquitectura unificada

La idea central del proyecto: el mismo paradigma sirve para cualquier modalidad,
solo cambian el extractor y el codebook.

```
CONTENIDO → SPLIT (chunks) → EXTRACTOR (features) → CODEBOOK → ÍNDICE INVERTIDO → BÚSQUEDA
```

| Etapa | Texto | Imagen |
|-------|-------|--------|
| Split | párrafos / estrofas | la imagen completa del producto |
| Extractor | tokens (nltk) | SIFT (128-d por keypoint) |
| Codebook | top-k palabras (lingüístico) | K-Means sobre los SIFT (visual words) |
| Índice | SPIMI → índice invertido | el mismo índice invertido |
| Búsqueda | coseno sobre histogramas | el mismo coseno |

El núcleo de búsqueda (`search_sparse`) es **idéntico** para ambas modalidades:
recibe un histograma disperso y no le importa de dónde salió. Eso es lo que hace
la arquitectura "agnóstica".

## Dataset

- **Música:** *Audio features and lyrics of Spotify songs* (Kaggle,
  `imuhammad/...`) — 18.454 canciones con letra, multilingüe (mayoría inglés,
  algo de español). Se usan `track_name`, `track_artist`, `lyrics`.
- **Imagen:** *Fashion Product Images (Small)* (Kaggle, `paramaggarwal/...`) —
  ~44.000 imágenes de productos de moda.

El ingest acepta `.csv` / `.json` / `.parquet` y mapea columnas por alias, así
que el dataset es intercambiable.

## Implementación por módulo

- **Split** — [paragraph.py](backend/app/engine/split/paragraph.py): parte el
  texto en párrafos (robusto a letras vacías/NaN).
- **Extractor** — [sift.py](backend/app/engine/extractor/sift.py): SIFT con
  OpenCV (import perezoso). Para texto el extractor es la tokenización.
- **Codebook** — [linguistic.py](backend/app/engine/codebook/linguistic.py)
  (top-k con tokenizar/stopwords/stemming) y
  [kmeans.py](backend/app/engine/codebook/kmeans.py) (MiniBatchKMeans → visual
  words). Ambos producen histogramas L2-normalizados.
- **Índice invertido** — [spimi.py](backend/app/engine/index/spimi.py)
  (Single-Pass In-Memory Indexing, obligatorio para texto: vuelca bloques a
  disco y hace k-way merge) + [inverted.py](backend/app/engine/index/inverted.py).
- **Búsqueda** — [similarity.py](backend/app/engine/search/similarity.py):
  coseno con accumulator (solo toca los chunks que comparten codewords).
- **Persistencia** — [repository.py](backend/app/db/repository.py): vuelca
  codebook, histogramas, metadatos e índice invertido a PostgreSQL.
- **Comparativas** — [comparisons/](backend/app/comparisons/): la misma consulta
  por los 3 enfoques (índice propio, GIN, pgvector).

## Resultados experimentales

Benchmark de texto sobre datos reales (Spotify), cargas de 1K/5K/10K canciones,
25 consultas cada una. Generado con
[experiments/benchmark.py](backend/experiments/benchmark.py).

| Corpus | Enfoque | Latencia media | QPS | Recall@10 vs propio | Memoria índice |
|--------|---------|---------------:|----:|--------------------:|---------------:|
| 10K | Índice invertido (propio) | **4.5 ms** | **223** | 1.00 (ref.) | 38.8K postings |
| 10K | GIN full-text | 11.6 ms | 87 | 0.08 | 12.6 MB |
| 10K | pgvector coseno | 15.1 ms | 66 | 0.97 | 1.8 MB |

![Latencia vs tamaño](backend/experiments/results/latency_vs_size.png)
![Recall vs tamaño](backend/experiments/results/recall_vs_size.png)

### Análisis y trade-offs

- **Latencia / throughput:** gana el índice invertido propio, y escala mejor
  (crece casi lineal y plano; los nativos se disparan con el tamaño).
- **pgvector** recupera casi lo mismo que el motor propio (recall ~0.97-0.99),
  lógico porque ambos hacen coseno sobre los mismos histogramas. Su índice ocupa
  mucho menos disco que el GIN, pero es el más lento en consulta.
- **GIN** recupera un conjunto distinto (recall bajo): hace match booleano de
  términos + `ts_rank`, no similitud de histograma. No es "peor", es otra
  semántica de búsqueda; conviene cuando importa el match exacto de palabras.

**Conclusión:** para búsqueda por similitud, el índice invertido + codebook gana
en velocidad y memoria a costa de la cuantización (pierde matices). pgvector es
la alternativa nativa más cercana en resultados; GIN es para full-text clásico.

## Instalación y uso

Requisitos: Docker. La base trae PostgreSQL + pgvector y los scripts de init.

```bash
docker compose up --build        # db (5432) + backend (8000) + frontend (5173)
```

Cargar datos y persistir en Postgres (dentro del contenedor backend):

```bash
# música (texto)
docker compose exec backend python -m pipelines.ingest \
  --app music --data /data/raw/spotify/spotify_songs.csv --k 256 --persist

# e-commerce (imagen)
docker compose exec backend python -m pipelines.ingest \
  --app ecommerce --modality image --data /data/raw/fashion/images --k 256 --persist
```

- UI: http://localhost:5173 — pestañas para las 2 apps.
- API docs: http://localhost:8000/docs
- Comparativa: `GET /compare/text?q=...` devuelve los 3 enfoques con su latencia.

Benchmark de la Fase 4 (genera CSV + gráficos en `experiments/results/`):

```bash
docker compose exec backend python -m experiments.benchmark \
  --data /data/raw/spotify/spotify_songs.csv --sizes 1000 5000 10000
```

## Tests

```bash
cd backend && pytest          # 16 tests; los de Postgres se saltan si no hay DB
```

## Estructura

```
backend/
  app/
    api/          routers: ecommerce, music, compare
    engine/       arquitectura unificada (split/extractor/codebook/index/search)
    apps/         orquestadores por app (text_index, visual_index)
    db/           sesión, repositorio y adaptadores de persistencia
    comparisons/  índice propio vs GIN vs pgvector
    core/         config
  pipelines/      ingest offline (+ --persist a Postgres)
  experiments/    benchmark Fase 4 + resultados
frontend/         UI React + Vite (demo de las 2 apps)
db/init/          01_init.sql (extensión vector + schema)
docker-compose.yml
```
