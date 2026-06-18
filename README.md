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

### Cómo obtener los datos

Los datasets **no están en el repo** (son pesados; van en `.gitignore`). Cada
quien los baja una vez de Kaggle y los deja en `data/raw/`. Necesitas una cuenta
de Kaggle y su CLI (`pip install kaggle` + tu token en `~/.kaggle/kaggle.json`).

```bash
# Música (CSV de letras)
kaggle datasets download -d imuhammad/audio-features-and-lyrics-of-spotify-songs \
  -p data/raw/spotify --unzip

# Imágenes (versión small, ~592 MB)
kaggle datasets download -d paramaggarwal/fashion-product-images-small \
  -p data/raw/fashion --unzip
```

Tras esto debes tener `data/raw/spotify/spotify_songs.csv` y las imágenes en
`data/raw/fashion/images/`. Sin token de Kaggle puedes bajarlos a mano desde la
web del dataset y descomprimirlos en esas mismas carpetas. Luego corre el ingest
(ver más abajo) para construir los índices.

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

Benchmark de texto sobre datos reales (Spotify), cargas de 1K/5K/10K/18.454
canciones, 30 consultas cada una. Generado con
[experiments/benchmark.py](backend/experiments/benchmark.py).

Resultados con el corpus completo (18.454 canciones):

| Enfoque | Latencia media | p95 | Throughput | Recall@10 vs propio |
|---------|---------------:|----:|-----------:|--------------------:|
| Índice invertido (propio) | **12.0 ms** | 15.9 ms | **83 q/s** | 1.00 (ref.) |
| GIN full-text | 18.6 ms | 24.9 ms | 54 q/s | 0.04 |
| pgvector coseno | 44.9 ms | 58.2 ms | 22 q/s | 0.97 |

![Latencia vs tamaño](backend/experiments/results/latency_vs_size.png)
![Throughput vs tamaño](backend/experiments/results/throughput_vs_size.png)
![Recall vs tamaño](backend/experiments/results/recall_vs_size.png)
![Memoria de índices nativos](backend/experiments/results/memory_vs_size.png)

### Análisis y trade-offs

- **Latencia / throughput:** gana el índice invertido propio en todas las cargas
  y escala mejor (crece casi lineal; los nativos se separan al subir el tamaño).
- **Recall:** pgvector recupera casi lo mismo que el motor propio (~0.97), lógico
  porque ambos hacen coseno sobre los mismos histogramas. GIN recupera un
  conjunto distinto (recall ~0.04): hace match booleano de términos + `ts_rank`,
  no similitud de histograma. No es "peor", es otra semántica de búsqueda.
- **Memoria:** ver gráfico. *Nota:* la tabla `histograms` guarda también los
  histogramas de las imágenes (e-commerce), así que su tamaño no es solo del
  texto; tómese como cota superior.

**Conclusión:** para búsqueda por similitud, el índice invertido + codebook gana
en velocidad a costa de la cuantización (pierde matices). pgvector es la
alternativa nativa más cercana en *resultados*; GIN sirve para full-text clásico.

> **Limitación honesta:** pgvector se midió **sin índice HNSW/IVF** (está
> comentado en [01_init.sql](db/init/01_init.sql)), por lo que corre como scan
> lineal — de ahí su latencia alta. Crear el índice ANN lo aceleraría bastante,
> a cambio de una recall aproximada. Queda como mejora pendiente de la Fase 3.

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
