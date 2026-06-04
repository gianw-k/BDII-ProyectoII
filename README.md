# Sistema Multimodal de Recuperación y Búsqueda

Proyecto 2 — Base de Datos 2 (2026-1). Motor unificado de búsqueda sobre
texto, imagen y audio (split, extractor, codebook, indice invertido, busqueda),
comparado contra PostgreSQL nativo (GIN/GiST, pgvector).

**Apps implementadas:** Busqueda Visual E-commerce (imagen) y Busqueda Musical (audio + texto).

## Estructura

```
backend/          API FastAPI + motor (Python)
  app/
    api/          routers: ecommerce, music, compare
    engine/       arquitectura unificada (split/extractor/codebook/index/search)
    apps/         lógica de cada app sobre el motor
    db/           modelos / repositorios
    comparisons/  GIN/GiST y pgvector
    core/         config
  pipelines/      ingest offline (construye codebook + índice + carga DB)
  experiments/    benchmarks + gráficos (Fase 4)
frontend/         UI React + Vite (demo de las 2 apps)
db/init/          init.sql (extensión vector + schema)
data/             datasets (gitignored)
docs/             informe + diagramas
docker-compose.yml
```

## Arranque (cuando esté implementado)

```bash
docker compose up --build      # db + backend (8000) + frontend (5173)
# 1) cargar datos:  docker compose exec backend python -m pipelines.ingest ...
# 2) abrir UI:      http://localhost:5173
# 3) API docs:      http://localhost:8000/docs
```