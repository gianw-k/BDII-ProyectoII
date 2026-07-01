from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
from app.api import ecommerce, music, compare, query

app = FastAPI(title="Sistema Multimodal de Recuperacion y Busqueda")

# Servir imagenes de e-commerce si existen (útil para el frontend)
image_path = "/data/raw/fashion/images" if os.path.exists("/data/raw/fashion/images") else "./data/raw/fashion/images"
if os.path.exists(image_path):
    app.mount("/images", StaticFiles(directory=image_path), name="images")

# Servir archivos de audio GTZAN para reproduccion en el frontend
audio_path = "/data/raw/music/genres_original"
if os.path.exists(audio_path):
    app.mount("/audio-files", StaticFiles(directory=audio_path), name="audio")

app.include_router(ecommerce.router, prefix="/ecommerce", tags=["ecommerce"])
app.include_router(music.router, prefix="/music", tags=["music"])
app.include_router(compare.router, prefix="/compare", tags=["compare"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.get("/health")
def health():
    return {"status": "ok"}
