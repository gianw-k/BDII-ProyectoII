from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
from app.api import ecommerce, music, compare

app = FastAPI(title="Sistema Multimodal de Recuperacion y Busqueda")

# Servir imagenes de e-commerce si existen (útil para el frontend)
image_path = "/data/raw/fashion/images" if os.path.exists("/data/raw/fashion/images") else "./data/raw/fashion/images"
if os.path.exists(image_path):
    app.mount("/images", StaticFiles(directory=image_path), name="images")

app.include_router(ecommerce.router, prefix="/ecommerce", tags=["ecommerce"])
app.include_router(music.router, prefix="/music", tags=["music"])
app.include_router(compare.router, prefix="/compare", tags=["compare"])


@app.get("/health")
def health():
    return {"status": "ok"}
