from fastapi import FastAPI
from app.api import ecommerce, music, compare

app = FastAPI(title="Sistema Multimodal de Recuperacion y Busqueda")

app.include_router(ecommerce.router, prefix="/ecommerce", tags=["ecommerce"])
app.include_router(music.router, prefix="/music", tags=["music"])
app.include_router(compare.router, prefix="/compare", tags=["compare"])


@app.get("/health")
def health():
    return {"status": "ok"}
