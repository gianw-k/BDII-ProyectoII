from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://bdii:bdii@localhost:5432/multimodal"
    data_dir: str = "/data"
    # Tamano del codebook de texto (perilla central de trade-offs: mas k =
    # vocabulario mas rico y mejor recall, mas memoria). Tope practico: 2000,
    # que es el maximo de dimensiones que indexa el HNSW de pgvector.
    codebook_k: int = 1024
    top_n: int = 10        # resultados por consulta
    image_max_side: int = 256


settings = Settings()
