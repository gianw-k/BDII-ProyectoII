from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://bdii:bdii@localhost:5432/multimodal"
    data_dir: str = "/data"
    codebook_k: int = 256  # tamano del codebook (perilla central de trade-offs)
    top_n: int = 10        # resultados por consulta


settings = Settings()
