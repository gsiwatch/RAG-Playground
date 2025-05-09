from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    ssl_cert_file: str | None = None
    requests_ca_bundle: str | None = None
    
    openai_api_key: str
    openai_model_name: str
    openai_embedding_model: str
    openai_embedding_dimensions: int
    
    chunk_size: int = 500
    chunk_overlap: int = 50

    mongodb_uri: str
    mongodb_db_name: str
    mongodb_collection_name: str
    mongodb_summary_collection: str = "compiled-answers-test-summary"
    mongodb_chunks_collection: str = "compiled-answers-test-chunks"

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
