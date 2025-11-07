from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    db_url: str = "sqlite:///data/carryon.db"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "data/faiss.index"
    cors_allow_origins: List[str] = ["*"]
    entropy_weight: float = 0.35
    alpha: float = 0.50  # semantic
    beta: float = 0.25   # recency
    gamma: float = 0.15  # graph degree
    delta: float = 0.10  # entropy

    class Config:
        env_prefix = "CARRYON_"

settings = Settings()