# app/core/config.py
from typing import List

try:
    # Try Pydantic v2 first
    from pydantic_settings import BaseSettings
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PINNACLE Fine-tuning API"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # Data paths
    EMBEDDINGS_PATH: str = "./pinnacle_protein_embed.pth"
    LABELS_PATH: str = "./pinnacle_protein_labels_dict.txt"
    
    # Model training defaults
    DEFAULT_HIDDEN_DIM: int = 256
    DEFAULT_DROPOUT_RATES: List[float] = [0.3, 0.2, 0.1]
    DEFAULT_LEARNING_RATE: float = 0.001
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 30
    
    class Config:
        case_sensitive = True

settings = Settings()