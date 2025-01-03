from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    SESSION_ID: str

    DEVICE: str
    TEMP_DIR: str
    CACHE_DIR: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None

    GENERATION_TEMPERATURE: float = None
    SUMMERIZATION_TEMPERATURE: float = None

    VECTOR_DB_PATH: str = None
    
    class Config:
        env_file = ".env"
        arbitrary_types_allowed = True

def get_settings():
    return Settings()
