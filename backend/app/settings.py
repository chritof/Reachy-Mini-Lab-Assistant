from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    API_KEY: str | None = None
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8080

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:14b"

    WHISPER_MODEL: str = "medium"
    WHISPER_COMPUTE_TYPE: str = "int8"
    STT_LANGUAGE: str = "no"


settings = Settings()