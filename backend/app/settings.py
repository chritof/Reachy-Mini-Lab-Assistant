from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Backend
    API_KEY: str | None = None
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8080

    # Ollama / LLM
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:14b"

    # STT
    WHISPER_MODEL: str = "medium"
    WHISPER_COMPUTE_TYPE: str = "int8"
    STT_LANGUAGE: str = "no"

    # RAG
    RAG_INDEX_DIR: Path = Path("data/rag_index")
    RAG_EMBED_MODEL: str = "nomic-embed-text"
    RAG_TOP_K: int = 3

    # TTS
    PIPER_MODEL_PATH: str = "models/no_NO-talesyntese-medium.onnx"


settings = Settings()