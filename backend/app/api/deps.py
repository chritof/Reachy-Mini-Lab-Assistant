"""
Oppretter og cacher backend-tjenester.

Ansvar:
- Lage STTService
- Lage TTSService
- Lage RagService
- Lese konfig fra settings
- Finne riktige model paths

Routerne henter tjenestene herfra.
"""
from pathlib import Path

from backend.app.settings import settings

from reachy_assistant.stt.faster_whisper_silero import FasterWhisperSileroSTT
from reachy_assistant.services.stt_service import STTService
from reachy_assistant.services.rag_service import RagService
from reachy_assistant.rag.rag_engine import RagEngine
from reachy_assistant.llm.llm_engine import LLMEngine
from reachy_assistant.tts.piper_tts import PiperTTSEngine
from reachy_assistant.services.tts_service import TTSService

_stt_engine = None
_stt_service = None
_rag_service = None
_tts_service = None



def get_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "backend").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()

def get_stt_service() -> STTService:
    global _stt_engine, _stt_service
    if _stt_service is None:
        _stt_engine = FasterWhisperSileroSTT(
            whisper_model=settings.WHISPER_MODEL,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            language=settings.STT_LANGUAGE,
        )
        _stt_service = STTService(engine=_stt_engine)
    return _stt_service

def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        root = get_repo_root()

        model_path = Path(settings.PIPER_MODEL_PATH)
        if not model_path.is_absolute():
            model_path = (root / model_path).resolve()

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found at: {model_path}\n"
                f"Repo root resolved to: {root}\n"
                f"settings.PIPER_MODEL_PATH was: {settings.PIPER_MODEL_PATH}\n"
                f"Expected files in: {root / 'models'}"
            )

        engine = PiperTTSEngine(model_path=str(model_path))
        _tts_service = TTSService(engine=engine)

    return _tts_service


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        root = get_repo_root()

        index_dir = Path(settings.RAG_INDEX_DIR)
        if not index_dir.is_absolute():
            index_dir = (root / index_dir).resolve()
        index_dir.mkdir(parents=True, exist_ok=True)

        rag_engine = RagEngine(
            index_dir=index_dir,
            ollama_base_url=settings.OLLAMA_BASE_URL,
            embed_model=getattr(settings, "RAG_EMBED_MODEL", "nomic-embed-text"),
            top_k=int(getattr(settings, "RAG_TOP_K", 3)),
        )

        llm_engine = LLMEngine(
            base_url=settings.OLLAMA_BASE_URL,
            request_timeout=int(getattr(settings, "OLLAMA_TIMEOUT", 120)),
        )

        _rag_service = RagService(rag_engine=rag_engine, llm_engine=llm_engine)

    return _rag_service