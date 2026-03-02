from pathlib import Path

from backend.app.settings import settings

from reachy_assistant.stt.faster_whisper_silero import FasterWhisperSileroSTT
from reachy_assistant.services.stt_service import STTService
from reachy_assistant.services.rag_service import RagService
from reachy_assistant.rag.rag_engine import RagEngine
from reachy_assistant.llm.llm_engine import LLMEngine

_stt_engine = None
_stt_service = None
_rag_service = None


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


def get_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "backend").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        root = get_repo_root()
        index_dir = root / "data" / "rag_index"

        rag_engine = RagEngine(
            index_dir=index_dir,
            ollama_base_url=settings.OLLAMA_BASE_URL,
            embed_model=getattr(settings, "RAG_EMBED_MODEL", "nomic-embed-text"),
            top_k=int(getattr(settings, "RAG_TOP_K", 3)),
        )

        llm_engine = LLMEngine(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            request_timeout=int(getattr(settings, "OLLAMA_TIMEOUT", 120)),
        )

        _rag_service = RagService(rag_engine=rag_engine, llm_engine=llm_engine)

    return _rag_service