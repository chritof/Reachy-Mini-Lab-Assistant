from reachy_assistant.stt.faster_whisper_silero import FasterWhisperSileroSTT
from reachy_assistant.services.stt_service import STTService
from backend.app import settings

_stt_engine = None
_stt_service = None

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