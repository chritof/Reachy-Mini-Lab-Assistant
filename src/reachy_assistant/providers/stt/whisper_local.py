from pathlib import Path
from reachy_assistant.core.interfaces import STTEngine
from reachy_assistant.core.types import Transcript

from reachy_assistant.stt.whisper_local import transcribe_file


class WhisperLocalSTT(STTEngine):
    def transcribe_file(self, path: Path, language: str | None = None) -> Transcript:
        out = transcribe_file(path, language=language)      # dict
        return Transcript(text=out["text"])                 # standardized type