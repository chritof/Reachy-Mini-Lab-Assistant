from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class STTService:
    engine: object

    def transcribe_wav(self, wav_path: Path) -> str:
        return self.engine.transcribe_wav(wav_path)

    def transcribe_audio_array(self, audio_16k):
        return self.engine.transcribe(audio_16k)