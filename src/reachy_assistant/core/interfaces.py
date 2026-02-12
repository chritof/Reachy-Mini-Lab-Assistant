from typing import Protocol
from pathlib import Path
from .types import Transcript

# contracts, they are implemented in providers

class STTEngine(Protocol):
    def transcribe_file(self, path: Path, language: str | None = None) -> Transcript: ...

class WakewordEngine(Protocol):
    def process_frame(self, frame: bytes, sample_rate: int) -> bool:
        """Feed audio chunk; return True when wakeword detected"""

    def reset(self) -> None: ...

class Recorder(Protocol):
    def record_to_wav(self, out_path: Path, duration_sec: float) -> Path:
        """Records audio and writes a WAV file, returns the written path"""