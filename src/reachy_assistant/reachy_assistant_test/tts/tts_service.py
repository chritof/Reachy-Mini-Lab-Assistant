from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTSService:
    engine: object

    def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""
        return self.engine.synthesize_wav_bytes(text)