"""
Service-lag for tekst-til-tale (TTS).

Har en enkel metode: synthesize(text)
som returnerer WAV-bytes.

Selve syntesen gjøres av en TTS-engine.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTSService:
    engine: object  # PiperTTSEngine

    def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""
        return self.engine.synthesize_wav_bytes(text)