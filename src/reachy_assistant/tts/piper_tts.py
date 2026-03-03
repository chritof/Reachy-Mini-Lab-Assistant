"""
TTS-implementasjon med Piper.

Laster en ONNX stemmemodell
og konverterer tekst til WAV-lyd.

Inneholder lavnivå synteselogikk.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

from piper import PiperVoice


@dataclass
class PiperTTSEngine:
    model_path: str  # pass full path from deps.py

    def __post_init__(self) -> None:
        mp = Path(self.model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Piper model not found: {mp.resolve()}")
        self._voice = PiperVoice.load(str(mp))

    def synthesize_wav_bytes(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)

        return buf.getvalue()