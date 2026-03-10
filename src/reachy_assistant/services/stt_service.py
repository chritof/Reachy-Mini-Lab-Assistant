"""
Service-lag for tale-til-tekst (STT).

Gir en enkel metode for å transkribere lyd.
Selve STT-logikken ligger i en egen engine-klasse.

Formål:
- Skille API fra modell-implementasjon
- Gjøre det enkelt å bytte STT-motor senere
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class STTService:
    engine: object

    def transcribe_wav(self, wav_path: Path) -> str:
        return self.engine.transcribe_wav(wav_path)
