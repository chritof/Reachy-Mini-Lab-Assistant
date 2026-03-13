from __future__ import annotations

from pathlib import Path

import numpy as np
import whisper
from scipy.io.wavfile import write as wav_write


class WhisperEngine:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_16k: np.ndarray) -> str:
        if audio_16k.size == 0:
            return ""

        temp_path = Path("data/audio/recordings/_temp_stt.wav")
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        audio_i16 = np.clip(audio_16k, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
        wav_write(str(temp_path), 16000, audio_i16)

        result = self.model.transcribe(str(temp_path), language="no", fp16=False)
        return (result.get("text") or "").strip()

    def transcribe_wav(self, wav_path: Path) -> str:
        result = self.model.transcribe(str(wav_path), language="no", fp16=False)
        return (result.get("text") or "").strip()