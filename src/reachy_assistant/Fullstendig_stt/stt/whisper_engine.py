# stt/whisper_engine.py
# tar bare “tale-lyden” og gjør den om til tekst
from faster_whisper import WhisperModel

WHISPER_MODEL = "medium"
COMPUTE_TYPE = "int8"
LANGUAGE = "no"


class WhisperEngine:
    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type=COMPUTE_TYPE
        )

    def transcribe(self, audio_16k_voiced):
        if audio_16k_voiced.size == 0:
            return ""

        segments, info = self.model.transcribe(
            audio_16k_voiced,
            language=LANGUAGE,
            vad_filter=False,
            temperature=0.0,
            beam_size=5,
        )

        return " ".join(
            seg.text.strip()
            for seg in segments
            if seg.text
        ).strip()