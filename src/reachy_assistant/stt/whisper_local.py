import whisper
from pathlib import Path


# reusable STT logic to be used in a provider

_model = whisper.load_model("large")

# returns a dict, use result["text"] for transcript(string)
def transcribe_file(audio_path: Path, language: str = "en") -> dict:
    return _model.transcribe(
        str(audio_path),
        language = language,
        fp16 = False
    )
