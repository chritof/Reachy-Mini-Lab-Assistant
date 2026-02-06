import whisper
from pathlib import Path


# reusable STT logic to be used in main pipeline

_model = whisper.load_model("small")

# returns a dict, use result["text"] for transcript(string)
def transcribe_audio(audio_path: Path, language: str = "no") -> dict:
    return _model.transcribe(
        str(audio_path),
        language = language,
        fp16 = False
    )
