from pathlib import Path
from stt_service import transcribe_audio

# demo script

ROOT = Path(__file__).resolve().parents[3]  # stt -> reachy_assistant -> src -> repo root
AUDIO = ROOT / "data" / "audio" / "test-audio5.m4a"

if __name__ == "__main__":
    out = transcribe_audio(AUDIO, language="no")
    print(out["text"])
