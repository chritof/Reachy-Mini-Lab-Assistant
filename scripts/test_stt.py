from pathlib import Path
from reachy_assistant.stt.whisper_local import transcribe_file

# demo script

ROOT = Path(__file__).resolve().parents[3]  # stt -> reachy_assistant -> src -> repo root
AUDIO = ROOT / "data" / "audio" / "test-audio5.m4a"

if __name__ == "__main__":
    out = transcribe_file(AUDIO, language="no")
    print(out["text"])
