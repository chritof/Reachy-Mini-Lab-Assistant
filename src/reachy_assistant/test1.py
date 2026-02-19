from pathlib import Path
from audio_io.recorder import record_audio
from stt.whisper_local import transcribe_file

ROOT = Path(__file__).resolve().parent
AUDIO_PATH = ROOT / "test_recording.wav"

recordedPath = record_audio(AUDIO_PATH, 5)
transcribeRes = transcribe_file(recordedPath)
print(transcribeRes["text"])