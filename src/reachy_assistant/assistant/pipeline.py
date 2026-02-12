from reachy_assistant.audio_io import recorder
from reachy_assistant.stt import whisper_local
from reachy_assistant.llm.v5 import call_ollama
from reachy_assistant.stt.whisper_local import transcribe_file
from pathlib import Path

# foreløpig test av pipeline

ROOT = Path(__file__).resolve().parents[3]
audio_path = ROOT / "data" / "audio" / "test-audio5.m4a"

transcription = transcribe_file(audio_path)
prompt = transcription["text"]

answer = call_ollama(prompt, context="")
print(answer)