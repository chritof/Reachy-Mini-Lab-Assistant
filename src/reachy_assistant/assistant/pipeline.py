from reachy_assistant.audio_io import recorder
from reachy_assistant.stt import stt_service
from reachy_assistant.llm.v5 import call_ollama
from reachy_assistant.stt.stt_service import transcribe_audio
from pathlib import Path

# foreløpig test av pipeline

ROOT = Path(__file__).resolve().parents[3]
audio_path = ROOT / "data" / "audio" / "test-audio5.m4a"

transcription = transcribe_audio(audio_path)
prompt = transcription["text"]

answer = call_ollama(prompt, context="")
print(answer)