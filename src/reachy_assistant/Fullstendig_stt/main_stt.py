# main_stt.py

from pathlib import Path

from audio.recorder import record_audio_wav
from audio.preprocessing import load_wav_as_float32_mono_16k
from audio.vad import get_vad_segments, extract_voiced_audio
from stt.whisper_engine import WhisperEngine

DURATION_SEC = 60.0
TARGET_SR = 16000


def main():
    root = Path(__file__).resolve().parent
    wav_path = root / "data/audio/recordings/reachy_best.wav"

    print("Tar opp...")
    wav_path, orig_sr = record_audio_wav(wav_path, DURATION_SEC)

    audio_16k = load_wav_as_float32_mono_16k(wav_path, orig_sr)

    segments = get_vad_segments(audio_16k)
    voiced = extract_voiced_audio(audio_16k, segments)

    if voiced.size < TARGET_SR * 0.2:
        print("Ingen tale funnet.")
        return

    engine = WhisperEngine()
    text = engine.transcribe(voiced)

    print("\nResultat:")
    print(text)


if __name__ == "__main__":
    main()