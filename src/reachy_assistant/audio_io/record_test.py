from pathlib import Path
from recorder import record_audio

if __name__ == "__main__":
    recorded_audio_path = Path("data/audio/reachy_test.wav")
    record_audio(recorded_audio_path, duration_sec=5.0)
    print("Saved:", recorded_audio_path)