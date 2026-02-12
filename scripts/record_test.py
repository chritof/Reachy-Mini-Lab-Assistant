from pathlib import Path
from reachy_assistant.audio_io.recorder import record_audio

def main():
    ROOT = Path(__file__).resolve().parents[1]

    recorded_audio_path = ROOT / "data" / "audio" / "recordings" / "reachy-test.wav"
    recorded_audio_path.parent.mkdir(parents=True, exist_ok=True)

    print("recoding 5 seconds from reachys microphone...")
    path = record_audio(recorded_audio_path, duration_sec=5.0)

    if not path.exists():
        raise RuntimeError("recording failed; file was not created")

    print("saved: ", path.resolve())


if __name__ == "__main__":
    main()