# Test for Reachy Mini opptak av lyd (lager wav som faktisk har lyd)
# Kjør: python record_test.py

from pathlib import Path
import time
import numpy as np
from scipy.io.wavfile import write
from reachy_mini import ReachyMini


def record_audio(recorded_audio_path: Path, duration_sec: float = 5.0) -> Path:
    recorded_audio_path.parent.mkdir(parents=True, exist_ok=True)

    with ReachyMini(media_backend="default") as reachy:
        # Start opptak + liten "warmup"
        reachy.media.start_recording()
        time.sleep(0.2)

        t0 = time.time()
        chunks = []

        while time.time() - t0 < duration_sec:
            sample = reachy.media.get_audio_sample()
            if sample is not None:
                chunks.append(sample)
            else:
                time.sleep(0.01)

        sample_rate = reachy.media.get_input_audio_samplerate()
        reachy.media.stop_recording()

    if not chunks:
        raise RuntimeError("No audio frames received from Reachy Mini mic")

    # Slå sammen frames
    samples = np.concatenate(chunks, axis=0)

    # Debug: se om vi får ekte signal eller bare nuller
    print("sample_rate:", sample_rate)
    print("raw dtype:", samples.dtype, "shape:", samples.shape)
    print("raw min/max:", float(np.min(samples)), float(np.max(samples)))

    # Vanligst: float32 i [-1, 1] -> må skaleres til int16 før wav
    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)
    else:
        samples = samples.astype(np.int16)

    print("int16 min/max:", int(samples.min()), int(samples.max()))

    # Lagre WAV
    write(recorded_audio_path, sample_rate, samples)
    return recorded_audio_path


def main():
    root = Path(__file__).resolve().parent
    out_path = root / "data" / "audio" / "recordings" / "reachy-test.wav"

    print(f"🎙️  Recorder {5} sekunder fra Reachy Mini...")
    path = record_audio(out_path, duration_sec=5.0)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Recording failed; file was not created or is empty")

    print("✅ Lagret:", path.resolve())
    print("Tips: Hvis 'raw min/max' er 0.0/0.0 -> sjekk macOS mic-permissions for Terminal/VS Code.")


if __name__ == "__main__":
    main()