from pathlib import Path
import time
import numpy as np
from scipy.io.wavfile import write
from reachy_mini import ReachyMini

# audio input logic for reuse in pipeline

def record_audio(recorded_audio_path: Path, duration_sec: float = 5.0,) -> Path:
    recorded_audio_path.parent.mkdir(parents=True, exist_ok=True)

    with ReachyMini(media_backend="default") as reachy:
        reachy.media.start_recording()
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

    samples = np.concatenate(chunks, axis=0)
    samples = np.asarray(samples, dtype=np.int16)
    write(recorded_audio_path, sample_rate, samples)
    return recorded_audio_path