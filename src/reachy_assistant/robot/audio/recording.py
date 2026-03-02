import time
from pathlib import Path
from scipy.io.wavfile import write as wav_write
import numpy as np
from reachy_mini import ReachyMini


def record_audio_wav(path: Path, duration_sec: float) -> tuple[Path, int]:
    from reachy_mini import ReachyMini
    path.parent.mkdir(parents=True, exist_ok=True)

    with ReachyMini(media_backend="default") as reachy:
        reachy.media.start_recording()
        time.sleep(0.2)  # warmup

        t0 = time.time()
        chunks = []
        while time.time() - t0 < duration_sec:
            sample = reachy.media.get_audio_sample()
            if sample is not None:
                chunks.append(sample)
            else:
                time.sleep(0.01)

        sr = int(reachy.media.get_input_audio_samplerate())
        reachy.media.stop_recording()

    if not chunks:
        raise RuntimeError("No audio frames received from Reachy Mini mic")

    audio = np.concatenate(chunks, axis=0)

    # typisk float32 [-1,1] -> int16 WAV
    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767.0).astype(np.int16)
    else:
        audio_i16 = audio.astype(np.int16)

    wav_write(str(path), sr, audio_i16)
    return path, sr