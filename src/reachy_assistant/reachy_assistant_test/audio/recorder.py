from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write as wav_write
from reachy_mini import ReachyMini


class AudioRecorder:
    def __init__(self, media_backend: str = "default"):
        self.media_backend = media_backend

    def record_audio_wav(self, path: Path, duration_sec: float) -> tuple[Path, int]:
        path.parent.mkdir(parents=True, exist_ok=True)

        with ReachyMini(media_backend=self.media_backend) as reachy:
            reachy.media.start_recording()
            time.sleep(0.2)

            t0 = time.time()
            chunks: list[np.ndarray] = []

            while time.time() - t0 < duration_sec:
                sample = reachy.media.get_audio_sample()
                if sample is not None:
                    chunks.append(sample)
                else:
                    time.sleep(0.01)

            sr = int(reachy.media.get_input_audio_samplerate())
            reachy.media.stop_recording()

        if not chunks:
            raise RuntimeError("No audio frames received from Reachy")

        audio = np.concatenate(chunks, axis=0)

        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        else:
            audio = audio.astype(np.int16)

        wav_write(str(path), sr, audio)
        return path, sr