from dataclasses import dataclass
from pathlib import Path
import time
import wave
import numpy as np


def float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


@dataclass
class ReachyRecorder:
    mini: object
    out_path: Path
    duration_sec: float = 4.0

    def record_to_file(self) -> Path:
        sr_in = int(self.mini.media.get_input_audio_samplerate())
        self.mini.media.start_recording()

        chunks: list[np.ndarray] = []
        t_end = time.time() + self.duration_sec

        try:
            while time.time() < t_end:
                samples = self.mini.media.get_audio_sample()

                if samples is None:
                    continue

                samples = np.asarray(samples, dtype=np.float32)

                if samples.ndim == 0 or samples.size == 0:
                    continue

                # If multichannel, force mono by taking first channel
                if samples.ndim > 1:
                    samples = samples[:, 0]
                else:
                    samples = samples.flatten()

                chunks.append(samples)

        finally:
            self.mini.media.stop_recording()

        if not chunks:
            raise RuntimeError("No valid audio chunks captured from Reachy.")

        audio = np.concatenate(chunks, axis=0)
        pcm = float32_to_pcm16(audio)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(self.out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr_in)
            wf.writeframes(pcm.tobytes())

        return self.out_path