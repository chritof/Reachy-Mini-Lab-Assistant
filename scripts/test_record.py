from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
from reachy_mini import ReachyMini


def float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def main():
    seconds = 4.0
    out_path = Path("reachy_record.wav")

    with ReachyMini(media_backend="default") as mini:
        sr_in = int(mini.media.get_input_audio_samplerate())
        in_channels = int(mini.media.get_input_channels())
        print("Input samplerate:", sr_in)
        print("Input channels:", in_channels)

        mini.media.start_recording()

        chunks: list[np.ndarray] = []
        t_end = time.time() + seconds

        while time.time() < t_end:
            samples = mini.media.get_audio_sample()

            # Skip missing chunks
            if samples is None:
                continue

            samples = np.asarray(samples, dtype=np.float32)

            # Skip scalar / invalid chunks
            if samples.ndim == 0 or samples.size == 0:
                continue

            # Normalize shape to (N, C)
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)

            chunks.append(samples)

        mini.media.stop_recording()

    if not chunks:
        raise RuntimeError(
            "No valid audio chunks were captured. "
            "The mic may be returning empty data or silence."
        )

    audio = np.concatenate(chunks, axis=0)
    pcm = float32_to_pcm16(audio)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr_in)
        wf.writeframes(pcm.tobytes())

    print("Saved:", out_path)
    print("Audio shape:", audio.shape)
    print("Peak abs level:", float(np.max(np.abs(audio))))


if __name__ == "__main__":
    main()