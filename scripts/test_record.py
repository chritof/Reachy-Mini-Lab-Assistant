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
    out_path = Path("/tmp/reachy_record.wav")

    with ReachyMini(media_backend="default") as mini:
        sr_in = int(mini.media.get_input_audio_samplerate())
        print("Input samplerate:", sr_in)

        mini.media.start_recording()

        chunks = []
        t_end = time.time() + seconds
        while time.time() < t_end:
            samples = mini.media.get_audio_sample()  # numpy float array
            samples = np.asarray(samples, dtype=np.float32)
            chunks.append(samples)

        mini.media.stop_recording()

    audio = np.concatenate(chunks, axis=0)

    # Ensure shape (N, C)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    pcm = float32_to_pcm16(audio)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr_in)
        wf.writeframes(pcm.tobytes())

    print("Saved:", out_path)


if __name__ == "__main__":
    main()