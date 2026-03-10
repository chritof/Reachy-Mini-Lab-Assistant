"""
Avspilling av lyd på Reachy Mini.

Konverterer WAV-bytes til lydsamples
og spiller dem via Reachy SDK.

Inneholder robot-spesifikk lydlogikk.
"""
from __future__ import annotations

import io
import wave
import time
import numpy as np


def wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sampwidth={sampwidth}")

    audio_i16 = np.frombuffer(frames, dtype=np.int16)
    audio = (audio_i16.astype(np.float32) / 32768.0)
    audio = audio.reshape(-1, nch)
    return audio, sr


def play_on_reachy(mini, wav_bytes: bytes) -> None:
    audio, sr = wav_bytes_to_float32(wav_bytes)

    mini.media.start_playing()
    out_sr = int(mini.media.get_output_audio_samplerate())

    # If rates differ, simplest is: don’t resample, just play anyway if it matches.
    # Your Piper output is 22050 Hz; many devices are 48000 Hz. If it mismatches, you need resampling.
    if sr != out_sr:
        # Lightweight resample without scipy: linear interpolation
        import numpy as np

        n = len(audio)
        new_n = int(n * out_sr / sr)
        x_old = np.linspace(0.0, 1.0, n, endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_n, endpoint=False)
        audio = np.vstack(
            [np.interp(x_new, x_old, audio[:, ch]) for ch in range(audio.shape[1])]
        ).T.astype(np.float32)

    mini.media.push_audio_sample(audio)

    duration_s = len(audio) / out_sr
    time.sleep(duration_s + 0.05)
    mini.media.stop_playing()