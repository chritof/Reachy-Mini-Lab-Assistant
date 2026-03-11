from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

"""
Klasse for å spille lyd fra pc.

Brukt for testing av pipeline-logikk.
"""
@dataclass
class PcPlayer:
    def play_wav(self, wav_bytes: bytes) -> None:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)

        if channels > 1:
            audio = audio.reshape(-1, channels)

        audio = audio.astype(np.float32) / 32767.0

        print("Playing audio on PC speakers...")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()