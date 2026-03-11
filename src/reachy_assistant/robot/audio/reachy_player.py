from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass

import numpy as np
from reachy_mini import ReachyMini


def pcm16_to_float32(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32767.0


@dataclass
class ReachyPlayer:
    mini: object

    def play_wav(self, wav_bytes: bytes) -> None:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)

        if channels > 1:
            audio = audio.reshape(-1, channels)

        audio = pcm16_to_float32(audio)

        self.mini.media.start_playing()
        self.mini.media.push_audio_sample(audio)
        time.sleep(len(audio) / sample_rate)
        self.mini.media.stop_playing()