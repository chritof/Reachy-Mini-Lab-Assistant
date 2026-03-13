import io
import wave
import numpy as np
import time
from dataclasses import dataclass


def pcm16_to_float32(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32767.0


@dataclass
class ReachyPlayer:
    mini: object
    chunk_size: int = 2048

    def play_wav(self, wav_bytes: bytes) -> None:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)

        # reshape if stereo
        if channels > 1:
            audio = audio.reshape(-1, channels)
            audio = audio[:, 0]  # force mono

        audio = pcm16_to_float32(audio)
        audio = audio.astype(np.float32).flatten()

        self.mini.media.start_playing()

        # stream chunks instead of one large push
        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            self.mini.media.push_audio_sample(chunk)
            time.sleep(len(chunk) / sample_rate)

        self.mini.media.stop_playing()