from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
"""
Klasse for ta opp lyd fra pc.

Brukt for testing av pipeline-logikk.
"""

@dataclass
class PcRecorder:
    out_path: Path
    duration_sec: float = 4.0
    sample_rate: int = 16000
    channels: int = 1

    def record_to_file(self) -> Path:
        print(f"Recording from PC mic for {self.duration_sec} seconds...")

        num_frames = int(self.duration_sec * self.sample_rate)

        audio = sd.rec(
            frames=num_frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()

        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(self.out_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())

        print(f"Saved recording to {self.out_path}")
        return self.out_path