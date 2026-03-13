from __future__ import annotations

import tempfile
from pathlib import Path
from scipy.io.wavfile import write as wav_write
import numpy as np
import simpleaudio as sa


class ReachyMotionController:
    def idle(self):
        print("[MOTION] Reachy er idle")

    def listening_pose(self):
        print("[MOTION] Reachy går til listening-pose")

    def thinking_pose(self):
        print("[MOTION] Reachy går til thinking-pose")

    def speaking_pose(self):
        print("[MOTION] Reachy går til speaking-pose")

    def play_wav_bytes_blocking(self, wav_bytes: bytes):
        if not wav_bytes:
            return

        temp_path = Path(tempfile.gettempdir()) / "reachy_tts_playback.wav"
        temp_path.write_bytes(wav_bytes)

        # simpleaudio leser PCM/WAV fra fil via WaveObject
        wave_obj = sa.WaveObject.from_wave_file(str(temp_path))
        play_obj = wave_obj.play()
        play_obj.wait_done()