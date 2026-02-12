import time
import numpy as np
import pvporcupine
from dataclasses import dataclass
import os
import sounddevice as sd


@dataclass
class PorcupineConfig:
    access_key: str
    keyword_path: str
    sensitivity: float = 1.0
    cooldown_seconds: float = 1.0


class PorcupineWakeword:

    def __init__(self, config: PorcupineConfig):
        self._porcupine = pvporcupine.create(
            access_key=config.access_key,
            keyword_paths=[config.keyword_path],
            sensitivities=[config.sensitivity],
        )

        self.sample_rate = self._porcupine.sample_rate
        self.frame_length = self._porcupine.frame_length

        self._cooldown_seconds = config.cooldown_seconds
        self._last_trigger_time = 0.0

    def process_frame(self, frame: np.ndarray) -> bool:

        if frame.ndim == 2:
            frame = frame[:, 0]

        pcm = (frame * 32768).astype(np.int16)

        result = self._porcupine.process(pcm)

        now = time.time()
        if result >= 0 and (now - self._last_trigger_time) > self._cooldown_seconds:
            self._last_trigger_time = now
            return True

        return False

    def close(self):
        self._porcupine.delete()


def _test_wakeword():
    config = PorcupineConfig(
        access_key=os.environ["PICOVOICE_ACCESS_KEY"],
        keyword_path="models/hei-reachy.ppn"
    )

    detector = PorcupineWakeword(config)

    print("Listening for wakeowrd..")

    with sd.InputStream(
        samplerate=detector.sample_rate,
        blocksize=detector.frame_length,
        channels=1,
        dtype="float32",
    ) as stream:

        while True:
            frames, _ = stream.read(detector.frame_length)

            if detector.process_frame(frames):
                print("Wakeword detected😁😏👍💕😘👌😊😂🤣")

    detector.close()


if __name__ == "__main__":
    _test_wakeword()