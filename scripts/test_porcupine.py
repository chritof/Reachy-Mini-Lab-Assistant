import os
import time
import numpy as np
import sounddevice as sd
import pvporcupine
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")


ROOT = Path(__file__).resolve().parents[1]
KEYWORD_PATH = ROOT / "src" / "reachy_assistant" / "wakeword" / "models" / "hei-reachy.ppn"

if not KEYWORD_PATH.exists():
    raise FileNotFoundError(f"Keyword not found: {KEYWORD_PATH}")

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[str(KEYWORD_PATH)],
    sensitivities=[1.0],
)

SAMPLE_RATE = porcupine.sample_rate
FRAME_LENGTH = porcupine.frame_length

last_trigger = 0.0
COOLDOWN = 1.0

print("Listening for 'hei reachy'...)")

try:
    with sd.InputStream(
        device=sd.default.device[0],
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_LENGTH,
    ) as stream:
        while True:
            frames, _ = stream.read(FRAME_LENGTH)  # (FRAME_LENGTH, 1)
            pcm = (frames[:, 0] * 32768).astype(np.int16)

            result = porcupine.process(pcm)

            now = time.time()
            if result >= 0 and (now - last_trigger) > COOLDOWN:
                last_trigger = now
                print("Wakeword detected: hei reachy")

except KeyboardInterrupt:
    print("Stopped.")
finally:
    porcupine.delete()
