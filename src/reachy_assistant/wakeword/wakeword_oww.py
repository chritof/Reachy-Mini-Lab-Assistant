import openwakeword
from openwakeword.model import Model
import time
import numpy as np
import sounddevice as sd

# one-time download
openwakeword.utils.download_models()

WAKEWORD_NAME = "alexa"
THRESHOLD = 0.65
SAMPLE_RATE = 16000
FRAME_MS = 20
COOLDOWN = 2.0


model = Model(
    inference_framework="onnx" #force ONNX instead of tensorflowlite, tflite is better for linux
)
print("Loaded models:", list(model.models.keys()))

frame_size = int(SAMPLE_RATE * FRAME_MS / 1000)
last_trigger = 0


def audio_callback(indata, frames, time_info, status):
    global last_trigger

    # convert float32 -> int16PCM (raw analog signals)
    audio = np.clip(indata[:, 0], -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    scores = model.predict(audio)
    score = scores.get(WAKEWORD_NAME, 0.0)

    now = time.time()
    if score > THRESHOLD and (now-last_trigger) > COOLDOWN:
        last_trigger = now
        print(f"Wakeword detected! score={score:.3f}")

print("Listening for wakeword...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=frame_size,
    channels=1,
    dtype="float32",
    callback=audio_callback,

):
    while True:
        time.sleep(0.1)