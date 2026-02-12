# mega vibe code det her, kun for test haha😁😏
# bbruker reachy media, istedet for sounddevice

import os
import time
from pathlib import Path
from collections import deque

import numpy as np
from dotenv import load_dotenv
from scipy.io.wavfile import write

import pvporcupine
from reachy_mini import ReachyMini


def to_int16_mono(x: np.ndarray) -> np.ndarray:
    """
    Convert Reachy audio samples to int16 mono 1D array.
    Handles float32 [-1,1] or int16.
    """
    x = np.asarray(x)

    # If shape is (N, 1) -> (N,)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]

    # If float, assume [-1, 1]
    if x.dtype.kind == "f":
        x = np.clip(x, -1.0, 1.0)
        x = (x * 32767.0).astype(np.int16)
    else:
        x = x.astype(np.int16, copy=False)

    return x


def main():
    load_dotenv()

    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("PICOVOICE_ACCESS_KEY is not set (env or .env).")

    # repo root from scripts/
    root = Path(__file__).resolve().parents[1]

    # Store recordings here (gitignore this folder)
    out_dir = root / "data" / "audio" / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)

    keyword_path = root / "src" / "reachy_assistant" / "wakeword" / "models" / "hei-reachy.ppn"
    if not keyword_path.exists():
        raise FileNotFoundError(f"Wakeword model not found: {keyword_path}")

    # Build Porcupine
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[str(keyword_path)],
        sensitivities=[1.0],
    )

    frame_len = porcupine.frame_length
    porcupine_sr = porcupine.sample_rate

    # Tuning
    pre_roll_seconds = 1.0   # audio kept before wakeword fires
    post_roll_seconds = 4.0  # audio recorded after wakeword fires
    cooldown_seconds = 1.0

    pending = np.zeros((0,), dtype=np.int16)  # accumulator for exact frame_len chunks
    ring = deque(maxlen=int(pre_roll_seconds * porcupine_sr))  # pre-roll buffer
    last_trigger = 0.0

    print("Starting Reachy audio + Porcupine wakeword...")
    print(f"Porcupine: sr={porcupine_sr}, frame_len={frame_len}")
    print("Listening for wakeword... (Ctrl+C to stop)")

    try:
        with ReachyMini(media_backend="default") as reachy:
            # Important: use media.start/stop consistently
            reachy.media.start_recording()
            reachy_sr = reachy.media.get_input_audio_samplerate()
            print(f"Reachy input samplerate: {reachy_sr}")

            if reachy_sr != porcupine_sr:
                # For tomorrow testing: fail fast so behavior is predictable.
                # If you hit this, tell me the reachy_sr value and we’ll add resampling.
                raise RuntimeError(
                    f"Sample rate mismatch: Reachy={reachy_sr}, Porcupine={porcupine_sr}. "
                    "Need resampling or correct input samplerate."
                )

            while True:
                sample = reachy.media.get_audio_sample()
                if sample is None:
                    time.sleep(0.01)
                    continue

                chunk = to_int16_mono(sample)
                if chunk.size == 0:
                    continue

                # Add to pre-roll ring buffer
                ring.extend(chunk.tolist())

                # Accumulate until we can form exact porcupine frames
                pending = np.concatenate([pending, chunk])

                while pending.size >= frame_len:
                    frame = pending[:frame_len]
                    pending = pending[frame_len:]

                    result = porcupine.process(frame)
                    now = time.time()

                    if result >= 0 and (now - last_trigger) > cooldown_seconds:
                        last_trigger = now
                        print("Wakeword detected!")

                        # Build audio to save: pre-roll + post-roll
                        pre = np.array(ring, dtype=np.int16)

                        # Collect post-roll seconds
                        needed = int(post_roll_seconds * porcupine_sr)
                        post_parts = []
                        collected = 0
                        t0 = time.time()

                        while collected < needed:
                            s2 = reachy.media.get_audio_sample()
                            if s2 is None:
                                time.sleep(0.01)
                                # safety to avoid infinite hang
                                if time.time() - t0 > (post_roll_seconds + 2.0):
                                    break
                                continue

                            c2 = to_int16_mono(s2)
                            if c2.size == 0:
                                continue

                            take = min(c2.size, needed - collected)
                            post_parts.append(c2[:take])
                            collected += take

                        post = np.concatenate(post_parts) if post_parts else np.zeros((0,), dtype=np.int16)
                        audio = np.concatenate([pre, post])

                        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
                        out_path = out_dir / f"reachy_wakeword_{ts}.wav"
                        write(out_path, porcupine_sr, audio)
                        print(f"Saved: {out_path}")

                        # Clear buffers so the next trigger starts fresh
                        ring.clear()
                        pending = np.zeros((0,), dtype=np.int16)

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        porcupine.delete()


if __name__ == "__main__":
    main()
