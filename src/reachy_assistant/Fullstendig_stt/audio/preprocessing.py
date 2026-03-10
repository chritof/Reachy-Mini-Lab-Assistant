# audio/preprocessing.py
# leser WAV, gjør den om til “Whisper-vennlig” lyd (mono, float32, 16kHz)
import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_wav_as_float32_mono_16k(wav_path, orig_sr):
    sr, audio = wav_read(str(wav_path))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    audio = audio - float(np.mean(audio))

    if orig_sr != TARGET_SR:
        g = np.gcd(orig_sr, TARGET_SR)
        up = TARGET_SR // g
        down = orig_sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)

    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0:
        audio = np.clip(audio / peak * 0.9, -1.0, 1.0)

    return audio