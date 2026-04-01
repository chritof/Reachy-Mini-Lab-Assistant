from __future__ import annotations

import base64
from math import gcd

import numpy as np
from scipy.signal import resample_poly


REALTIME_SAMPLE_RATE = 24000


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr[:, 0].astype(np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    mono = ensure_mono(audio)
    if source_rate == target_rate:
        return mono.astype(np.float32, copy=False)
    ratio = gcd(source_rate, target_rate)
    up = target_rate // ratio
    down = source_rate // ratio
    return resample_poly(mono, up, down).astype(np.float32)


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    mono = np.clip(ensure_mono(audio), -1.0, 1.0)
    return (mono * 32767.0).astype(np.int16).tobytes()


def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0


def to_base64_pcm16(audio: np.ndarray, source_rate: int, target_rate: int = REALTIME_SAMPLE_RATE) -> str:
    resampled = resample_audio(audio, source_rate, target_rate)
    pcm = float32_to_pcm16_bytes(resampled)
    return base64.b64encode(pcm).decode("ascii")


def from_base64_pcm16(payload: str, sample_rate: int = REALTIME_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    audio_bytes = base64.b64decode(payload)
    return pcm16_bytes_to_float32(audio_bytes), sample_rate
