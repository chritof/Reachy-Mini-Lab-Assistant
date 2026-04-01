import base64

import numpy as np

from reachy_assistant.realtime.audio import (
    REALTIME_SAMPLE_RATE,
    from_base64_pcm16,
    pcm16_bytes_to_float32,
    to_base64_pcm16,
)


def test_pcm16_roundtrip_preserves_shape() -> None:
    audio = np.linspace(-0.5, 0.5, num=4800, dtype=np.float32)
    payload = to_base64_pcm16(audio, REALTIME_SAMPLE_RATE, REALTIME_SAMPLE_RATE)

    decoded, sample_rate = from_base64_pcm16(payload, REALTIME_SAMPLE_RATE)

    assert sample_rate == REALTIME_SAMPLE_RATE
    assert decoded.shape == audio.shape
    assert np.allclose(decoded[:50], audio[:50], atol=5e-4)


def test_pcm16_decoder_handles_raw_bytes() -> None:
    raw = (np.array([0, 1000, -1000], dtype=np.int16)).tobytes()
    decoded = pcm16_bytes_to_float32(raw)

    assert decoded.shape == (3,)
    assert decoded[0] == 0.0
    assert decoded[1] > 0.0
    assert decoded[2] < 0.0


def test_base64_payload_is_ascii() -> None:
    audio = np.zeros(240, dtype=np.float32)
    payload = to_base64_pcm16(audio, REALTIME_SAMPLE_RATE, REALTIME_SAMPLE_RATE)

    assert isinstance(payload, str)
    assert base64.b64decode(payload) == b"\x00\x00" * 240
