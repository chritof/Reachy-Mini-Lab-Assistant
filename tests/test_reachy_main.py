from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.reachy_main import _apply_reachy_lab_audio_defaults


def test_apply_reachy_lab_audio_defaults_sets_conservative_vad(monkeypatch) -> None:
    config = RealtimeConfig(api_key="test")

    _apply_reachy_lab_audio_defaults(config)

    assert config.vad_threshold == 0.72
    assert config.vad_silence_ms == 950
    assert config.vad_prefix_padding_ms == 180


def test_apply_reachy_lab_audio_defaults_overrides_existing_values() -> None:
    config = RealtimeConfig(
        api_key="test",
        vad_threshold=0.6,
        vad_silence_ms=700,
        vad_prefix_padding_ms=300,
    )

    _apply_reachy_lab_audio_defaults(config)

    assert config.vad_threshold == 0.72
    assert config.vad_silence_ms == 950
    assert config.vad_prefix_padding_ms == 180
