import asyncio

import numpy as np

from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.client import OpenAIRealtimeClient
from reachy_assistant.realtime.engine import RealtimeConversationEngine
from reachy_assistant.realtime.io_base import RealtimeSessionRestart
from reachy_assistant.realtime.openwakeword_source import OpenWakewordAudioSource
from reachy_assistant.robot.motion import gestures


class DummySource:
    sample_rate = 24000
    starts_asleep = False

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def chunks(self):
        if False:
            yield None

    def is_awake(self) -> bool:
        return True


class DummySink:
    def __init__(self) -> None:
        self.cleared = 0

    async def open(self, sample_rate: int) -> None:
        return None

    async def close(self) -> None:
        return None

    async def write(self, audio, sample_rate: int) -> None:
        return None

    async def clear(self) -> None:
        self.cleared += 1


class DummyMotion:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def idle(self) -> None:
        self.calls.append("idle")

    def listening(self) -> None:
        self.calls.append("listening")

    def thinking(self) -> None:
        self.calls.append("thinking")

    def speaking(self) -> None:
        self.calls.append("speaking")

    def acknowledge(self) -> None:
        self.calls.append("acknowledge")

    def sleeping(self) -> None:
        self.calls.append("sleeping")


def test_motion_calls_dispatch_methods() -> None:
    motion = DummyMotion()
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=DummySource(),
        sink=DummySink(),
        client=object(),
        motion=motion,
    )

    for name in ["idle", "listening", "thinking", "speaking", "acknowledge"]:
        engine._motion_call(name)

    assert motion.calls == ["idle", "listening", "thinking", "speaking", "acknowledge"]


class FakeMini:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def goto_target(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_waking_uses_explicit_safe_sequence() -> None:
    mini = FakeMini()

    gestures.waking(mini)

    assert [call["duration"] for call in mini.calls] == [1.0, 0.6, 0.6]
    assert mini.calls[0]["antennas"] == [0.0, 0.0]
    assert mini.calls[-1]["antennas"] == [0.0, 0.0]


def test_openwakeword_callback_errors_do_not_crash() -> None:
    source = OpenWakewordAudioSource(base_source=DummySource(), model_path="model.onnx")

    def explode() -> None:
        raise RuntimeError("boom")

    source._run_callback(explode, "wake")


def test_openwakeword_sleep_model_resets_to_sleep() -> None:
    class FakeModel:
        def reset(self) -> None:
            return None

    source = OpenWakewordAudioSource(
        base_source=DummySource(),
        model_path="hei_reachee.onnx",
        sleep_model_path="Stop_20260303_000650.onnx",
    )
    source._model = FakeModel()
    source._active = True
    calls: list[str] = []
    source.on_sleep = lambda: calls.append("sleep")

    def fake_detect(_chunk, model_name, _threshold):
        return model_name == "Stop_20260303_000650"

    source._detect_model = fake_detect

    async def chunks():
        yield np.ones(2400, dtype=np.float32)

    source.base_source.chunks = chunks

    async def run_once() -> None:
        async for _ in source.chunks():
            pass

    try:
        asyncio.run(run_once())
    except RealtimeSessionRestart as exc:
        assert "Sleep word detected" in str(exc)
    else:
        raise AssertionError("sleep word should restart the realtime session")

    assert calls == ["sleep"]
    assert source._active is False


def test_openwakeword_does_not_forward_wake_audio() -> None:
    class FakeModel:
        def reset(self) -> None:
            return None

    source = OpenWakewordAudioSource(
        base_source=DummySource(),
        model_path="hei_reachee.onnx",
    )
    source._model = FakeModel()
    wake_calls: list[str] = []
    source.on_wake = lambda: wake_calls.append("wake")
    chunks_to_send = [
        np.ones(2400, dtype=np.float32),
        np.full(2400, 2.0, dtype=np.float32),
    ]

    def fake_detect(_chunk, model_name, _threshold):
        return model_name == "hei_reachee" and not source._active

    source._detect_model = fake_detect

    async def chunks():
        for chunk in chunks_to_send:
            yield chunk

    source.base_source.chunks = chunks

    async def collect():
        outputs = []
        async for chunk in source.chunks():
            outputs.append(chunk)
        return outputs

    outputs = asyncio.run(collect())

    assert wake_calls == ["wake"]
    assert len(outputs) == 1
    assert np.array_equal(outputs[0], chunks_to_send[1])


def test_sleep_phrase_matching_requires_configured_phrase() -> None:
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=DummySource(),
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
    )

    assert engine._is_sleep_phrase("Reachy stopp")
    assert engine._is_sleep_phrase("god natt, Reachy")
    assert not engine._is_sleep_phrase("kan du stoppe musikken")


def test_tool_definitions_include_sleep_tool() -> None:
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=DummySource(),
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
        rag_tool=None,
    )

    names = [tool["name"] for tool in engine._tool_definitions()]

    assert "go_to_sleep" in names


def test_realtime_client_uses_ga_realtime_resource() -> None:
    client = OpenAIRealtimeClient(RealtimeConfig(api_key="test"))

    assert client.connect().__class__.__module__.startswith("openai.resources.realtime")


def test_session_config_uses_current_realtime_shape() -> None:
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=DummySource(),
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
        rag_tool=None,
    )

    session = engine._session_config()

    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    assert "modalities" not in session
    assert "input_audio_format" not in session
    assert "temperature" not in session
    assert session["max_output_tokens"] == "inf"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["input"]["turn_detection"]["type"] == "server_vad"


def test_enter_sleep_mode_resets_source_and_motion() -> None:
    class ResettableSource(DummySource):
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset_session_state(self) -> None:
            self.reset_calls += 1

    source = ResettableSource()
    sink = DummySink()
    motion = DummyMotion()
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=source,
        sink=sink,
        client=object(),
        motion=motion,
    )

    asyncio.run(engine._enter_sleep_mode())

    assert source.reset_calls == 1
    assert sink.cleared == 1
    assert motion.calls == ["sleeping"]


def test_enter_sleep_mode_restarts_for_wakeword_sources() -> None:
    class ResettableSleepSource(DummySource):
        starts_asleep = True

        def __init__(self) -> None:
            self.reset_calls = 0
            self.awake = True

        def reset_session_state(self) -> None:
            self.reset_calls += 1
            self.awake = False

        def is_awake(self) -> bool:
            return self.awake

    source = ResettableSleepSource()
    sink = DummySink()
    motion = DummyMotion()
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=source,
        sink=sink,
        client=object(),
        motion=motion,
    )

    try:
        asyncio.run(engine._enter_sleep_mode())
    except RealtimeSessionRestart as exc:
        assert "Sleep requested" in str(exc)
    else:
        raise AssertionError("wakeword-controlled sleep should restart the realtime session")

    assert source.reset_calls == 1
    assert sink.cleared == 1
    assert motion.calls == ["sleeping"]
    assert source.is_awake() is False


def test_sleep_phrase_sets_skip_response_flag() -> None:
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=DummySource(),
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
    )

    engine._skip_next_response_create = True
    engine._reset_conversation_state()

    assert engine._skip_next_response_create is False


def test_wake_state_uses_source_when_source_controls_wake() -> None:
    class SleepAwareSource(DummySource):
        starts_asleep = True

        def __init__(self, awake: bool) -> None:
            self.awake = awake

        def is_awake(self) -> bool:
            return self.awake

    asleep_engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=SleepAwareSource(awake=False),
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
    )
    awake_engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=SleepAwareSource(awake=True),
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
    )

    assert asleep_engine._wake_is_active() is False
    assert awake_engine._wake_is_active() is True


def test_keep_source_awake_if_busy_touches_activity() -> None:
    class BusySource(DummySource):
        starts_asleep = True

        def __init__(self) -> None:
            self.touched = 0

        def touch_activity(self) -> None:
            self.touched += 1

        def is_awake(self) -> bool:
            return True

    source = BusySource()
    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=source,
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
    )
    engine._response_create_pending = True

    engine._keep_source_awake_if_busy()

    assert source.touched == 1


def test_should_hold_sleep_pose_when_source_is_asleep() -> None:
    class SleepAwareSource(DummySource):
        starts_asleep = True

        def is_awake(self) -> bool:
            return False

    engine = RealtimeConversationEngine(
        config=RealtimeConfig(api_key="test"),
        source=SleepAwareSource(),
        sink=DummySink(),
        client=object(),
        motion=DummyMotion(),
    )

    assert engine._should_hold_sleep_pose() is True
