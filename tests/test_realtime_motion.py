from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.engine import RealtimeConversationEngine


class DummySource:
    sample_rate = 24000

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def chunks(self):
        if False:
            yield None


class DummySink:
    async def open(self, sample_rate: int) -> None:
        return None

    async def close(self) -> None:
        return None

    async def write(self, audio, sample_rate: int) -> None:
        return None


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
