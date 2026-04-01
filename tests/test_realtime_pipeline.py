from dataclasses import dataclass

from reachy_assistant.realtime.pipeline import RealtimeConversationPipeline


@dataclass
class FakeEngine:
    runs: int = 0
    stopped: int = 0

    async def run(self) -> None:
        self.runs += 1

    def stop(self) -> None:
        self.stopped += 1


def test_realtime_pipeline_runs_engine() -> None:
    engine = FakeEngine()
    pipeline = RealtimeConversationPipeline(engine=engine)

    pipeline.run()

    assert engine.runs == 1
