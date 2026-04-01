from pathlib import Path

from reachy_assistant.robot.pipeline.conversation_pipeline import ConversationPipeline
from reachy_assistant.robot.pipeline.states import ConversationState


class FakeRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record_to_file(self) -> Path:
        return self.path


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe_wav(self, wav_path: Path) -> str:
        return self.text


class FakeRag:
    def ask(self, question: str):
        return type("RagAnswer", (), {"answer": f"Svar: {question}"})()


class FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return b"wav"


class FakePlayer:
    def __init__(self) -> None:
        self.play_count = 0

    def play_wav(self, wav_bytes: bytes) -> None:
        self.play_count += 1


class FakeMotion:
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


def test_pipeline_runs_full_turn(tmp_path: Path) -> None:
    tts = FakeTTS()
    player = FakePlayer()
    motion = FakeMotion()
    pipeline = ConversationPipeline(
        stt=FakeSTT("Hei Reachy"),
        rag=FakeRag(),
        tts=tts,
        recorder=FakeRecorder(tmp_path / "input.wav"),
        player=player,
        motion=motion,
        intro_text=None,
    )

    for _ in range(4):
        pipeline.run_once()

    assert pipeline.state == ConversationState.IDLE
    assert player.play_count == 1
    assert tts.calls == ["Svar: Hei Reachy"]
    assert motion.calls == ["idle", "listening", "thinking", "speaking"]


def test_pipeline_speaks_intro_once(tmp_path: Path) -> None:
    tts = FakeTTS()
    player = FakePlayer()
    pipeline = ConversationPipeline(
        stt=FakeSTT(""),
        rag=FakeRag(),
        tts=tts,
        recorder=FakeRecorder(tmp_path / "input.wav"),
        player=player,
        motion=None,
        intro_text="Hei der",
    )

    pipeline.run_once()
    pipeline.run_once()

    assert tts.calls == ["Hei der"]
    assert player.play_count == 1
