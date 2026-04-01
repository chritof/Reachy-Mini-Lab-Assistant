from pathlib import Path

from reachy_assistant.llm.llm_engine import LLMEngine
from reachy_assistant.pc.audio.player import PcPlayer
from reachy_assistant.pc.audio.recorder import PcRecorder
from reachy_assistant.rag.rag_engine import RagEngine
from reachy_assistant.robot.pipeline.conversation_pipeline import ConversationPipeline
from reachy_assistant.services.rag_service import RagService
from reachy_assistant.services.stt_service import STTService
from reachy_assistant.services.tts_service import TTSService
from reachy_assistant.stt.faster_whisper_silero import FasterWhisperSileroSTT
from reachy_assistant.tts.piper_tts import PiperTTSEngine

"""
Pipeline-logikk som bruker input og output fra PC.

Brukt for testing av pipeline.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    recorder = PcRecorder(out_path=DATA_DIR / "audio" / "input.wav")
    player = PcPlayer()

    stt = STTService(FasterWhisperSileroSTT())
    rag = RagService(RagEngine(index_dir=DATA_DIR / "rag_index"), LLMEngine())
    tts = TTSService(PiperTTSEngine(str(MODELS_DIR / "no_NO-talesyntese-medium.onnx")))

    pipeline = ConversationPipeline(
        recorder=recorder,
        stt=stt,
        rag=rag,
        tts=tts,
        player=player,
        motion=None,
    )
    pipeline.run_forever()


if __name__ == "__main__":
    main()
