from pathlib import Path

from reachy_mini import ReachyMini

from reachy_assistant.llm.llm_engine import LLMEngine
from reachy_assistant.rag.rag_engine import RagEngine
from reachy_assistant.robot.audio.reachy_player import ReachyPlayer
from reachy_assistant.robot.audio.reachy_recorder import ReachyRecorder
from reachy_assistant.robot.motion.motion_controller import ReachyMotionController
from reachy_assistant.robot.pipeline.conversation_pipeline import ConversationPipeline
from reachy_assistant.services.rag_service import RagService
from reachy_assistant.services.stt_service import STTService
from reachy_assistant.services.tts_service import TTSService
from reachy_assistant.stt.faster_whisper_silero import FasterWhisperSileroSTT
from reachy_assistant.tts.piper_tts import PiperTTSEngine

"""
Pipeline-logikk som bruker lokale ressurser + Reachy Mini mikrofon og høyttaler.
Kjøres koblet opp til Reachy Mini Lite med reachy-mini-daemon kjørende.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "audio"
MODELS_DIR = PROJECT_ROOT / "models"
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"


def main() -> None:
    with ReachyMini(connection_mode="auto") as mini:
        recorder = ReachyRecorder(mini=mini, out_path=DATA_DIR / "input.wav")
        player = ReachyPlayer(mini=mini)
        motion = ReachyMotionController(mini=mini)

        stt = STTService(FasterWhisperSileroSTT())
        rag = RagService(RagEngine(index_dir=RAG_INDEX_DIR), LLMEngine())
        tts = TTSService(
            PiperTTSEngine(model_path=str(MODELS_DIR / "no_NO-talesyntese-medium.onnx"))
        )

        pipeline = ConversationPipeline(
            stt=stt,
            rag=rag,
            tts=tts,
            recorder=recorder,
            player=player,
            motion=motion,
        )
        pipeline.run_forever()


if __name__ == "__main__":
    main()
