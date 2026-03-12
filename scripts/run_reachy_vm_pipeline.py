import time

from reachy_mini import ReachyMini
from pathlib import Path

from reachy_assistant.robot.audio.reachy_recorder import ReachyRecorder
from reachy_assistant.robot.audio.reachy_player import ReachyPlayer

from reachy_assistant.robot.pipeline.conversation_pipeline import ConversationPipeline
from reachy_assistant.services.api_stt_service import ApiSTTService
from reachy_assistant.services.api_rag_service import ApiRagService
from reachy_assistant.services.api_tts_service import ApiTTSService

from reachy_assistant.stt.faster_whisper_silero import FasterWhisperSileroSTT
from reachy_assistant.llm.llm_engine import LLMEngine
from reachy_assistant.rag.rag_engine import RagEngine
from reachy_assistant.tts.piper_tts import PiperTTSEngine
"""
Pipeline-logikk som bruker modeller kjørende på backend.
Til bruk på Reachy Mini.
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "audio"
MODELS_DIR = PROJECT_ROOT / "models"
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"

def main():

    with ReachyMini(connection_mode="auto") as mini:
        recorder = ReachyRecorder(mini=mini,out_path=DATA_DIR / "input.wav")
        player = ReachyPlayer(mini=mini)

        stt = ApiSTTService()

        rag = ApiRagService()

        tts = ApiTTSService()

        pipeline = ConversationPipeline(
            recorder=recorder,
            stt=stt,
            rag=rag,
            tts=tts,
            player=player,
            motion=None,
        )

        while True:
            pipeline.run_once()
            time.sleep(0.1)


if __name__ == "__main__":
    main()



