import time
import os
from pathlib import Path

from reachy_assistant.robot.pipeline.conversation_pipeline import ConversationPipeline

from reachy_assistant.services.api_stt_service import ApiSTTService
from reachy_assistant.services.api_rag_service import ApiRagService
from reachy_assistant.services.api_tts_service import ApiTTSService

from reachy_assistant.pc.audio.recorder import PcRecorder
from reachy_assistant.pc.audio.player import PcPlayer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "audio"


def main():

    recorder = PcRecorder(DATA_DIR / "input.wav")
    player = PcPlayer()

    stt = ApiSTTService()
    rag = ApiRagService()
    tts = ApiTTSService()

    pipeline = ConversationPipeline(
        recorder=recorder,
        stt=stt,
        rag=rag,
        tts=tts,
        player=player,
        motion=None
    )

    while True:
        pipeline.run_once()
        time.sleep(0.1)


if __name__ == "__main__":
    main()