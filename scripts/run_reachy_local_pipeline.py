import argparse
from pathlib import Path

from reachy_mini import ReachyMini

from reachy_assistant.llm.llm_engine import LLMEngine
from reachy_assistant.llm.openai_chatgpt_engine import OpenAIChatGPTEngine
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the original Reachy conversation pipeline.")
    parser.add_argument("--llm-backend", choices=["ollama", "openai"], default="openai")
    return parser


def build_llm_engine(backend: str):
    if backend == "openai":
        return OpenAIChatGPTEngine()
    return LLMEngine()


def main() -> None:
    args = build_parser().parse_args()

    print(f"[local-pipeline] Starting with llm_backend={args.llm_backend}")
    print("[local-pipeline] Connecting to Reachy Mini...")
    with ReachyMini(connection_mode="auto") as mini:
        print("[local-pipeline] Reachy connected.")
        print("[local-pipeline] Initializing audio and motion interfaces...")
        recorder = ReachyRecorder(mini=mini, out_path=DATA_DIR / "input.wav")
        player = ReachyPlayer(mini=mini)
        motion = ReachyMotionController(mini=mini)

        print("[local-pipeline] Loading STT model...")
        stt = STTService(FasterWhisperSileroSTT())
        print("[local-pipeline] Loading RAG and LLM...")
        rag = RagService(RagEngine(index_dir=RAG_INDEX_DIR), build_llm_engine(args.llm_backend))
        print("[local-pipeline] Loading TTS...")
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
        print("[local-pipeline] Pipeline ready. Waiting for speech.")
        pipeline.run_forever()


if __name__ == "__main__":
    main()
