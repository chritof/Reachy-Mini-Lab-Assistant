from pathlib import Path

from reachy_assistant_test.audio.recorder import AudioRecorder
from reachy_assistant_test.llm.ollama_llm_engine import OllamaLLMEngine
from reachy_assistant_test.rag.rag_engine import RagEngine
from reachy_assistant_test.rag.rag_service import RagService
from reachy_assistant_test.robot.controller import ReachyMotionController
from reachy_assistant_test.robot.pipeline.conversation_pipeline import ConversationPipeline
from reachy_assistant_test.stt.stt_service import STTService
from reachy_assistant_test.stt.whisper_engine import WhisperEngine
from reachy_assistant_test.tts.piper_tts_engine import PiperTTSEngine
from reachy_assistant_test.tts.tts_service import TTSService


def main():
    root = Path(__file__).resolve().parent
    index_dir = root / "data" / "rag_index"

    recorder = AudioRecorder(media_backend="default")

    stt_engine = WhisperEngine(model_name="base")
    stt_service = STTService(engine=stt_engine)

    llm_engine = OllamaLLMEngine(
        model="mistral:latest",
        base_url="http://localhost:11434",
        request_timeout=120,
    )

    rag_engine = RagEngine(
        index_dir=index_dir,
        embed_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
        top_k=5,
    )
    rag_service = RagService(rag_engine=rag_engine, llm_engine=llm_engine)

    tts_engine = PiperTTSEngine(
    model_path=str(root / "models" / "no_NO-talesyntese-medium.onnx"),
    config_path=str(root / "models" / "no_NO-talesyntese-medium.onnx.json"),
)
    tts_service = TTSService(engine=tts_engine)

    motion = ReachyMotionController()

    pipeline = ConversationPipeline(
        recorder=recorder,
        stt=stt_service,
        rag=rag_service,
        tts=tts_service,
        motion=motion,
        listen_duration_sec=6.0,
    )

    print("Reachy samtalepipeline er startet. Trykk Ctrl+C for å stoppe.")
    while True:
        pipeline.run_once()


if __name__ == "__main__":
    main()