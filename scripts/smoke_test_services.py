from pathlib import Path
from backend.app.api.deps import get_repo_root, get_stt_service, get_tts_service, get_rag_service

def main():
    root = get_repo_root()

    # STT
    wav = (root / "data" / "audio" / "recordings" / "reachy-test2.wav").resolve()
    if wav.exists():
        stt = get_stt_service()
        text = stt.transcribe_wav(wav)  # pass Path, not bytes
        print("STT:", text)
    else:
        print("STT: skipped (wav not found):", wav)

    # TTS
    tts = get_tts_service()
    audio = tts.synthesize("Hei! Dette er en test.")
    print(f"TTS: type={type(audio).__name__} bytes={len(audio)}")

    Path("test_output.wav").write_bytes(audio)
    print("TTS: wrote test_output.wav")

    # RAG + LLM
    try:
        rag = get_rag_service()
        ans = rag.ask("Hva er wifi-passordet i laben?")
        print("RAG used_query:", ans.used_query)
        print("RAG answer:", ans.answer)
    except Exception as e:
        print("RAG: skipped:", repr(e))

if __name__ == "__main__":
    main()