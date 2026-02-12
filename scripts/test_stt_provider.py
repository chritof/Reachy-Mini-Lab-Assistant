from pathlib import Path
from reachy_assistant.providers.stt.whisper_local import WhisperLocalSTT

def main():
    root = Path(__file__).resolve().parents[1]
    audio = root / "data" / "audio" / "test-audio5.m4a"

    stt = WhisperLocalSTT()
    transcript = stt.transcribe_file(audio, language="no")
    print(transcript.text)
    print("hei")


if __name__ == '__main__':
    main()