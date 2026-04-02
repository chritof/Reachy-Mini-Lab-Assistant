from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RealtimeConfig:
    api_key: str
    model: str = "gpt-realtime"
    voice: str = "ballad"
    instructions: str = (
        "Du er Reachy, en norsk labassistent på Læringslabben ved HVL. "
        "Du hjelper brukere av labben med utstyr og spørsmål knyttet til læringslabben. "
        "For spørsmål om utlån, regler, rom, utstyr, tilgjengelighet eller hvordan noe brukes, "
        "skal du bruke dokumentasjonstoolen når den er tilgjengelig og basere svaret på den. "
        "I denne modusen har du bare tilgang til tale, ikke kamera eller bilder. "
        "Du må aldri late som om du kan se brukeren, rommet eller objekter foran deg. "
        "Hvis noen spør hva du ser, hva de holder, eller hva som ligger foran deg, "
        "skal du svare at du ikke kan se det i denne modusen."
    )
    language: str = "no"
    transcription_model: str = "gpt-4o-mini-transcribe"
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    chunk_ms: int = 100
    vad_threshold: float = 0.5
    vad_silence_ms: int = 700
    vad_prefix_padding_ms: int = 300
    vad_create_response: bool = True
    allow_interruptions: bool = True
    temperature: float = 0.8
    max_output_tokens: str | int = "inf"

    @classmethod
    def from_env(cls) -> "RealtimeConfig":
        cls._load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for the realtime pipeline. "
                "Set it in the environment or in the project's .env file."
            )

        max_output_tokens_raw = os.getenv("OPENAI_REALTIME_MAX_OUTPUT_TOKENS", "inf").strip()
        if max_output_tokens_raw.isdigit():
            max_output_tokens: str | int = int(max_output_tokens_raw)
        else:
            max_output_tokens = "inf"

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime").strip(),
            voice=os.getenv("OPENAI_REALTIME_VOICE", "alloy").strip(),
            instructions=os.getenv(
                "OPENAI_REALTIME_INSTRUCTIONS",
                "You are Reachy, a concise Norwegian-speaking lab assistant for the Learning Lab. "
                "Use the documentation search tool for factual questions about equipment, loan rules, rooms, "
                "availability, and how to use lab resources whenever it is available. "
                "In this mode you only have audio, not camera or image input. "
                "Do not claim to see the user, the room, or any object. "
                "If asked visual questions, say that vision is not enabled in this mode.",
            ).strip(),
            language=os.getenv("OPENAI_REALTIME_LANGUAGE", "no").strip(),
            transcription_model=os.getenv(
                "OPENAI_REALTIME_TRANSCRIPTION_MODEL",
                "gpt-4o-mini-transcribe",
            ).strip(),
            input_sample_rate=int(os.getenv("OPENAI_REALTIME_INPUT_SAMPLE_RATE", "24000")),
            output_sample_rate=int(os.getenv("OPENAI_REALTIME_OUTPUT_SAMPLE_RATE", "24000")),
            chunk_ms=int(os.getenv("OPENAI_REALTIME_CHUNK_MS", "100")),
            vad_threshold=float(os.getenv("OPENAI_REALTIME_VAD_THRESHOLD", "0.5")),
            vad_silence_ms=int(os.getenv("OPENAI_REALTIME_VAD_SILENCE_MS", "700")),
            vad_prefix_padding_ms=int(os.getenv("OPENAI_REALTIME_VAD_PREFIX_PADDING_MS", "300")),
            vad_create_response=os.getenv("OPENAI_REALTIME_VAD_CREATE_RESPONSE", "true").lower()
            in {"1", "true", "yes", "on"},
            allow_interruptions=os.getenv("OPENAI_REALTIME_ALLOW_INTERRUPTIONS", "true").lower()
            in {"1", "true", "yes", "on"},
            temperature=float(os.getenv("OPENAI_REALTIME_TEMPERATURE", "0.8")),
            max_output_tokens=max_output_tokens,
        )

    @property
    def chunk_frames(self) -> int:
        return max(1, int(self.input_sample_rate * self.chunk_ms / 1000))

    @staticmethod
    def _load_dotenv() -> None:
        dotenv_path = RealtimeConfig._find_dotenv()
        if dotenv_path is None:
            return

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue

            cleaned = value.strip().strip("\"'")
            os.environ[key] = cleaned

    @staticmethod
    def _find_dotenv() -> Path | None:
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / ".env"
            if candidate.is_file():
                return candidate
        return None
