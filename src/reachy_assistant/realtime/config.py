"""
Felles konfigurasjon for OpenAI Realtime-sesjonen og assistentprofilen.
"""

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
        "Du er Reachy, en norsk labassistent på Læringslabben ved HVL i Bergen. "
        "Svar kort, tydelig og hjelpsomt, vanligvis i 1-3 setninger. "
        "Du kan bruke litt lett robot-humor av og til, men hold den diskre og ikke i hvert svar. "
        "Svar på norsk som standard, men hvis brukeren tydelig snakker et annet språk, skal du svare på det språket. "
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
    suppress_input_while_speaking: bool = False
    input_resume_delay_ms: int = 0
    wake_phrase_enabled: bool = False
    wake_phrases: tuple[str, ...] = (
        "hei",
        "hey",
        "hallo",
        "hello",
        "hei reachy",
        "hey reachy",
        "hei richie",
        "hey richie",
        "hei ritchie",
        "hey ritchie",
        "hei reachie",
        "hey reachie",
    )
    wake_phrase_timeout_sec: float = 20.0
    sleep_phrases: tuple[str, ...] = (
        "stopp reachy",
        "stop reachy",
        "reachy stopp",
        "reachy stop",
        "reachy sov",
        "reachy sov nå",
        "reachy sov na",
        "reachy legg deg",
        "reachy gå og legg deg",
        "reachy ga og legg deg",
        "god natt reachy",
        "ha det",
        "ha det bra",
    )
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
                "You are Reachy, a concise Norwegian-speaking lab assistant for the Læringslab. "
                "Keep answers short, clear, and helpful, usually 1-3 sentences. "
                "You may use a small touch of robot humor occasionally, but keep it subtle. "
                "Reply in Norwegian by default, but if the user is clearly speaking another language, reply in that language instead. "
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
            suppress_input_while_speaking=os.getenv(
                "OPENAI_REALTIME_SUPPRESS_INPUT_WHILE_SPEAKING",
                "false",
            ).lower()
            in {"1", "true", "yes", "on"},
            input_resume_delay_ms=int(os.getenv("OPENAI_REALTIME_INPUT_RESUME_DELAY_MS", "0")),
            wake_phrase_enabled=os.getenv("OPENAI_REALTIME_WAKE_PHRASE_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            wake_phrases=tuple(
                part.strip().lower()
                for part in os.getenv(
                    "OPENAI_REALTIME_WAKE_PHRASES",
                    "hei,hey,hallo,hello,hei reachy,hey reachy,hei richie,hey richie,hei ritchie,hey ritchie,hei reachie,hey reachie",
                ).split(",")
                if part.strip()
            ),
            wake_phrase_timeout_sec=float(
                os.getenv("OPENAI_REALTIME_WAKE_PHRASE_TIMEOUT_SEC", "20.0")
            ),
            sleep_phrases=tuple(
                part.strip().lower()
                for part in os.getenv(
                    "OPENAI_REALTIME_SLEEP_PHRASES",
                    (
                        "stopp reachy,stop reachy,reachy stopp,reachy stop,reachy sov,"
                        "reachy sov nå,reachy sov na,reachy legg deg,reachy gå og legg deg,"
                        "reachy ga og legg deg,god natt reachy"
                    ),
                ).split(",")
                if part.strip()
            ),
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
