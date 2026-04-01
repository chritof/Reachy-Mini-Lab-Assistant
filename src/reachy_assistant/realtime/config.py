from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class RealtimeConfig:
    api_key: str
    model: str = "gpt-realtime"
    voice: str = "ash"
    instructions: str = (
        "Du er Reachy, en norsk labassistent på Læringslabben ved HVL. Du hjelper brukere av labben med utstyr og spørsmål knyttet til læringslabben."
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
    temperature: float = 0.8
    max_output_tokens: str | int = "inf"

    @classmethod
    def from_env(cls) -> "RealtimeConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the realtime pipeline.")

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
                "You are Reachy, a concise Norwegian-speaking lab assistant for the Learning Lab.",
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
            temperature=float(os.getenv("OPENAI_REALTIME_TEMPERATURE", "0.8")),
            max_output_tokens=max_output_tokens,
        )

    @property
    def chunk_frames(self) -> int:
        return max(1, int(self.input_sample_rate * self.chunk_ms / 1000))
