from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from reachy_assistant.realtime.audio import REALTIME_SAMPLE_RATE, from_base64_pcm16, to_base64_pcm16
from reachy_assistant.realtime.client import OpenAIRealtimeClient
from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.io_base import RealtimeAudioSink, RealtimeAudioSource


@dataclass
class RealtimeConversationEngine:
    config: RealtimeConfig
    source: RealtimeAudioSource
    sink: RealtimeAudioSink
    client: OpenAIRealtimeClient | None = None
    last_user_transcript: str = ""
    last_assistant_transcript: str = ""
    _stop_event: asyncio.Event = field(init=False, default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OpenAIRealtimeClient(self.config)

    async def run(self) -> None:
        await self.source.open()
        await self.sink.open(self.config.output_sample_rate)

        try:
            async with self.client.connect() as connection:
                await connection.session.update(
                    session={
                        "modalities": ["text", "audio"],
                        "instructions": self.config.instructions,
                        "voice": self.config.voice,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": self.config.transcription_model,
                            "language": self.config.language,
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": self.config.vad_threshold,
                            "silence_duration_ms": self.config.vad_silence_ms,
                            "prefix_padding_ms": self.config.vad_prefix_padding_ms,
                            "create_response": self.config.vad_create_response,
                            "interrupt_response": True,
                        },
                        "temperature": self.config.temperature,
                        "max_response_output_tokens": self.config.max_output_tokens,
                    }
                )

                send_task = asyncio.create_task(self._send_audio(connection))
                receive_task = asyncio.create_task(self._receive_events(connection))

                done, pending = await asyncio.wait(
                    {send_task, receive_task},
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
        finally:
            self._stop_event.set()
            await self.source.close()
            await self.sink.close()

    def stop(self) -> None:
        self._stop_event.set()

    async def _send_audio(self, connection) -> None:
        async for chunk in self.source.chunks():
            if self._stop_event.is_set():
                break
            payload = to_base64_pcm16(chunk, self.source.sample_rate, REALTIME_SAMPLE_RATE)
            await connection.input_audio_buffer.append(audio=payload)

    async def _receive_events(self, connection) -> None:
        async for event in connection:
            event_type = getattr(event, "type", "")

            if event_type == "response.audio.delta":
                audio, sample_rate = from_base64_pcm16(event.delta, REALTIME_SAMPLE_RATE)
                await self.sink.write(audio, sample_rate)

            elif event_type == "response.audio_transcript.delta":
                self.last_assistant_transcript += event.delta

            elif event_type == "response.audio_transcript.done":
                transcript = getattr(event, "transcript", "").strip()
                if transcript:
                    self.last_assistant_transcript = transcript
                    print(f"Assistant: {transcript}")

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = getattr(event, "transcript", "").strip()
                if transcript:
                    self.last_user_transcript = transcript
                    print(f"User: {transcript}")

            elif event_type == "error":
                error = getattr(event, "error", None)
                raise RuntimeError(f"Realtime API error: {error}")
