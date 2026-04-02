from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reachy_assistant.realtime.audio import REALTIME_SAMPLE_RATE, from_base64_pcm16, to_base64_pcm16
from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.io_base import RealtimeAudioSink, RealtimeAudioSource
from reachy_assistant.realtime.rag_tool import OpenAIRagRealtimeTool

if TYPE_CHECKING:
    from reachy_assistant.realtime.client import OpenAIRealtimeClient


@dataclass
class RealtimeConversationEngine:
    config: RealtimeConfig
    source: RealtimeAudioSource
    sink: RealtimeAudioSink
    client: "OpenAIRealtimeClient | None" = None
    motion: object | None = None
    rag_tool: OpenAIRagRealtimeTool | None = None
    last_user_transcript: str = ""
    last_assistant_transcript: str = ""
    _stop_event: asyncio.Event = field(init=False, default_factory=asyncio.Event)
    _active_response_id: str | None = field(init=False, default=None)
    _active_item_id: str | None = field(init=False, default=None)
    _active_content_index: int | None = field(init=False, default=None)
    _active_audio_end_ms: int = field(init=False, default=0)
    _interrupting_response_id: str | None = field(init=False, default=None)
    _response_create_pending: bool = field(init=False, default=False)
    _tool_calls: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.client is None:
            from reachy_assistant.realtime.client import OpenAIRealtimeClient

            self.client = OpenAIRealtimeClient(self.config)
        if self.rag_tool is None:
            self.rag_tool = OpenAIRagRealtimeTool.from_env(self.config.api_key)

    async def run(self) -> None:
        print("[realtime] Opening audio devices...")
        await self.source.open()
        await self.sink.open(self.config.output_sample_rate)
        self._motion_call("idle")

        try:
            print("[realtime] Connecting to OpenAI Realtime API...")
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
                            "create_response": (
                                self.config.vad_create_response
                                if self.config.allow_interruptions
                                else False
                            ),
                            "interrupt_response": self.config.allow_interruptions,
                        },
                        "temperature": self.config.temperature,
                        "max_response_output_tokens": self.config.max_output_tokens,
                        "tools": self._tool_definitions(),
                        "tool_choice": "auto",
                    }
                )
                print("[realtime] Session ready. Speak into the microphone. Press Ctrl+C to stop.")

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
                self._motion_call("speaking")
                self._active_response_id = getattr(event, "response_id", None)
                self._active_item_id = getattr(event, "item_id", None)
                self._active_content_index = getattr(event, "content_index", None)
                audio, sample_rate = from_base64_pcm16(event.delta, REALTIME_SAMPLE_RATE)
                await self.sink.write(audio, sample_rate)
                self._active_audio_end_ms += int(round(len(audio) * 1000 / sample_rate))

            elif event_type == "response.audio.done":
                self._clear_active_response()

            elif event_type == "response.created":
                response = getattr(event, "response", None)
                self._active_response_id = getattr(response, "id", None)
                self._response_create_pending = False

            elif event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", "") == "function_call":
                    call_id = getattr(item, "call_id", None)
                    name = getattr(item, "name", None)
                    if call_id and name:
                        self._tool_calls[call_id] = name

            elif event_type == "response.audio_transcript.delta":
                self.last_assistant_transcript += event.delta

            elif event_type == "response.audio_transcript.done":
                transcript = getattr(event, "transcript", "").strip()
                if transcript:
                    self.last_assistant_transcript = transcript
                    print(f"Assistant: {transcript}")
                self._motion_call("idle")

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = getattr(event, "transcript", "").strip()
                if transcript:
                    self.last_user_transcript = transcript
                    print(f"User: {transcript}")
                    self._motion_call("acknowledge")

            elif event_type == "input_audio_buffer.speech_started":
                print("[realtime] Speech detected.")
                if self.config.allow_interruptions:
                    await self._interrupt_assistant_response(connection)
                self._motion_call("listening")

            elif event_type == "input_audio_buffer.speech_stopped":
                print("[realtime] Processing speech...")
                if not self.config.allow_interruptions:
                    await self._request_response_if_idle(connection)
                self._motion_call("thinking")

            elif event_type == "response.done":
                self._clear_active_response()
                self._motion_call("idle")

            elif event_type == "response.function_call_arguments.done":
                await self._handle_function_call(connection, event)

            elif event_type == "error":
                error = getattr(event, "error", None)
                error_code = getattr(error, "code", None)
                if error_code == "response_cancel_not_active":
                    print("[realtime] Interrupt race: response was already stopped.")
                    self._clear_active_response()
                    continue
                if error_code == "conversation_already_has_active_response":
                    print("[realtime] Response race: server already has an active response.")
                    self._response_create_pending = False
                    continue
                raise RuntimeError(f"Realtime API error: {error}")

    def _motion_call(self, name: str) -> None:
        if self.motion is None or not hasattr(self.motion, name):
            return
        try:
            getattr(self.motion, name)()
        except Exception as exc:
            print(f"[realtime] Motion '{name}' failed: {exc}")

    async def _interrupt_assistant_response(self, connection) -> None:
        if self._active_response_id is None:
            if hasattr(self.sink, "clear"):
                await self.sink.clear()
            return

        if self._interrupting_response_id == self._active_response_id:
            return

        print("[realtime] Interrupting assistant response.")
        response_id = self._active_response_id
        self._interrupting_response_id = response_id
        await connection.response.cancel(response_id=response_id)

        if self._active_item_id is not None and self._active_content_index is not None:
            await connection.conversation.item.truncate(
                item_id=self._active_item_id,
                content_index=self._active_content_index,
                audio_end_ms=max(0, self._active_audio_end_ms),
            )

        if hasattr(self.sink, "clear"):
            await self.sink.clear()

        self._clear_active_response()

    def _clear_active_response(self) -> None:
        self._active_response_id = None
        self._active_item_id = None
        self._active_content_index = None
        self._active_audio_end_ms = 0
        self._interrupting_response_id = None
        self._response_create_pending = False

    async def _request_response_if_idle(self, connection) -> None:
        if self._active_response_id is not None or self._response_create_pending:
            return
        self._response_create_pending = True
        await connection.response.create()

    def _tool_definitions(self) -> list[dict[str, object]]:
        if self.rag_tool is None:
            return []
        return [self.rag_tool.definition()]

    async def _handle_function_call(self, connection, event) -> None:
        call_id = getattr(event, "call_id", None)
        arguments = getattr(event, "arguments", "") or "{}"

        if not call_id:
            return

        name = self._tool_calls.get(call_id, "")

        self._clear_active_response()

        if name != "openai_rag_search" or self.rag_tool is None:
            output = json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
        else:
            print("[realtime] Running tool: openai_rag_search")
            debug = await asyncio.to_thread(self.rag_tool.debug_summary, arguments)
            print(
                "[realtime][rag] "
                f"query={debug.get('query')!r} "
                f"category={debug.get('category')!r} "
                f"limit={debug.get('limit')} "
                f"results={debug.get('results_count')} "
                f"sources={debug.get('sources')}"
            )
            answer_preview = (debug.get("answer") or "")[:240]
            if answer_preview:
                print(f"[realtime][rag] answer={answer_preview}")
            if debug.get("error"):
                print(f"[realtime][rag] error={debug['error']}")
            output = await asyncio.to_thread(self.rag_tool.execute, arguments)
        self._tool_calls.pop(call_id, None)

        await connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        )
        await self._request_response_if_idle(connection)
